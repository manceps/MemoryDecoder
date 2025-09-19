import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import faiss
import numpy as np
from tqdm import tqdm
import random
import os
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# Setup and Configuration
# ==============================================================================

# Seed for reproducibility
SEED = 501
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model and tokenizer setup
MODEL_NAME = 'google/gemma-2b'
MAX_SEQ_LENGTH = 128
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Training parameters
NUM_EPOCHS = 3
DECODER_LR = 1e-4
BATCH_SIZE = 16
DECODER_DIM = 256 # Must be a multiple of the number of heads (32)
VOCAB_SIZE = tokenizer.vocab_size

# kNN parameters
K = 32
INDEX_METRIC = faiss.METRIC_L2

# Dataset configuration
DATASET_NAME = "ibunescu/qa_legal_dataset_train"
DATASET_SPLIT = "train"

# File paths for saving
SAVE_PATH = "./trained_models"
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "memory_decoder.pt")
INDEX_SAVE_PATH = os.path.join(SAVE_PATH, "faiss_index.bin")
VALUES_SAVE_PATH = os.path.join(SAVE_PATH, "datastore_values.npy")
os.makedirs(SAVE_PATH, exist_ok=True)

# ==============================================================================
# Custom Dataset for DataLoader
# ==============================================================================

class DatastoreDataset(Dataset):
    """
    A PyTorch Dataset to wrap the keys and values from the datastore.
    """
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        return {
            'key': torch.from_numpy(self.keys[idx]).float(),
            'value': torch.tensor(self.values[idx]).long()
        }

# ==============================================================================
# Data and Datastore Generation
# ==============================================================================

def create_datastore_from_dataset(dataset_name: str, split: str):
    """
    Generates a datastore by processing a real-world dataset.
    This version processes the entire dataset.
    """
    print(f"Loading and processing entire dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}. Cannot proceed without a dataset.")
        return None, None

    def process_example(example):
        text = f"Title: {example['Title']} Question: {example['Question']} Answer: {example['Answer']}"
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True, padding=True)

        if inputs.input_ids.shape[1] < 2:
            return {'key': None, 'value': None}

        with torch.no_grad():
            outputs = base_model.model(inputs.input_ids.to(DEVICE), output_hidden_states=True)
            # The key is the hidden state before the last token
            key = outputs.hidden_states[-1][0, -2, :].cpu().numpy()
            # The value is the last token itself
            value = inputs.input_ids[0, -1].item()
            return {'key': key, 'value': value}

    processed_dataset = dataset.map(
        process_example,
        desc="Processing dataset samples",
        cache_file_name=f'datastore_cache_{dataset_name.replace("/", "_")}_{split}_full.arrow'
    )

    valid_examples = processed_dataset.filter(lambda x: x['key'] is not None)
    keys = np.stack(valid_examples['key'])
    values = np.array(valid_examples['value'])

    return keys, values

# ==============================================================================
# Memory Decoder Model
# ==============================================================================

class MemoryDecoder(nn.Module):
    """
    The Memory Decoder model with a transformer stack.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Set attention heads to 32 and number of layers to 6
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=32, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.norm(x) # Apply LayerNorm
        x = self.transformer_encoder(x)
        x = self.output_projection(x.squeeze(1))
        return x

# ==============================================================================
# Training
# ==============================================================================

def train_memory_decoder(decoder, faiss_index, dataloader, num_epochs):
    """
    Trains the Memory Decoder model using a DataLoader.
    """
    print("Training Memory Decoder...")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=DECODER_LR)
    
    for epoch in range(num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            batch_keys_tensor = batch['key'].to(DEVICE)
            true_tokens = batch['value'].to(DEVICE)

            # Get nearest neighbors from FAISS index
            batch_keys_np = batch_keys_tensor.cpu().numpy()
            distances, indices = faiss_index.search(batch_keys_np, k=K)

            # Build kNN distributions
            knn_distributions = np.zeros((len(true_tokens), VOCAB_SIZE))
            datastore_values = dataloader.dataset.values
            for j in range(len(true_tokens)):
                for neighbor_idx in indices[j]:
                    token_id = datastore_values[neighbor_idx]
                    knn_distributions[j, token_id] += 1
            knn_distributions = knn_distributions / (knn_distributions.sum(axis=1, keepdims=True) + 1e-9)
            knn_distributions_tensor = torch.from_numpy(knn_distributions).float().to(DEVICE)
            
            # Get decoder logits
            decoder_logits = decoder(batch_keys_tensor)
            decoder_log_probs = F.log_softmax(decoder_logits, dim=-1)
            
            # Compute hybrid loss
            kl_loss = F.kl_div(decoder_log_probs, knn_distributions_tensor, reduction='batchmean', log_target=False)
            ce_loss = F.cross_entropy(decoder_logits, true_tokens)
            
            loss = kl_loss + ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(
                total_loss=f"{loss.item():.4f}",
                kl_loss=f"{kl_loss.item():.4f}",
                ce_loss=f"{ce_loss.item():.4f}"
            )

# ==============================================================================
# Main Execution and Saving
# ==============================================================================

if __name__ == "__main__":
    print("Starting Memory Decoder training pipeline...")
    
    # Create the domain datastore from the entire dataset
    datastore_keys, datastore_values = create_datastore_from_dataset(
        dataset_name=DATASET_NAME,
        split=DATASET_SPLIT
    )
    
    if datastore_keys is None:
        print("Datastore creation failed. Exiting.")
        exit()

    print(f"Datastore created with {len(datastore_values)} samples.")
    
    if len(datastore_keys) == 0:
        print("Datastore is empty. Cannot build FAISS index or train the model. Exiting.")
    else:
        actual_key_dim = datastore_keys.shape[1]
        print(f"Actual key dimension from datastore: {actual_key_dim}")
        
        # Build a FAISS index
        print("Building FAISS index...")
        faiss_index = faiss.IndexFlatL2(actual_key_dim)
        faiss_index.add(datastore_keys)
        print(f"FAISS index built with {faiss_index.ntotal} vectors.")

        # Create DataLoader for batching
        datastore_dataset = DatastoreDataset(datastore_keys, datastore_values)
        dataloader = DataLoader(datastore_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
        # Initialize and train the Memory Decoder
        memory_decoder = MemoryDecoder(input_dim=actual_key_dim, hidden_dim=DECODER_DIM, output_dim=VOCAB_SIZE).to(DEVICE)
        train_memory_decoder(memory_decoder, faiss_index, dataloader, num_epochs=NUM_EPOCHS)
        
        # Save the trained model and FAISS index
        print(f"\nSaving trained model to {MODEL_SAVE_PATH}")
        torch.save(memory_decoder.state_dict(), MODEL_SAVE_PATH)
        print(f"Saving FAISS index to {INDEX_SAVE_PATH}")
        faiss.write_index(faiss_index, INDEX_SAVE_PATH)
        print(f"Saving datastore values to {VALUES_SAVE_PATH}")
        np.save(VALUES_SAVE_PATH, datastore_values)

        print("\nTraining complete. Files saved for inference.")
