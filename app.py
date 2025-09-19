import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np
import os

# ==============================================================================
# Setup and Configuration
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = 'google/gemma-2b'
MAX_SEQ_LENGTH = 128
DECODER_DIM = 256

# File paths for loading
SAVE_PATH = "./trained_models"
MODEL_SAVE_PATH = os.path.join(SAVE_PATH, "memory_decoder.pt")
INDEX_SAVE_PATH = os.path.join(SAVE_PATH, "faiss_index.bin")
VALUES_SAVE_PATH = os.path.join(SAVE_PATH, "datastore_values.npy")

# ==============================================================================
# Model and Datastore Loading
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

def load_models_and_datastore():
    """
    Loads all necessary components for inference.
    """
    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    
    print("Loading FAISS index and datastore...")
    if not os.path.exists(INDEX_SAVE_PATH) or not os.path.exists(VALUES_SAVE_PATH):
        raise FileNotFoundError(
            "FAISS index or datastore values not found. "
            "Please run the training script first."
        )
    
    faiss_index = faiss.read_index(INDEX_SAVE_PATH)
    datastore_values = np.load(VALUES_SAVE_PATH, allow_pickle=True)
    
    # Get the actual key dimension from the FAISS index
    actual_key_dim = faiss_index.d
    
    print("Loading memory decoder...")
    memory_decoder = MemoryDecoder(
        input_dim=actual_key_dim, 
        hidden_dim=DECODER_DIM, 
        output_dim=tokenizer.vocab_size
    ).to(DEVICE)
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(
            "Memory decoder model not found. "
            "Please run the training script first."
        )
    memory_decoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    memory_decoder.eval()
    
    return tokenizer, base_model, memory_decoder, faiss_index, datastore_values

# Load everything once at the start of the application
try:
    TOKENIZER, BASE_MODEL, MEMORY_DECODER, FAISS_INDEX, DATASTORE_VALUES = load_models_and_datastore()
    LOADED = True
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure you have run `train_memory_decoder.py` to generate the necessary files.")
    LOADED = False

# ==============================================================================
# Inference Function for Gradio
# ==============================================================================

def generate_text_from_memory(prompt, alpha, temperature, top_k, history):
    """
    Generates text by interpolating between the base model and the memory decoder.
    This function is designed to be used with Gradio's ChatInterface.
    """
    if not LOADED:
        return "Error: Model files not found. Please run the training script first."

    # History is a list of tuples (user_message, bot_message)
    # We combine the past conversation to form the full prompt
    full_prompt = ""
    for user_msg, bot_msg in history:
        full_prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    full_prompt += f"User: {prompt}\nAssistant: "
    
    input_ids = TOKENIZER.encode(full_prompt, return_tensors="pt").to(DEVICE)

    # Use a state for the generated output
    generated_output = ""
    
    with torch.no_grad():
        for i in range(50):  # Generate up to 50 new tokens
            # Get logits from the base model
            base_model_outputs = BASE_MODEL(input_ids)
            base_logits = base_model_outputs.logits[:, -1, :]
            
            # Get the hidden state for the last token to use as a query
            last_hidden_state = BASE_MODEL.model(input_ids, output_hidden_states=True).hidden_states[-1][0, -1, :].cpu().numpy()
            
            # Search for the k-nearest neighbors in the FAISS index
            K = 32
            _, indices = FAISS_INDEX.search(np.expand_dims(last_hidden_state, axis=0), k=K)
            
            # Create a distribution from the neighbors' values
            knn_distribution = np.zeros(TOKENIZER.vocab_size, dtype=np.float32)
            for neighbor_idx in indices[0]:
                token_id = DATASTORE_VALUES[neighbor_idx]
                knn_distribution[token_id] += 1
            knn_distribution /= np.sum(knn_distribution) + 1e-9
            
            knn_logits = torch.from_numpy(knn_distribution).unsqueeze(0).to(DEVICE)
            
            # Pass the hidden state through the memory decoder to get logits
            decoder_logits = MEMORY_DECODER(torch.from_numpy(last_hidden_state).unsqueeze(0).float().to(DEVICE))

            # Interpolate the logits
            # `alpha` controls the influence of the memory decoder
            final_logits = (1 - alpha) * base_logits + alpha * decoder_logits
            
            # Apply temperature
            logits_with_temp = final_logits / temperature
            
            # Apply top-k filtering
            top_k_tensor = torch.topk(logits_with_temp, top_k)
            filtered_logits = torch.full_like(logits_with_temp, -float('inf'))
            filtered_logits.scatter_(1, top_k_tensor.indices, top_k_tensor.values)
            
            # Sample the next token from the filtered distribution
            probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1).item()
            
            # Append the new token to the input_ids
            new_token_tensor = torch.tensor([[next_token_id]], device=DEVICE)
            input_ids = torch.cat([input_ids, new_token_tensor], dim=-1)
            
            # Decode the newly generated token and append to the output string
            new_text = TOKENIZER.decode(new_token_tensor[0], skip_special_tokens=True)
            generated_output += new_text

            # Check for end-of-sequence token to stop generation
            if new_token_tensor[0].item() == TOKENIZER.eos_token_id:
                break
    
    # Return the generated text, stripped of any extra whitespace
    return generated_output.strip()

# ==============================================================================
# Gradio Interface
# ==============================================================================

# Use a state for all generation parameters
params_state = gr.State({"alpha": 0.5, "temperature": 0.7, "top_k": 50})

def update_params(new_alpha, new_temp, new_k):
    """Updates the state variable for generation parameters."""
    return {"alpha": new_alpha, "temperature": new_temp, "top_k": new_k}

# Define a wrapper function for the chat interface
def wrapped_generate_fn(prompt, history, alpha, temperature, top_k):
    """
    A wrapper function to handle arguments for the chat interface.
    """
    return generate_text_from_memory(prompt, alpha, temperature, top_k, history)

with gr.Blocks(title="Memory Decoder Chat") as demo:
    gr.Markdown(
        """
        # Memory-Augmented Chatbot
        
        This application demonstrates a memory-augmented language model for legal Q&A. You can chat with the model and use the sliders below to control the influence of the external memory and the style of generation.
        The *google/gemma-2b* model is used as the base language model, and the memory decoder is trained with the *ibunescu/qa_legal_dataset_train* dataset.

        - **Alpha (Memory Influence)**: Control the blend between the base model and the memory decoder.
        - **Temperature**: Adjust the randomness of the output. Higher values lead to more creative, less predictable text.
        - **Top-K**: Limit the model's choices to a fixed number of tokens. A lower value makes the model more focused.
        
        Adjust the sliders and see how the answers change!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            alpha_slider = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                value=0.5, 
                step=0.01, 
                label="Alpha (Memory Influence)"
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature"
            )
            top_k_number = gr.Number(
                minimum=1,
                value=50,
                step=1,
                label="Top-K"
            )
            
        with gr.Column(scale=2):
            # The ChatInterface component uses the wrapper function and additional inputs
            chat_interface = gr.ChatInterface(
                fn=wrapped_generate_fn,
                additional_inputs=[alpha_slider, temperature_slider, top_k_number],
                examples=[
                    ["Does the man who tripped the Syrian Refugee have any legal basis to sue Twitter?", 0.5, 0.7, 50],
                    ["Landlord broke lease agreement, what are my rights?", 0.5, 0.7, 50],
                    ["I'm an uber driver who just got arrested for my passenger's drugs. what do I do?", 0.5, 0.7, 50]
                ],
            )
            
demo.launch()
