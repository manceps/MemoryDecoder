# **Memory-Augmented Language Model**

This project is a demo implementation of the concepts presented in the paper **"The Case for a Deep-Learning-Based Memory Decoder"**. The core objective is to enhance the capabilities of a pre-trained language model by augmenting it with a non-parametric, external memory. This hybrid architecture allows the model to retrieve and incorporate information from a specific knowledge base, producing more informed and grounded responses.

### **1\. Implementation and Architectural Choices**

The system is composed of two main components that work together at inference time: a base language model and a separate memory decoder.

#### **Base Language Model (Gemma-2b)**

We use the Gemma-2b model as the foundation for text generation. Its role is to provide a strong base for generating coherent and contextually relevant text. We extract its hidden states, which serve as the query vectors for our memory system.

#### **Memory Decoder (MemoryDecoder class)**

The MemoryDecoder is a small neural network trained to predict the next token distribution based on a query hidden state from the base model. The architecture is a key element of the system's performance:

* **Deep Transformer Stack:** It uses a stack of **6 Transformer Encoder layers**. This design choice allows the decoder to learn more complex, hierarchical mappings between the hidden state and the vocabulary space.  
* With **32 attention heads**, the model can attend to different aspects of the input vector simultaneously, capturing a rich set of relationships and improving its ability to learn from the datastore.  
* **Layer Normalization:** Layer normalization is applied to stabilize and regularize the training process, which is crucial for the performance of a deeper network.

#### **External Memory (Datastore)**

The external memory is a non-parametric datastore created from a specific domain (in this case, a legal Q\&A dataset).

* **FAISS Index:** We use the FAISS library for building an efficient k-nearest neighbors index. This allows us to quickly search for the most relevant memories (hidden states) for a given query, a critical step for real-time inference.  
* **Key-Value Pairs:** The datastore is built by encoding a dataset where keys are hidden states from the base model and values are the corresponding next-token IDs.

#### **Hybrid Generation**

During text generation, the final output is a blend of the base model's predictions and the memory decoder's predictions. The alpha parameter, which can be configured at runtime, controls the influence of the memory decoder:

final\_logits=(1−α)⋅base\_logits+α⋅decoder\_logits  

This approach ensures that the model can gracefully fall back to its base knowledge when the memory is not relevant, while still being able to rely heavily on the datastore for domain-specific tasks.

#### **Sampling Strategy**

To generate complete and varied responses, the inference script uses a combination of **temperature sampling** and **top-k sampling** instead of simple greedy decoding. This adds a controlled amount of randomness, preventing the model from getting stuck in repetitive loops and enabling more natural-sounding text.

### **2\. Configurable Parameters**

| Parameter | Script | Description |
| :---- | :---- | :---- |
| NUM\_EPOCHS | train\_memory\_decoder.py | The number of times to iterate over the dataset during training. |
| DECODER\_LR | train\_memory\_decoder.py | The learning rate for the memory decoder. |
| DECODER\_DIM | train\_memory\_decoder.py | The hidden dimension of the memory decoder's layers. |
| K | train\_memory\_decoder.py & app.py | The number of nearest neighbors to retrieve from the datastore. |
| alpha | app.py | The interpolation factor between the base model and the memory decoder. |
| temperature | app.py | Controls the randomness of the output during generation. |
| top\_k | app.py | The number of top tokens to consider during sampling. |

### **3\. Setup and Usage**

To run this project, you need to set up a Python virtual environment and install the necessary libraries.

#### **Step 1: Clone this repo**

git clone https://github.com/manceps/MemoryDecoder.git
cd MemoryDecoder

#### **Step 2: Set up Virtual Environment and Install Libraries**

Create a virtual environment, activate it, and install the required packages.

python3 \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`  
pip install \-r requirements.txt

#### **Step 3: Train the Memory Decoder**

Before you can run the application, you must train the memory decoder and create the datastore. This will generate the necessary files (memory\_decoder.pt, faiss\_index.bin, and datastore\_values.npy).

python train\_memory\_decoder.py

#### **Step 4: Run the Application**

Once the training is complete, you can launch the Gradio web interface.

python app.py

The application will start, and you can access the chatbot interface in your web browser.