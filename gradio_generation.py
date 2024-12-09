import pickle
import tiktoken
import torch
import torch.nn.functional as F
import gradio as gr

from PresGPT2 import GPTConfig, PresGPT2

# Load tokenizer and model configuration
with open("pres_tokenizer.pkl", "rb") as f:
    pres_enc: tiktoken.Encoding = pickle.load(f)

config: GPTConfig = GPTConfig(
    1024,
    len(pres_enc._mergeable_ranks) + len(pres_enc._special_tokens),
    12,
    12,
    768
)

checkpoint = torch.load('./checkpoint.pt', map_location=torch.device("cpu"))  # Adjust device if needed

# Initialize model
model: PresGPT2 = PresGPT2(config)
model.load_state_dict(checkpoint)
model.eval()

# Function to generate text based on input
def generate_text(input_text: str, generation_len: int, top_k: int, temperature: float):
    # Tokenize the input text
    text_tokens = torch.tensor(pres_enc.encode(input_text, allowed_special='all'))
    text_tokens = torch.unsqueeze(text_tokens, 0)

    # Generate text
    for i in range(generation_len):
        logits, _ = model(text_tokens)
        
        logits = logits[:, -1, :] / temperature  # B x T x vocab_size -> B x vocab_size
        
        values, _ = torch.topk(logits, top_k)  # values is descending
        logits[logits < values[:, [-1]]] = -float('Inf')  # for all possible logits for a sequence, if less than smallest topk set to negative inf
        
        # Softmax over logits to get probabilities
        probs = F.softmax(logits, dim=1)  # dim=1 means we compute softmax over COLUMNS IN A ROW!!
        
        # Introduce variability in the generated text
        next_idx = torch.multinomial(probs, num_samples=1)
        
        text_tokens = torch.cat((text_tokens, next_idx), dim=1)
    
    # Decode the generated tokens into text
    generated_text = pres_enc.decode(text_tokens.squeeze().tolist())
    return generated_text

# Create Gradio interface
iface = gr.Interface(fn=generate_text, 
                     inputs=[
                         gr.Textbox(label="Enter a prompt", lines=2),
                         gr.Slider(minimum=10, maximum=200, value=100, label="Generation Length"),
                         gr.Slider(minimum=1, maximum=500, value=200, label="Top-k"),
                         gr.Slider(minimum=0.1, maximum=1.5, step=0.1, value=0.8, label="Temperature")
                     ],
                     outputs=gr.Textbox(label="Generated Text", lines=10),
                     title="PresGPT2 Text Generator",
                     description="Enter a prompt starting with <President: Name> to generate a presidential speech.")

# Launch the app
iface.launch()
