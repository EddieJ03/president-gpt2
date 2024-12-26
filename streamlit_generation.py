import pickle
import tiktoken
import torch
import torch.nn.functional as F
import streamlit as st

from pres_gpt2 import PresGPT2, GPTConfig
from const import presidents

# Load tokenizer and model configuration
@st.cache_resource
def load_model():
    with open("./tokenizer/pres_tokenizer.pkl", "rb") as f:
        pres_enc: tiktoken.Encoding = pickle.load(f)

    config: GPTConfig = GPTConfig(
        1024,
        len(pres_enc._mergeable_ranks) + len(pres_enc._special_tokens),
        12,
        12,
        768,
        0.0
    )

    checkpoint = torch.load('./model/checkpoint.pt', map_location=torch.device("cpu"))  # Adjust device if needed

    # Initialize model
    model: PresGPT2 = PresGPT2(config)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, pres_enc

# Function to stream text generation
def stream_text(model: PresGPT2, pres_enc: tiktoken.Encoding, president: str, input_text: str, top_k: int, temperature: float):
    input_text = input_text.strip()
    
    prompt_text = f"<President: {president}> {input_text}"

    # Tokenize the input text
    text_tokens = torch.tensor(pres_enc.encode(prompt_text, allowed_special='all'))
    text_tokens = torch.unsqueeze(text_tokens, 0)
    
    final_token = ''
    additional_tokens = []
    
    with torch.no_grad():
        # Placeholder for streaming
        output_placeholder = st.empty()
        current_text = input_text

        # Generate text
        while True:
            logits, _ = model(text_tokens)
            logits = logits[:, -1, :] / temperature  # B x T x vocab_size -> B x vocab_size
            values, _ = torch.topk(logits, top_k)  # values is descending
            logits[logits < values[:, [-1]]] = -float('Inf')  # Set low-probability logits to negative infinity

            # Softmax over logits to get probabilities
            probs = F.softmax(logits, dim=1)

            # Introduce variability in the generated text
            next_idx = torch.multinomial(probs, num_samples=1)
            text_tokens = torch.cat((text_tokens, next_idx), dim=1)
            
            final_token = pres_enc.decode([next_idx.squeeze().tolist()])
            additional_tokens.append(final_token)
            
            finished = ('.' in final_token or '!' in final_token or '?' in final_token)
            
            if finished:
                last_index = max(final_token.rfind('.'), final_token.rfind('!'), final_token.rfind('?'))
                if last_index+1 < len(final_token) and final_token[last_index+1] == '"':
                    final_token = final_token[:last_index + 2]
                else:
                    final_token = final_token[:last_index + 1]

            # Update the streaming text
            current_text = current_text + final_token
            output_placeholder.text_area("Generated Text (Streaming)", value=current_text, height=200, disabled=True)
            
            if finished:
                break

    return current_text

def main():
    model, encoder = load_model()

    # Streamlit app setup
    st.title("PresGPT2 Text Generator")
    st.write("Have a president complete a sentence for you!")

    # Sidebar for inputs
    president = st.selectbox("Select a President", presidents)
    input_text = st.text_area("Enter a prompt", value="", height=100)
    top_k = st.slider("Top-k", min_value=1, max_value=500, value=100)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, step=0.1, value=0.9)

    # Generate button
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            stream_text(model, encoder, president, input_text, top_k, temperature)
    
if __name__ == "__main__":
    main()
