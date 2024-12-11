import pickle
import tiktoken
import torch
import torch.nn.functional as F
import streamlit as st

from PresGPT2 import GPTConfig, PresGPT2

# Load tokenizer and model configuration
@st.cache_resource
def load_model():
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
    
    return model, pres_enc

# Function to generate text based on input
def generate_text(model: PresGPT2, pres_enc: tiktoken.Encoding, president: str, input_text: str, generation_len: int, top_k: int, temperature: float):
    prompt_text = f"<President: {president}> {input_text}"

    # Tokenize the input text
    text_tokens = torch.tensor(pres_enc.encode(prompt_text, allowed_special='all'))
    text_tokens = torch.unsqueeze(text_tokens, 0)

    with torch.no_grad():
        # Generate text
        for i in range(generation_len):
            logits, _ = model(text_tokens)
            logits = logits[:, -1, :] / temperature  # B x T x vocab_size -> B x vocab_size
            values, _ = torch.topk(logits, top_k)  # values is descending
            logits[logits < values[:, [-1]]] = -float('Inf')  # Set low-probability logits to negative infinity

            # Softmax over logits to get probabilities
            probs = F.softmax(logits, dim=1)

            # Introduce variability in the generated text
            next_idx = torch.multinomial(probs, num_samples=1)
            text_tokens = torch.cat((text_tokens, next_idx), dim=1)

    # Decode the generated tokens into text
    generated_text = pres_enc.decode(text_tokens.squeeze().tolist())
    
    return generated_text.replace(f"<President: {president}>", '')

def main():
    president_names = [
        "Abraham Lincoln", "Barack Obama", "Bill Clinton", "Donald Trump", "George Washington",
        "Harry S. Truman", "Joe Biden", "John F. Kennedy", "Lyndon B. Johnson", "Richard Nixon"
    ]
    
    model, encoder = load_model()

    # Streamlit app setup
    st.title("PresGPT2 Text Generator")
    st.write("Enter a prompt to generate something a president might say!")

    # Sidebar for inputs
    president = st.selectbox("Select a President", president_names)
    input_text = st.text_area("Enter a prompt", value="", height=100)
    generation_len = st.slider("Generation Length", min_value=10, max_value=200, value=100)
    top_k = st.slider("Top-k", min_value=1, max_value=500, value=200)
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.5, step=0.1, value=0.8)

    # Generate button
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            generated_text = generate_text(model, encoder, president, input_text, generation_len, top_k, temperature)
        st.text_area("Generated Text", value=generated_text, height=200, disabled=True)
    
if __name__ == "__main__":
    main()
