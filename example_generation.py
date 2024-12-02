import pickle
import tiktoken
import torch
import torch.nn.functional as F

from PresGPT2 import GPTConfig, PresGPT2

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

model: PresGPT2 = PresGPT2(config)

model.load_state_dict(checkpoint)

model.eval()

generation_len = 20

text = "<President: Barack Obama> Today marks the first day of"

text_tokens = torch.tensor(pres_enc.encode(text, allowed_special='all'))

text_tokens = torch.unsqueeze(text_tokens, 0)

for i in range(generation_len):
    logits, _ = model(text_tokens)
    
    logits = logits[:, -1, :] # B x T x vocab_size -> B x vocab_size
    
    # softmax over logits to get probabilities
    probs = F.softmax(logits, dim=1)
    
    max_idx = torch.argmax(probs, dim=1)
    
    max_idx = torch.unsqueeze(max_idx, 0)
    
    text_tokens = torch.cat((text_tokens, max_idx), dim=1)
    
for conv in list(text_tokens):
    print(pres_enc.decode(list(conv)))


