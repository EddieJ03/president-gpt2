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

generation_len = 100

top_k = 200

temperature = 0.8 

text = "<President: Donald Trump> Today marks the first day of"

text_tokens = torch.tensor(pres_enc.encode(text, allowed_special='all'))

text_tokens = torch.unsqueeze(text_tokens, 0)

for i in range(generation_len):
    logits, _ = model(text_tokens)
    
    logits = logits[:, -1, :] / temperature # B x T x vocab_size -> B x vocab_size
    
    values, _ = torch.topk(logits, top_k) # values is descending
    logits[logits < values[:, [-1]]] = -float('Inf') # for all possible logtis for a sequence, if less than smallest topk set to negative inf
    
    # softmax over logits to get probabilities
    probs = F.softmax(logits, dim=1) # dim=1 means we compute softmax over COLUMNS IN A ROW!!
    
    # introduce a little variability in the generated text
    next_idx = torch.multinomial(probs, num_samples=1)
    
    text_tokens = torch.cat((text_tokens, next_idx), dim=1)
    
for conv in list(text_tokens):
    print(pres_enc.decode(list(conv)))


