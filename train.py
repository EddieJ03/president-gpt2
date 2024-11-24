from pres_gpt2 import PresGPT, GPTConfig 
from Dataset import PresidentDataset

import pickle
import tiktoken

with open("pres_tokenizer.pkl", "rb") as f:
    pres_enc: tiktoken.Encoding = pickle.load(f)
    
with open("train.pkl", "rb") as f:
    train = pickle.load(f)
    
with open("validation.pkl", "rb") as f:
    validation = pickle.load(f)

train_dataset = PresidentDataset(train)
validation_dataset = PresidentDataset(validation)



