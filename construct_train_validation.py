from collections import defaultdict

import pandas as pd
import torch
from pres_gpt2 import GPTConfig 
import torch.nn.functional as F

import pickle as pkl
import tiktoken

with open("pres_tokenizer.pkl", "rb") as f:
    pres_enc = pkl.load(f)

tokenizer = pres_enc
samples = defaultdict(list)

pres_df = pd.read_csv('./presidential_speeches.csv', sep=';', encoding='utf-8', quotechar="'")

# remove duplicate spaces
pres_df['speech'] = pres_df['speech'].str.replace(r'\s+', ' ', regex=True)

pres_df['speech_tokenized'] = pres_df['speech'].apply(lambda x: tokenizer.encode(x))

pad_id = tokenizer.encode('<PAD>', allowed_special='all')[0]
president_encodings = {}

max_len = GPTConfig.block_size

def __pad_tensor(tensor, desired_length, pad_id):
    # Calculate padding (pad on the right)
    padding_length = desired_length - tensor.size(0)

    # Apply padding
    padded_tensor = F.pad(tensor, (0, padding_length), mode='constant', value=pad_id)
    
    return padded_tensor

for _, row in pres_df.iterrows():
    # Get the president's name
    president_name = row['President']
    
    # Retrieve or compute the encoding for the president
    if president_name not in president_encodings:
        president_encoding = tokenizer.encode(f'<President: {president_name}>', allowed_special='all')[0]
        president_encodings[president_name] = president_encoding
    else:
        president_encoding = president_encodings[president_name]
        
    speech_tokens = row['speech_tokenized']
    
    pres_tensors = samples[president_name]
    
    for i in range(0, len(speech_tokens), max_len-1):
        input = [president_encoding, *speech_tokens[i:min(i+max_len-1, len(speech_tokens))]]
        output = speech_tokens[i:min(i+max_len, len(speech_tokens))]
        
        pres_tensors.append([__pad_tensor(torch.tensor(input), max_len, pad_id), __pad_tensor(torch.tensor(output), max_len, pad_id)])

train = []
validation = []

tot_len = 0

for president, tensor_list in samples.items():
    train.extend(tensor_list[:int(0.8*len(tensor_list))])
    validation.extend(tensor_list[int(0.8*len(tensor_list)):])
    
    tot_len += len(tensor_list)
    
with open('train.pkl', 'wb') as f:
    pkl.dump(train, f)
    
with open('validation.pkl', 'wb') as f:
    pkl.dump(validation, f)