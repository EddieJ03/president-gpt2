import tiktoken
import pandas as pd
import pickle as pkl

pres_df = pd.read_csv('presidential_speeches.csv' , sep=';', encoding='utf-8', quotechar="'")

unique_presidents = pres_df['President'].value_counts().keys()

gpt2_base = tiktoken.get_encoding("gpt2")

special_tokens = dict()

for i, pres in enumerate(unique_presidents):
    special_tokens[f'<President: {pres}>'] = len(gpt2_base._mergeable_ranks)+i+1
    
pres_enc = tiktoken.Encoding(
    name="pres_encoding",
    pat_str=gpt2_base._pat_str,
    mergeable_ranks=gpt2_base._mergeable_ranks,
    special_tokens={**gpt2_base._special_tokens, **special_tokens}
)

# pickle the tokenizer obj
with open('pres_tokenizer.pkl', 'wb') as f:
    pkl.dump(pres_enc, f)

