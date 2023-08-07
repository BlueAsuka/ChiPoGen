"""
Tokenization
Train-val dataset spliting
Saves the poetry training and valuating dataset to a binary file 
"""

import numpy as np
import pickle
import os


data_dir = ('data/raw_data/poetry.txt')
with open(data_dir, encoding='utf-8') as f:
    poetry_text = f.read()

# get all unique chars in the Chinese poetry dataset, there are total 12966 unique Chinese chars
# this method tokenizes the text by char-level spliting, so each Chinese char is a token for training and generation
# TODO: investigating tokenization methods such as subword-level tokenlization for Chinese text 
tokens = sorted(list(set(poetry_text)))

# check whether the tokens folder exists
if not os.path.exists('data/tokens'):
    os.makedirs('data/tokens')

# then save the tokens into the token.pkl
with open('data/tokens/tokens.pkl', 'wb') as f:
    pickle.dump(tokens, f)

# the dictionary for interger and chars mapping
stoi = { w:i for i,w in enumerate(tokens)}
itos = { i:w for i,w in enumerate(tokens)}

# the decoding and encoding 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode all poetry text 
data = encode(poetry_text)
n = len(data)

# split the text 
train_val_ratio = 0.9
train_data = data[:int(train_val_ratio * n)]
val_data = data[int(train_val_ratio * n):]

# save the encoded trained and val poetry into a binary file
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
print('Saving the .bin file successfully...')
train_ids.tofile('data/train.bin')
val_ids.tofile('data/val.bin')
print('Done')