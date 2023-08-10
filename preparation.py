"""
Tokenization
Train-val dataset spliting
Saves the poetry training and valuating dataset to a binary file 
"""

import numpy as np
import pickle
import os
from easydict import EasyDict


print("Data preparing...")

data_dir = 'data'
with open(os.path.join(data_dir, 'poetry.txt'), encoding='utf-8') as f:
    poetry_text = f.read()

# get all unique chars in the Chinese poetry dataset, there are total 12966 unique Chinese chars
# this method tokenizes the text by char-level spliting, so each Chinese char is a token for training and generation
# TODO: investigating tokenization methods such as subword-level tokenlization for Chinese text 
tokens = sorted(list(set(poetry_text)))

# then save the tokens into the token.pkl
with open(os.path.join(data_dir, 'tokens.pkl'), 'wb') as f:
    pickle.dump(tokens, f)

# the dictionary for interger and chars mapping
stoi = { w:i for i,w in enumerate(tokens)}
itos = { i:w for i,w in enumerate(tokens)}

# the decoding and encoding 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode all poetry text 
data = encode(poetry_text)

# split the text 
n = len(data)
train_val_ratio = 0.9
train_data = data[:int(train_val_ratio * n)]
val_data = data[int(train_val_ratio * n):]

# save the encoded trained and val poetry into a binary file
train_ids = np.array(train_data, dtype=np.uint16)
val_ids = np.array(val_data, dtype=np.uint16)
print('Saving the .bin file successfully...')
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))
print('Done')