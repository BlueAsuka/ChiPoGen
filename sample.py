import os
import json
import pickle
from contextlib import nullcontext
from easydict import EasyDict
from colorama import Fore, Style

import torch
import torch.nn.functional as F

import sys
sys.path.append('../')
from model import GPT


# ===============================
# The file path for reading configuration files and trained weights
configs_dir = 'configs'
data_dir = 'data'
params_dir = 'params'
save_dir = 'out'
if_save = True
if_seed = True # reproduciable
if_prompt = True # if not prompt then input from the console, else read from the sample.json default is '\n'


# =================================
# Read model, sampling and tokens files
with open(os.path.join(configs_dir, 'sample.json'), 'r') as f:
    sample_config = json.load(f)
sample_config = EasyDict(sample_config)

with open(os.path.join(configs_dir, 'model.json'), 'r') as f:
    model_config = json.load(f)
model_config = EasyDict(model_config)

with open(os.path.join(data_dir, 'tokens.pkl'), 'rb') as f:
    tokens = pickle.load(f)


# ============================
# configurate for generation 
if if_seed:
    torch.manual_seed(sample_config.seed)
    torch.cuda.manual_seed(sample_config.seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# =============================
# Load the trained model params 
# list all .pth/.pt files in the params folder
pth_count = 0
for file in os.listdir(params_dir):
    if file.endswith('.pth') or file.endswith('.pt'):
        print(Fore.GREEN + file)
        pth_count += 1
        print(Style.RESET_ALL)
if pth_count > 0:
    print(f"{pth_count} .pth/.pt found, can be used for generation, select one for generation: ", end='')
else:
    print("No avaliable .pth file for generation.")
checkpoint_file = input()
checkpoint = torch.load(os.path.join(params_dir, checkpoint_file), map_location=device)
gpt_model = GPT(model_config)
state_dict = checkpoint['model']
gpt_model.load_state_dict(state_dict)
gpt_model.to(device)

# interger and chars mapping with encoding and decoding methods
stoi = { w:i for i,w in enumerate(tokens)}
itos = { i:w for i,w in enumerate(tokens)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the start char defined in the sample configuration files '/n'
start_ids = encode(sample_config.start)
if if_prompt:
    print("input for prompting: ", end='')
    prompt = input()
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device).view(-1, 1)
else:    
    x = torch.tensor(start_ids, dtype=torch.long, device=device).view(-1, 1)


# =============================
# define the generation method
gpt_model.eval()
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature, top_k):
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model_config.block_size else idx[:, -model_config.block_size:]
        # forward the model to get the logits
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature # pluck the logits at the final step and scale by desired temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

print('=======================')
print('Chinese poetry generating...')
with torch.no_grad():
    with ctx:
        y = generate(gpt_model, x, 
                     sample_config.max_new_tokens, 
                     sample_config.temperature,
                     sample_config.top_k)
        ids = y[0].tolist()
        res = decode(ids)
        print(res)

# save the generated text
if if_save:
    print('Saving the generted text...')
    with open(os.path.join(save_dir, 'generated_poetry.txt',), 'w', encoding='utf-8') as f:
        f.write(res) 
    print('Done')