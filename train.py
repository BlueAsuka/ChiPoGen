import sys
sys.path.append('../')
from model import GPT

import os
import math
import json
import inspect
import wandb
import numpy as np
import time

from contextlib import nullcontext
from easydict import EasyDict
from colorama import Fore, Style

import torch
import torch.nn as nn


configs_dir = 'configs'
data_dir = 'data'
checkpoint_dir = 'checkpoint'
params_dir = 'params'
wandb_log = True


#==================================
# Read all configurations files
with open(os.path.join(configs_dir, 'model.json')) as f:
    model_config = json.load(f)
model_config = EasyDict(model_config)

with open(os.path.join(configs_dir, 'train.json')) as f:
    train_config = json.load(f)

# initialize the wandb    
if wandb_log:
    run = wandb.init(
        project='ChiPoGen',
        config=train_config
    )

train_config = EasyDict(train_config)


#===================================
# device and computation configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn


#===============================
# loading data
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# get batch from the data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    # Randomly select chunk of text for training
    ix = torch.randint(len(data) - model_config.block_size, (model_config.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+model_config.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+model_config.block_size].astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y


#=================================
# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
temp_val_loss = 1e9

# define model and optimization, use the init_from to control 
# 'scratch' : training from scratch
# 'resume'  : resume training from a checkpoint
gpt_model = GPT(model_config)
if train_config.init_from == 'scratch':
    # from scratch
    print('Initializing a new model from scratch')
elif train_config.init_from == 'resume':
    # from existed checkpoint file
    ckpt_path = os.path.join(checkpoint_dir, 'ckpt.pt')
    assert os.path.exists(ckpt_path), "The checkpoint dir is not exist, please train the model from scratch"
    print(f'Resuming training from {checkpoint_dir}')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gpt_model.load_state_dict(checkpoint['model'])
    iter_num = checkpoint['iter_num']
    temp_val_loss = checkpoint['temp_val_loss']
gpt_model.to(device)

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
        """ define the optimizer with weight decaying """
    
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # create AdamW optimizer and use the fused version if it is available
        # the foreach and fused implementations are typically faster than the for-loop, single-tensor implementation
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

optimizer = configure_optimizers(gpt_model, train_config.weight_decay, train_config.learning_rate, 
                                (train_config.beta1, train_config.beta2), device_type=device)
if train_config.init_from == 'resume':
    # load the optimizer from the existed checkpoint file
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None


#=================================
# utils functions
@torch.no_grad()
def estimate_loss():
    """ evaluating the model on train and val set after a training period """
    
    out = {}
    gpt_model.eval() # set the model in evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = gpt_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    gpt_model.train() # set the model in training mode
    return out

def get_lr(it):
    """ use cosine learning rate decay to get learning rate """
    
    # 1. linear warmup for warmup_iters steps
    if it < train_config.warmup_iters:
        return train_config.learning_rate * it / train_config.warmup_iters
    # 2. if it > lr_decay_iters, return min learning rate
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    # 3. in between, use cosine decay down to min learning rate
    decay_ratio = (it - train_config.warmup_iters) / (train_config.lr_decay_iters - train_config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (train_config.learning_rate - train_config.min_lr)

def estimate_mfu(model, fwdbwd_per_iter, dt):  
    """ estimate model flops utilization (MFU) in units of GPU bfloat16 peak FLOPS """
     
    N = model.get_num_params()
    cfg = model_config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 22e12 # RTX 4070 GPU bfloat16 peak flops is 22.61 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu


#=========================================
# training initalization
X, Y = get_batch('train')
t0 = time.time()
# local_iter_num = 0

# the training loop
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % train_config.eval_interval == 0:
        losses = estimate_loss()
        print('==================================')
        print(Fore.GREEN + f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(Style.RESET_ALL)
        
        # Use wandb to log the process
        if wandb_log:
            wandb.log({
                "val/loss": losses['val'],
            })        
        
        # save the current result to the checkpoints
        temp_val_loss = losses['val'] # update the val loss
        if iter_num > 0:
            checkpoint = {
                'model': gpt_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'temp_val_loss': temp_val_loss,
                'config': model_config,
            }
            print(f'saving checkpoint to {checkpoint_dir}')
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'ckpt.pth'))
            print('Done')
        
    for micro_step in range(train_config.gradient_accumulation_steps):
        with ctx:
            logits, loss = gpt_model(X, Y)
            # scale the loss to account for gradient accumulation
            loss = loss / train_config.gradient_accumulation_steps
            
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    
    # clip the gradient
    if train_config.grad_clip != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(gpt_model.parameters(), train_config.grad_clip)
        
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True) # release the gradients 
    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % train_config.log_interval == 0:
        lossf = loss.item() * train_config.gradient_accumulation_steps
        # if local_iter_num >= 5:
            # mfu = estimate_mfu(gpt_model, model_config.batch_size * train_config.gradient_accumulation_steps, dt)
            # running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*train_config.log_interval:.2f}s")
        
        # log the training loss
        if wandb_log:
            wandb.log({
                "train/loss": lossf,
            })    
        
    # update iters num
    iter_num += 1
    # local_iter_num += 1
    
    # terminate conditions
    if iter_num > train_config.max_iters:
        break


# =============================
# save the final result
# the file name include filename 'model', the size of the model, a random number and a suffix .pth 
model_params_name = 'model' + str(int(gpt_model.get_num_params() // 1e6)) + 'M-' + str(np.random.randint(0, 65536)) + '.pth'
print(f"saving model")
torch.save(gpt_model.state_dict(), os.path.join(params_dir, model_params_name))
print('save model successfully')