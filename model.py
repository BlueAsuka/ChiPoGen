import math

import torch
import torch.nn as nn
import torch.nn.functional as F




#======================================================
# All building blocks for the GPT: LayerNorm, Self Attention and MLP 
class LayerNorm(nn.Module):
    
    def __init__(self, n_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
    
    def forward(self, input):
        # Bias is not used in this model
        return F.layer_norm(input, self.weight.shape, self.weight, bias=None, eps=1e-5)

class SelfAttention(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "The remainder of embedding and head number should be zero."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.register_buffer('bias', torch.tril(torch.ones(self.block_size, self.block_size))
                                          .view(1, 1, self.block_size, self.block_size))
    
    def forward(self, x):
        B, T, C = x.shape # batch_size, sequence length (block_size), embedding dimension (n_embd)
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

#=======================================
# The GPT model
class GPT(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        # The transformer architecture
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            attn = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight tying
        # share the same weight between the input embedding and output layer
        self.transformer.wte.weight = self.lm_head.weight
        
        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                
         # report number of parameters in the model
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # if module.bias is not None:
            #     nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            # the position parameter will be subtracted 
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # the forward pass
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.attn:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # final outcome based on different mode
        if targets is not None: # in training mode
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1)
        else: # in the inference mode
            logits = self.lm_head(x[:, [-1], :]) # using list [-1] to preserve the time dim
            loss = None
        
        return logits, loss