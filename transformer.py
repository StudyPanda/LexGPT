import wandb
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
from dataloader import get_tokenizer, text, vocab_size

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config.yaml')


#training hyperparameters
batch_size = config['training']['batch_size'] #how many sequences to process in parallel
block_size = config['training']['block_size'] #maxlength of a sequence 
max_iters = config['training']['max_iters']
learning_rate = config['training']['learning_rate']
eval_interval = config['training']['eval_interval']
eval_iters = config['training']['eval_iters']

#model hyperparameters
embed_dim = config['model']['embed_dim']
num_heads = config['model']['num_heads']
num_decoder_blocks = config['model']['num_decoder_blocks']
dropout = config['model']['dropout']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(54)
'''
#import Lex podcast data
file_name = 'Lex.csv'
df = pd.read_csv(file_name)
text = df['text'].str.cat(sep='')

#we use character level tokens
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)} #string to int
itos = { i:ch for i,ch in enumerate(chars)} #into to string
encode = lambda s: [stoi[c] for c in s] #input string, output list of ints
decode = lambda l: ''.join([itos[i] for i in l]) #input list of ints, output string

#encode entire text and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype = torch.long)

#separate into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

#Dataloader
def get_batch(split):
  #generate a small batch of data of inputs x and target y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x, y
'''


class Head(nn.Module):
  def __init__(self, head_dim):
    super().__init__()
    self.head_dim = head_dim
    self.key = nn.Linear(embed_dim, head_dim, bias = False)
    self.query = nn.Linear(embed_dim, head_dim, bias = False)
    self.value = nn.Linear(embed_dim, head_dim, bias = False)
    self.register_buffer('tril_mask', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout) 
  
  def forward(self, x):
    B,T,C = x.shape #Channel is d_k, the dimension of head_space
    k = self.key(x) #(B, T, head_space)
    q = self.query(x) #(B, T, head_space)
    v = self.value(x) #(B, T, head_space)

    #well we want (B, T, T) weight vector, for each token in the sequence,
    #we want to know for much attention it should pay to other tokens
    #so therefore, attn = softmax(Q K.t)...
    wei = q @ k.transpose(-2,-1)
    wei = wei * self.head_dim**(-0.5)
    wei = wei.masked_fill(self.tril_mask[:T,:T] ==0, float('-inf')) #(B, T , T)
    wei = F.softmax(wei, dim = -1) #(B, T , T)
    wei = self.dropout(wei)

    out = wei @ v
    return out

class MultiHead(nn.Module):
  def __init__(self, num_heads):
    super().__init__()
    self.num_heads = num_heads
    assert embed_dim % num_heads == 0, "Number of heads and embedding dimension incompatible"
    self.head_dim = embed_dim // num_heads
    self.heads = nn.ModuleList([Head(self.head_dim) for _ in range(num_heads)])
    self.W_o = nn.Linear(num_heads * self.head_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)


  def forward(self, x):
    head_outs = [head(x) for head in self.heads] #(B, T, head_dim) head_dim = embed_dim/num_heads
    concatenated = torch.cat(head_outs, dim = -1) #concatenate outputs from heads by channel dimension
    out = self.W_o(concatenated) #(B, T, embed_dim)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.FF = nn.Sequential(
      nn.Linear(embed_dim, 4 * embed_dim), #internal dimension of feed forward is 4*d_model
      nn.ReLU(),
      nn.Linear(4* embed_dim, embed_dim), #projection back into embed_dim similar to W_o
      nn.Dropout(dropout)
    )
    
  def forward(self, x):
    return self.FF(x)

class Block(nn.Module):
  def __init__(self, num_heads):
    super().__init__()
    self.heads = MultiHead(num_heads)
    self.FF = FeedForward()
    self.ln1 = nn.LayerNorm(embed_dim) #normalization happens to the embedding_dimension
    self.ln2 = nn.LayerNorm(embed_dim) # (B,T) both act as batches and per token in normalized
  
  def forward(self, x):
    #pre-norm formulation rather than as in original paper
    x = x + self.heads(self.ln1(x))
    x = x + self.FF(self.ln2(x))
    return x

  

class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding_table = nn.Embedding(block_size, embed_dim)
    self.blocks = nn.Sequential(*[Block(num_heads) for _ in range(num_decoder_blocks)])
    self.ln = nn.LayerNorm(embed_dim)
    self.lm_head = nn.Linear(embed_dim, vocab_size)



  def forward(self, idx, targets = None):
    #idx and targets are 2D tensors of size (B,T)
    B,T = idx.shape
    token_embed = self.token_embedding_table(idx) #(B,T,C)
    pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)

    x = token_embed + pos_embed #(B, T, C) positional embeddings get broadcasted across batch
    x = self.blocks(x) #(B,T,C)
    x = self.ln(x)
    logits = self.lm_head(x) # (B,T, vocab_size)

    if targets == None:
      loss = None
    else:
      #reshape tensor so channel (logits) are the second dimension
      #which is what pytorch expects
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    #take idx (B,T) and generate max_new_tokens more tokens in time dimension
    for _ in range(max_new_tokens):
      idx_cond = idx[:,-block_size:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      #apply softmax to get probability distribution
      probs = F.softmax(logits, dim = -1)
      #sample from distribution
      idx_next = torch.multinomial(probs, num_samples = 1)
      idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
    return idx