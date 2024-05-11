import wandb
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

#initialise wandb project
wandb.login()
wandb.init(project='Transformer2_Lex')

#hyperparameters
batch_size = 32 #how many sequences to process in parallel
block_size = 8 #max length of a sequence
max_iters = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 200
embed_dim = 32
num_heads = 4

torch.manual_seed(54)

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

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for iter in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      losses[iter] = loss
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  def __init__(self, head_dim):
    super().__init__()
    self.head_dim = head_dim
    self.key = nn.Linear(embed_dim, head_dim, bias = False)
    self.query = nn.Linear(embed_dim, head_dim, bias = False)
    self.value = nn.Linear(embed_dim, head_dim, bias = False)
    self.register_buffer('tril_mask', torch.tril(torch.ones(block_size, block_size))) 
  
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


  def forward(self, x):
    head_outs = [Head(x) for Head in self.heads] #(B, T, head_dim) head_dim = embed_dim/num_heads
    concatenated = torch.cat(head_outs, dim = -1) #concatenate outputs from heads by channel dimension
    out = self.W_o(concatenated) #(B, T, embed_dim)
    return out
    
  

class TransformerDecoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
    self.position_embedding_table = nn.Embedding(block_size, embed_dim)
    self.heads = MultiHead(num_heads)
    self.lm_head = nn.Linear(embed_dim, vocab_size)



  def forward(self, idx, targets = None):
    #idx and targets are 2D tensors of size (B,T)
    B,T = idx.shape
    token_embed = self.token_embedding_table(idx) #(B,T,C)
    pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #(T,C)

    x = token_embed + pos_embed #(B, T, C) positional embeddings get broadcaster across batch
    x = self.heads(x) #(B,T,C)
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
  
#initialise model and train
model = TransformerDecoder()
model = model.to(device)

#create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
  
  #evaluate the loss on train and val set
  if iter % eval_interval == 0:
    losses = estimate_loss()
    wandb.log(losses)
    print(f"step {iter}, training loss is {losses['train']:.4f}, validation loss is {losses['val']:.4f}")

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()


#generate from model
idx = torch.zeros((1,1), dtype = torch.long)
#idx[0] = len(itos)-1

print(decode(model.generate(idx, max_new_tokens=50)[0].tolist()))

