import wandb
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F

#initialise wandb project
wandb.login()
wandb.init(project='Bigram_Lex')

#hyperparameters
batch_size = 32 #how many sequences to process in parallel
block_size = 8 #max length of a sequence
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
eval_iters = 200

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

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    #each embedding represents the logits for the next token
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets = None):
    #idx and targets are 2D tensors of size (B,T)
    logits = self.token_embedding_table(idx) #(B,T,C)

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
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      #apply softmax to get probability distribution
      probs = F.softmax(logits, dim = -1)
      #sample from distribution
      idx_next = torch.multinomial(probs, num_samples = 1)

      idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
    return idx
  
#initialise model and train
model = BigramLanguageModel(vocab_size)
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
print(decode(model.generate(idx, max_new_tokens=100)[0].tolist()))
