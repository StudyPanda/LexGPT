import wandb
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml
from dataloader import get_tokenizer, text, vocab_size
from transformer import Transformer

torch.manual_seed(54)

tokenizer = get_tokenizer()

#encode entire text and store it into a torch.Tensor
data = torch.tensor(tokenizer.encode(text).ids, dtype = torch.long)

#separate into train and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  #generate a small batch of data of inputs x and target y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x,y = x.to(device), y.to(device)
  return x, y

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


wandb.login()
wandb.init(project='Transformer2_Lex')

#initialise model and train
model = Transformer()
model = model.to(device)

print(sum(p.numel() for  p in model.parameters())/1e6, 'M parameters')

#create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


def save_checkpoint(model, optimizer, epoch, path):
  checkpoint = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
  }

  torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer = None):
  checkpoint = torch.load(path, map_location=device)
  if optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      print("Optimizer Loaded")
  model.load_state_dict(checkpoint['model_state_dict'])
  print("Model checkpoint Loaded")
  epoch = checkpoint['epoch']
  

  return epoch

checkpoint_path = 'checkpoint.pth'

if os.path.exists(checkpoint_path):
  last_epoch = load_checkpoint(checkpoint_path, model, optimizer)
  start_epoch = last_epoch + 1
  print(f"Resuming training from epoch {start_epoch}")
else:
  start_epoch = 0
  print("Starting training from scratch")



for iter in range(start_epoch, max_iters):
  
  #evaluate the loss on train and val set
  if iter % eval_interval == 0 or iter == (max_iters-1):
    losses = estimate_loss()
    wandb.log(losses)
    print(f"step {iter}, training loss is {losses['train']:.4f}, validation loss is {losses['val']:.4f}")
    save_checkpoint(model, optimizer, iter, checkpoint_path)

  xb, yb = get_batch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()