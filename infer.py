from transformer import Transformer
from dataloader import get_tokenizer
import torch

def load_checkpoint(path, model, optimizer = None):
    checkpoint = torch.load(path, map_location=device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer Loaded")
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model checkpoint Loaded")
    epoch = checkpoint['epoch']
    

    return epoch

tokenizer = get_tokenizer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_path = 'checkpoint.pth'
model = Transformer()

load_checkpoint(checkpoint_path, model, None)

#generate from model
idx = torch.zeros((1,1), dtype = torch.long).to(device = device)
#idx[0] = len(itos)-1

with open('output.txt', 'w') as file:
    
    output = tokenizer.decode(model.generate(idx, max_new_tokens=200)[0].tolist())
    file.write(output)  # Write a string to the file
