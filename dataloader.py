from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch
import pandas as pd
import yaml
import re
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 20000

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


config = load_config('config.yaml')

batch_size = config['training']['batch_size'] #how many sequences to process in parallel
block_size = config['training']['block_size'] #max length of a sequence 

def filter_characters(text):
    # Allow only English letters, standard punctuation, and numbers
    filtered_text = re.sub('[^a-zA-Z0-9!?"\',.\\- ]', '', text)
    return filtered_text

def clean_punctuation_and_whitespace(text):
    # Replace multiple spaces with a single space
    text = re.sub('\\s+', ' ', text)
    # Optionally, handle or simplify punctuation as needed
    text = text.strip()
    return text



# Dictionary of contractions and their expanded forms
contractions_dict = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
    "let's": "let us",
    "don't": "do not",
    "you're": "you are",
    "i'm": "i am",
    # Add more contractions as needed
}

# Function to expand contractions
def expand_contractions(text, contractions_dict=contractions_dict):
    pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')
    expanded_text = pattern.sub(lambda x: contractions_dict[x.group()], text)
    return expanded_text

#import Lex podcast data
file_name = 'Lex.csv'
df = pd.read_csv(file_name)

df['text'] = df['text'].apply(filter_characters)
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(clean_punctuation_and_whitespace)
text = df['text'].str.cat(sep='')
text = expand_contractions(text)
print(text[54000:55000])

def get_tokenizer():

    #Character encoding
    '''stoi = { ch:i for i,ch in enumerate(chars)} #string to int
    itos = { i:ch for i,ch in enumerate(chars)} #into to string
    encode = lambda s: [stoi[c] for c in s] #input string, output list of ints
    decode = lambda l: ''.join([itos[i] for i in l]) #input list of ints, output string'''

    file_path = 'text_data.txt'
    with open(file_path, 'w') as file:
        file.write(text)

    # Initialize a tokenizer
    tok_file = "bpe_tokenizer.json"

    if os.path.exists(tok_file):
        #load tokenizer
        tokenizer = Tokenizer.from_file(tok_file)
        print("tokenizer loaded")
    else:
        #train new tokenizer
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        print("tokenizer training")

        # Initialize a trainer, specify the vocabulary size
        trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[PAD]","[SEP]"])

        # Train the tokenizer
        tokenizer.train([file_path], trainer)

        # Save the tokenizer
        tokenizer.save(tok_file)
    return tokenizer



