# LexGPT
Transformer with Lex Fridman podcast transcript.

CONFIG
Model and training hyperparameters can be changed in config.yaml

TRAINING
To train the model, run train.py. Please ensure that 'Lex.csv' is in the same directory as transformer.py, train.py, config.yaml and dataloader.py.

If data has never been loaded before, a tokenizer file will be created in the running directory. 

INFERENCE
For inference, run infer.py in the same directory as above. This requires a saved model checkpoint named 'checkpoint.pth'.

