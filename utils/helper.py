import os
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from utils.models import SkipGram, CBOW

"""
yaml_example:
    model_name: cbow

    dataset: WikiText2
    data_dir: data/
    train_batch_size: 96
    val_batch_size: 96
    shuffle: True

    optimizer: Adam
    learning_rate: 0.025
    epochs: 5
    train_steps: 
    val_steps: 

    checkpoint_frequency: 
    model_dir: weights/cbow_WikiText2
"""

def getModel(model_name, vocabsize):
    assert model_name in ['SkipGram', 'CBOW'], "Only Support：Skipgram or CBOW"
    
    if model_name == 'SkipGram':
        model = SkipGram(vocabsize=vocabsize)
    else:
        model = CBOW(vocabsize=vocabsize)
    return model
    
def getOptimizer(optimizer_name):
    if optimizer_name == 'Adam':
        return optim.Adam
    else:
        raise ValueError('Please choose optimzer from: Adam')
        return

def getScheduler(optimizer, fn, verbose:bool=True):
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=fn,
        verbose = verbose
    )
    return lr_scheduler

def saveConfig(config:dict, yaml_path):
    yaml_path = os.path.join(yaml_path, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f)

def saveVocab(vocab, vocab_path):
    vocab_path = os.path.join(vocab_path, 'vocab.pt')
    torch.save(vocab, vocab_path)
    
