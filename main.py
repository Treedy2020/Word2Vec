import os
import yaml
import argparse
from torch.nn import CrossEntropyLoss
from utils.trainer import Trainer
from utils.dataloader import getDataLoaderAndVocab
from utils.helper import (
    getModel,
    getOptimizer,
    getScheduler,
    saveVocab,
    saveConfig,
)

"""
yaml_example:
    model_name: cbow

    dataset: WikiText2
    data_dir: .data
    train_batch_size: 96
    val_batch_size: 96
    shuffle: True

    optimizer: Adam
    learning_rate: 0.025
    epochs: 5
    train_steps: 10
    val_steps: 10

    checkpoint_frequency: 100
    model_dir: weights/cbow_WikiText2
"""

def train(config:dict):
    if not os.path.isdir(config["model_dir"]):
        os.makedirs(config["model_dir"])
    
    # get DataLoader 
    train_dataloder, vocab = getDataLoaderAndVocab(
        model_name=config['model_name'],
        data_set = config['dataset'],
        data_dir=config['data_dir'],
        ds_type='train',
        batch_size= config['train_batch_size'],
        shuffle=config['shuffle'],
    )
    
    val_dataloader, _ = getDataLoaderAndVocab(
        model_name=config['model_name'],
        data_set=config['dataset'],
        data_dir=config['data_dir'],
        ds_type='valid',
        batch_size= config['val_batch_size'],
        shuffle=config['shuffle'],
        vocab=vocab,
    )
    
    vocab_size=len(vocab.get_stoi())
    print(f'The vocab size is {vocab_size}')
    
    model = getModel(config['model_name'], vocabsize=vocab_size)
    
    Adam = getOptimizer(config['optimizer'])
    optimizer = Adam(params=model.parameters(), lr=config['learning_rate'])
    
    criterion = CrossEntropyLoss()
    lr_scheduler = getScheduler(optimizer=optimizer, fn = lambda x: (config['epochs'] - x)/config['epochs'])
    
    trainer = Trainer(
        model=model,
        epoch=config['epochs'],
        train_dataloader=train_dataloder,
        val_dataloader=val_dataloader,
        training_step=config['train_steps'],
        valid_step=config['val_steps'],
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkfrequency=config['checkpoint_frequency'],
        model_savepath=config['model_dir'],
        checkpoint_path=config['model_dir'],
        loss_path=config['model_dir'],
    )
    
    trainer.train()
    saveConfig(config, config['model_dir'])
    saveVocab(vocab, config['model_dir'])
    
    print('The training process is finised, please check the model and vocab in {}'.format(config['model_dir']))

if __name__ == "__main__":
    paser = argparse.ArgumentParser()
    paser.add_argument('--config', type=str, help='Path of  config.yaml')
    args = paser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)