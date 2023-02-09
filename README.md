# Summary
  This is a Word2Vec project finished by the guiding by OlgaChernytska, you can also check [her project on github at here.](https://github.com/OlgaChernytska/word2vec-pytorch)
  
## Usage
```
python3 train.py --config config.yaml
```

## config temple
```
model_name: CBOW # SkipGram or CBOW

dataset: WikiText2 # WikiText2 or Wiki Text103
data_dir: .data/
train_batch_size: 96
val_batch_size: 96
shuffle: True

optimizer: Adam
learning_rate: 0.025
epochs: 5
train_steps: 10
val_steps: 10

checkpoint_frequency: 100
model_dir: weights//cbow_WikiText2
```
