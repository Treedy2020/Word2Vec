import torch
import torch.nn as nn
from utils.cons import (CBOW_N, SKIP_GRAM_N, MAX_LENGTH, NORM, EMBED_SIZE)

class CBOW(nn.Module):
    def __init__(self, vocabsize:int):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings = vocabsize,
            embedding_dim = EMBED_SIZE,
            max_norm = NORM
        )
        self.liner = nn.Linear(
            in_features=EMBED_SIZE,
            out_features=vocabsize
        )
    def forward(self, inputs):
        return self.liner(self.embeddings(inputs).mean(axis=1))

class SkipGram(nn.Module):
    def __init__(self, vocabsize:int):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings = vocabsize,
            embedding_dim = EMBED_SIZE,
            max_norm = NORM
        )
        self.liner = nn.Linear(
            in_features=EMBED_SIZE,
            out_features=vocabsize
        )
    def forward(self, inputs):
        return self.liner(self.embeddings(inputs))

if __name__ == '__main__':
    model = SkipGram(vocabsize=200)
    inputs = torch.tensor([1,2,3,4])
    print(model(inputs).shape)