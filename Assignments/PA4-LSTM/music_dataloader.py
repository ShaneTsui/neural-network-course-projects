import torch
from torch.utils.data import Dataset, Dataloader

import numpy as np

class MusicDataset(Dataset):

    def __init__(self, dir='./pa4Data/train.txt', chunk_size=100, batch_size=10):
        self.chunk_size = chunk_size
        # self.batch_size = batch_size
        with open(dir, 'r') as f:
            self.raw = f.read()
        self.create_encoding()

    def __len__(self):
        return len(self.raw) - self.chunk_size + 1

    def __getitem__(self, idx):
        assert idx <= len(self.raw) - self.chunk_size
        this = self.raw[idx:idx+self.chunk_size]
        nxt = self.raw[idx+1:idx+self.chunk_size+1]
        return sel.encode(this), self.encode(nxt)

    def create_encoding(self):
        # create one-hot encoding dict
        self.encoding = []
        for c in self.raw:
            if c not in self.encoding:
                self.encoding.append(c)
        self.input_size = len(self.encoding)

    def encode(self, chunk):
        # encode chunks
        onehot = torch.zeros(self.chunk_size, 1, self.input_size)
        idx = torch.FloatTensor([self.encoding.index(c) for c in chunk])
        onehot.scatter_(2, idx, 1)
        return onehot

class MusicDataloader(Dataloader):

    def __init__(self):
        None