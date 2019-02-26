import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

class MusicDataset(Dataset):

    def __init__(self, dirstr, encoding, input_size, chunk_size=100):
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.input_size = input_size
        # self.batch_size = batch_size
        with open(dirstr, 'r') as f:
            self.raw = f.read()

    def __len__(self):
        return len(self.raw) // self.chunk_size + 1

    def __getitem__(self, idx):
        assert idx <= len(self.raw) // self.chunk_size
        this = self.raw[idx*self.chunk_size : (idx+1)*self.chunk_size]
        nxt = self.raw[idx*self.chunk_size+1 : (idx+1)*self.chunk_size+1]
        return self.encode(this), self.encode(nxt)

    def encode(self, chunk):
        # encode chunks
        onehot = torch.zeros(self.chunk_size, 1, self.input_size)
        idx = torch.LongTensor([self.encoding.index(c) for c in chunk])
        idx = idx.view(self.chunk_size, 1, 1)
        onehot.scatter_(2, idx, 1)
        return onehot


# class encoder:

#     def __init__(self):
#         None

#     def encode(self, chunk):
#         None


def create_encoding(dir='./pa4Data'):

    raw = ''
    encoding = []
    for s in ['/train.txt', '/val.txt', '/test.txt']:
        with open(dir+s, 'r') as f:
            raw = raw + f.read()

    for c in raw:
        if c not in encoding:
            encoding.append(c)
    input_size = len(encoding)

    return encoding, input_size


# def encode(chunk, encoding, chunk_size, input_size):
#     assert len(chunk) == chunk_size
#     onehot = torch.zeros(chunk_size, 1, input_size)
#     idx = torch.LongTensor([encoding.index(c) for c in chunk])
#     idx = idx.view(chunk_size, 1, 1)
#     onehot.scatter_(2, idx, 1)
#     return onehot