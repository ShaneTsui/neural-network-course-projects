from basic_lstm import *
from data.music_dataloader import *

import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import pathlib


def main():

    # Set up folder for model saving
    model_path = '{}/LSTMs/basic/{}/'.format(os.getcwd(), time.strftime("%Y%m%d-%H%M%S"))
    model_pathlib = pathlib.Path(model_path)
    if not model_pathlib.exists():
        pathlib.Path(model_pathlib).mkdir(parents=True, exist_ok=True)

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 4, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    # config
    config = {}
    config['chunk_size'] = 100
    config['num_epoch'] = 100
    config['batch_size'] = 1
    config['learning_rate'] = 0.01
    encoding, config['input_size'] = create_encoding()

    # load data
    train = MusicDataset(dirstr='./pa4Data/train.txt', encoding=encoding, input_size=config['input_size'], chunk_size=config['chunk_size'])
    val = MusicDataset(dirstr='./pa4Data/val.txt', encoding=encoding, input_size=config['input_size'], chunk_size=config['chunk_size'])
    test = MusicDataset(dirstr='./pa4Data/test.txt', encoding=encoding, input_size=config['input_size'], chunk_size=config['chunk_size'])
    train_loader = DataLoader(train, batch_size=config['batch_size'])
    val_loader = DataLoader(val, batch_size=config['batch_size'])
    test_loader = DataLoader(test, batch_size=config['batch_size'])

    
    
    # create model
    model = BasicLSTM(config['input_size'], config['input_size'], config['input_size'])
    model = model.to(computing_device)
    softmax = nn.Softmax()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # losses
    train_loss = []
    val_loss = []

    # training loops
    for epoch in range(config['num_epoch']):

        for batch_count, (this, nxt) in enumerate(train_loader):
            
            # assume sgd
            this, nxt = this[0], nxt[0]

            # use GPU if supported
            this, nxt = this.to(computing_device), nxt.to(computing_device)

            # zero out gradient
            optimizer.zero_grad()

            # compute outputs and loss
            output = model(this)
            loss = criterion((output, dim=2), nxt)
            train_loss.append(loss.item)

            # backprop
            loss.backward()

            # update the parameters
            optimizer.step()
    


if __name__ == '__main__':
    main()