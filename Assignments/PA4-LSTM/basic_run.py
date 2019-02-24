from basic_lstm import *
from music_dataloader import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matlplotlib.pyplot as plt

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
    config['num_epoch'] = 100
    config['batch_size'] = 1
    config['learning_rate'] = 0.01

    # load data
    train = MusicDataset(dir='./pa4Data/train.txt')
    val = MusicDataset(dir='./pa4Data/val.txt')
    test = MusicDataset(dir='./pa4Data/test.txt')
    train_loader = Dataloader(train, batch_size=config['batch_size'])
    val_loader = Dataloader(val, batch_size=config['batch_size'])
    test_loader = Dataloader(test, batch_size=config['batch_size'])

    
    # create model
    model = BasicLSTM()
    model = model.to(computing_device)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # training loops
    for epoch in range(config['num_epoch']):

        for batch_count, (this, nxt) in enumerate(train_loader):

            # use GPU if supported
            this, nxt = this.to(computing_device), nxt.to(computing_device)

            # zero out gradient
            optimizer.zero_grad()

            # compute outputs and loss
            outputs = model(this)
            loss = criterion(this, nxt)

            # backprop
            loss.backward()

            # update the parameters
            optimizer.step()
    


if __name__ == '__main__':
    main()