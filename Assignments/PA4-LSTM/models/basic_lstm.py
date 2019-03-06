import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class BasicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BasicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size  = output_size

        self.init_hidden()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.output = nn.Linear(hidden_size, output_size)
        

    def init_hidden(self):
        # initialize hidden state
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def forward(self, input):
        output, self.hidden = self.lstm(input, self.hidden)
        return output
        
