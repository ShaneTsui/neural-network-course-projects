import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as func

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, type='LSTM', batch_size=1):
        super(RNN, self).__init__()
        self.type = type
        if type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif type == 'Vanilla':
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise Exception('Only support LSTM, GRU and Vanilla, got {}'.format(type))

        self.num_layers, self.batch_size, self.hidden_size = num_layers, batch_size, hidden_size
        self.decoder = nn.Linear(hidden_size, output_size)

    # Input dim: (batch_size, seq_len, num_class)
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        if self.type == 'LSTM':
            return self.decoder(output), (hidden[0].detach(), hidden[1].detach())
        elif self.type == 'GRU' or self.type == 'Vanilla':
            return self.decoder(output), hidden.detach()
        else:
            raise Exception('Only support LSTM, GRU and Vanilla, got {}'.format(type))

    def init_hidden(self, type='LSTM'):
        if type == 'LSTM':
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda")),
             Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda")))
        elif type == 'GRU' or self.type == 'Vanilla':
            return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda"))
        else:
            raise Exception('Only support LSTM, GRU and Vanilla, got {}'.format(type))