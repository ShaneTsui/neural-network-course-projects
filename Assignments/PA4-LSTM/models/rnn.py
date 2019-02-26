import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as func

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, type='LSTM', batch_size=1):
        super(RNN, self).__init__()
        # torch.set_default_tensor_type('torch.cuda.LongTensor')
        if type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise Exception('Only support LSTM or GRU, got {}'.format(type))

        self.num_layers, self.batch_size, self.hidden_size = num_layers, batch_size, hidden_size
        self.decoder = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    # Input dim: (batch_size, seq_len, num_class)
    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        return self.decoder(output), hidden

    def init_hidden(self, type='LSTM'):
        if type == 'LSTM':
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda")),
             Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda")))
        elif type == 'GRU':
            return Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)).to(torch.device("cuda"))
        else:
            raise Exception('Only support LSTM or GRU, got {}'.format(type))