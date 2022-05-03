import torch
import torch.nn as nn

import per_sample_util as util

class PerSampleRNN(nn.Module):
    # module = nn.RNN | nn.GRU | nn.LSTM
    def __init__(self, input_size, output_size, hidden_rnn_size = 32, num_layers = 3, module = nn.RNN, dropout = 0.2, bidirectional = True, double_hidden_fc_size = False):
        super(PerSampleRNN, self).__init__()
        
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.double_hidden_fc_size = double_hidden_fc_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_rnn_size = hidden_rnn_size
        self.hidden_fc_size = self.hidden_rnn_size * (2 if self.bidirectional else 1) * (2 if self.double_hidden_fc_size else 1)
        
        self.rnn = module(self.input_size, self.hidden_rnn_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        self.is_lstm = (module == nn.LSTM)
        
        activation = nn.ELU()
        self.linear_net = nn.Sequential(nn.Linear(self.hidden_rnn_size * (2 if self.bidirectional else 1), self.hidden_fc_size),
                                        nn.BatchNorm1d(self.hidden_fc_size),
                                        nn.Dropout(self.dropout),
                                        activation,
                                        nn.Linear(self.hidden_fc_size, self.hidden_fc_size),
                                        nn.BatchNorm1d(self.hidden_fc_size),
                                        nn.Dropout(self.dropout),
                                        activation,
                                        nn.Linear(self.hidden_fc_size, self.output_size))

    def forward(self, x):
        x = x.squeeze().unsqueeze(-1 if self.input_size == 1 else 1)

        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.shape[0], self.hidden_rnn_size, device = util.torch_device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.shape[0], self.hidden_rnn_size, device = util.torch_device)

        if self.is_lstm:
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        if self.bidirectional:
            out = torch.cat((out[:, -1, :self.hidden_rnn_size], out[:, 0, self.hidden_rnn_size:]), 1)
        else:
            out = out[:, -1, :]
        
        out = self.linear_net(out)
        return out

if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset, in_len, out_len = util.load_split_dataset()
    model = PerSampleRNN(1, out_len)
    print(model)
    util.train_model(model, train_dataset, validation_dataset, test_dataset)
