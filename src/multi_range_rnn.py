import torch
import torch.nn as nn
import numpy as np

import multi_range_util as util

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import per_range_rnn

class MultiRangeSecondStageRNN(nn.Module):
    # module = nn.RNN | nn.GRU | nn.LSTM
    def __init__(self, input_size, output_size, hidden_rnn_size = 32, num_layers = 3, module = nn.RNN, dropout = 0.2, bidirectional = True, double_hidden_fc_size = True):
        super(MultiRangeSecondStageRNN, self).__init__()
        
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
        batch_size = x.batch_sizes[0]

        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_rnn_size, device = util.torch_device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_rnn_size, device = util.torch_device)

        if self.is_lstm:
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        
        unpacked_out, unpacked_lengths = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True) # see: https://stackoverflow.com/a/69968933
        
        fwd_out = []
        for i in range(batch_size):
            fwd_out.append(unpacked_out[i, :unpacked_lengths[i], :self.hidden_rnn_size])
        fwd_out = torch.vstack(fwd_out)
        
        if self.bidirectional:
            bwd_out = []
            for i in range(batch_size):
                bwd_out.append(unpacked_out[i, -unpacked_lengths[i]:, self.hidden_rnn_size:])
            bwd_out = torch.vstack(bwd_out)

            out = torch.cat((fwd_out, bwd_out), 1)
        else:
            out = fwd_out
        
        out = self.linear_net(out)        
        return out

    
class MultiRangeTwoStageRNN(nn.Module):
    def __init__(self, input_size, output_size, latent_size = 64):
        super(MultiRangeTwoStageRNN, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.per_range_rnn = per_range_rnn.PerRangeRNN(input_size, latent_size)
        self.multi_range_rnn = MultiRangeSecondStageRNN(latent_size, output_size)

    def forward(self, x, sample_num_ranges):
        out = self.per_range_rnn(x)
        
        offset = 0
        latent_seqs = []
        latent_lengths = []
        for num_ranges in sample_num_ranges:
            latent_seqs.append(out[offset:offset+num_ranges])
            latent_lengths.append(num_ranges)
            offset += num_ranges
            assert num_ranges > 0
        
        # see:
        # [1] https://stackoverflow.com/a/69968933
        # [2] https://stackoverflow.com/a/51030945
        ps = torch.nn.utils.rnn.pad_sequence(latent_seqs, batch_first=True)
        pps = torch.nn.utils.rnn.pack_padded_sequence(ps, lengths=latent_lengths, batch_first=True, enforce_sorted=False)
        x = pps.to(util.torch_device)
        
        out = self.multi_range_rnn(x)
        return out

if __name__ == "__main__":
    train_dataset, validation_dataset, test_dataset = util.load_split_dataset()
    model = MultiRangeTwoStageRNN(train_dataset.dataset.in_len, train_dataset.dataset.out_len)
    print(model)
    util.train_model(model, train_dataset, validation_dataset, test_dataset)
