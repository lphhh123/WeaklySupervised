import warnings
import torch
from torch import nn as nn


class BidirectionalGRU(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        """
            Initialization of BidirectionalGRU instance
        Args:
            n_in: int, number of input features (from CNN output)
            n_hidden: int, number of hidden units per direction
            dropout: float, dropout probability
            num_layers: int, number of layers
        """
        super(BidirectionalGRU, self).__init__()

        self.rnn = nn.GRU(
            n_in,
            n_hidden,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
            num_layers=num_layers,
        )

    def forward(self, input_feat):
        # input_feat shape: [Batch, Time, Features]

        recurrent, _ = self.rnn(input_feat)
        return recurrent




class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BidirectionalLSTM, self).__init__()




        self.rnn = nn.LSTM(
            n_in,
            n_hidden,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )



    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent