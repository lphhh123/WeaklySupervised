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
        # IMU 数据的时间序列特性被 GRU 自然捕获
        recurrent, _ = self.rnn(input_feat)
        return recurrent


# [可选] 如果你想用 LSTM，这是修复后适配 CRNN 接口的版本
# 如果你只用默认的 GRU，下面这个类完全可以删掉
class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, dropout=0, num_layers=1):
        super(BidirectionalLSTM, self).__init__()

        # 修正 1: 接口参数与 GRU 保持一致 (去掉了 nOut)
        # 修正 2: 移除了多余的 Linear 层 (CRNN 外部有 Classifier)

        self.rnn = nn.LSTM(
            n_in,
            n_hidden,  # 注意：PyTorch LSTM 的 hidden_size 是单向的
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
            num_layers=num_layers,
        )

        # 输出维度将是 n_hidden * 2，这对齐了 CRNN 里 dense 层的预期

    def forward(self, input_feat):
        recurrent, _ = self.rnn(input_feat)
        return recurrent