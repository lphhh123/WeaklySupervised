import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class VGG1DBackbone(nn.Module):
    """
    VGG-16 style for 1D sequence.
    Input : [B, C_in, T]
    Output: [B, 512, T/32] (roughly; depends on pooling floors)
    """
    def __init__(self, in_channels=30, feat_dim=512, batch_norm=True, dropout=0.0):
        super().__init__()
        assert feat_dim == 512

        def conv(cin, cout):
            layers = [nn.Conv1d(cin, cout, kernel_size=3, padding=1, bias=not batch_norm)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        def block(cin, cout, n_conv):
            layers = []
            for i in range(n_conv):
                layers += conv(cin if i == 0 else cout, cout)
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            return nn.Sequential(*layers)

        # VGG16: [2,2,3,3,3]
        self.b1 = block(in_channels, 64,  2)   # /2
        self.b2 = block(64,         128, 2)   # /4
        self.b3 = block(128,        256, 3)   # /8
        self.b4 = block(256,        512, 3)   # /16
        self.b5 = block(512,        512, 3)   # /32

        self.out_dim = 512

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return x  # [B,512,T’]