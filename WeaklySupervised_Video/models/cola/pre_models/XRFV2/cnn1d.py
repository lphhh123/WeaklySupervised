# pre_models/xrfv2/cnn1d.py
import torch
import torch.nn as nn


class CNN1DBackbone(nn.Module):
    """
    CNN1DBackbone from pre_model_7s.py

    Total Downsampling Factor: 16
    Input:  [B, in_channels, 2048]
    Output: [B, feat_dim, 128]
    """

    def __init__(self, in_channels=30, feat_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            # Layer 1: /2
            # (30, T) -> (64, T/2)
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 2: /2
            # (64, T/2) -> (64, T/4)
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Layer 3: /2
            # (64, T/4) -> (128, T/8)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Layer 4: /2
            # (128, T/8) -> (256, T/16)
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

                                 
            # (256, T/16) -> (512, T/16)
            nn.Conv1d(256, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(feat_dim),
        )
        self.out_dim = feat_dim

    def forward(self, x):
        return self.layers(x)