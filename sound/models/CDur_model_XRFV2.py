# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ============================================================
# ============================================================
class DilatedResBlock(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(cout, cout, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(cout),
        )
        self.shortcut = nn.Sequential()
        if cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=1, bias=False),
                nn.BatchNorm1d(cout)
            )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.relu(out)


# ============================================================
# ============================================================
class LinearSoftPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, x, time_decision):
        # time_decision: [B, T, C]

        return (time_decision ** 2).sum(self.pooldim) / (time_decision.sum(self.pooldim) + 1e-8)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, x, decision):
        return x.mean(self.pooldim)


def parse_poolingfunction(poolingfunction_name='linear', **kwargs):
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)

    return LinearSoftPool(pooldim=1)


# ============================================================
# ============================================================
class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()



        self.features = nn.Sequential(
            DilatedResBlock(inputdim, 64, dilation=1),
            nn.LPPool1d(4, 2),
            DilatedResBlock(64, 128, dilation=2),
            DilatedResBlock(128, 128, dilation=4),
            nn.LPPool1d(4, 2),
            DilatedResBlock(128, 256, dilation=8),
            nn.Dropout(0.5)
        )

        with torch.no_grad():

            dummy_x = torch.randn(1, inputdim, 512)
            feat_out = self.features(dummy_x)
            rnn_input_dim = feat_out.shape[1]


        self.gru = nn.GRU(rnn_input_dim, 128, bidirectional=True, batch_first=True)


        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'linear'),
                                               inputdim=256, outputdim=outputdim)


        self.apply(init_weights)

    def forward(self, x, upsample=True):
        """
        x shape: [Batch, Time, Dim]
        """
        batch, time, dim = x.shape


        x = x.transpose(1, 2)


        x = self.features(x)  # [B, 256, T_down]


        x = x.transpose(1, 2).contiguous()


        x, _ = self.gru(x)  # [B, T_down, 256]



        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.0)


        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.0)


        if upsample:

            decision_time = F.interpolate(
                decision_time.transpose(1, 2),
                size=time,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        return decision, decision_time


class MilSEDCNN(CDur):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__(inputdim, outputdim, **kwargs)