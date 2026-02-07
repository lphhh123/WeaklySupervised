from itertools import zip_longest
import numpy as np

import torch
import torch.nn as nn


def init_weights(m):

    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)




# ... ...



class MilSEDCNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()

        # assert inputdim == 128
        self._inputdim = inputdim

        self.network = nn.Sequential(
            nn.BatchNorm1d(inputdim),

            # Block 1

            nn.Conv1d(inputdim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            # BLock 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            # Block 4
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),


            nn.Conv1d(128, 256, kernel_size=8, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        def calculate_cnn_size(input_size):

            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]


        cnn_outputdim = calculate_cnn_size((inputdim, 500))

        linear_input_dim = cnn_outputdim[0]

        # During training, pooling in time
        self.outputlayer = nn.Linear(linear_input_dim, outputdim)


        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'soft'),
                                               inputdim=linear_input_dim,
                                               outputdim=outputdim)
        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):



        x = x.transpose(1, 2)


        x = self.network(x)


        x = x.transpose(1, 2).contiguous()






        decision_time = torch.sigmoid(self.outputlayer(x))
        decision = self.temp_pool(x, decision_time).squeeze(1)
        decision = torch.clamp(decision, min=1e-7, max=1.)
        return decision, decision_time


class Block1d(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(cin),
            nn.Conv1d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)




class cATPSDS(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()



        filters = [inputdim] + kwargs.get('filters', [160, 160, 160])
        kernels = [5, 5, 3]
        paddings = [2, 2, 1]
        self.dimensions = kwargs.get('dimensions',
                                     [46, 22, 92, 42, 82, 17, 13, 160, 74, 85])
        self.outputdim = outputdim
        features = nn.ModuleList([nn.BatchNorm1d(inputdim, eps=1e-4, momentum=0.01)])
        for h0, h1, kernel, padding in zip(filters, filters[1:], kernels,
                                           paddings):
            features.append(
                nn.Sequential(
                    nn.Conv1d(h0,
                              h1,
                              kernel_size=kernel,
                              padding=padding,
                              bias=False),
                    nn.BatchNorm1d(h1, eps=1e-4, momentum=0.01), nn.ReLU(True),

                    nn.MaxPool1d(4)))
        self.features = nn.Sequential(*features)
        init_weights(self.features)
        self.attentions = nn.ModuleList(
            [cATP(self.dimensions[f]) for f in range(outputdim)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.dimensions[f], 1) for f in range(outputdim)])

    def forward(self, x):
        # x shape: [Batch, Time, Dim]

        x = x.transpose(1, 2)


        x = self.features(x).flatten(-2).permute(0, 2, 1).contiguous()

        decision, decision_time = [], []
        for c in range(self.outputdim):


            sds = x[:, :, :self.dimensions[c]]
            embedding_level, time_level = self.attentions[c](sds)
            decision.append(self.classifiers[c](embedding_level))
            decision_time.append(time_level)
        decision_time = torch.sigmoid(torch.cat(decision_time, dim=-1))
        decision = torch.sigmoid(torch.cat(decision, dim=-1)).squeeze(1)
        decision = torch.clamp(decision, min=1e-7, max=1.)
        return decision, decision_time


class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        features = nn.ModuleList()
        self.features = nn.Sequential(

            Block1d(inputdim, 32),

            nn.LPPool1d(4, 2),
            Block1d(32, 128),
            Block1d(128, 128),

            nn.LPPool1d(4, 2),
            Block1d(128, 128),
            Block1d(128, 128),

            nn.LPPool1d(4, 4),
            nn.Dropout(0.3),
        )
        with torch.no_grad():


            rnn_input_dim = self.features(torch.randn(1, inputdim, 500)).shape


            rnn_input_dim = rnn_input_dim[1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.temp_pool = parse_poolingfunction(kwargs.get(
            'temppool', 'linear'),
            inputdim=256,
            outputdim=outputdim)
        self.outputlayer = nn.Linear(256, outputdim)
        self.features.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x, upsample=True):
        # x: [Batch, Time, Dim]
        batch, time, dim = x.shape


        x = x.transpose(1, 2)


        x = self.features(x)


        x = x.transpose(1, 2).contiguous()





        x, _ = self.gru(x)

        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)

        if upsample:

            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return decision, decision_time