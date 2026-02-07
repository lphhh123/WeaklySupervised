from itertools import zip_longest
import numpy as np

import torch
import torch.nn as nn


def init_weights(m):
    # 修复：移除重复的 check，增加对 1d 的支持
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


# ... (中间的 MaxPool, LinearSoftPool, MeanPool 等 Pooling 类保持不变，不需要改) ...
# 为了节省篇幅，这里省略 Pooling 类的代码，请保留你原文件中的这部分
# ... ...

# === 以下是重点修改的模型部分 ===

class MilSEDCNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        # 1. 删除只能用于 128 维 Mel 谱的断言
        # assert inputdim == 128
        self._inputdim = inputdim

        self.network = nn.Sequential(
            nn.BatchNorm1d(inputdim),  # 2. 输入并不是 1 通道，而是 inputdim (113) 通道

            # Block 1
            # 3. 第一个 Conv 的 in_channels 改为 inputdim
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

            # 4. 修复 Tuple Kernel: (1, 8) -> 8 (只在时间轴卷积)
            nn.Conv1d(128, 256, kernel_size=8, padding=0),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        def calculate_cnn_size(input_size):
            # 5. 修复 dummy input 的形状: (Batch, Channels, Time)
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]

        # input_size 改为 (Channels, Time)
        cnn_outputdim = calculate_cnn_size((inputdim, 500))
        # 1D CNN 输出是 (Channels, Time)，Flatten 时需要计算通道数
        linear_input_dim = cnn_outputdim[0]

        # During training, pooling in time
        self.outputlayer = nn.Linear(linear_input_dim, outputdim)

        # pooling 的 inputdim 修正
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'soft'),
                                               inputdim=linear_input_dim,
                                               outputdim=outputdim)
        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        # x shape: [Batch, Time, Dim] (例如 [B, 15643, 113])

        # 6. 维度调整: 1D CNN 需要 [Batch, Dim, Time]
        x = x.transpose(1, 2)

        # 移除 unsqueeze(1)，因为 Dim 已经是通道了
        x = self.network(x)

        # 此时 x 是 [Batch, Feature, Time]
        x = x.transpose(1, 2).contiguous()  # 变回 [Batch, Time, Feature]

        # MilSEDCNN 在最后做了一个特殊的 view/flatten 操作
        # 但这里的逻辑如果是针对 time pooling 的话，得看 classifier 怎么接
        # 原逻辑：x = x.view(x.shape[0], x.shape[1], -1)
        # 对于 1D CNN，transpose 后已经是 [B, T, F]，无需再次 view 除非有特殊合并

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


# cATP 类保持不变...

class cATPSDS(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        # assert outputdim == 10 # 建议注释掉，你的数据集可能不是 10 类

        # 7. filters 第一个元素改为 inputdim (113)
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
                    # 8. 修复 Tuple MaxPool: (1, 4) -> 4
                    nn.MaxPool1d(4)))
        self.features = nn.Sequential(*features)
        init_weights(self.features)
        self.attentions = nn.ModuleList(
            [cATP(self.dimensions[f]) for f in range(outputdim)])
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.dimensions[f], 1) for f in range(outputdim)])

    def forward(self, x):
        # x shape: [Batch, Time, Dim]
        # 9. 调整维度 [B, Dim, Time]
        x = x.transpose(1, 2)

        # 移除 unsqueeze(1)
        x = self.features(x).flatten(-2).permute(0, 2, 1).contiguous()

        decision, decision_time = [], []
        for c in range(self.outputdim):
            # 注意：这里的切片逻辑依赖于特定的 filters 数量，如果你修改了网络结构，这里可能会越界
            # 暂时保留原逻辑
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
            # 10. 第一层 Block1d 输入改为 inputdim (113)
            Block1d(inputdim, 32),
            # 11. 修复 Tuple Pool: (2, 4) -> 2 (或者 4, 视具体时间下采样需求而定，这里假设时间下采样为 2)
            nn.LPPool1d(4, 2),
            Block1d(32, 128),
            Block1d(128, 128),
            # 修复 Tuple Pool
            nn.LPPool1d(4, 2),
            Block1d(128, 128),
            Block1d(128, 128),
            # 修复 Tuple Pool: (1, 4) -> 4
            nn.LPPool1d(4, 4),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            # 12. 修复 dummy input 形状: (1, InputDim, Time)
            # 原来的 500 是时间长度，inputdim 是通道数
            rnn_input_dim = self.features(torch.randn(1, inputdim, 500)).shape
            # 1D CNN 输出: [Batch, Channels, Time]
            # GRU 的 input_size 应该是 Channels 数
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

        # 13. 维度调整 [Batch, Dim, Time]
        x = x.transpose(1, 2)

        # 移除 unsqueeze
        x = self.features(x)

        # CNN 输出 [B, C, T]，GRU 需要 [B, T, C]
        x = x.transpose(1, 2).contiguous()

        # 这里的 flatten(-2) 在原代码里可能是处理 (Freq, Time) 的，现在是 (T, C)
        # 只要保证最后维度是 C 即可，不需要 flatten，因为已经是 3D
        # x = x.flatten(-2) # 对于 1D CNN + RNN，这一步通常不需要，或者 x 已经是 [B, T, C]

        x, _ = self.gru(x)

        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.).squeeze(1)

        if upsample:
            # 上采样回原始时间长度
            decision_time = torch.nn.functional.interpolate(
                decision_time.transpose(1, 2),
                time,
                mode='linear',
                align_corners=False).transpose(1, 2)
        return decision, decision_time