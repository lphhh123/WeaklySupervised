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
# 基础模块：包含空洞卷积和残差结构
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
# 池化层定义
# ============================================================
class LinearSoftPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, x, time_decision):
        # time_decision: [B, T, C]
        # 使用决策概率作为权重进行池化
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
    # 默认回退到 linear，因为在弱监督 TAL 中表现最稳
    return LinearSoftPool(pooldim=1)


# ============================================================
# 主模型：CDur
# ============================================================
class CDur(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()

        # 1. 特征提取层：改用空洞卷积残差块
        # 相比原版，我们将下采样从 16x 减少到 4x，保留更多时序细节
        self.features = nn.Sequential(
            DilatedResBlock(inputdim, 64, dilation=1),
            nn.LPPool1d(4, 2),  # 下采样 2x
            DilatedResBlock(64, 128, dilation=2),
            DilatedResBlock(128, 128, dilation=4),
            nn.LPPool1d(4, 2),  # 下采样 2x
            DilatedResBlock(128, 256, dilation=8),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            # 自动探测 GRU 输入维度
            dummy_x = torch.randn(1, inputdim, 512)
            feat_out = self.features(dummy_x)
            rnn_input_dim = feat_out.shape[1]

        # 2. 时序建模层：双向 GRU
        self.gru = nn.GRU(rnn_input_dim, 128, bidirectional=True, batch_first=True)

        # 3. 分类与池化层
        self.outputlayer = nn.Linear(256, outputdim)
        self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'linear'),
                                               inputdim=256, outputdim=outputdim)

        # 初始化
        self.apply(init_weights)

    def forward(self, x, upsample=True):
        """
        x shape: [Batch, Time, Dim]
        """
        batch, time, dim = x.shape

        # 转换为 [Batch, Dim, Time] 适配 1D CNN
        x = x.transpose(1, 2)

        # 提取空间/通道特征
        x = self.features(x)  # [B, 256, T_down]

        # 转换为 [Batch, T_down, 256] 适配 RNN
        x = x.transpose(1, 2).contiguous()

        # GRU 捕捉长程时序依赖
        x, _ = self.gru(x)  # [B, T_down, 256]

        # 获取每一帧的决策概率 (Logits -> Sigmoid)
        # 注意：这里我们使用 clamp 增强数值稳定性
        decision_time = torch.sigmoid(self.outputlayer(x)).clamp(1e-7, 1.0)

        # 视频级决策 (Temporal Pooling)
        decision = self.temp_pool(x, decision_time).clamp(1e-7, 1.0)

        # 针对测试模式，将预测上采样回原始长度
        if upsample:
            # interpolate 期望 [B, C, T]
            decision_time = F.interpolate(
                decision_time.transpose(1, 2),
                size=time,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        return decision, decision_time


# 为了保持兼容性，保留 MilSEDCNN 的壳子，但内部可以调用 CDur 的逻辑或独立实现
class MilSEDCNN(CDur):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__(inputdim, outputdim, **kwargs)