import sys
import torch
import torch.nn as nn
causal_conv1d_path = '/data/zhenkui.yzk/wifi_location/video-mamba-suite/causal-conv1d'
mamba_path         = '/data/zhenkui.yzk/wifi_location/video-mamba-suite/mamba'
project_path       = '/home/yangzhenkui/code/WSDDN'

if causal_conv1d_path not in sys.path:
    sys.path.append(causal_conv1d_path)
if mamba_path not in sys.path:
    sys.path.append(mamba_path)
# 导入核心 Mamba 实现
try:
    from mamba_ssm import Mamba as CoreMamba
except ImportError:
    from mamba_ssm.models.mixer_seq_simple import Mamba as CoreMamba

class SimpleMambaBlock(nn.Module):
    """
    最小版 Mamba block:
      x -> LayerNorm -> CoreMamba -> 残差
    输入输出: (B, L, D)
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 这里显式传 bimamba_type="v2"，满足你这个 mamba_simple.py 里的断言
        self.mamba = CoreMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v2",
            use_fast_path=True,   # 一般这个实现也支持，开着就行
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.norm(x)
        h = self.mamba(h)     # (B, L, D)
        return x + h          # 残差


class MambaBackbone_7s(nn.Module):
    """
    Mamba 特征提取 backbone，用于 7s IMU 片段预训练 & 后续 WSDDN 主干：

      输入:  x ∈ R^{B, 30, 478}
      输出:  feat ∈ R^{B, feat_dim(=512), T(≈478)}

    结构: 1x1 Conv 把 30->d_model，堆 n_layers 个 SimpleMambaBlock，再投影到 feat_dim
    """
    def __init__(
        self,
        in_channels: int = 30,
        d_model: int = 256,
        n_layers: int = 4,
        feat_dim: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.feat_dim = feat_dim

        # 30 维 IMU 通道 -> d_model 维 embedding
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)

        # 堆几层 Mamba block（不下采样，只做时序编码）
        self.layers = nn.ModuleList([
            SimpleMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # 输出投影到 feat_dim = 512，方便和你现有的 CNN/TCN 等保持统一
        self.output_proj = nn.Conv1d(d_model, feat_dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm1d(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 30, 478)
        return: (B, feat_dim, T)  一般 T ≈ 478
        """
        # (B, 30, T) -> (B, d_model, T)
        x = self.input_proj(x)

        # 变成 (B, T, d_model) 以匹配 Mamba 的 (B,L,D) 接口
        x = x.permute(0, 2, 1)   # (B, T, d_model)

        for layer in self.layers:
            x = layer(x)         # (B, T, d_model)

        # 变回 (B, d_model, T)
        x = x.permute(0, 2, 1)   # (B, d_model, T)

        # 投影到 feat_dim = 512
        x = self.output_proj(x)  # (B, feat_dim, T)
        x = self.out_bn(x)

        return x


class MambaClassifier_7s(nn.Module):
    """
    用于 7s 分类预训练的 Mamba 模型：
      - backbone: MambaBackbone_7s
      - classifier: 全局池化 + Linear

    接口与 CNN1DClassifier_7s / TCNClassifier_7s 等完全一致。
    """
    def __init__(self, num_classes: int, task: str = 'single', feat_dim: int = 512,in_channels: int = 30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = MambaBackbone_7s(
            self.in_channels,
            d_model=256,     # 可调：更大更强，也更耗显存
            n_layers=4,
            feat_dim=feat_dim,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),              # (B, feat_dim)
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 478)
        feat = self.backbone(x)        # (B, feat_dim, T)
        logits = self.classifier(feat) # (B, num_classes)
        return logits