import torch
import torch.nn as nn
import sys

# [关键修改]
# 为了支持 bimamba_type="v2"，我们必须使用本地集成好的 VideoMamba/VisionMamba 实现
# 而不是使用 pip 安装的官方 mamba_ssm (官方版不支持 v2)
# 假设你之前已经按步骤在 model/mamba/ 目录下放置了 mamba_new.py 和 blocks.py
from mamba_ssm import Mamba as CoreMamba


class SimpleMambaBlock(nn.Module):
    """
    最小版 Mamba block (保留学姐的逻辑):
      x -> LayerNorm -> CoreMamba(v2) -> 残差
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

        # [关键] 保留 bimamba_type="v2"
        # 这要求 CoreMamba 必须是支持双向的版本
        self.mamba = CoreMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bimamba_type="v2",  # <--- 必须保留，匹配权重
            use_fast_path=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.mamba(h)
        return x + h


class MambaBackbone(nn.Module):
    """
    MambaBackbone_7s 的复刻版
    输入: [B, C, T]
    输出: [B, 512, T] (无下采样)
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
        self.out_dim = feat_dim

        # 1. Input Projection
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)

        # 2. Mamba Layers
        self.layers = nn.ModuleList([
            SimpleMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            for _ in range(n_layers)
        ])

        # 3. Output Projection
        self.output_proj = nn.Conv1d(d_model, feat_dim, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm1d(feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, d_model, T)
        x = self.input_proj(x)

        # (B, d_model, T) -> (B, T, d_model)
        x = x.permute(0, 2, 1)

        for layer in self.layers:
            x = layer(x)

        # (B, T, d_model) -> (B, d_model, T)
        x = x.permute(0, 2, 1)

        x = self.output_proj(x)
        x = self.out_bn(x)

        return x