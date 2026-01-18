# adapters.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAdapter1D(nn.Module):
    """
    轻量时序适配器：输入输出同 shape [B, C, T]
    目的：冻结 backbone 时，靠少量参数做 domain adaptation / 伪标签噪声适配
    """
    def __init__(
        self,
        channels: int,
        bottleneck: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        scale: float = 0.1,
        use_dwconv: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 建议用奇数，便于 same padding"

        self.pw1 = nn.Conv1d(channels, bottleneck, kernel_size=1, bias=True)

        if use_dwconv:
            self.dw = nn.Conv1d(
                bottleneck, bottleneck,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=bottleneck,
                bias=True
            )
        else:
            self.dw = nn.Conv1d(
                bottleneck, bottleneck,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=True
            )

        self.pw2 = nn.Conv1d(bottleneck, channels, kernel_size=1, bias=True)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        # 残差缩放（让初始更接近 identity，训练更稳）
        self.alpha = nn.Parameter(torch.tensor(float(scale)))

        # 初始化：让 pw2 初始接近 0（更像 identity）
        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        y = self.pw1(x)
        y = self.act(y)
        y = self.dw(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pw2(y)
        return x + self.alpha * y
