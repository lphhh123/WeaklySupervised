import torch
import torch.nn as nn
import torch.nn.functional as F


def minmax_norm_1d_per_sample(a: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a_min = a.amin(dim=1, keepdim=True)
    a_max = a.amax(dim=1, keepdim=True)
    return (a - a_min) / (a_max - a_min + eps)


class VGG1DBackbone(nn.Module):
    """
    VGG-16 style for 1D sequence with BUAA (Bottom-Up Attention Aggregation).
    Input : [B, C_in, T] (e.g., 2048)
    Output: [B, 512, T/32] (e.g., 64)
    """

    def __init__(self, in_channels=30, feat_dim=512, batch_norm=True, dropout=0.0, use_buaa=True):
        super().__init__()

        assert feat_dim == 512
        self.use_buaa = use_buaa
        self.out_dim = 512

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


        self.b1 = block(in_channels, 64, 2)  # T -> T/2
        self.b2 = block(64, 128, 2)  # T/2 -> T/4
        self.b3 = block(128, 256, 3)  # T/4 -> T/8
        self.b4 = block(256, 512, 3)  # T/8 -> T/16
        self.b5 = block(512, 512, 3)  # T/16 -> T/32

    def forward(self, x):

        x1 = self.b1(x)  # [B,64, T/2]
        x2 = self.b2(x1)  # [B,128,T/4]
        x3 = self.b3(x2)  # [B,256,T/8]
        x4 = self.b4(x3)  # [B,512,T/16]
        x5 = self.b5(x4)  # [B,512,T/32]  <-- Top Level Features

        if not self.use_buaa:
            return x5

        # --- BUAA (Bottom-Up Attention Aggregation) ---
        T_top = x5.shape[-1]

        def att(feat):

            feat_r = F.adaptive_max_pool1d(feat, output_size=T_top)  # [B, C, T_top]

            a = feat_r.mean(dim=1)
            # 3) per-sample min-max -> [B, T_top]
            a = minmax_norm_1d_per_sample(a)
            return a



        a1 = att(x1)
        a2 = att(x2)
        a3 = att(x3)
        a5 = att(x5)


        A = torch.stack([a1, a2, a3, a5], dim=0).amax(dim=0)  # [B, T_top]



        x5 = x5 * (1.0 + A.unsqueeze(1))  # [B, 512, T_top]

        return x5