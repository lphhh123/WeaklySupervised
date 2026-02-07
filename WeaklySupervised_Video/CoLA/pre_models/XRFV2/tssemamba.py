import torch
import torch.nn as nn
import sys
import os


WSDDN_PATH = '/home/lipei/project/WSDDN'
if WSDDN_PATH not in sys.path:
    sys.path.append(WSDDN_PATH)








from .impl_tsse_mamba import TSSE_MambaBackbone_7s


class TSSEMambaBackbone(nn.Module):
    def __init__(self, in_channels=30):
        super().__init__()



        # mamba_cfg={"layer": 4, "mamba_type": "dbm"}
        self.backbone = TSSE_MambaBackbone_7s(
            in_channels=in_channels,
            feat_dim=512,
            input_length=2048,
            embed_type="TSSE",
            tsse_layers=2,
            mamba_cfg={"layer": 4, "mamba_type": "dbm"}
        )

        self.out_dim = 512

    def forward(self, x):
        # x: [B, C, T]
        return self.backbone(x)