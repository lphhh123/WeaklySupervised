import torch
import torch.nn as nn
import sys
import os

# 1. 动态添加 WSDDN 路径，以便能 import 其中的 models 模块
WSDDN_PATH = '/home/lipei/project/WSDDN'
if WSDDN_PATH not in sys.path:
    sys.path.append(WSDDN_PATH)

# 2. 导入学姐写的预训练模型定义
# 注意：你需要把 pre_tsse_mamba_model_7s.py 放在 WSDDN 的 pre_train 目录下
# 或者如果文件在你手边，直接 import 它 (假设它在 pre_models/xrfv2 下)
# 这里假设你把 pre_tsse_mamba_model_7s.py 的内容直接粘贴到了这里，
# 或者作为一个辅助文件放在旁边。为了方便，我们假设你把上面的代码保存为了
# pre_models/xrfv2/impl_tsse_mamba.py，然后在这里引用。

from .impl_tsse_mamba import TSSE_MambaBackbone_7s  # 引用你刚才发给我的那个文件内容


class TSSEMambaBackbone(nn.Module):
    def __init__(self, in_channels=30):
        super().__init__()

        # 实例化 Backbone
        # 参数根据学姐的 pre_imu.py 里的配置来
        # mamba_cfg={"layer": 4, "mamba_type": "dbm"}
        self.backbone = TSSE_MambaBackbone_7s(
            in_channels=in_channels,
            feat_dim=512,
            input_length=2048,  # CoLA 是长序列，这里填 2048
            embed_type="TSSE",
            tsse_layers=2,
            mamba_cfg={"layer": 4, "mamba_type": "dbm"}
        )

        self.out_dim = 512

    def forward(self, x):
        # x: [B, C, T]
        return self.backbone(x)