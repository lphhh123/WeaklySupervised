import torch
import torch.nn as nn

from models.WSDDN_model import WSDDN
from models.PCL_OICR_model import IMU_PCL_OICR

def count_parameters(model: nn.Module):
    """
    计算并打印 PyTorch 模型的总参数量和可训练参数量。

    Args:
        model (nn.Module): 需要计算参数的 PyTorch 模型。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")


# 实例化模型
example_model = IMU_PCL_OICR(feat_dim=512,num_classes=21,use_pcl=True)

# 调用函数计算参数
count_parameters(example_model)

