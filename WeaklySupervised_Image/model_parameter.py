import torch
import torch.nn as nn

from models.WSDDN_model import WSDDN
from models.PCL_OICR_model import IMU_PCL_OICR

def count_parameters(model: nn.Module):
    """
    Message PyTorch Message。

    Args:
        model (nn.Module): Message PyTorch Message。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")



example_model = IMU_PCL_OICR(feat_dim=512,num_classes=21,use_pcl=True)


count_parameters(example_model)

