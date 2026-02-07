# builder_pretrainbackbone.py
import os
from typing import Type, Tuple

from torch.nn import init

from pre_train.pre_model import *


PRETRAINED_ZOO = {
    "CNN1D": {
        "cls": CNN1DBackbone,
        # "ckpt": "/home/lipei/project/WSDDN/pre_train/person2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "ckpt": "/home/lipei/project/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_pretrain_best.pth",
        # "ckpt": "/home/lipei/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "VGG1D_BUAA": {
        "cls": VGG1DBackboneBUAA,
        "ckpt": "/home/lipei/project/WSDDN/pre_train/XRFV2_all_VGG1DClassifier_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "VGG1D": {
        "cls": VGG1DBackbone,
        "ckpt": "/home/lipei/project/WSDDN/pre_train/XRFV2_all_VGG1DClassifier_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "CNN1D_person2": {
        "cls": CNN1DBackbone,
        # "ckpt": "/home/lipei/project/WSDDN/pre_train/person2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "ckpt": "/home/lipei/project/WSDDN/pre_train/person2/XRFV2_all_CNN1DClassifier_pretrain_best.pth",
        # "ckpt": "/home/lipei/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "CNN1D_person4": {
        "cls": CNN1DBackbone,
        # "ckpt": "/home/lipei/project/WSDDN/pre_train/person2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "ckpt": "/home/lipei/project/WSDDN/pre_train/person4/XRFV2_all_CNN1DClassifier_pretrain_best.pth",
        # "ckpt": "/home/lipei/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "CNN1D_person5": {
        "cls": CNN1DBackbone,
        # "ckpt": "/home/lipei/project/WSDDN/pre_train/person2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "ckpt": "/home/lipei/project/WSDDN/pre_train/person5/XRFV2_all_CNN1DClassifier_pretrain_best.pth",
        # "ckpt": "/home/lipei/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "CNN1D_person6": {
        "cls": CNN1DBackbone,
        # "ckpt": "/home/lipei/project/WSDDN/pre_train/person2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "ckpt": "/home/lipei/project/WSDDN/pre_train/person6/XRFV2_all_CNN1DClassifier_pretrain_best.pth",
        # "ckpt": "/home/lipei/WSDDN/pre_train/XRFV2_all_CNN1DClassifier_7s_pretrain_best.pth",
        "feat_dim": 512,
        "input_length": 2048,
    },
    "TCN": TCNBackbone_7s,
    "ResTCN": ResTCNBackbone_7s,
    "MSTCN": MSTCNBackbone_7s,
    "LSTM": LSTMBackbone_7s,
    "Transformer": TransformerBackbone_7s,
    "ActionFormer": ActionFormerBackbone_7s,
}

def _clean_state_dict(state):
    """
    Message：
      - Message state_dict
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - DDP "module." Message
    """
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint type: {type(state)}")

    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state

def get_pretrained_spec(pretrained_name: str) -> dict:
    pretrained_name = str(pretrained_name).strip()
    if pretrained_name not in PRETRAINED_ZOO:
        raise ValueError(f"Unknown pretrained_name={pretrained_name}, available={list(PRETRAINED_ZOO.keys())}")
    return PRETRAINED_ZOO[pretrained_name]

def _infer_T_global(backbone: nn.Module, in_channels: int, device, input_length: int = 2048) -> int:
    backbone.eval()
    with torch.no_grad():
        dummy = torch.randn(2, in_channels, input_length, device=device)
        feat = backbone(dummy)  # expect [B,C,T]
        if feat.dim() != 3:
            raise ValueError(f"Backbone output must be [B,C,T], got {feat.shape}")
        return int(feat.shape[-1])


def _init_backbone_weights(m: nn.Module):
    """
    Message：Conv/Linear Message kaiming，Norm Message weight=1 bias=0
    """
    if isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0.)
    elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
        if hasattr(m, "weight") and m.weight is not None:
            init.constant_(m.weight, 1.)
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias, 0.)

def load_pretrained_backbone(pretrained_name: str, in_channels: int = 30, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = get_pretrained_spec(pretrained_name)
    backbone_class = spec["cls"]
    ckpt_path = spec.get("ckpt", None)
    feat_dim = int(spec.get("feat_dim", 512))
    input_length = int(spec.get("input_length", 2048))

    # 1) build backbone
    try:
        backbone = backbone_class(in_channels=in_channels, feat_dim=feat_dim, input_length=input_length)
    except TypeError:
        try:
            backbone = backbone_class(in_channels=in_channels, feat_dim=feat_dim)
        except TypeError:
            backbone = backbone_class(in_channels=in_channels)

    backbone = backbone.to(device)

    # 2) load weights
    if ckpt_path is not None and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = _clean_state_dict(state)
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        print(f"[Backbone] Message")
        print(f"[Backbone] name={pretrained_name}")
        print(f"  ckpt={ckpt_path}")
        print(f"  missing={missing}")
        print(f"  unexpected={unexpected}")
    else:
        print(f"[Backbone] Message（Messageckpt）")
        print(f"[Backbone] name={pretrained_name}")


    # 3) infer T_global
    T_global = _infer_T_global(backbone, in_channels=in_channels, device=device, input_length=input_length)
    print(f"[Backbone] output: feat_dim={feat_dim}, T_global={T_global}")

    return backbone, T_global