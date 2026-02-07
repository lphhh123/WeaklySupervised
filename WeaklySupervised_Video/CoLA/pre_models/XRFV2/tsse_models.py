# pre_models/xrfv2/tsse_models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


WSDDN_PATH = '/home/lipei/project/WSDDN'
if WSDDN_PATH not in sys.path:
    sys.path.append(WSDDN_PATH)


from models.TAD.embedding import Embedding
from models.TAD.backbone import TSSE
from models.mamba.backbones import MambaBackbone





def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class TADEmbedding_7s(nn.Module):

    def __init__(self, in_channels, out_channels=512, layer=3, input_length=478, embedding_stride=1):
        super().__init__()
        self.embedding = Embedding(in_channels, stride=embedding_stride)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            emb = self.embedding(dummy)
        L = int(emb.size(-1))
        self.skip_tsse = nn.ModuleList()
        for i in range(int(layer)):
            L_out = _ceil_div(L, 2)
            self.skip_tsse.append(TSSE(in_channels=out_channels, out_channels=256, length=L_out))
            L = L_out
        self.out_len = L

    def forward(self, x):
        x = self.embedding(x)
        for blk in self.skip_tsse:
            x = blk(x)
        return x


def _pad_to_multiple(x: torch.Tensor, multiple: int):
    L = x.size(-1)
    pad = (multiple - (L % multiple)) % multiple
    if pad > 0:
        x = F.pad(x, (0, pad), value=0.0)
    return x, pad


def _crop_pyramid_to_valid(feats, valid_len0: int, scale_factor: int = 2):
    if not isinstance(feats, (list, tuple)): return feats
    out = []
    L = int(valid_len0)
    for f in feats:
        Li = min(f.size(-1), L)
        out.append(f[..., :Li])
        L = _ceil_div(L, scale_factor)
    return out


class Mamba_config:
    def __init__(self, cfg=None):
        self.layer = 4
        self.n_embd = 512
        self.n_embd_ks = 3
        self.scale_factor = 2
        self.with_ln = True
        self.mamba_type = "dbm"
        if cfg:
            for k, v in cfg.items():
                setattr(self, k, v)
        self.arch = (2, int(self.layer), 4)


class Mamba(nn.Module):
    def __init__(self, config: Mamba_config):
        super().__init__()
        self.cfg = config
        self.mamba_model = MambaBackbone(
            n_in=512,
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            mamba_type=config.mamba_type,
        )

    def forward(self, x: torch.Tensor, batched_masks: torch.Tensor = None):
        B, C, L = x.size()
        if batched_masks is None:
            batched_masks = torch.ones(B, 1, L, dtype=torch.bool, device=x.device)
        feats, masks = self.mamba_model(x, batched_masks)
        return feats



class TSSE_MambaBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512, input_length=478, embed_type="TSSE", embedding_stride=1,
                 tsse_layers=3, mamba_cfg=None):
        super().__init__()

        if embed_type == "Norm":
            self.embedding = Embedding(in_channels, stride=embedding_stride)
        else:
            self.embedding = TADEmbedding_7s(in_channels=in_channels, out_channels=512, layer=tsse_layers,
                                             input_length=input_length, embedding_stride=embedding_stride)

        cfg = Mamba_config(mamba_cfg or {})
        self.backbone = Mamba(cfg)
        self.scale_factor = int(cfg.scale_factor)
        self.arch = cfg.arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding(x)
        B, C, L0 = x.shape
        multiple = self.scale_factor ** (int(self.arch[0]) + int(self.arch[-1]))
        x_pad, pad = _pad_to_multiple(x, multiple)
        Lp = x_pad.size(-1)
        masks = torch.ones(B, 1, Lp, dtype=torch.bool, device=x.device)
        if pad > 0: masks[:, :, -pad:] = False
        feats = self.backbone(x_pad, masks)
        feats = _crop_pyramid_to_valid(feats, valid_len0=L0, scale_factor=self.scale_factor)
        y = feats[0] if isinstance(feats, (list, tuple)) else feats
        return y



class TSSE_7s(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512, input_length=478, embedding_stride=1, tsse_layers=3):
        super().__init__()
        self.embedding = TADEmbedding_7s(in_channels=in_channels, out_channels=512, layer=tsse_layers,
                                         input_length=input_length, embedding_stride=embedding_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)