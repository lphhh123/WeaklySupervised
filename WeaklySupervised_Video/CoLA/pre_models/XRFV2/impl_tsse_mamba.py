# pre_train/pre_tsse_mamba_model_7s.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
if '/home/lipei/project/WSDDN' not in sys.path:
    sys.path.append('/home/lipei/project/WSDDN')

from models.TAD.embedding import Embedding
from models.TAD.backbone import TSSE
from models.mamba.backbones import MambaBackbone



def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

class TADEmbedding_7s(nn.Module):
    """
    xrfv2 Message：Embedding -> TSSE x layer
    Message length Message，Message dummy Message，Message 478 Message stride/ceil Message。
    """
    def __init__(self, in_channels, out_channels=512, layer=3, input_length=478, embedding_stride=1):
        super().__init__()
        self.embedding = Embedding(in_channels, stride=embedding_stride)  # xrfv2 Norm embedding


        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            emb = self.embedding(dummy)  # [1,512,L]
        if emb.dim() != 3 or emb.size(1) != out_channels:
            raise ValueError(f"[TADEmbedding_7s] embedding output mismatch: {emb.shape}")

        L = int(emb.size(-1))
        self.skip_tsse = nn.ModuleList()
        for i in range(int(layer)):
            L_out = _ceil_div(L, 2)

            self.skip_tsse.append(TSSE(in_channels=out_channels, out_channels=256, length=L_out))
            L = L_out

        self.out_len = L

    def forward(self, x):
        x = self.embedding(x)      # [B,512,L0]
        for blk in self.skip_tsse:
            x = blk(x)
        return x                   # [B,512,L_final]

def _pad_to_multiple(x: torch.Tensor, multiple: int):
    """
    x: [B,C,L]
    return x_pad, pad
    """
    L = x.size(-1)
    pad = (multiple - (L % multiple)) % multiple
    if pad > 0:
        x = F.pad(x, (0, pad), value=0.0)
    return x, pad


def _crop_pyramid_to_valid(feats, valid_len0: int, scale_factor: int = 2):
    """
    feats: list of [B,C,L_i]
    valid_len0: level0 Message（Message padding Message）
    Message /scale_factor（Message xrfv2/FPNIdentity Message）
    """
    if not isinstance(feats, (list, tuple)):
        return feats

    out = []
    L = int(valid_len0)
    for f in feats:
        Li = min(f.size(-1), L)
        out.append(f[..., :Li])
        L = _ceil_div(L, scale_factor)
    return out


class Mamba_config:
    """
    Message xrfv2 Message Mamba_config Message（Message register Message）
    """
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
    """
    Message xrfv2 Message Mamba wrapper：
      MambaBackbone + (Message)FPNIdentity
    Message forward Message batched_masks（Message padding/mask）
    """
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
    """
    Message xrfv2 Message：
      embedding = Embedding Message TADEmbedding_7s
      backbone  = Mamba(config)
    forward:
      x = embedding(x)
      (pad+mask)
      feats = backbone(x, mask)
      return feats[0] (Message)
    """
    def __init__(
        self,
        in_channels: int = 30,
        feat_dim: int = 512,
        input_length: int = 478,
        embed_type: str = "Norm",       # "Norm" or "TSSE"
        embedding_stride: int = 1,
        tsse_layers: int = 3,
        mamba_cfg: dict = None,
    ):
        super().__init__()
        if feat_dim != 512:
            raise ValueError("xrfv2 Message TSSE/Mamba Message 512 Message，Message feat_dim=512")

        # ---- 1) embedding (xrfv2) ----
        if embed_type == "Norm":
            self.embedding = Embedding(in_channels, stride=embedding_stride)
        else:
            self.embedding = TADEmbedding_7s(
                in_channels=in_channels,
                out_channels=512,
                layer=tsse_layers,
                input_length=input_length,
                embedding_stride=embedding_stride,
            )

        # ---- 2) backbone = Mamba(xrfv2) ----
        cfg = Mamba_config(mamba_cfg or {})
        self.backbone = Mamba(cfg)


        self.scale_factor = int(cfg.scale_factor)
        self.arch = cfg.arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,in_channels,T]
        x = self.embedding(x)  # [B,512,L0]
        B, C, L0 = x.shape

        # ---- pad to safe multiple ----

        multiple = self.scale_factor ** (int(self.arch[0]) + int(self.arch[-1]))
        x_pad, pad = _pad_to_multiple(x, multiple)
        Lp = x_pad.size(-1)

        masks = torch.ones(B, 1, Lp, dtype=torch.bool, device=x.device)
        if pad > 0:
            masks[:, :, -pad:] = False

        feats = self.backbone(x_pad, masks)  # list pyramid

        # ---- crop pyramid back to valid lengths (remove padding influence) ----
        feats = _crop_pyramid_to_valid(feats, valid_len0=L0, scale_factor=self.scale_factor)

        y = feats[0] if isinstance(feats, (list, tuple)) else feats  # highest resolution
        # for i, f in enumerate(feats):
        #     print(i, f.shape)
        return y  # [B,512,T_feat]


class TSSEMambaClassifier_7s(nn.Module):
    def __init__(self, num_classes, task="single", in_channels=30, input_length=478, embed_type="TSSE", tsse_layers=3, mamba_cfg=None):
        super().__init__()
        self.task = task
        self.backbone = TSSE_MambaBackbone_7s(
            in_channels=in_channels,
            feat_dim=512,
            input_length=input_length,
            embed_type=embed_type,
            embedding_stride=1,
            tsse_layers=tsse_layers,
            mamba_cfg=mamba_cfg,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)  # [B,512,T]

        pooled = feat.mean(dim=-1)   # [B,512]
        return self.fc(pooled)


class TSSE_7s(nn.Module):
    """
    TADEmbedding_7s
    forward:
      x = embedding(x)
      (pad+mask)
    """
    def __init__(
        self,
        in_channels: int = 30,
        feat_dim: int = 512,
        input_length: int = 478,
        embedding_stride: int = 1,
        tsse_layers: int = 3,
    ):
        super().__init__()
        if feat_dim != 512:
            raise ValueError("xrfv2 Message TSSE/Mamba Message 512 Message，Message feat_dim=512")

        # ---- 1) embedding (TSSE) ----
        self.embedding = TADEmbedding_7s(
            in_channels=in_channels,
            out_channels=512,
            layer=tsse_layers,
            input_length=input_length,
            embedding_stride=embedding_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,in_channels,T]
        x = self.embedding(x)  # [B,512,L0]

        return x


class TSSEClassifier_7s(nn.Module):
    def __init__(self, num_classes, task="single", in_channels=30, input_length=478, tsse_layers=3):
        super().__init__()
        self.task = task
        self.backbone = TSSE_7s(
            in_channels=in_channels,
            feat_dim=512,
            input_length=input_length,
            embedding_stride=1,
            tsse_layers=tsse_layers,
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)  # [B,512,T]

        pooled = feat.mean(dim=-1)   # [B,512]
        return self.fc(pooled)