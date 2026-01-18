# pre_train/pre_tsse_mamba_model_7s.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TAD.embedding import Embedding
from models.TAD.backbone import TSSE
from models.mamba.backbones import MambaBackbone



def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

class TADEmbedding_7s(nn.Module):
    """
    xrfv2 形态：Embedding -> TSSE x layer
    但 length 不写死，用 dummy 推断，避免 478 这种长度在 stride/ceil 上踩坑。
    """
    def __init__(self, in_channels, out_channels=512, layer=3, input_length=478, embedding_stride=1):
        super().__init__()
        self.embedding = Embedding(in_channels, stride=embedding_stride)  # xrfv2 Norm embedding

        # 先用 dummy 推断 embedding 输出长度
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_length)
            emb = self.embedding(dummy)  # [1,512,L]
        if emb.dim() != 3 or emb.size(1) != out_channels:
            raise ValueError(f"[TADEmbedding_7s] embedding output mismatch: {emb.shape}")

        L = int(emb.size(-1))
        self.skip_tsse = nn.ModuleList()
        for i in range(int(layer)):
            L_out = _ceil_div(L, 2)  # TSSE stride=2 近似输出长度
            # 注意：xrfv2 这里 out_channels 传 256 也行（原实现里其实写死了 512/1024/2048）
            self.skip_tsse.append(TSSE(in_channels=out_channels, out_channels=256, length=L_out))
            L = L_out

        self.out_len = L  # 记录最终长度，便于调试

    def forward(self, x):
        x = self.embedding(x)      # [B,512,L0]
        for blk in self.skip_tsse:
            x = blk(x)             # 每层大约 /2
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
    valid_len0: level0 的有效长度（未 padding 前）
    假设每降一层长度大约 /scale_factor（与 xrfv2/FPNIdentity 一致）
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
    复刻 xrfv2 的 Mamba_config 形态（但不依赖 register 系统）
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
    复刻 xrfv2 的 Mamba wrapper：
      MambaBackbone + (可选)FPNIdentity
    但 forward 支持传入外部 batched_masks（为了 padding/mask）
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

        feats, masks = self.mamba_model(x, batched_masks)  # xrfv2 原接口

        return feats


class TSSE_MambaBackbone_7s(nn.Module):
    """
    完全 xrfv2 形态：
      embedding = Embedding 或 TADEmbedding_7s
      backbone  = Mamba(config)
    forward:
      x = embedding(x)
      (pad+mask)
      feats = backbone(x, mask)
      return feats[0] (最高分辨率)
    """
    def __init__(
        self,
        in_channels: int = 30,
        feat_dim: int = 512,
        input_length: int = 478,
        embed_type: str = "Norm",       # "Norm" or "TSSE"
        embedding_stride: int = 1,
        tsse_layers: int = 3,           # 对齐 xrfv2 TADEmbedding 默认 3
        mamba_cfg: dict = None,
    ):
        super().__init__()
        if feat_dim != 512:
            raise ValueError("xrfv2 的 TSSE/Mamba 通道写死为 512 系，预训练先固定 feat_dim=512")

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

        # 用于 pad_multiple 计算（避免 7/8 mismatch）
        self.scale_factor = int(cfg.scale_factor)
        self.arch = cfg.arch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,in_channels,T]
        x = self.embedding(x)  # [B,512,L0]
        B, C, L0 = x.shape

        # ---- pad to safe multiple ----
        # 常见 arch=(2,*,4), scale=2 => multiple=2^(2+4)=64
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
        # 更稳：用 mean 而不是 AdaptiveAvgPool，避免你以后改动长度/裁剪出问题
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
        tsse_layers: int = 3,           # 对齐 xrfv2 TADEmbedding 默认 3
    ):
        super().__init__()
        if feat_dim != 512:
            raise ValueError("xrfv2 的 TSSE/Mamba 通道写死为 512 系，预训练先固定 feat_dim=512")

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
        # 更稳：用 mean 而不是 AdaptiveAvgPool，避免你以后改动长度/裁剪出问题
        pooled = feat.mean(dim=-1)   # [B,512]
        return self.fc(pooled)