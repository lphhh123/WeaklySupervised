# pre_train/mamba_wrapper.py
import torch
import torch.nn as nn
import torch.nn.init as init

from .mamba.backbones import MambaBackbone
# from .mamba.necks import FPNIdentity


class MambaConfig:
    """
    替代 xrfv2 的 Config / Mamba_config（不依赖 register / utils.basic_config）
    """
    def __init__(self, cfg=None):
        # xrfv2 默认值
        self.layer = 4
        self.n_embd = 512
        self.n_embd_ks = 3
        self.scale_factor = 2
        self.with_ln = True
        self.mamba_type = "dbm"
        self.arch = (2, self.layer, 4)  # (base_conv, stem, branch)

        if cfg is not None:
            for k, v in cfg.items():
                setattr(self, k, v)
            # 如果用户只改了 layer，但没改 arch，则跟着更新一下 arch
            if "layer" in cfg and "arch" not in cfg:
                self.arch = (2, self.layer, 4)


class Mamba(nn.Module):
    """
    等价于你贴的 xrfv2 那个 Mamba wrapper：
      MambaBackbone -> FPNIdentity -> return fpn_feats (多尺度 list/tuple)
    """
    def __init__(self, config: MambaConfig, n_in: int = 512):
        super().__init__()

        self.mamba_model = MambaBackbone(
            n_in=n_in,
            n_embd=config.n_embd,
            n_embd_ks=config.n_embd_ks,
            arch=config.arch,
            scale_factor=config.scale_factor,
            with_ln=config.with_ln,
            mamba_type=config.mamba_type,
        )

        # # 注意：arch[-1] + 1 是因为输出 feat levels = branch_depth + 1
        # self.neck = FPNIdentity(
        #     in_channels=[config.n_embd] * (config.arch[-1] + 1),
        #     out_channel=config.n_embd,
        #     scale_factor=config.scale_factor,
        #     with_ln=config.with_ln
        # )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.neck.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, L = x.size()
        masks = torch.ones(B, 1, L, dtype=torch.bool, device=x.device)
        feats, masks = self.mamba_model(x, masks)
        # fpn_feats, fpn_masks = self.neck(feats, masks)
        # return fpn_feats   # 多尺度 tuple/list
        use_feat = feats[0]  # 最长尺度
        return use_feat
