import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import Mamba   # video-mamba-suite 里的实现
from .WSDDN_model import TemporalSPP1D

class ProposalMambaBlock(nn.Module):
    """Basic Mamba block on proposal dimension:
       x: [B, P, D]  (P=proposals, D=hidden_dim)
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            bimamba_type="v2"
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, P, D]
        h = self.norm(x)
        h = self.mamba(h)               # Mamba 期望 [B, L, D]，这里 L=P
        return x + self.dropout(h)      # residual

class ProposalMambaEncoderBi(nn.Module):
    """K-layer bidirectional Mamba encoder (xrfv2-style backbone for proposals)."""
    def __init__(
        self,
        dim,
        depth=3,
        dropout=0.1,
        share_weights=True,
        merge_mode="sum",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            ProposalBiMambaBlock(
                dim, dropout=dropout,
                share_weights=share_weights,
                merge_mode=merge_mode,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: [B, P, D]
        for blk in self.blocks:
            x = blk(x)
        return x

class ProposalBiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block over proposals:
      - forward Mamba
      - backward Mamba (在 P 维度上翻转再翻回来)
    """
    def __init__(self, dim, dropout=0.1, share_weights=True, merge_mode="sum"):
        super().__init__()
        assert merge_mode in ("sum", "concat")
        self.norm = nn.LayerNorm(dim)
        self.share_weights = share_weights
        self.merge_mode = merge_mode

        self.mamba_f = Mamba(
            d_model=dim,
            bimamba_type="v2"
        )
        self.mamba_b = self.mamba_f if share_weights else Mamba(dim)

        if merge_mode == "concat":
            self.merge_proj = nn.Linear(2 * dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, P, D]
        h = self.norm(x)

        # forward
        h_f = self.mamba_f(h)              # [B, P, D]

        # backward（在 proposal 维度上翻转）
        h_rev = torch.flip(h, dims=[1])    # [B, P, D]
        h_b = self.mamba_b(h_rev)
        h_b = torch.flip(h_b, dims=[1])    # 再翻回来

        if self.merge_mode == "sum":
            h_m = h_f + h_b                # 双向加和
        else:
            h_m = torch.cat([h_f, h_b], dim=-1)  # [B, P, 2D]
            h_m = self.merge_proj(h_m)           # → [B, P, D]

        return x + self.dropout(h_m)       # residual

class ProposalMambaEncoderBasic(nn.Module):
    """K-layer unidirectional Mamba encoder over proposals."""
    def __init__(self, dim, depth=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ProposalMambaBlock(dim, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x):
        # x: [B, P, D]
        for blk in self.blocks:
            x = blk(x)
        return x


class WSDDN_MambaBasic(nn.Module):
    def __init__(
        self,
        num_classes=30,
        feat_dim=512,
        spp_levels=(1, 2, 4),
        pool_type="avg",
        hidden_dim=512,
        mamba_depth=2,
        mamba_dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # 1D SPP
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)

        # 先映射到 hidden_dim 再送入 Mamba
        self.in_proj = nn.Linear(self.spp_out_dim, hidden_dim)

        # basic Mamba encoder（单向，堆 mamba_depth 层）
        self.mamba_encoder = ProposalMambaEncoderBasic(
            dim=hidden_dim,
            depth=mamba_depth,
            dropout=mamba_dropout,
        )

        # MIL 双分支
        self.class_branch = nn.Linear(hidden_dim, num_classes)
        self.det_branch   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, proposal_boxes):
        """
        x: [B, C, T_global]   — backbone 输出
        proposal_boxes: [B, P, 2]
        """
        B, C, T_global = x.shape
        P = proposal_boxes.shape[1]

        proposal_features = []

        # ===== 对每个 proposal 做 1D SPP =====
        for b in range(B):
            for p in range(P):
                start = int(proposal_boxes[b, p, 0].item())
                end   = int(proposal_boxes[b, p, 1].item())
                start = max(0, min(start, T_global - 1))
                end   = max(start + 1, min(end, T_global))

                seg = x[b:b+1, :, start:end]         # [1, C, L_p]
                feat = self.spp(seg)                 # [1, C * sum(levels)]
                proposal_features.append(feat)

        # [B*P, C'] -> [B,P,C']
        proposal_features = torch.cat(proposal_features, dim=0)   # [B*P, spp_out_dim]
        proposal_features = self.in_proj(proposal_features)       # [B*P, hidden_dim]
        feat = proposal_features.view(B, P, -1)                  # [B, P, hidden_dim]

        # ===== Mamba encoder over proposals =====
        feat = self.mamba_encoder(feat)                          # [B, P, hidden_dim]

        feat_fc7 = feat

        # ===== MIL 双分支 =====
        feat_flat = feat.view(B * P, -1)                         # [B*P, hidden_dim]

        class_logits = self.class_branch(feat_flat).view(B, P, -1)   # [B,P,C]
        det_logits   = self.det_branch(feat_flat).view(B, P, -1)     # [B,P,C]

        class_prob = F.softmax(class_logits, dim=2)  # 按类别 softmax
        det_prob   = F.softmax(det_logits,   dim=1)  # 按 proposal softmax

        joint_prob = class_prob * det_prob           # [B,P,C]
        video_prob = joint_prob.sum(dim=1)           # [B,C]


        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat_fc7,                    # [B,P,hidden_dim]
        }


class WSDDN_MambaBi(nn.Module):
    def __init__(
        self,
        num_classes=30,
        feat_dim=512,
        spp_levels=(1, 2, 4),
        pool_type="max",
        hidden_dim=512,
        mamba_depth=3,
        mamba_dropout=0.1,
        share_weights=True,
        merge_mode="sum",
    ):
        super().__init__()
        self.num_classes = num_classes

        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul

        self.in_proj = nn.Linear(self.spp_out_dim, hidden_dim)

        self.mamba_encoder = ProposalMambaEncoderBi(
            dim=hidden_dim,
            depth=mamba_depth,
            dropout=mamba_dropout,
            share_weights=share_weights,
            merge_mode=merge_mode,
        )

        self.class_branch = nn.Linear(hidden_dim, num_classes)
        self.det_branch   = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, proposal_boxes):
        B, C, T_global = x.shape
        P = proposal_boxes.shape[1]

        proposal_features = []
        for b in range(B):
            for p in range(P):
                start = int(proposal_boxes[b, p, 0].item())
                end   = int(proposal_boxes[b, p, 1].item())
                start = max(0, min(start, T_global - 1))
                end   = max(start + 1, min(end, T_global))
                seg = x[b:b+1, :, start:end]
                feat = self.spp(seg)
                proposal_features.append(feat)

        proposal_features = torch.cat(proposal_features, dim=0)
        proposal_features = self.in_proj(proposal_features)      # [B*P, hidden_dim]
        feat = proposal_features.view(B, P, -1)                  # [B,P,hidden_dim]

        feat = self.mamba_encoder(feat)                          # 双向 Mamba 堆

        feat_fc7 = feat

        feat_flat = feat.view(B * P, -1)
        class_logits = self.class_branch(feat_flat).view(B, P, -1)
        det_logits   = self.det_branch(feat_flat).view(B, P, -1)

        class_prob = F.softmax(class_logits, dim=2)
        det_prob   = F.softmax(det_logits,   dim=1)

        joint_prob = class_prob * det_prob
        video_prob = joint_prob.sum(dim=1)

        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat,
        }
