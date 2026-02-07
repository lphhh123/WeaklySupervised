# PCLHead.py
from pre_train.pre_tsse_mamba_model_7s import TSSE_MambaBackbone_7s
from .PCL_OICR_model import pcl_1d, oicr_1d
from .adapters import TemporalAdapter1D
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.TAD.backbone import TSSE
from .WSDDN_model import TemporalSPP1D
from .mamba.backbones import MambaBackbone
from .pcl_model_blocks import mil_outputs, mil_losses, OICRLosses, refine_outputs


class Head_PCL(nn.Module):
    """
    Head_PCL includes adapter + pcl/oicr; pretrained backbone is TSSE + Mamba.
    """
    def __init__(self,
                 feat_dim,
                 num_classes,
                 refine_times=3,
                 use_pcl=True,
                 fg_thresh=0.5,
                 bg_thresh=0.1,
                 graph_iou_thresh=0.5,
                 max_pc_num=3,
                 hidden_dim=4096,
                 spp_levels=(1, 2, 4),
                 pool_type="avg",
                 adapter_cfg=None,
                 ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.refine_times = refine_times
        self.use_pcl = use_pcl
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.graph_iou_thresh = graph_iou_thresh
        self.max_pc_num = max_pc_num

        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)

        self.fc1 = nn.Linear(self.spp_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # MIL + refine heads
        self.mil_head = mil_outputs(hidden_dim, num_classes)
        self.refine_head = refine_outputs(hidden_dim, num_classes + 1, self.refine_times)
        self.refine_losses = nn.ModuleList(
            [OICRLosses() for _ in range(refine_times)]
        )

        adapter_cfg = adapter_cfg or {}
        adapter_enable = bool(adapter_cfg.get("enable", False))
        if adapter_enable:
            self.adapter = TemporalAdapter1D(
                channels=feat_dim,
                bottleneck=int(adapter_cfg.get("bottleneck", 128)),
                kernel_size=int(adapter_cfg.get("kernel_size", 3)),
                dropout=float(adapter_cfg.get("dropout", 0.1)),
                scale=float(adapter_cfg.get("scale", 0.1)),
                use_dwconv=bool(adapter_cfg.get("use_dwconv", True)),
            )
        else:
            self.adapter = nn.Identity()

    def pool_proposals_1d_spp(self, global_feat, proposal_boxes):
        """
        Proposal pooling with 1D SPP.
        global_feat:    [B, C, T]
        proposal_boxes: [B, P, 2] (start, end)
        Return:         [B, P, C * sum(levels)]
        """
        B, C, T = global_feat.shape
        B2, P, _ = proposal_boxes.shape
        assert B == B2
        pooled = []

        for b in range(B):
            feat_b = global_feat[b:b+1]       # [1, C, T]
            boxes_b = proposal_boxes[b]       # [P, 2]
            feats_b = []
            for p in range(P):
                s = int(boxes_b[p, 0].item())
                e = int(boxes_b[p, 1].item())
                s = max(0, min(s, T - 1))
                e = max(s + 1, min(e, T))

                seg = feat_b[:, :, s:e]      # [1, C, L_p]
                # 1D SPP → [1, C * sum(levels)]
                feat_p = self.spp(seg)       # [1, C']
                feats_b.append(feat_p.squeeze(0))  # [C']

            feats_b = torch.stack(feats_b, dim=0)  # [P, C']
            pooled.append(feats_b)

        return torch.stack(pooled, dim=0)          # [B, P, C']

    def forward(self, global_feat, proposal_boxes, labels=None):
        """
        global_feat: [B, C, T] or [B, T, C]
        proposal_boxes: [B, P, 2]
        labels: [B, num_classes] (0/1), required in training
        """
        if global_feat.dim() != 3:
            raise ValueError("global_feat must be [B, C, T] or [B, T, C]")

        if global_feat.size(1) == self.feat_dim:
            feat = global_feat
        elif global_feat.size(2) == self.feat_dim:
            feat = global_feat.transpose(1, 2)
        else:
            raise ValueError(
                f"global_feat shape mismatch feat_dim={self.feat_dim}, got {global_feat.shape}"
            )

        feat = self.adapter(feat)  # [B, C, T]

        B, C, T = feat.shape
        B2, P, _ = proposal_boxes.shape
        assert B == B2

        # 1) SPP pooling
        proposal_feats = self.pool_proposals_1d_spp(feat, proposal_boxes)  # [B, P, C']
        _, _, D = proposal_feats.shape
        x = proposal_feats.view(B * P, D)  # [B*P, C']

        # 2) MLP
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # 3) MIL + refine
        # mil_score = self.mil_head(x)             # [B*P, num_classes]
        # refine_scores_flat = self.refine_head(x) # list of [B*P, num_classes+1]
        x_bp = x.view(B, P, -1)
        mil_score = self.mil_head(x_bp)

        device = x.device
        output = {}

        if self.training:
            if labels is None:
                raise ValueError("labels must be provided during training")

            # image-level MIL loss
            mil_score_vid = mil_score.view(B, P, self.num_classes).sum(dim=1)  # [B, C]
            loss_im_cls = mil_losses(mil_score_vid, labels.float())
            output["losses"] = {"loss_im_cls": loss_im_cls}

            boxes_np = proposal_boxes.detach().cpu().numpy()           # [B, P, 2]
            labels_np = labels.detach().cpu().numpy()                  # [B, C]
            mil_np = mil_score.detach().cpu().numpy().reshape(B, P, self.num_classes)
            refine_np = [
                rs.detach().cpu().numpy().reshape(B, P, self.num_classes + 1)
                for rs in refine_scores_flat
            ]

            for i_refine in range(self.refine_times):
                loss_refine_all = 0.0

                for b in range(B):
                    boxes_b = boxes_np[b]                 # [P, 2]
                    im_labels_b = labels_np[b][None, :]   # [1, C]

                    if i_refine == 0:
                        cls_prob = mil_np[b]              # [P, C]
                    else:
                        cls_prob = refine_np[i_refine - 1][b, :, 1:]  # [P, C]

                    if self.use_pcl:
                        pcl_out = pcl_1d(
                            boxes_b, cls_prob, im_labels_b,
                            fg_thresh=self.fg_thresh,
                            bg_thresh=self.bg_thresh,
                            graph_iou_thresh=self.graph_iou_thresh,
                            max_pc_num=self.max_pc_num
                        )
                    else:
                        pcl_out = oicr_1d(
                            boxes_b, cls_prob, im_labels_b,
                            fg_thresh=self.fg_thresh,
                            bg_thresh=self.bg_thresh
                        )

                    lbl = torch.from_numpy(pcl_out["labels"].reshape(-1)).long().to(device)
                    w = torch.from_numpy(pcl_out["cls_loss_weights"].reshape(-1)).float().to(device)
                    gt_assign = torch.from_numpy(pcl_out["gt_assignment"].reshape(-1)).long().to(device)

                    prob_b = refine_scores_flat[i_refine].view(
                        B, P, self.num_classes + 1
                    )[b]  # [P, C+1]
                    loss_b = self.refine_losses[i_refine](prob_b, lbl, w, gt_assign)
                    loss_refine_all = loss_refine_all + loss_b

                loss_refine_avg = loss_refine_all / B
                if i_refine == 0:
                    loss_refine_avg = loss_refine_avg * 3.0

                output["losses"][f"refine_loss{i_refine}"] = loss_refine_avg

        output["mil_score"] = mil_score.view(B, P, self.num_classes)
        output["refine_scores"] = [
            rs.view(B, P, self.num_classes + 1) for rs in refine_scores_flat
        ]

        return output


def map_boxes_input_to_feat(boxes: torch.Tensor, T_in: int, T_feat: int) -> torch.Tensor:
    """
    Map raw sequence coordinates (0..T_in) to feature coordinates (0..T_feat).
    boxes: [B,P,2], end is exclusive (consistent with pooling)
    return: LongTensor [B,P,2] in [0..T_feat]
    """
    if boxes.dtype.is_floating_point:
        s = torch.floor(boxes[..., 0] * T_feat / T_in)
        e = torch.ceil (boxes[..., 1] * T_feat / T_in)
    else:
        s = (boxes[..., 0].float() * T_feat / T_in).floor()
        e = (boxes[..., 1].float() * T_feat / T_in).ceil()

    s = s.clamp(min=0, max=T_feat - 1)
    e = e.clamp(min=1, max=T_feat)

    e = torch.maximum(e, s + 1)

    return torch.stack([s.long(), e.long()], dim=-1)


class Change_Block_PCL_Model(nn.Module):
    """
    Overall model:
      x:[B,C_in,T_in] -> backbone -> feat:[B,512,T_feat]
      proposals (input axis or feat axis) -> head -> losses/scores
    """

    def __init__(self,
                 feat_dim,
                 num_classes,
                 refine_times=3,
                 use_pcl=False,
                 fg_thresh=0.5,
                 bg_thresh=0.1,
                 graph_iou_thresh=0.5,
                 max_pc_num=3,
                 hidden_dim=4096,
                 spp_levels=(1, 2, 4),
                 pool_type="avg",
                 roi_head="fc"):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.refine_times = refine_times
        self.use_pcl = use_pcl
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.graph_iou_thresh = graph_iou_thresh
        self.max_pc_num = max_pc_num

        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)


        # MIL + refine heads
        self.mil_head = mil_outputs(hidden_dim, num_classes)
        self.refine_head = refine_outputs(hidden_dim, num_classes + 1, self.refine_times)
        self.refine_losses = nn.ModuleList(
            [OICRLosses() for _ in range(refine_times)]
        )

        self.roi_head_name = roi_head
        self.roi_head = _build_roi_head(
            self.roi_head_name,
            feat_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            spp_levels=spp_levels,
        )

    def pool_proposals_1d_spp(self, global_feat, proposal_boxes):
        """
        Proposal pooling with 1D SPP.
        global_feat:    [B, C, T]
        proposal_boxes: [B, P, 2] (start, end)
        Return:         [B, P, C * sum(levels)]
        """
        B, C, T = global_feat.shape
        B2, P, _ = proposal_boxes.shape
        assert B == B2
        pooled = []

        for b in range(B):
            feat_b = global_feat[b:b + 1]  # [1, C, T]
            boxes_b = proposal_boxes[b]  # [P, 2]
            feats_b = []
            for p in range(P):
                s = int(boxes_b[p, 0].item())
                e = int(boxes_b[p, 1].item())
                s = max(0, min(s, T - 1))
                e = max(s + 1, min(e, T))

                seg = feat_b[:, :, s:e]  # [1, C, L_p]
                # 1D SPP → [1, C * sum(levels)]
                feat_p = self.spp(seg)  # [1, C']
                feats_b.append(feat_p.squeeze(0))  # [C']

            feats_b = torch.stack(feats_b, dim=0)  # [P, C']
            pooled.append(feats_b)

        return torch.stack(pooled, dim=0)  # [B, P, C']

    def forward(self, global_feat, proposal_boxes, labels=None):
        """
        global_feat: [B, C, T] or [B, T, C]
        proposal_boxes: [B, P, 2]
        labels: [B, num_classes] (0/1), required in training
        """
        if global_feat.dim() != 3:
            raise ValueError("global_feat must be [B, C, T] or [B, T, C]")

        if global_feat.size(1) == self.feat_dim:
            feat = global_feat
        elif global_feat.size(2) == self.feat_dim:
            feat = global_feat.transpose(1, 2)
        else:
            raise ValueError(
                f"global_feat shape mismatch feat_dim={self.feat_dim}, got {global_feat.shape}"
            )

        B, C, T = feat.shape
        B2, P, _ = proposal_boxes.shape
        assert B == B2

        # 1) SPP pooling
        proposal_feats = self.pool_proposals_1d_spp(feat, proposal_boxes)  # [B, P, C']
        _, _, D = proposal_feats.shape
        x = proposal_feats.view(B * P, D)  # [B*P, C']

        x = self.roi_head(x)  # [B*P, hidden_dim]

        # 3) MIL + refine
        mil_score = self.mil_head(x)  # [B*P, num_classes]
        refine_scores_flat = self.refine_head(x)  # list of [B*P, num_classes+1]

        device = x.device
        output = {}

        if self.training:
            if labels is None:
                raise ValueError("labels must be provided during training")

            # image-level MIL loss
            mil_score_vid = mil_score.view(B, P, self.num_classes).sum(dim=1)  # [B, C]
            loss_im_cls = mil_losses(mil_score_vid, labels.float())
            output["losses"] = {"loss_im_cls": loss_im_cls}

            boxes_np = proposal_boxes.detach().cpu().numpy()  # [B, P, 2]
            labels_np = labels.detach().cpu().numpy()  # [B, C]
            mil_np = mil_score.detach().cpu().numpy().reshape(B, P, self.num_classes)
            refine_np = [
                rs.detach().cpu().numpy().reshape(B, P, self.num_classes + 1)
                for rs in refine_scores_flat
            ]

            for i_refine in range(self.refine_times):
                loss_refine_all = 0.0

                for b in range(B):
                    boxes_b = boxes_np[b]  # [P, 2]
                    im_labels_b = labels_np[b][None, :]  # [1, C]

                    if i_refine == 0:
                        cls_prob = mil_np[b]  # [P, C]
                    else:
                        cls_prob = refine_np[i_refine - 1][b, :, 1:]  # [P, C]

                    if self.use_pcl:
                        pcl_out = pcl_1d(
                            boxes_b, cls_prob, im_labels_b,
                            fg_thresh=self.fg_thresh,
                            bg_thresh=self.bg_thresh,
                            graph_iou_thresh=self.graph_iou_thresh,
                            max_pc_num=self.max_pc_num
                        )
                    else:
                        pcl_out = oicr_1d(
                            boxes_b, cls_prob, im_labels_b,
                            fg_thresh=self.fg_thresh,
                            bg_thresh=self.bg_thresh
                        )

                    lbl = torch.from_numpy(pcl_out["labels"].reshape(-1)).long().to(device)
                    w = torch.from_numpy(pcl_out["cls_loss_weights"].reshape(-1)).float().to(device)
                    gt_assign = torch.from_numpy(pcl_out["gt_assignment"].reshape(-1)).long().to(device)

                    prob_b = refine_scores_flat[i_refine].view(
                        B, P, self.num_classes + 1
                    )[b]  # [P, C+1]
                    loss_b = self.refine_losses[i_refine](prob_b, lbl, w, gt_assign)
                    loss_refine_all = loss_refine_all + loss_b

                loss_refine_avg = loss_refine_all / B
                if i_refine == 0:
                    loss_refine_avg = loss_refine_avg * 3.0

                output["losses"][f"refine_loss{i_refine}"] = loss_refine_avg

        output["mil_score"] = mil_score.view(B, P, self.num_classes)
        output["refine_scores"] = [
            rs.view(B, P, self.num_classes + 1) for rs in refine_scores_flat
        ]

        return output

def _flat_to_seq(x_flat: torch.Tensor, C: int, L: int):
    # [N, C*L] -> [N, C, L]
    N, D = x_flat.shape
    if D != C * L:
        raise ValueError(f"D mismatch: got {D}, expect {C*L} (=C*L)")
    return x_flat.view(N, C, L).contiguous()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b



class ROIHead_FC(nn.Module):
    """Equivalent implementation of the original 2*FC (default)."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_flat):
        x = F.relu(self.fc1(x_flat), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        return x

class ROIHead_TSSELite(nn.Module):
    """
    Use a single TSSE by default (with Lspp=7, deeper stacks shrink quickly).
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        if self.feat_dim != 512:
            self.to512 = nn.Conv1d(self.feat_dim, 512, kernel_size=1, bias=False)
        else:
            self.to512 = nn.Identity()

        length_down = _ceil_div(self.Lspp, 2)  # L=7 -> 4

        self.tsse = TSSE(in_channels=512, out_channels=256, length=length_down, kernel_size=3, stride=2)

        self.proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [N, feat_dim*Lspp]  (N=B*P)
        return: [N, hidden_dim]
        """
        x = _flat_to_seq(x_flat, self.feat_dim, self.Lspp)  # [N, C, Lspp]
        x = self.to512(x)                                   # [N, 512, Lspp]

        x = self.tsse(x)                                    # [N, 512, ~ceil(Lspp/2)]
        x = x.mean(dim=-1)                                  # [N, 512]
        x = self.proj(x)                                    # [N, hidden_dim]
        return x

class ROIHead_MambaLite(nn.Module):
    """
    Use the existing MambaBackbone as an ROI encoder (no downsampling to avoid short Lspp).
    Default parameters are defined here rather than read from config.
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.Lspp = Lspp

        mamba_layers = 4
        n_embd_ks = 3
        with_ln = True
        mamba_type = "dbm"

        self.mamba = MambaBackbone(
            n_in=feat_dim,
            n_embd=feat_dim,
            n_embd_ks=n_embd_ks,
            arch=(0, int(mamba_layers), 0),
            scale_factor=1,
            with_ln=with_ln,
            mamba_type=mamba_type,
        )

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_flat):
        x = _flat_to_seq(x_flat, self.feat_dim, self.Lspp)   # [N,C,L]
        mask = torch.ones(x.size(0), 1, x.size(-1), dtype=torch.bool, device=x.device)

        feats, _ = self.mamba(x, mask)
        y = feats[0] if isinstance(feats, (list, tuple)) else feats  # [N,C,L]
        y = y.mean(dim=-1)                                          # [N,C]
        return self.proj(y)                                         # [N,H]


class ROIHead_TSSEMamba(nn.Module):
    """
    ROI head for PCL/OICR (replaces fc1+fc2):

    Input:
      x_flat: [N, feat_dim * Lspp]   (N=B*P)
        - feat_dim = backbone output channels (typically 512)
        - Lspp     = total SPP bins (e.g., levels=(1,2,4) -> Lspp=7)
    Output:
      [N, hidden_dim] (default 4096), feeds mil_head / refine_head

    Structure:
      reshape -> (optional 1x1 conv to 512) -> TSSE -> Mamba (no downsampling)
      -> mean pool -> MLP(->4096)
    """

    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)
        self.hidden_dim = int(hidden_dim)

        # -----------------------
        # -----------------------
        if self.feat_dim != 512:
            self.to512 = nn.Conv1d(self.feat_dim, 512, kernel_size=1, bias=False)
        else:
            self.to512 = nn.Identity()

        # -----------------------
        # -----------------------
        length_down = _ceil_div(self.Lspp, 2)  # Lspp=7 -> 4
        self.tsse = TSSE(
            in_channels=512,
            out_channels=256,
            length=length_down,
            kernel_size=3,
            stride=2,
        )

        # -----------------------
        # -----------------------
        mamba_layers = 2
        n_embd_ks = 3
        with_ln = True
        mamba_type = "dbm"

        self.mamba = MambaBackbone(
            n_in=512,
            n_embd=512,
            n_embd_ks=n_embd_ks,
            arch=(0, int(mamba_layers), 0),
            scale_factor=1,
            with_ln=with_ln,
            mamba_type=mamba_type,
        )

        # -----------------------
        # -----------------------
        self.proj = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [N, feat_dim*Lspp]
        return: [N, hidden_dim]
        """
        # 1) [N, C*L] -> [N, C, L]
        x = _flat_to_seq(x_flat, self.feat_dim, self.Lspp)  # [N, feat_dim, Lspp]

        # 2) -> [N, 512, Lspp]
        x = self.to512(x)

        x = self.tsse(x)

        mask = torch.ones(x.size(0), 1, x.size(-1), dtype=torch.bool, device=x.device)
        feats, _ = self.mamba(x, mask)
        y = feats[0] if isinstance(feats, (list, tuple)) else feats  # [N,512,L2]

        # 5) pool + proj -> [N,4096]
        y = y.mean(dim=-1)          # [N,512]
        y = self.proj(y)            # [N,hidden_dim]
        return y

class ROIHead_Transformer(nn.Module):
    """
    x_flat after SPP: [N, feat_dim*Lspp]
    -> reshape to [N, feat_dim, Lspp] as token sequence
    -> Transformer token mixing
    -> mean pool -> [N, d_model] -> proj -> [N, hidden_dim]
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        d_model = 512
        nhead = 4
        num_layers = 2
        dim_feedforward = 2048
        dropout = 0.1

        # [N, feat_dim, L] -> [N, L, d_model]
        self.in_proj = nn.Linear(self.feat_dim, d_model)

        self.pos = nn.Parameter(torch.zeros(1, self.Lspp, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [N, feat_dim*Lspp]
        return: [N, hidden_dim]
        """
        x = _flat_to_seq(x_flat, self.feat_dim, self.Lspp)  # [N, C, L]
        x = x.transpose(1, 2)                               # [N, L, C]
        x = self.in_proj(x)                                 # [N, L, d_model]
        x = x + self.pos                                    # [N, L, d_model]

        y = self.encoder(x)                                 # [N, L, d_model]
        y = y.mean(dim=1)

        return self.proj(y)                                  # [N, hidden_dim]


class ROIHead_LSTM(nn.Module):
    """
    Treat Lspp bins as a sequence of length Lspp and use LSTM for token mixing.
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        d_model = 512
        lstm_hidden = 512
        num_layers = 2
        dropout = 0.1
        bidirectional = False

        self.in_proj = nn.Linear(self.feat_dim, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,     # [N, L, d_model]
            dropout=(dropout if num_layers > 1 else 0.0),
            bidirectional=bidirectional,
        )

        out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.proj = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [N, feat_dim*Lspp]
        return: [N, hidden_dim]
        """
        x = _flat_to_seq(x_flat, self.feat_dim, self.Lspp)  # [N, C, L]
        x = x.transpose(1, 2)                               # [N, L, C]
        x = self.in_proj(x)                                 # [N, L, d_model]

        y, (hn, cn) = self.lstm(x)                          # y: [N, L, out_dim]

        last = y[:, -1, :]                                  # [N, out_dim]

        # last = y.mean(dim=1)

        return self.proj(last)                               # [N, hidden_dim]

def _build_roi_head(name: str, feat_dim: int, hidden_dim: int, spp_levels):
    name = str(name).lower()
    Lspp = int(sum(tuple(spp_levels)))
    in_dim = feat_dim * Lspp

    if name == "fc":
        return ROIHead_FC(in_dim, hidden_dim)
    elif name == "tsse":
        return ROIHead_TSSELite(feat_dim, Lspp, hidden_dim)
    elif name == "mamba":
        return ROIHead_MambaLite(feat_dim, Lspp, hidden_dim)
    elif name == ["tsse_mamba"]:
        return ROIHead_TSSEMamba(feat_dim, Lspp, hidden_dim)
    elif name == "lstm":
        return ROIHead_LSTM(feat_dim, Lspp, hidden_dim)
    elif name == "transformer":
        return ROIHead_Transformer(feat_dim, Lspp, hidden_dim)
    else:
        raise ValueError(f"Unknown roi_head='{name}', supported: fc/tsse/mamba/tsse_mamba")
