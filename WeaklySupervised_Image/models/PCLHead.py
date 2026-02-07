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
    Head_PCL只有adapter+pcl/oicr，预训练模型是TSSE+Mamba
    """
    def __init__(self,
                 feat_dim,         # backbone 输出的通道数 C
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
                 adapter_cfg=None,   # ★ 新增,控制是否有适配器
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

        # -------- 1D SPP 模块 --------
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)

        # 相当于 roi_2mlp_head：对 pooled feature 做两层 MLP
        self.fc1 = nn.Linear(self.spp_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # MIL + refine heads
        self.mil_head = mil_outputs(hidden_dim, num_classes)
        self.refine_head = refine_outputs(hidden_dim, num_classes + 1, self.refine_times)
        self.refine_losses = nn.ModuleList(
            [OICRLosses() for _ in range(refine_times)]
        )

        # ============ Adapter（可选）===========
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
        使用 1D SPP 的 proposal pooling
        global_feat:    [B, C, T]
        proposal_boxes: [B, P, 2] (start, end)
        返回:           [B, P, C * sum(levels)]
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
        global_feat: [B, C, T] 或 [B, T, C]
        proposal_boxes: [B, P, 2]
        labels: [B, num_classes] (0/1)，训练必传
        """
        if global_feat.dim() != 3:
            raise ValueError("global_feat 应该是 [B, C, T] 或 [B, T, C]")

        # 标准化为 [B, C, T]
        if global_feat.size(1) == self.feat_dim:
            feat = global_feat
        elif global_feat.size(2) == self.feat_dim:
            feat = global_feat.transpose(1, 2)
        else:
            raise ValueError(
                f"global_feat 形状不匹配 feat_dim={self.feat_dim}, got {global_feat.shape}"
            )

        # 做特征适配（冻结 backbone 时，这里是主要可学习部分之一）
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
        x_bp = x.view(B, P, -1)  # ★恢复 [B,P,D]
        mil_score = self.mil_head(x_bp)  # ★mil_head 要按 [B,P,D] 计算

        device = x.device
        output = {}

        # 4) 训练：计算 loss
        if self.training:
            if labels is None:
                raise ValueError("训练模式下必须传入 labels")

            # image-level MIL loss
            mil_score_vid = mil_score.view(B, P, self.num_classes).sum(dim=1)  # [B, C]
            loss_im_cls = mil_losses(mil_score_vid, labels.float())
            output["losses"] = {"loss_im_cls": loss_im_cls}

            # numpy 版数据，用于生成 pseudo labels
            boxes_np = proposal_boxes.detach().cpu().numpy()           # [B, P, 2]
            labels_np = labels.detach().cpu().numpy()                  # [B, C]
            mil_np = mil_score.detach().cpu().numpy().reshape(B, P, self.num_classes)
            refine_np = [
                rs.detach().cpu().numpy().reshape(B, P, self.num_classes + 1)
                for rs in refine_scores_flat
            ]

            # 逐个 refine 分支
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
                    loss_refine_avg = loss_refine_avg * 3.0  # 论文里的 trick

                output["losses"][f"refine_loss{i_refine}"] = loss_refine_avg

        # 5) 无论 train / eval，都返回 score 方便调试和测试
        output["mil_score"] = mil_score.view(B, P, self.num_classes)
        output["refine_scores"] = [
            rs.view(B, P, self.num_classes + 1) for rs in refine_scores_flat
        ]

        return output


def map_boxes_input_to_feat(boxes: torch.Tensor, T_in: int, T_feat: int) -> torch.Tensor:
    """
    把原始序列坐标 (0..T_in) 映射到 feature 坐标 (0..T_feat)
    boxes: [B,P,2]，end 为 exclusive（和你 pooling 里的写法一致）
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

    # 保证 e > s
    e = torch.maximum(e, s + 1)

    return torch.stack([s.long(), e.long()], dim=-1)


class Change_Block_PCL_Model(nn.Module):
    """
    整体模型：
      x:[B,C_in,T_in] -> backbone -> feat:[B,512,T_feat]
      proposals(输入轴 or feat轴) -> head -> losses/scores
    """

    def __init__(self,
                 feat_dim,  # backbone 输出的通道数 C
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

        # -------- 1D SPP 模块 --------
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)


        # MIL + refine heads
        self.mil_head = mil_outputs(hidden_dim, num_classes)
        self.refine_head = refine_outputs(hidden_dim, num_classes + 1, self.refine_times)
        self.refine_losses = nn.ModuleList(
            [OICRLosses() for _ in range(refine_times)]
        )

        # 可切换模块
        self.roi_head_name = roi_head
        self.roi_head = _build_roi_head(
            self.roi_head_name,
            feat_dim=self.feat_dim,
            hidden_dim=hidden_dim,
            spp_levels=spp_levels,
        )

    def pool_proposals_1d_spp(self, global_feat, proposal_boxes):
        """
        使用 1D SPP 的 proposal pooling
        global_feat:    [B, C, T]
        proposal_boxes: [B, P, 2] (start, end)
        返回:           [B, P, C * sum(levels)]
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
        global_feat: [B, C, T] 或 [B, T, C]
        proposal_boxes: [B, P, 2]
        labels: [B, num_classes] (0/1)，训练必传
        """
        if global_feat.dim() != 3:
            raise ValueError("global_feat 应该是 [B, C, T] 或 [B, T, C]")

        # 标准化为 [B, C, T]
        if global_feat.size(1) == self.feat_dim:
            feat = global_feat
        elif global_feat.size(2) == self.feat_dim:
            feat = global_feat.transpose(1, 2)
        else:
            raise ValueError(
                f"global_feat 形状不匹配 feat_dim={self.feat_dim}, got {global_feat.shape}"
            )

        B, C, T = feat.shape
        B2, P, _ = proposal_boxes.shape
        assert B == B2

        # 1) SPP pooling
        proposal_feats = self.pool_proposals_1d_spp(feat, proposal_boxes)  # [B, P, C']
        _, _, D = proposal_feats.shape
        x = proposal_feats.view(B * P, D)  # [B*P, C']

        # 2) 可切换模块（替代原来的2*fc）
        x = self.roi_head(x)  # [B*P, hidden_dim]

        # 3) MIL + refine
        mil_score = self.mil_head(x)  # [B*P, num_classes]
        refine_scores_flat = self.refine_head(x)  # list of [B*P, num_classes+1]

        device = x.device
        output = {}

        # 4) 训练：计算 loss
        if self.training:
            if labels is None:
                raise ValueError("训练模式下必须传入 labels")

            # image-level MIL loss
            mil_score_vid = mil_score.view(B, P, self.num_classes).sum(dim=1)  # [B, C]
            loss_im_cls = mil_losses(mil_score_vid, labels.float())
            output["losses"] = {"loss_im_cls": loss_im_cls}

            # numpy 版数据，用于生成 pseudo labels
            boxes_np = proposal_boxes.detach().cpu().numpy()  # [B, P, 2]
            labels_np = labels.detach().cpu().numpy()  # [B, C]
            mil_np = mil_score.detach().cpu().numpy().reshape(B, P, self.num_classes)
            refine_np = [
                rs.detach().cpu().numpy().reshape(B, P, self.num_classes + 1)
                for rs in refine_scores_flat
            ]

            # 逐个 refine 分支
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
                    loss_refine_avg = loss_refine_avg * 3.0  # 论文里的 trick

                output["losses"][f"refine_loss{i_refine}"] = loss_refine_avg

        # 5) 无论 train / eval，都返回 score 方便调试和测试
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
    """原版 2*FC 的等价实现（默认就用这个）"""
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
    默认只用 1 个 TSSE（因为 Lspp=7，再堆会很快变 1）
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        # 原版 TSSE 内部写死 512 通道，所以这里保证进 TSSE 之前是 512
        if self.feat_dim != 512:
            self.to512 = nn.Conv1d(self.feat_dim, 512, kernel_size=1, bias=False)
        else:
            self.to512 = nn.Identity()

        # TSSE 的 length 参数：应当是 Downscale 之后的长度
        # TADEmbedding 里 length=(input_length//2)//(2**i) 就是在传 downscale 后的长度
        length_down = _ceil_div(self.Lspp, 2)  # L=7 -> 4

        self.tsse = TSSE(in_channels=512, out_channels=256, length=length_down, kernel_size=3, stride=2)

        # 输出到 4096（相当于替代原来的 fc1+fc2 的表征升维）
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
    用你现有的 MambaBackbone 做 ROI encoder（不下采样，避免 Lspp 太短出问题）
    默认参数都写在这里，不从 config 读。
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.Lspp = Lspp

        # ---- 默认超参 ----
        mamba_layers = 4
        n_embd_ks = 3
        with_ln = True
        mamba_type = "dbm"

        # 关键：不下采样 / 不建金字塔
        self.mamba = MambaBackbone(
            n_in=feat_dim,
            n_embd=feat_dim,
            n_embd_ks=n_embd_ks,
            arch=(0, int(mamba_layers), 0),   # 不做前后降采样
            scale_factor=1,                   # 不缩短长度
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

        feats, _ = self.mamba(x, mask)  # xrfv2 风格：可能返回 list
        y = feats[0] if isinstance(feats, (list, tuple)) else feats  # [N,C,L]
        y = y.mean(dim=-1)                                          # [N,C]
        return self.proj(y)                                         # [N,H]


class ROIHead_TSSEMamba(nn.Module):
    """
    用在 PCL/OICR 的 ROI head（替代原来的 fc1+fc2）：

    输入：
      x_flat: [N, feat_dim * Lspp]   (N=B*P)
        - feat_dim = backbone输出通道数（通常512）
        - Lspp     = SPP 的总 bin 数（如 levels=(1,2,4) -> Lspp=7）
    输出：
      [N, hidden_dim] （默认4096），可以直接喂给 mil_head / refine_head

    结构：
      reshape -> (可选 1x1 conv 到512) -> TSSE -> Mamba(不下采样) -> mean pool -> MLP(->4096)
    """

    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)
        self.hidden_dim = int(hidden_dim)

        # -----------------------
        # 0) 让 TSSE 输入满足 512 通道
        # -----------------------
        if self.feat_dim != 512:
            self.to512 = nn.Conv1d(self.feat_dim, 512, kernel_size=1, bias=False)
        else:
            self.to512 = nn.Identity()

        # -----------------------
        # 1) TSSE（只放 1 层）
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
        # 2) Mamba（ROI 内不下采样）
        # -----------------------
        mamba_layers = 2
        n_embd_ks = 3
        with_ln = True
        mamba_type = "dbm"

        self.mamba = MambaBackbone(
            n_in=512,
            n_embd=512,
            n_embd_ks=n_embd_ks,
            arch=(0, int(mamba_layers), 0),  # 不做前后降采样/不建金字塔
            scale_factor=1,                  # 不缩短长度
            with_ln=with_ln,
            mamba_type=mamba_type,
        )

        # -----------------------
        # 3) 输出投影到 4096（替代原 fc1+fc2 的“表征升维”）
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

        # 3) TSSE: [N,512,Lspp] -> [N,512,L2] （大概 L2=ceil(Lspp/2)）
        x = self.tsse(x)

        # 4) Mamba: 不下采样，保持长度 L2
        mask = torch.ones(x.size(0), 1, x.size(-1), dtype=torch.bool, device=x.device)
        feats, _ = self.mamba(x, mask)  # xrfv2 风格：可能返回 list
        y = feats[0] if isinstance(feats, (list, tuple)) else feats  # [N,512,L2]

        # 5) pool + proj -> [N,4096]
        y = y.mean(dim=-1)          # [N,512]
        y = self.proj(y)            # [N,hidden_dim]
        return y

class ROIHead_Transformer(nn.Module):
    """
    SPP 后的 x_flat=[N, feat_dim*Lspp]
    -> reshape 成 [N, feat_dim, Lspp] 当成 token 序列
    -> Transformer 做 token mixing
    -> mean pool -> [N, d_model] -> proj -> [N, hidden_dim]
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        # ===== 默认超参 =====
        d_model = 512
        nhead = 4
        num_layers = 2
        dim_feedforward = 2048
        dropout = 0.1

        # [N, feat_dim, L] -> [N, L, d_model]
        self.in_proj = nn.Linear(self.feat_dim, d_model)

        # learnable positional embedding (L 很短，不用复杂 PE)
        self.pos = nn.Parameter(torch.zeros(1, self.Lspp, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 关键：用 [N, L, d_model]
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # 输出到 PCL 需要的 hidden_dim（替代原 fc1+fc2）
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
        y = y.mean(dim=1)                                   # [N, d_model]  (每个 proposal 一个向量)

        return self.proj(y)                                  # [N, hidden_dim]


class ROIHead_LSTM(nn.Module):
    """
    把 Lspp 个 bin 当成长度为 Lspp 的序列，用 LSTM 做 token mixing。
    """
    def __init__(self, feat_dim: int, Lspp: int, hidden_dim: int = 4096):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.Lspp = int(Lspp)

        # ===== 默认超参 =====
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

        # 方案A：取最后一个 time step（更像“序列总结”）
        last = y[:, -1, :]                                  # [N, out_dim]

        # 方案B：mean pool（也可以，二选一）
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