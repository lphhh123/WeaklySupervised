
from models.pcl_model_blocks import *
from models.PCL_OICR_model import TemporalSPP1D


# =========================
# 1D IoU + OICR pseudo label
# =========================

def segment_iou_1d_torch(boxes_a: torch.Tensor, boxes_b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    boxes_a: [P,2]  (start,end)
    boxes_b: [G,2]
    return:  [P,G] IoU
    """
    a_s = boxes_a[:, 0].unsqueeze(1)  # [P,1]
    a_e = boxes_a[:, 1].unsqueeze(1)  # [P,1]
    b_s = boxes_b[:, 0].unsqueeze(0)  # [1,G]
    b_e = boxes_b[:, 1].unsqueeze(0)  # [1,G]

    inter_s = torch.maximum(a_s, b_s)
    inter_e = torch.minimum(a_e, b_e)
    inter = (inter_e - inter_s).clamp(min=0.0)

    len_a = (a_e - a_s).clamp(min=eps)
    len_b = (b_e - b_s).clamp(min=eps)
    union = len_a + len_b - inter
    return inter / (union + eps)


@torch.no_grad()
def oicr_pseudo_labels_1d(
    boxes: torch.Tensor,        # [P,2] float
    cls_prob: torch.Tensor,
    im_labels: torch.Tensor,    # [C] 0/1
    fg_thresh: float = 0.5,
    bg_thresh: float = 0.1,
    eps: float = 1e-6,
):
    """
    OICR 1D pseudo label：
      - Message c：Message scoreMessage proposal Message pseudo GT
      - Message IoU Message proposals Message pseudo GT，Message label / weight / gt_assignment

    return:
      labels: [P]  (0..C) 0=bg
      weights: [P]
      gt_assignment: [P] (-1 for bg)
    """
    device = boxes.device
    P = boxes.shape[0]
    C = cls_prob.shape[1]

    pos_cls = (im_labels > 0).nonzero(as_tuple=False).view(-1)  # [N_pos]
    if pos_cls.numel() == 0:
        labels = torch.zeros(P, dtype=torch.long, device=device)
        weights = torch.zeros(P, dtype=torch.float32, device=device)
        gt_assignment = torch.full((P,), -1, dtype=torch.long, device=device)
        return labels, weights, gt_assignment


    cls_tmp = cls_prob.clone()
    gt_boxes = []
    gt_classes = []
    gt_scores = []

    for c in pos_cls.tolist():
        scores_c = cls_tmp[:, c]
        p_star = int(torch.argmax(scores_c).item())
        gt_boxes.append(boxes[p_star].view(1, 2))
        gt_classes.append(torch.tensor([c + 1], device=device, dtype=torch.long))      # 1..C
        gt_scores.append(scores_c[p_star].clamp(min=eps, max=1.0).view(1))            # [1]

        cls_tmp[p_star, :] = -1.0

    gt_boxes = torch.cat(gt_boxes, dim=0)      # [G,2]
    gt_classes = torch.cat(gt_classes, dim=0)  # [G]
    gt_scores = torch.cat(gt_scores, dim=0)    # [G]
    G = gt_boxes.shape[0]


    overlaps = segment_iou_1d_torch(boxes.float(), gt_boxes.float())


    max_overlaps, gt_assignment = overlaps.max(dim=1)     # [P], [P]
    labels = gt_classes[gt_assignment]                    # [P] in 1..C
    weights = gt_scores[gt_assignment].float()            # [P]

    # fg/bg/ignore
    bg_mask = max_overlaps < fg_thresh
    ig_mask = max_overlaps < bg_thresh

    labels[bg_mask] = 0
    gt_assignment[bg_mask] = -1
    weights[ig_mask] = 0.0

    return labels, weights, gt_assignment

def mil_losses_per_sample(cls_score: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    cls_score: [B,C] (0..1)
    labels:    [B,C] (0/1)
    return:    [B]
    """
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)  # [B,C]
    return loss.mean(dim=1)  # [B]



# =========================
#  OICR + PA-Loss (IMU)
# =========================

class IMU_OICR_PALoss(nn.Module):
    """
    OICR + PA-Loss（Message PCL）
    - global_feat: [B,C,T]
    - proposal_boxes: [B,P,2]
    - labels: [B,C]  (Message)
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        refine_times: int = 3,
        hidden_dim: int = 4096,
        spp_levels=(1, 2, 4),
        pool_type="avg",
        fg_thresh: float = 0.5,
        bg_thresh: float = 0.1,
        stage0_boost: float = 3.0,
        pa_mode: str = "sigmoid",
        enhance_weight: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.refine_times = refine_times
        self.fg_thresh = float(fg_thresh)
        self.bg_thresh = float(bg_thresh)
        self.stage0_boost = float(stage0_boost)
        self.pa_mode = str(pa_mode)
        self.enhance_weight = bool(enhance_weight)

        # 1D-SPP
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul

        # 2xFC
        self.fc1 = nn.Linear(self.spp_out_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # MIL + refine
        self.mil_head = mil_outputs(hidden_dim, num_classes)
        self.refine_head = refine_outputs(hidden_dim, num_classes + 1, refine_times)
        self.refine_losses = nn.ModuleList([OICRLosses() for _ in range(refine_times)])

    def pool_proposals_1d_spp(self, global_feat, proposal_boxes):
        """
        global_feat: [B,C,T]
        proposal_boxes: [B,P,2]
        return: [B,P,C*spp_mul]
        """
        B, C, T = global_feat.shape
        _, P, _ = proposal_boxes.shape
        pooled_all = []

        for b in range(B):
            feat_b = global_feat[b:b+1]    # [1,C,T]
            boxes_b = proposal_boxes[b]    # [P,2]
            feats_b = []
            for p in range(P):
                s = int(boxes_b[p, 0].item())
                e = int(boxes_b[p, 1].item())
                s = max(0, min(s, T - 1))
                e = max(s + 1, min(e, T))
                seg = feat_b[:, :, s:e]        # [1,C,L]
                f = self.spp(seg)              # [1, C*spp_mul]
                feats_b.append(f.squeeze(0))   # [C*spp_mul]
            feats_b = torch.stack(feats_b, dim=0)  # [P, C*spp_mul]
            pooled_all.append(feats_b)

        return torch.stack(pooled_all, dim=0)       # [B,P,C*spp_mul]

    def _make_w_pa(self, loss_im_cls_per: torch.Tensor) -> torch.Tensor:
        """
        loss_im_cls_per: [B]  (detachMessage)
        return w_pa: [B]
        """
        if self.pa_mode == "exp":

            return 1.0 + torch.exp(-loss_im_cls_per)

        return 1.0 + torch.sigmoid(-loss_im_cls_per)

    def forward(self, global_feat, proposal_boxes, labels=None):
        if global_feat.dim() != 3:
            raise ValueError(f"global_feat must be [B,C,T], got {global_feat.shape}")
        if global_feat.size(1) != self.feat_dim:
            raise ValueError(f"feat_dim mismatch: expect C={self.feat_dim}, got {global_feat.shape}")

        B, C, T = global_feat.shape
        _, P, _ = proposal_boxes.shape
        device = global_feat.device

        # 1) proposal pooling -> [B,P,D]
        proposal_feats = self.pool_proposals_1d_spp(global_feat, proposal_boxes)  # [B,P,C*spp_mul]

        # 2) MLP -> [B,P,H]
        x = F.relu(self.fc1(proposal_feats), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # 3) MIL & refine
        mil_score = self.mil_head(x)                 # [B,P,C]
        refine_scores = self.refine_head(x)          # list of [B,P,C+1]

        out = {}

        # ========== Inference ==========
        if not self.training:

            final = refine_scores[0][:, :, 1:]
            for k in range(1, self.refine_times):
                final = final + refine_scores[k][:, :, 1:]
            final = final / float(self.refine_times)

            out["joint_prob"] = final
            out["mil_score"] = mil_score            # [B,P,C]
            out["refine_scores"] = refine_scores    # list([B,P,C+1])
            return out

        # ========== Training ==========
        if labels is None:
            raise ValueError("training requires labels [B,C]")

        labels = labels.float()

        # ---- image-level MIL loss (per-sample) ----
        im_cls_score = mil_score.sum(dim=1)                 # [B,C]
        loss_im_cls_per = mil_losses_per_sample(im_cls_score, labels)   # [B]
        loss_im_cls = loss_im_cls_per.mean()
        out["losses"] = {"loss_im_cls": loss_im_cls}

        # ---- PA weight per-sample ----
        w_pa = self._make_w_pa(loss_im_cls_per.detach())    # [B]

        # ---- refinement losses ----
        for k in range(self.refine_times):
            loss_k = 0.0
            for b in range(B):
                boxes_b = proposal_boxes[b].float()   # [P,2]
                im_labels_b = labels[b]               # [C]

                if k == 0:
                    cls_prob_b = mil_score[b].detach()                     # [P,C]
                else:
                    cls_prob_b = refine_scores[k - 1][b, :, 1:].detach()   # [P,C] exclude bg

                lbl, w, gt_assign = oicr_pseudo_labels_1d(
                    boxes=boxes_b,
                    cls_prob=cls_prob_b,
                    im_labels=im_labels_b,
                    fg_thresh=self.fg_thresh,
                    bg_thresh=self.bg_thresh,
                )  # [P], [P], [P]

                if self.enhance_weight:
                    w = w * torch.exp(w)

                prob_b = refine_scores[k][b]  # [P,C+1]
                loss_b = self.refine_losses[k](prob_b, lbl.long(), w.float(), gt_assign)


                loss_b = loss_b * w_pa[b]
                loss_k = loss_k + loss_b

            loss_k = loss_k / float(B)
            if k == 0 and self.stage0_boost != 1.0:
                loss_k = loss_k * self.stage0_boost

            out["losses"][f"refine_loss{k}"] = loss_k

        out["mil_score"] = mil_score
        out["refine_scores"] = refine_scores
        return out
