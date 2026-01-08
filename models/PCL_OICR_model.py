import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

from .pcl_model_blocks import mil_outputs, mil_losses, OICRLosses, refine_outputs

# SSP版
class IMU_PCL_OICR(nn.Module):
    """
    IMU 上的 PCL/OICR 头（带 1D SPP）：
      输入: global_feat [B, C, T] 或 [B, T, C], proposal_boxes [B, P, 2], labels [B, C]
      训练时: 返回 losses + 各级 score
      测试时: 返回 mil_score / refine_scores
    """
    def __init__(self,
                 feat_dim,         # backbone 输出的通道数 C
                 num_classes,
                 refine_times=3,
                 use_pcl=False,
                 fg_thresh=0.5,
                 bg_thresh=0.1,
                 graph_iou_thresh=0.5,
                 max_pc_num=3,
                 hidden_dim=4096,
                 spp_levels=(1, 2, 4),
                 pool_type="avg"):
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
        mil_score = self.mil_head(x)             # [B*P, num_classes]
        refine_scores_flat = self.refine_head(x) # list of [B*P, num_classes+1]

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


def segment_overlaps_1d(a, b):
    """
    1D IoU 计算:
      a: [N, 2], b: [M, 2]
      返回 overlaps: [N, M]
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    start_a = a[:, 0][:, None]
    end_a   = a[:, 1][:, None]
    start_b = b[:, 0][None, :]
    end_b   = b[:, 1][None, :]

    inter_start = np.maximum(start_a, start_b)
    inter_end   = np.minimum(end_a, end_b)
    inter_len   = np.maximum(0.0, inter_end - inter_start)

    len_a = np.maximum(1e-6, end_a - start_a)
    len_b = np.maximum(1e-6, end_b - start_b)
    union = len_a + len_b - inter_len

    return inter_len / union


def _get_top_ranking_indices(probs, num_clusters=3, rng_seed=1):
    """
    原 PCL 里的 k-means 挑高分簇：
      probs: [P]，某一类别下所有 proposal 的得分
    返回：属于高分簇的 index
    """
    probs = probs.reshape(-1, 1)
    N = probs.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64)
    if N < num_clusters:
        return np.arange(N, dtype=np.int64)

    kmeans = KMeans(n_clusters=num_clusters, random_state=rng_seed).fit(probs)
    centers = kmeans.cluster_centers_.reshape(-1)
    high_label = np.argmax(centers)
    idx = np.where(kmeans.labels_ == high_label)[0]
    if idx.size == 0:
        idx = np.array([np.argmax(probs)], dtype=np.int64)
    return idx


def _build_graph_1d(segments, iou_thresh):
    overlaps = segment_overlaps_1d(segments, segments)
    return (overlaps > iou_thresh).astype(np.float32)


def _get_graph_centers_1d(boxes, cls_prob, im_labels,graph_iou_thresh=0.5,max_pc_num=3,num_kmeans_cluster=3,rng_seed=1):
    """
    PCL 的 graph center 选 pseudo GT 的逻辑，1D 版。
    boxes: [P,2]
    cls_prob: [P,C]
    im_labels: [1,C]（image-level label）
    """
    num_imgs, num_classes = im_labels.shape
    assert num_imgs == 1
    im_labels = im_labels[0].copy()

    gt_segments = []
    gt_classes = []
    gt_scores = []

    boxes_pool = boxes.copy()
    cls_pool = cls_prob.copy()

    for c in range(num_classes):
        if im_labels[c] != 1:
            continue

        scores_c = cls_pool[:, c].copy()
        valid_idx = np.where(scores_c >= 0)[0]
        if valid_idx.size == 0:
            continue

        top_rel = _get_top_ranking_indices(
            scores_c[valid_idx],
            num_clusters=num_kmeans_cluster,
            rng_seed=rng_seed
        )
        top_idx = valid_idx[top_rel]
        boxes_c = boxes_pool[top_idx, :]         # [Nc,2]
        scores_sel = scores_c[top_idx]           # [Nc]

        graph = _build_graph_1d(boxes_c, graph_iou_thresh)

        keep_centers = []
        center_scores = []
        remaining = scores_sel.size

        while True:
            degree = graph.sum(axis=1)
            order = degree.argsort()[::-1]
            center = int(order[0])
            keep_centers.append(center)

            neighbors = np.where(graph[center, :] > 0)[0]
            center_scores.append(scores_sel[neighbors].max())

            graph[:, neighbors] = 0
            graph[neighbors, :] = 0
            remaining -= len(neighbors)
            if remaining <= 5:
                break

        centers_boxes = boxes_c[keep_centers, :]
        centers_scores = np.array(center_scores, dtype=np.float32)

        if centers_boxes.shape[0] > max_pc_num:
            idx = np.argsort(centers_scores)[-max_pc_num:]
            centers_boxes = centers_boxes[idx, :]
            centers_scores = centers_scores[idx]

        gt_segments.append(centers_boxes)
        gt_scores.append(centers_scores.reshape(-1, 1))
        gt_classes.append((c + 1) * np.ones((centers_boxes.shape[0], 1), dtype=np.int32))

        # 删掉这些 center，从 pool 中去掉
        delete_idx = top_idx[keep_centers]
        # cls_pool = np.delete(cls_pool, delete_idx, axis=0)
        cls_pool[delete_idx, :] = -1.0  # 不做 np.delete，直接标记为无效，避免数组拷贝
        # boxes_pool = np.delete(boxes_pool, delete_idx, axis=0)

    if len(gt_segments) == 0:
        gt_segments = np.zeros((1, 2), dtype=np.float32)
        gt_classes = np.zeros((1, 1), dtype=np.int32)
        gt_scores  = np.zeros((1, 1), dtype=np.float32)
    else:
        gt_segments = np.vstack(gt_segments).astype(np.float32)
        gt_classes  = np.vstack(gt_classes).astype(np.int32)
        gt_scores   = np.vstack(gt_scores).astype(np.float32)

    return {
        "gt_segments": gt_segments,
        "gt_classes": gt_classes,
        "gt_scores":  gt_scores,
    }


def _get_highest_score_proposals_1d(boxes, cls_prob, im_labels):
    """
    OICR 的最高分 proposal 当 pseudo GT 的逻辑，1D 版。
    """
    num_imgs, num_classes = im_labels.shape
    assert num_imgs == 1
    im_labels = im_labels[0].copy()

    gt_segments = []
    gt_classes = []
    gt_scores = []

    cls_prob_tmp = cls_prob.copy()

    for c in range(num_classes):
        if im_labels[c] != 1:
            continue
        scores_c = cls_prob_tmp[:, c].copy()
        max_idx = int(np.argmax(scores_c))

        gt_segments.append(boxes[max_idx, :].reshape(1, -1))
        gt_classes.append(np.array([[c + 1]], dtype=np.int32))
        gt_scores.append(np.array([[scores_c[max_idx]]], dtype=np.float32))

        cls_prob_tmp[max_idx, :] = 0.0

    if len(gt_segments) == 0:
        gt_segments = np.zeros((1, 2), dtype=np.float32)
        gt_classes  = np.zeros((1, 1), dtype=np.int32)
        gt_scores   = np.zeros((1, 1), dtype=np.float32)
    else:
        gt_segments = np.vstack(gt_segments).astype(np.float32)
        gt_classes  = np.vstack(gt_classes).astype(np.int32)
        gt_scores   = np.vstack(gt_scores).astype(np.float32)

    return {
        "gt_segments": gt_segments,
        "gt_classes":  gt_classes,
        "gt_scores":   gt_scores,
    }


def _assign_clusters_1d(all_segments, proposals, im_labels,
                        fg_thresh=0.5, bg_thresh=0.1):
    """
    与原 get_proposal_clusters 对应：
      返回 labels, cls_loss_weights, gt_assignment
    """
    gt_segments = proposals["gt_segments"]
    gt_classes  = proposals["gt_classes"]
    gt_scores   = proposals["gt_scores"]

    overlaps = segment_overlaps_1d(
        all_segments.astype(np.float32, copy=False),
        gt_segments.astype(np.float32, copy=False)
    )  # [P, G]

    gt_assignment = overlaps.argmax(axis=1)     # [P]
    max_overlaps  = overlaps.max(axis=1)       # [P]
    labels = gt_classes[gt_assignment, 0]      # [P]
    cls_loss_weights = gt_scores[gt_assignment, 0]  # [P]

    fg_inds = np.where(max_overlaps >= fg_thresh)[0]
    bg_inds = np.where(max_overlaps < fg_thresh)[0]
    ig_inds = np.where(max_overlaps < bg_thresh)[0]

    cls_loss_weights[ig_inds] = 0.0
    labels[bg_inds] = 0
    gt_assignment[bg_inds] = -1

    return labels, cls_loss_weights, gt_assignment


def pcl_1d(
    boxes,
    cls_prob,
    im_labels,
    fg_thresh=0.5,
    bg_thresh=0.1,
    graph_iou_thresh=0.5,
    max_pc_num=3,
    num_kmeans_cluster=3,
    rng_seed=1,
):
    """
    PCL 的 1D 版本：图聚类 + proposal cluster。
    """
    cls_prob = cls_prob.copy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        # 有背景列时去背景
        cls_prob = cls_prob[:, 1:]

    eps = 1e-9
    cls_prob = np.clip(cls_prob, eps, 1 - eps)

    proposals = _get_graph_centers_1d(
        boxes.copy(),
        cls_prob.copy(),
        im_labels.copy(),
        graph_iou_thresh=graph_iou_thresh,
        max_pc_num=max_pc_num,
        num_kmeans_cluster=num_kmeans_cluster,
        rng_seed=rng_seed,
    )

    labels, cls_loss_weights, gt_assignment = _assign_clusters_1d(
        boxes.copy(),
        proposals,
        im_labels.copy(),
        fg_thresh=fg_thresh,
        bg_thresh=bg_thresh,
    )

    return {
        "labels":          labels.reshape(1, -1).astype(np.int64).copy(),
        "cls_loss_weights": cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "gt_assignment":   gt_assignment.reshape(1, -1).astype(np.int64).copy(),
    }


def oicr_1d(
    boxes,
    cls_prob,
    im_labels,
    fg_thresh=0.5,
    bg_thresh=0.1,
):
    """
    OICR 的 1D 版本：最高分 proposal 直接做 pseudo GT。
    """
    cls_prob = cls_prob.copy()
    if cls_prob.shape[1] != im_labels.shape[1]:
        cls_prob = cls_prob[:, 1:]

    eps = 1e-9
    cls_prob = np.clip(cls_prob, eps, 1 - eps)

    proposals = _get_highest_score_proposals_1d(
        boxes.copy(),
        cls_prob.copy(),
        im_labels.copy()
    )

    labels, cls_loss_weights, gt_assignment = _assign_clusters_1d(
        boxes.copy(),
        proposals,
        im_labels.copy(),
        fg_thresh=fg_thresh,
        bg_thresh=bg_thresh,
    )

    return {
        "labels":           labels.reshape(1, -1).astype(np.int64).copy(),
        "cls_loss_weights": cls_loss_weights.reshape(1, -1).astype(np.float32).copy(),
        "gt_assignment":    gt_assignment.reshape(1, -1).astype(np.int64).copy(),
    }


class TemporalSPP1D(nn.Module):
    """
    1D 版的 Spatial Pyramid Pooling：
    给定 [N, C, T]，输出 [N, C * sum(levels)]
    例如 levels=(1,2,4)，就会做：
      - 1 个 bin -> [N, C*1]
      - 2 个 bin -> [N, C*2]
      - 4 个 bin -> [N, C*4]
    拼在一起得到 [N, C*(1+2+4)]
    """
    def __init__(self, levels=(1, 2, 4), pool_type="avg"):
        super().__init__()
        self.levels = levels
        assert pool_type in ["avg", "max"]
        self.pool_type = pool_type

        self.poolers = nn.ModuleList()
        for L in levels:
            if pool_type == "avg":
                self.poolers.append(nn.AdaptiveAvgPool1d(L))
            else:
                self.poolers.append(nn.AdaptiveMaxPool1d(L))

    @property
    def out_mul(self):
        """输出维度是 C * out_mul （out_mul = sum(levels)）"""
        return sum(self.levels)

    def forward(self, x):
        """
        x: [N, C, T]
        return: [N, C * sum(levels)]
        """
        N, C, T = x.shape
        feats = []
        for L, pool in zip(self.levels, self.poolers):
            # [N, C, L]
            y = pool(x)
            # 展平时间维： [N, C*L]
            y = y.view(N, C * L)
            feats.append(y)
        # [N, C * sum(levels)]
        return torch.cat(feats, dim=1)