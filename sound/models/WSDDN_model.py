import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import json
from tqdm import tqdm
import random
import os
from tool import *
import torch.nn.functional as F


class WSDDN_avg(nn.Module):
    def __init__(self, num_classes=30, feat_dim=512):
        super().__init__()
        self.ssp = nn.AdaptiveAvgPool1d(1)  # nn.AdaptiveMaxPool1d(1)
        self.fc6 = nn.Linear(feat_dim, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.class_branch = nn.Linear(512, num_classes)
        self.det_branch   = nn.Linear(512, num_classes)

    def forward(self, x, proposal_boxes):
        B, C, T_global = x.shape
        P = proposal_boxes.shape[1]


        proposal_features = []
        for b in range(B):
            for p in range(P):

                start = int(proposal_boxes[b, p, 0].item())
                end   = int(proposal_boxes[b, p, 1].item())
                feat = x[b:b+1, :, start:end]       # [1, C, T_p]
                feat = self.ssp(feat).flatten(1)    # [1, C]
                proposal_features.append(feat)
        proposal_features = torch.cat(proposal_features, dim=0)  # [B*P, C]

        # 2) fc6, fc7
        feat_fc6 = F.relu(self.fc6(proposal_features), inplace=True)  # [B*P, 1024]
        feat_fc7 = F.relu(self.fc7(feat_fc6), inplace=True)  # [B*P, 512]


        feat_fc7_reshaped = feat_fc7.view(B, P, -1)  # [B, P, 512]


        class_logits = self.class_branch(feat_fc7)  # [B*P, num_classes]
        det_logits   = self.det_branch(feat_fc7)    # [B*P, num_classes]


        class_logits = class_logits.view(B, P, -1)
        det_logits   = det_logits.view(B, P, -1)


        class_prob = F.softmax(class_logits, dim=2)  # [B, P, C]
        det_prob   = F.softmax(det_logits,   dim=1)  # [B, P, C]


        joint_prob = class_prob * det_prob           # [B, P, C]


        video_prob = joint_prob.sum(dim=1)           # [B, C]

        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat_fc7_reshaped,
        }

class WSDDN(nn.Module):
    def __init__(self,num_classes=30,feat_dim=512,spp_levels=(1, 2, 4),pool_type="avg"):
        super().__init__()

        # 1D SPP
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)


        self.fc6 = nn.Linear(self.spp_out_dim, 1024)
        self.fc7 = nn.Linear(1024, 512)

        self.class_branch = nn.Linear(512, num_classes)
        self.det_branch   = nn.Linear(512, num_classes)

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

                # [1, C, T_p]
                feat = x[b:b+1, :, start:end]
                # SPP: [1, C * sum(levels)]
                feat = self.spp(feat)
                proposal_features.append(feat)

        # [B*P, C * sum(levels)]
        proposal_features = torch.cat(proposal_features, dim=0)

        # ======= fc6, fc7 =======
        feat_fc6 = F.relu(self.fc6(proposal_features), inplace=True)  # [B*P, 1024]
        feat_fc7 = F.relu(self.fc7(feat_fc6), inplace=True)  # [B*P, 512]


        feat_fc7_reshaped = feat_fc7.view(B, P, -1)


        class_logits = self.class_branch(feat_fc7)  # [B*P, C_cls]
        det_logits   = self.det_branch(feat_fc7)    # [B*P, C_cls]

        class_logits = class_logits.view(B, P, -1)
        det_logits   = det_logits.view(B, P, -1)

        class_prob = F.softmax(class_logits, dim=2)  # [B, P, C_cls]
        det_prob   = F.softmax(det_logits,   dim=1)  # [B, P, C_cls]

        joint_prob = class_prob * det_prob           # [B, P, C_cls]
        video_prob = joint_prob.sum(dim=1)           # [B, C_cls]

        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat_fc7_reshaped,
        }

class TemporalSPP1D(nn.Module):
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

            y = y.view(N, C * L)
            feats.append(y)
        # [N, C * sum(levels)]
        return torch.cat(feats, dim=1)

class WSDDNTransformerIMU(nn.Module):
    def __init__(
        self,
        num_classes: int = 30,
        feat_dim: int = 512,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        spp_levels=(1, 2, 4),
        pool_type: str = "avg",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding


        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)


        if self.spp_out_dim != d_model:
            self.input_proj = nn.Linear(self.spp_out_dim, d_model)
        else:
            self.input_proj = nn.Identity()


        if use_positional_encoding:
            self.pos_mlp = nn.Sequential(
                nn.Linear(2, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )
        else:
            self.pos_mlp = None

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        self.class_branch = nn.Linear(d_model, num_classes)
        self.det_branch   = nn.Linear(d_model, num_classes)

    def _pool_proposals(self, x, proposal_boxes):
        B, C, T_global = x.shape
        _, P, _ = proposal_boxes.shape


        proposal_feats = x.new_zeros(B, P, self.spp_out_dim)


        starts = proposal_boxes[..., 0].clamp(0, T_global - 1).long()
        ends   = proposal_boxes[..., 1].clamp(1, T_global).long()


        bad_mask = ends <= starts
        ends[bad_mask] = (starts[bad_mask] + 1).clamp(max=T_global)

        for b in range(B):
            for p in range(P):
                s = int(starts[b, p].item())
                e = int(ends[b, p].item())
                # [1, C, T_p]
                seg = x[b:b+1, :, s:e]
                if seg.size(2) == 0:

                    seg = x[b:b+1, :, s:s+1]
                # 1D SPP: [1, C * sum(levels)]
                feat_spp = self.spp(seg)          # [1, C * sum(levels)]
                feat_spp = feat_spp.view(-1)      # [C * sum(levels)]
                proposal_feats[b, p] = feat_spp

        return proposal_feats  # [B, P, spp_out_dim]

    def forward(self, x, proposal_boxes):
        B, C, T_global = x.shape
        _, P, _ = proposal_boxes.shape


        proposal_feats = self._pool_proposals(x, proposal_boxes)  # [B, P, spp_out_dim]


        starts_f = proposal_boxes[..., 0].float()
        ends_f   = proposal_boxes[..., 1].float()
        centers  = (starts_f + ends_f) / 2.0        # [B, P]
        lengths  = (ends_f - starts_f).clamp(min=1.0)

        T_global_f = float(T_global)
        center_norm = centers / T_global_f          # [0, 1]
        length_norm = lengths / T_global_f          # [0, 1]

        pos_feat = torch.stack([center_norm, length_norm], dim=-1)  # [B, P, 2]


        sort_idx = centers.argsort(dim=1)           # [B, P]
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(sort_idx)


        proposal_feats_sorted = proposal_feats[batch_idx, sort_idx, :]   # [B, P, spp_out_dim]
        if self.use_positional_encoding and self.pos_mlp is not None:
            pos_emb = self.pos_mlp(pos_feat)                             # [B, P, d_model]
            pos_emb_sorted = pos_emb[batch_idx, sort_idx, :]             # [B, P, d_model]
        else:
            pos_emb_sorted = 0


        tokens = self.input_proj(proposal_feats_sorted)                  # [B, P, d_model]
        tokens = tokens + pos_emb_sorted                                 # [B, P, d_model]


        tokens = tokens.transpose(0, 1)          # [P, B, d_model]
        tokens_out = self.transformer(tokens)    # [P, B, d_model]
        tokens_out = tokens_out.transpose(0, 1)


        inv_sort_idx = torch.argsort(sort_idx, dim=1)   # [B, P]
        proposal_ctx = tokens_out[batch_idx, inv_sort_idx, :]  # [B, P, d_model]


        feat_ctx_flat = proposal_ctx.reshape(B * P, -1)        # [B*P, d_model]

        class_logits = self.class_branch(feat_ctx_flat)        # [B*P, num_classes]
        det_logits   = self.det_branch(feat_ctx_flat)          # [B*P, num_classes]

        class_logits = class_logits.view(B, P, self.num_classes)
        det_logits   = det_logits.view(B, P, self.num_classes)


        class_prob = F.softmax(class_logits, dim=2)
        det_prob   = F.softmax(det_logits,   dim=1)

        joint_prob = class_prob * det_prob           # [B, P, C]
        video_prob = joint_prob.sum(dim=1)


        feat_fc7_reshaped = proposal_ctx             # [B, P, d_model]

        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat_fc7_reshaped,
        }

def generate_proposal_boxes(
    T_global=30,
    num_proposals=50,
    raw_data_length=1500,
    base_physical_sec=7,
    total_physical_sec=30,
    step_raw=2,
    min_raw_length=5,
    max_raw_length=15
):
    proposal_boxes = []



    raw_to_feat = T_global / raw_data_length


    base_raw_length = int((base_physical_sec / total_physical_sec) * raw_data_length) #7*（30/1500）


    base_feat_length = max(1, int(base_raw_length * raw_to_feat))


    step_feat = max(1, int(step_raw * 50 * raw_to_feat))


    start_feat = 0
    while start_feat + base_feat_length <= T_global:
        end_feat = start_feat + base_feat_length
        proposal_boxes.append([start_feat, end_feat])
        start_feat += step_feat




    while len(proposal_boxes) < num_proposals:

        raw_length = random.randint(min_raw_length, max_raw_length)

        feat_length = max(1, int(raw_length * 50 * raw_to_feat))


        max_raw_start = total_physical_sec - raw_length
        if max_raw_start <= 0:
            raw_start = 0
        else:
            raw_start = random.randint(0, max_raw_start)
        feat_start = int(raw_start * 50 * raw_to_feat)


        feat_end = min(feat_start + feat_length, T_global)
        proposal_boxes.append([feat_start, feat_end])



    return torch.tensor(proposal_boxes[:num_proposals], dtype=torch.long)

def generate_proposals(imu_data, num_proposals=50, base_len=476, step=256, min_len=340, max_len=884):
    C, T_total = imu_data.shape  # 30, 2048
    proposals = []


    start_positions = list(range(0, T_total - base_len + 1, step))
    for start in start_positions:
        proposal = imu_data[:, start:start+base_len]
        proposals.append(proposal)


    while len(proposals) < num_proposals:
        t_len = random.randint(min_len, max_len)
        start = random.randint(0, T_total - t_len)
        proposal = imu_data[:, start:start+t_len]
        proposals.append(proposal)

    return torch.stack(proposals)

def merge_overlapping_proposals(proposals, det_scores, overlap_thresh=0.5):

    sorted_idx = det_scores.mean(dim=1).argsort(descending=True)
    keep = []
    for idx in sorted_idx:
        p = proposals[idx]

        overlap = False
        for k in keep:

            p_start = (idx * 256) % 2048
            p_end = p_start + p.shape[1]
            k_start = (k * 256) % 2048
            k_end = k_start + proposals[k].shape[1]
            intersect = max(0, min(p_end, k_end) - max(p_start, k_start))
            union = p_end - p_start + k_end - k_start - intersect
            if intersect / union > overlap_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(idx)
            if len(keep) == 25:
                break
    return proposals[keep]

