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

        # 1) 截 proposal 特征
        proposal_features = []
        for b in range(B):
            for p in range(P):
                # 显式转成 int，防止 0-dim tensor 引起奇怪错误
                start = int(proposal_boxes[b, p, 0].item())
                end   = int(proposal_boxes[b, p, 1].item())
                feat = x[b:b+1, :, start:end]       # [1, C, T_p]
                feat = self.ssp(feat).flatten(1)    # [1, C]
                proposal_features.append(feat)
        proposal_features = torch.cat(proposal_features, dim=0)  # [B*P, C]

        # 2) fc6, fc7
        feat_fc6 = F.relu(self.fc6(proposal_features), inplace=True)  # [B*P, 1024]
        feat_fc7 = F.relu(self.fc7(feat_fc6), inplace=True)  # [B*P, 512]

        # === 这里新增：reshape 回 [B, P, 512]，给空间正则项用 ===
        feat_fc7_reshaped = feat_fc7.view(B, P, -1)  # [B, P, 512]

        # 3) 两个分支的 logits
        class_logits = self.class_branch(feat_fc7)  # [B*P, num_classes]
        det_logits   = self.det_branch(feat_fc7)    # [B*P, num_classes]

        # 4) reshape 回 [B, P, C]
        class_logits = class_logits.view(B, P, -1)
        det_logits   = det_logits.view(B, P, -1)

        # 5) WSDDN 两个 softmax
        class_prob = F.softmax(class_logits, dim=2)  # [B, P, C]
        det_prob   = F.softmax(det_logits,   dim=1)  # [B, P, C]

        # 6) proposal-level 置信度
        joint_prob = class_prob * det_prob           # [B, P, C]

        # 7) video-level 预测
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

        # 输入变成 spp_out_dim
        self.fc6 = nn.Linear(self.spp_out_dim, 1024)
        self.fc7 = nn.Linear(1024, 512)

        self.class_branch = nn.Linear(512, num_classes)
        self.det_branch   = nn.Linear(512, num_classes)

    def forward(self, x, proposal_boxes):
        """
        x: [B, C, T_global]  — backbone 输出
        proposal_boxes: [B, P, 2]  — 每个 proposal 的 [start, end]（时间索引）
        """
        B, C, T_global = x.shape
        P = proposal_boxes.shape[1]

        proposal_features = []

        # ======= 对每个 proposal 做 SPP =======
        for b in range(B):
            for p in range(P):
                start = int(proposal_boxes[b, p, 0].item())
                end   = int(proposal_boxes[b, p, 1].item())
                # 防止 start==end 或越界
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

        # 方便 spatial regularization，用 [B, P, 512] 的形状
        feat_fc7_reshaped = feat_fc7.view(B, P, -1)

        # ======= 分类/检测分支 =======
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

class WSDDNTransformerIMU(nn.Module):
    """
    WSDDN + Transformer 版本（proposal 先做 1D SPP）：
    - 每个 proposal 从 backbone 特征中截取 [1, C, T_p]
    - 用 TemporalSPP1D 做 1D SPP -> [1, C * sum(levels)]
    - 把所有 proposal 当成 token：[B, P, spp_out_dim] -> 投影到 d_model
    - 按中心时间排序 + 加时间位置编码 -> TransformerEncoder
    - 再反排序回原顺序，接 WSDDN 的 class / det 双分支
    - 返回 proposal_ctx 作为 feat_fc7，供空间正则使用
    """
    def __init__(
        self,
        num_classes: int = 30,
        feat_dim: int = 512,        # backbone 输出的 channel 数 C
        d_model: int = 512,         # Transformer 内部维度
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
        spp_levels=(1, 2, 4),       # 1D SPP 的金字塔层数
        pool_type: str = "avg",     # "avg" 或 "max"
    ):
        super().__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # 1D SPP：输入 [N, C, T] -> 输出 [N, C * sum(levels)]
        self.spp = TemporalSPP1D(levels=spp_levels, pool_type=pool_type)
        self.spp_out_dim = feat_dim * self.spp.out_mul  # C * sum(levels)

        # 把 SPP 特征投影到 d_model
        if self.spp_out_dim != d_model:
            self.input_proj = nn.Linear(self.spp_out_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        # 时间位置编码：输入 [center_norm, length_norm]，输出 [d_model]
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
            batch_first=False,   # 使用 [P, B, d_model]
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # WSDDN 的两个分支：分类 & 检测
        self.class_branch = nn.Linear(d_model, num_classes)
        self.det_branch   = nn.Linear(d_model, num_classes)

    def _pool_proposals(self, x, proposal_boxes):
        """
        从 backbone 特征 x 中，按 proposal_boxes 截取并做 1D SPP。
        x: [B, C, T_global]
        proposal_boxes: [B, P, 2] （start, end），是索引（帧下标）
        返回:
            proposal_feats: [B, P, spp_out_dim]
        """
        B, C, T_global = x.shape
        _, P, _ = proposal_boxes.shape

        # 预分配 [B, P, C * sum(levels)]
        proposal_feats = x.new_zeros(B, P, self.spp_out_dim)

        # clamp 索引范围
        starts = proposal_boxes[..., 0].clamp(0, T_global - 1).long()
        ends   = proposal_boxes[..., 1].clamp(1, T_global).long()

        # 确保 end > start
        bad_mask = ends <= starts
        ends[bad_mask] = (starts[bad_mask] + 1).clamp(max=T_global)

        for b in range(B):
            for p in range(P):
                s = int(starts[b, p].item())
                e = int(ends[b, p].item())
                # [1, C, T_p]
                seg = x[b:b+1, :, s:e]
                if seg.size(2) == 0:
                    # 极端情况下兜底
                    seg = x[b:b+1, :, s:s+1]
                # 1D SPP: [1, C * sum(levels)]
                feat_spp = self.spp(seg)          # [1, C * sum(levels)]
                feat_spp = feat_spp.view(-1)      # [C * sum(levels)]
                proposal_feats[b, p] = feat_spp

        return proposal_feats  # [B, P, spp_out_dim]

    def forward(self, x, proposal_boxes):
        """
        x: [B, C, T_global]  —— 预训练 backbone 的输出
        proposal_boxes: [B, P, 2] —— (start, end)，时间索引（单位：帧）
        """
        B, C, T_global = x.shape
        _, P, _ = proposal_boxes.shape

        # 1) 对每个 proposal 做 1D SPP 池化，得到 [B, P, spp_out_dim]
        proposal_feats = self._pool_proposals(x, proposal_boxes)  # [B, P, spp_out_dim]

        # 2) 构造时间位置编码（中心 + 时长）
        starts_f = proposal_boxes[..., 0].float()
        ends_f   = proposal_boxes[..., 1].float()
        centers  = (starts_f + ends_f) / 2.0        # [B, P]
        lengths  = (ends_f - starts_f).clamp(min=1.0)

        T_global_f = float(T_global)
        center_norm = centers / T_global_f          # [0, 1]
        length_norm = lengths / T_global_f          # [0, 1]

        pos_feat = torch.stack([center_norm, length_norm], dim=-1)  # [B, P, 2]

        # 3) 按时间中心排序（对每个样本独立）
        sort_idx = centers.argsort(dim=1)           # [B, P]
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand_as(sort_idx)

        # 排序后的 proposal 特征 & pos 编码
        proposal_feats_sorted = proposal_feats[batch_idx, sort_idx, :]   # [B, P, spp_out_dim]
        if self.use_positional_encoding and self.pos_mlp is not None:
            pos_emb = self.pos_mlp(pos_feat)                             # [B, P, d_model]
            pos_emb_sorted = pos_emb[batch_idx, sort_idx, :]             # [B, P, d_model]
        else:
            pos_emb_sorted = 0

        # 4) 投影到 Transformer 维度，并加位置编码
        tokens = self.input_proj(proposal_feats_sorted)                  # [B, P, d_model]
        tokens = tokens + pos_emb_sorted                                 # [B, P, d_model]

        # 5) 变成 [P, B, d_model]，喂进 TransformerEncoder
        tokens = tokens.transpose(0, 1)          # [P, B, d_model]
        tokens_out = self.transformer(tokens)    # [P, B, d_model]
        tokens_out = tokens_out.transpose(0, 1)  # [B, P, d_model]，排序后的结果

        # 6) 反排序回原来的 proposal 顺序，保证和 proposal_boxes 对齐
        inv_sort_idx = torch.argsort(sort_idx, dim=1)   # [B, P]
        proposal_ctx = tokens_out[batch_idx, inv_sort_idx, :]  # [B, P, d_model]

        # 7) WSDDN 两个分支
        feat_ctx_flat = proposal_ctx.reshape(B * P, -1)        # [B*P, d_model]

        class_logits = self.class_branch(feat_ctx_flat)        # [B*P, num_classes]
        det_logits   = self.det_branch(feat_ctx_flat)          # [B*P, num_classes]

        class_logits = class_logits.view(B, P, self.num_classes)
        det_logits   = det_logits.view(B, P, self.num_classes)

        # WSDDN 双 softmax
        class_prob = F.softmax(class_logits, dim=2)  # [B, P, C]，对类别 softmax
        det_prob   = F.softmax(det_logits,   dim=1)  # [B, P, C]，对 proposal softmax

        joint_prob = class_prob * det_prob           # [B, P, C]
        video_prob = joint_prob.sum(dim=1)           # [B, C]，视频级预测

        # 用 proposal_ctx 作为 feat_fc7，方便空间正则
        feat_fc7_reshaped = proposal_ctx             # [B, P, d_model]

        return {
            "video_prob": video_prob,
            "joint_prob": joint_prob,
            "class_logits": class_logits,
            "det_logits": det_logits,
            "feat_fc7": feat_fc7_reshaped,
        }

def generate_proposal_boxes(
    T_global=30,          # 特征提取后的时序维度
    num_proposals=50,     # 候选总数
    raw_data_length=1500, # 原始数据总长度（30s*50=1500）
    base_physical_sec=7,  # 候选对象的实际物理时长（秒）7s
    total_physical_sec=30,# 片段的总物理时长（秒）30s
    step_raw=2,         # 滑动步长的实际物理时长（秒）2s
    min_raw_length=5,   # 最短候选的实际物理时长（秒）5s
    max_raw_length=15   # 最长候选的实际物理时长（秒）15s
):
    """
    生成候选片段（特征维度的start_idx和end_idx）
    核心：基于原始数据尺度转换到特征尺度（T_global）
    """
    proposal_boxes = []

    # 转换系数：原始数据的1个点 → 特征维度的比例
    # 原来是30*50=1500，转为2048，再转为T_global,所以绝对=相对*（2048/1500）*（T_global/2048）=相对*（T_global/1500）
    raw_to_feat = T_global / raw_data_length

    # -------------------------- 1. 滑动窗口基础候选（基于物理时长） --------------------------
    base_raw_length = int((base_physical_sec / total_physical_sec) * raw_data_length) #7*（30/1500）

    # 基础候选长度转换到特征尺度
    base_feat_length = max(1, int(base_raw_length * raw_to_feat))

    # 滑动步长（原始数据→特征尺度）
    step_feat = max(1, int(step_raw * 50 * raw_to_feat))  # 50是采样率

    # 生成滑动窗口候选
    start_feat = 0
    while start_feat + base_feat_length <= T_global:
        end_feat = start_feat + base_feat_length
        proposal_boxes.append([start_feat, end_feat])
        start_feat += step_feat  # 按特征步长滑动


    # -------------------------- 2. 随机长度候选补充（基于原始数据长度范围） --------------------------
    # 补充候选至num_proposals
    while len(proposal_boxes) < num_proposals:
        # 随机生成原始数据时长（在[min_raw_length, max_raw_length]范围内，即5-15s）
        raw_length = random.randint(min_raw_length, max_raw_length)
        # 转换到特征尺度
        feat_length = max(1, int(raw_length * 50 * raw_to_feat))

        # 随机生成起始位置（原始数据→特征尺度）
        max_raw_start = total_physical_sec - raw_length  # 原始数据最大起始点（总时长-候选对象时长）
        if max_raw_start <= 0:
            raw_start = 0
        else:
            raw_start = random.randint(0, max_raw_start)
        feat_start = int(raw_start * 50 * raw_to_feat)

        # 计算结束位置（确保不超过特征长度）
        feat_end = min(feat_start + feat_length, T_global)
        proposal_boxes.append([feat_start, feat_end])


    # 截断至num_proposals并返回
    return torch.tensor(proposal_boxes[:num_proposals], dtype=torch.long)

# -------------------------- 3. 候选生成（滑动窗口+随机长度） --------------------------
def generate_proposals(imu_data, num_proposals=50, base_len=476, step=256, min_len=340, max_len=884):
    """
    :param imu_data: 单个30s IMU数据 [30, 2048]
    :param base_len: 7s对应的维度（50Hz×7s=350 → 插值后2048*(7/30)≈476）
    :param step: 滑动步长（256）
    :param min_len/max_len: 随机长度范围
    :return: 50个候选片段 [50, 30, T_proposal]
    """
    C, T_total = imu_data.shape  # 30, 2048
    proposals = []

    # 1. 滑动窗口生成基础候选（固定7s长度）
    start_positions = list(range(0, T_total - base_len + 1, step))
    for start in start_positions:
        proposal = imu_data[:, start:start+base_len]
        proposals.append(proposal)

    # 2. 随机起始+随机长度补充候选
    while len(proposals) < num_proposals:
        t_len = random.randint(min_len, max_len)
        start = random.randint(0, T_total - t_len)
        proposal = imu_data[:, start:start+t_len]
        proposals.append(proposal)

    return torch.stack(proposals)

# -------------------------- 4. 候选合并（重叠候选去重，保留25个） --------------------------
def merge_overlapping_proposals(proposals, det_scores, overlap_thresh=0.5):
    """
    :param proposals: 候选片段 [50, 30, 884]
    :param det_scores: 候选置信度 [50, num_classes]
    :return: 合并后的25个候选 [25, 30, 884]
    """
    # 简化逻辑：按置信度排序，保留高置信度且不重叠的候选
    sorted_idx = det_scores.mean(dim=1).argsort(descending=True)
    keep = []
    for idx in sorted_idx:
        p = proposals[idx]
        # 检查与已保留候选的重叠
        overlap = False
        for k in keep:
            # 计算时间维度重叠比例（简化：取候选起始/结束位置）
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

