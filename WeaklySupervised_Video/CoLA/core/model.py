import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
import sys
import os

try:
    from pre_models import backbone_factory
except ImportError:
    from pre_models import get_backbone as backbone_factory


class Actionness_Module(nn.Module):
    def __init__(self, len_feature, num_classes, cfg):
        super(Actionness_Module, self).__init__()

        # 1. 加载 Backbone
        self.backbone = backbone_factory(
            dataset_name='HANGTIME',
            backbone_type=cfg.BACKBONE_TYPE,
            in_channels=len_feature,
            pretrained_path=cfg.PRETRAINED_PATH
        )

        # 冻结逻辑
        train_backbone = getattr(cfg, 'TRAIN_BACKBONE', False)
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Adapter
        self.adapter = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Conv1d(1024, 2048, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 3. 分类头
        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1, bias=False),
        )
        self.dropout = nn.Dropout(p=0.5)

    def extract_features(self, x):
        """
        [新增接口] 仅提取特征，用于 test_full 的特征拼接
        输入 x: [B, T, C] (例如 [1, 1500, 3])
        输出: [B, 2048, T] (例如 [1, 2048, 1500])
        """
        T_input = x.shape[1]
        out = x.permute(0, 2, 1)  # [B, C, T]

        # Backbone
        if not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                out = self.backbone(out)
        else:
            out = self.backbone(out)

        # Adapter
        out = self.adapter(out)

        # 强制插值回输入长度 (这是 CoLA 能够进行 Pixel-level 预测的关键)
        # if out.shape[2] != T_input:
            # out = F.interpolate(out, size=T_input, mode='linear', align_corners=False)

        return out

    def predict(self, embeddings):
        """
        [新增接口] 仅分类，用于处理拼接后的长特征图
        输入 embeddings: [B, 2048, T_long]
        """
        out = self.dropout(embeddings)
        out = self.f_cls(out)  # [B, NumClass, T]

        cas = out.permute(0, 2, 1)  # [B, T, NumClass]
        actionness = cas.sum(dim=2)  # [B, T]

        # 为了保持接口一致，返回 embedding 需要转回 [B, T, C]
        return embeddings.permute(0, 2, 1), cas, actionness

    def forward(self, x):
        """
        [原始接口] 保持完全不变，用于训练和 Window 推理
        """
        # 1. 提取特征 (含 Backbone + Adapter + Interpolate)
        feats = self.extract_features(x)

        # 2. 分类预测
        return self.predict(feats)


class CoLA(nn.Module):
    def __init__(self, cfg):
        super(CoLA, self).__init__()
        self.len_feature = cfg.FEATS_DIM
        self.num_classes = cfg.NUM_CLASSES
        self.actionness_module = Actionness_Module(cfg.FEATS_DIM, cfg.NUM_CLASSES, cfg)
        self.softmax = nn.Softmax(dim=1)
        self.r_easy = cfg.R_EASY
        self.r_hard = cfg.R_HARD
        self.m = cfg.m
        self.M = cfg.M
        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)
        actionness_drop = actionness * select_idx
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx
        easy_act = self.select_topk_embeddings(actionness_drop, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_rev_drop, embeddings, k_easy)
        return easy_act, easy_bkg

    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1, self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def get_video_cls_scores(self, cas, k_easy):
        # 兼容变长 T
        T = cas.shape[1]
        k = max(1, min(k_easy, T))

        sorted_scores, _ = cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k, :]
        video_scores = self.softmax(topk_scores.mean(1))

        return video_scores

    def forward(self, x):
        # 1. Actionness Module (调用 split 后的接口)
        embeddings, cas, actionness = self.actionness_module(x)

        # 2. Mining 逻辑 (保持不变)
        num_segments = x.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard

        easy_act, easy_bkg = self.easy_snippets_mining(actionness, embeddings, k_easy)
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)

        video_scores = self.get_video_cls_scores(cas, k_easy)

        contrast_pairs = {
            'EA': easy_act,
            'EB': easy_bkg,
            'HA': hard_act,
            'HB': hard_bkg
        }

        return video_scores, contrast_pairs, actionness, cas