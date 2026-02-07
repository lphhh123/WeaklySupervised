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

                        
        self.backbone = backbone_factory(
            dataset_name='HANGTIME',
            backbone_type=cfg.BACKBONE_TYPE,
            in_channels=len_feature,
            pretrained_path=cfg.PRETRAINED_PATH
        )

              
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

                
        self.f_cls = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes, kernel_size=1, bias=False),
        )
        self.dropout = nn.Dropout(p=0.5)

    def extract_features(self, x):
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

                                                    
        # if out.shape[2] != T_input:
            # out = F.interpolate(out, size=T_input, mode='linear', align_corners=False)

        return out

    def predict(self, embeddings):
        out = self.dropout(embeddings)
        out = self.f_cls(out)  # [B, NumClass, T]

        cas = out.permute(0, 2, 1)  # [B, T, NumClass]
        actionness = cas.sum(dim=2)  # [B, T]

                                              
        return embeddings.permute(0, 2, 1), cas, actionness

    def forward(self, x):
                                                      
        feats = self.extract_features(x)

                 
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
                
        T = cas.shape[1]
        k = max(1, min(k_easy, T))

        sorted_scores, _ = cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k, :]
        video_scores = self.softmax(topk_scores.mean(1))

        return video_scores

    def forward(self, x):
                                              
        embeddings, cas, actionness = self.actionness_module(x)

                             
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