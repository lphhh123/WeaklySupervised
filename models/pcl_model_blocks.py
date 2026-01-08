import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

class OICRLosses(nn.Module):
    """
    和原版一样的 OICR 损失：
      prob: [P, C+1] softmax 概率（0 是背景）
      labels: [P] 0..C
      cls_loss_weights: [P]
      gt_assignments: [P] （这里只用不到，为了接口兼容保留）
    """
    def __init__(self):
        super().__init__()

    def forward(self, prob, labels, cls_loss_weights, gt_assignments, eps=1e-6):
        # 挑出每个 proposal 的当前标签的概率
        logp = torch.log(prob + eps)[torch.arange(prob.size(0), device=prob.device), labels]
        loss = -logp * cls_loss_weights
        return loss.mean()


def mil_losses(cls_score, labels):
    """
    image-level MIL loss，与原版 pcl_heads.mil_losses 一致：
      cls_score: [B, C]，视频级概率
      labels:    [B, C]，0/1 多标签
    """
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    return loss.mean()


# class mil_outputs(nn.Module):
#     """
#     原始 PCL/OICR 的 MIL head：
#       score0 在 proposal 维度 softmax
#       score1 在类维度 softmax
#       最后相乘
#     """
#     def __init__(self, dim_in, dim_out):
#         super().__init__()
#         self.mil_score0 = nn.Linear(dim_in, dim_out)
#         self.mil_score1 = nn.Linear(dim_in, dim_out)
#         self._init_weights()
#
#     def _init_weights(self):
#         init.normal_(self.mil_score0.weight, std=0.01)
#         init.constant_(self.mil_score0.bias, 0)
#         init.normal_(self.mil_score1.weight, std=0.01)
#         init.constant_(self.mil_score1.bias, 0)
#
#     def forward(self, x):
#         # 支持 [N, D] 或 [B, P, D]
#         if x.dim() == 3:
#             B, P, D = x.shape
#             x = x.view(B * P, D)
#
#         mil0 = self.mil_score0(x)       # [N, C]
#         mil1 = self.mil_score1(x)       # [N, C]
#         # 沿 proposal 维度和类别维度分别 softmax
#         score = F.softmax(mil0, dim=0) * F.softmax(mil1, dim=1)
#         return score

class mil_outputs(nn.Module):
    """
    MIL head：
      - proposal 维 softmax：应该在每个视频内部做（dim=1）
      - class   维 softmax：在类别维做（dim=2）
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.mil_score0 = nn.Linear(dim_in, dim_out)
        self.mil_score1 = nn.Linear(dim_in, dim_out)
        self._init_weights()

    def _init_weights(self):
        init.normal_(self.mil_score0.weight, std=0.01)
        init.constant_(self.mil_score0.bias, 0)
        init.normal_(self.mil_score1.weight, std=0.01)
        init.constant_(self.mil_score1.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            mil0 = self.mil_score0(x)              # [B,P,C]
            mil1 = self.mil_score1(x)              # [B,P,C]
            score0 = F.softmax(mil0, dim=1)        #  在 proposal 维 softmax（每个视频内部）
            score1 = F.softmax(mil1, dim=2)        #  在 class 维 softmax
            return score0 * score1                 # [B,P,C]

        # 保留旧接口：单视频 [P,D] -> [P,C]
        if x.dim() == 2:
            mil0 = self.mil_score0(x)              # [P,C]
            mil1 = self.mil_score1(x)              # [P,C]
            return F.softmax(mil0, dim=0) * F.softmax(mil1, dim=1)

        raise ValueError(f"mil_outputs expects 2D/3D, got {x.shape}")

class refine_outputs(nn.Module):
    """
    原始 PCL/OICR 的 refine head：
      共有 refine_times 个线性层，每个输出 [N, C+1]，按类别 softmax。
    """
    def __init__(self, dim_in, dim_out, refine_times):
        """
        dim_in:  输入特征维度（例如 hidden_dim 4096）
        dim_out: 输出通道数（C+1，含背景）
        refine_times: refine 的 stage 数 K
        """
        super().__init__()
        self.refine_times = refine_times
        self.refine_score = nn.ModuleList(
            [nn.Linear(dim_in, dim_out) for _ in range(refine_times)]
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.refine_score:
            init.normal_(layer.weight, std=0.01)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            B, P, D = x.shape
            x = x.view(B * P, D)

        outputs = []
        for layer in self.refine_score:
            logits = layer(x)
            prob = F.softmax(logits, dim=1)  # 按类别 softmax
            outputs.append(prob)
        return outputs