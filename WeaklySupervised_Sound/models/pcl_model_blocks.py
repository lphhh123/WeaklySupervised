import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

class OICRLosses(nn.Module):
    """
    Message OICR Message：
      prob: [P, C+1] softmax Message（0 Message）
      labels: [P] 0..C
      cls_loss_weights: [P]
      gt_assignments: [P] （Message，Message）
    """
    def __init__(self):
        super().__init__()

    def forward(self, prob, labels, cls_loss_weights, gt_assignments, eps=1e-6):

        logp = torch.log(prob + eps)[torch.arange(prob.size(0), device=prob.device), labels]
        loss = -logp * cls_loss_weights
        return loss.mean()


def mil_losses(cls_score, labels):
    """
    image-level MIL loss，Message pcl_heads.mil_losses Message：
      cls_score: [B, C]，Message
      labels:    [B, C]，0/1 Message
    """
    cls_score = cls_score.clamp(1e-6, 1 - 1e-6)
    labels = labels.clamp(0, 1)
    loss = -labels * torch.log(cls_score) - (1 - labels) * torch.log(1 - cls_score)
    return loss.mean()


class mil_outputs(nn.Module):
    """
    Message PCL/OICR Message MIL head：
      score0 Message proposal Message softmax
      score1 Message softmax
      Message
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
            mil0 = self.mil_score0(x)  # [B,P,C]
            mil1 = self.mil_score1(x)  # [B,P,C]
            score0 = F.softmax(mil0, dim=1)
            score1 = F.softmax(mil1, dim=2)
            return score0 * score1  # [B,P,C]


        if x.dim() == 2:
            mil0 = self.mil_score0(x)  # [P,C]
            mil1 = self.mil_score1(x)  # [P,C]
            return F.softmax(mil0, dim=0) * F.softmax(mil1, dim=1)

        raise ValueError(f"mil_outputs expects 1d/3D, got {x.shape}")


class refine_outputs(nn.Module):
    """
    Message PCL/OICR Message refine head：
      Message refine_times Message，Message [N, C+1]，Message softmax。
    """
    def __init__(self, dim_in, dim_out, refine_times):
        """
        dim_in:  Message（Message hidden_dim 4096）
        dim_out: Message（C+1，Message）
        refine_times: refine Message stage Message K
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
            prob = F.softmax(logits, dim=1)
            outputs.append(prob)
        return outputs