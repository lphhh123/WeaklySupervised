# model.py 里

import torch
import torch.nn as nn
from pre_train.pre_model import CNN1DBackbone   # 用你预训练用的那个 backbone

# --------- 1D 版的 ClassWisePool（完全仿照 2D 版，只是没有 H,W） ----------
class ClassWisePool1D(nn.Module):
    """
    输入: [B, C*K, T]
    输出: [B, C, T]
    每 C 个通道视为一组, 组内 K 个 map 做平均
    """
    def __init__(self, num_maps: int):
        super().__init__()
        self.num_maps = num_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Ck, T = x.size()
        assert Ck % self.num_maps == 0, \
            f"channels={Ck} 不能整除 num_maps={self.num_maps}"
        C = Ck // self.num_maps
        x = x.view(B, C, self.num_maps, T)      # [B, C, K, T]
        x = x.mean(dim=2)                       # 在 K 上平均 -> [B, C, T]
        return x


# --------- IMU 版 WSCNet（1D） ----------
class IMUWSCNet(nn.Module):
    """
    x: [B, 30, T]   (30 = 5个IMU * 6通道)
    backbone: CNN1DBackbone_7s (feat_dim = 512)
    输出:
      - logits_det: [B, C]   (detect 分支的 video-level logits)
      - logits_cls: [B, C]   (cls 分支的 video-level logits)
      - class_maps: [B, C, T']  (每类的 temporal sentiment map)
      - att_global: [B, 1, T']  (融合后的全局 attention map)
    """
    def __init__(self, num_classes: int, num_maps: int = 4,
                 in_channels: int = 30, feat_dim: int = 512):
        super().__init__()
        # 1) backbone：用 7s 预训练好的 CNN1DBackbone_7s 结构
        self.features = CNN1DBackbone_7s(in_channels=in_channels,
                                         feat_dim=feat_dim)

        # 2) WSCNet 结构
        self.num_classes = num_classes
        self.num_maps = num_maps

        # 每个类别 K 个子探测器: conv1d 变成 C*K 通道
        self.downconv = nn.Conv1d(feat_dim, num_classes * num_maps,
                                  kernel_size=1, stride=1)

        # class-wise pooling（第一个：K → per-class map）
        self.temporal_pool = ClassWisePool1D(num_maps)
        # 第二个：按类聚合 → 全局 attention map
        self.temporal_pool2 = ClassWisePool1D(num_classes)

        # 全局时间池化，用 avg pool
        self.GAP = nn.AdaptiveAvgPool1d(1)

        # classifier：拼接 原特征 + attention 过的特征
        self.classifier = nn.Linear(feat_dim * 2, num_classes)

    def forward(self, x):
        """
        x: [B, 30, T]  (T=2048)
        """
        # backbone 特征
        feat = self.features(x)       # [B, F, T'] F=512, T'~T/16
        feat_ori = feat

        # ------------- detect 分支 -------------
        x_conv = self.downconv(feat)          # [B, C*K, T']
        class_maps = self.temporal_pool(x_conv)  # [B, C, T']
        # 对时间做 GAP -> video-level logits (detect 分支)
        logits_det = self.GAP(class_maps).squeeze(-1)   # [B, C]

        # ------------- cls 分支（attention） -------------
        # 用每类得分加权 class-wise map -> 带权类 map
        # logits_det: [B, C] -> [B, C, 1]
        scores = logits_det.unsqueeze(-1)               # [B, C, 1]
        att_per_class = class_maps * scores             # [B, C, T']

        # 聚合所有类别得到一张全局 attention map: [B,1,T']
        att_global = self.temporal_pool2(att_per_class) # [B,1,T']
        att = torch.sigmoid(att_global)                 # 稍微 squashing 一下

        # broadcast 到 feature 维度
        att_broadcast = att.expand(-1, feat_ori.size(1), -1)  # [B,F,T']
        feat_gated = feat_ori * att_broadcast                 # [B,F,T']

        # 原特征 + gated 特征分别做 GAP -> concat -> 分类
        pooled_ori = self.GAP(feat_ori).squeeze(-1)           # [B,F]
        pooled_gated = self.GAP(feat_gated).squeeze(-1)       # [B,F]
        fused = torch.cat([pooled_ori, pooled_gated], dim=1)  # [B,2F]

        logits_cls = self.classifier(fused)                   # [B,C]

        return logits_det, logits_cls, class_maps, att_global

def fuse_wscnet_outputs(
    logits_det: torch.Tensor,   # [B, C]
    logits_cls: torch.Tensor,   # [B, C]
    class_maps: torch.Tensor,   # [B, C, T']
    att_global: torch.Tensor,   # [B, 1, T']
    alpha: float = 0.5,
    use_attention: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    测试阶段融合：
      - video_scores: [B, C]，融合两个分支的 video-level 打分
      - temporal_scores: [B, C, T']，融合 class-wise map + attention + video_scores

    融合策略：
      1) video_logits = alpha * logits_det + (1-alpha) * logits_cls
      2) video_scores = sigmoid(video_logits)
      3) class_act = sigmoid(class_maps)
      4) 如果 use_attention: class_act *= sigmoid(att_global)
      5) temporal_scores = class_act * video_scores.unsqueeze(-1)
    """
    # 1) 融合两个分支的 video-level logits
    video_logits = alpha * logits_det + (1.0 - alpha) * logits_cls    # [B, C]
    video_scores = torch.sigmoid(video_logits)                         # [B, C]

    # 2) class-wise 时序激活
    class_act = torch.sigmoid(class_maps)                              # [B, C, T']

    # 3) 融合全局 attention map
    if use_attention:
        att = torch.sigmoid(att_global)                                # [B,1,T']
        class_act = class_act * att                                    # 广播到 C 维

    # 4) 再乘上视频级得分，得到最终每类-每时刻得分
    temporal_scores = class_act * video_scores.unsqueeze(-1)           # [B, C, T']

    return video_scores, temporal_scores

