# models/DCASE_CRNN.py
import torch
import torch.nn as nn
import random
from .DCASE_CNN import CNN
from .DCASE_RNN import BidirectionalGRU


# ... (IMUSpecAugment 类保持不变) ...

class CRNN(nn.Module):
    # ... (__init__ 保持不变) ...

    def forward(self, x, pad_mask=None, embeddings=None, classes_mask=None):
        # 假设输入 x 的原始形状为 [Batch, Sensors=30, Time]

        # 1. 维度对齐适配 XRFV2 (核心修改点)
        # 目标：[Batch, 30, 1, Time]
        # 这样卷积核 weight[64, 30, 3, 3] 对应的 In-channel 就是 30
        if x.dim() == 3:
            x = x.unsqueeze(2)
        elif x.dim() == 4 and x.shape[1] == 1:
            # 如果进来的已经是 [B, 1, S, T]，我们需要把 S 换到 Channel 位
            x = x.transpose(1, 2)

            # 2. Apply Augmentation
        # 修改 SpecAugment 内部解包适配，或者在外部暂时跳过
        # 这里为了稳妥，如果是 4 维则进入
        # if x.dim() == 4:
        x = self.spec_augmenter(x)

        # 3. CNN Forward
        # 注意：此处 cnn_integration 必须设为 False 才能让 CNN 看到 30 个通道
        x = self.cnn(x)

        # 4. Flattening for RNN
        bs, chan, frames, freq = x.size()
        if freq != 1:
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)

        # 5. RNN Forward
        x = self.rnn(x)
        x = self.dropout(x)
        return self._get_logits(x, pad_mask, classes_mask)