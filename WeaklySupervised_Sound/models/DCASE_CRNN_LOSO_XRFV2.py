# models/DCASE_CRNN.py
import torch
import torch.nn as nn
import random
from .DCASE_CNN import CNN
from .DCASE_RNN import BidirectionalGRU



class CRNN(nn.Module):

    def forward(self, x, pad_mask=None, embeddings=None, classes_mask=None):

        if x.dim() == 3:
            x = x.unsqueeze(2)
        elif x.dim() == 4 and x.shape[1] == 1:
            x = x.transpose(1, 2)

            # 2. Apply Augmentation
        # if x.dim() == 4:
        x = self.spec_augmenter(x)

        # 3. CNN Forward
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