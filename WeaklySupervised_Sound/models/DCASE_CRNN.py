import warnings
import torch
import torch.nn as nn
import random

from .DCASE_CNN import CNN
from .DCASE_RNN import BidirectionalGRU


class IMUSpecAugment(nn.Module):
    """
    Pure PyTorch SpecAugment for IMU data.
    Replaces torchaudio.transforms.TimeMasking and FreqMasking.
    """

    def __init__(self, time_drop_width, time_stripes_num, sensor_drop_width, sensor_stripes_num):
        super().__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.sensor_drop_width = sensor_drop_width
        self.sensor_stripes_num = sensor_stripes_num

    def forward(self, x):
        # x shape: [Batch, Channel=1, Time, Sensors]

        if not self.training:
            return x

        x_aug = x.clone()
        batch_size, _, time_dim, sensor_dim = x_aug.shape

        for _ in range(self.time_stripes_num):
            t = random.randint(0, self.time_drop_width)
            t0 = random.randint(0, max(0, time_dim - t))
            x_aug[:, :, t0:t0 + t, :] = 0

        for _ in range(self.sensor_stripes_num):
            f = random.randint(0, self.sensor_drop_width)
            f0 = random.randint(0, max(0, sensor_dim - f))
            x_aug[:, :, :, f0:f0 + f] = 0

        return x_aug


class CRNN(nn.Module):
    def __init__(
            self,
            n_in_channel,
            nclass=10,
            attention=True,
            activation="glu",
            dropout=0.5,
            train_cnn=True,
            rnn_type="BGRU",
            n_RNN_cell=128,
            n_layers_RNN=2,
            dropout_recurrent=0,
            cnn_integration=False,
            freeze_bn=False,
            use_embeddings=False,
            embedding_size=527,
            embedding_type="global",
            frame_emb_enc_dim=512,
            aggregation_type="global",
            specaugm_t_p=0.2,
            specaugm_t_l=5,  # Time Mask Width
            specaugm_f_p=0.2,
            specaugm_f_l=2,
            dropstep_recurrent=0.0,
            dropstep_recurrent_len=5,
            **kwargs,
    ):
        super(CRNN, self).__init__()

        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn
        self.use_embeddings = use_embeddings
        self.embedding_type = embedding_type
        self.aggregation_type = aggregation_type
        self.nclass = nclass
        self.dropstep_recurrent = dropstep_recurrent
        self.dropstep_recurrent_len = dropstep_recurrent_len

        self.spec_augmenter = IMUSpecAugment(
            time_drop_width=specaugm_t_l,
            time_stripes_num=2,
            sensor_drop_width=specaugm_f_l,
            sensor_stripes_num=1
        )

        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn, activation=activation, conv_dropout=dropout, **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1]
            if self.cnn_integration:
                nb_in = nb_in * n_in_channel
            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)

        # Classification Head Logic
        if isinstance(self.nclass, (tuple, list)) and len(self.nclass) > 1:
            self.dense = torch.nn.ModuleList([])
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=-1)
            for current_classes in self.nclass:
                self.dense.append(nn.Linear(n_RNN_cell * 2, current_classes))
                if self.attention:
                    self.dense_softmax.append(
                        nn.Linear(n_RNN_cell * 2, current_classes)
                    )
        else:
            if isinstance(self.nclass, (tuple, list)):
                self.nclass = self.nclass[0]
            self.dense = nn.Linear(n_RNN_cell * 2, self.nclass)
            self.sigmoid = nn.Sigmoid()

            if self.attention:
                self.dense_softmax = nn.Linear(n_RNN_cell * 2, self.nclass)
                self.softmax = nn.Softmax(dim=-1)

        if self.use_embeddings:
            self._init_embeddings_layers(embedding_size, frame_emb_enc_dim, nb_in)

    def _init_embeddings_layers(self, embedding_size, frame_emb_enc_dim, nb_in):
        if self.aggregation_type == "frame":
            self.frame_embs_encoder = nn.GRU(
                batch_first=True, input_size=embedding_size, hidden_size=512, bidirectional=True
            )
            self.shrink_emb = torch.nn.Sequential(
                torch.nn.Linear(2 * frame_emb_enc_dim, nb_in),
                torch.nn.LayerNorm(nb_in),
            )
            self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)
        elif self.aggregation_type == "global":
            self.shrink_emb = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, nb_in), torch.nn.LayerNorm(nb_in)
            )
            self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)
        elif self.aggregation_type in ["interpolate", "pool1d"]:
            self.cat_tf = torch.nn.Linear(nb_in + embedding_size, nb_in)
        else:
            self.cat_tf = torch.nn.Linear(2 * nb_in, nb_in)

    def _get_logits(self, x, pad_mask, classes_mask=None):
        out_strong = []
        out_weak = []
        if isinstance(self.nclass, (tuple, list)):
            for indx, c_classes in enumerate(self.nclass):
                dense_softmax = (
                    self.dense_softmax[indx] if hasattr(self, "dense_softmax") else None
                )
                c_strong, c_weak = self._get_logits_one_head(
                    x, pad_mask, self.dense[indx], dense_softmax, classes_mask
                )
                out_strong.append(c_strong)
                out_weak.append(c_weak)
            return torch.cat(out_strong, 1), torch.cat(out_weak, 1)
        else:
            dense_softmax = (
                self.dense_softmax if hasattr(self, "dense_softmax") else None
            )
            return self._get_logits_one_head(
                x, pad_mask, self.dense, dense_softmax, classes_mask
            )

    def _get_logits_one_head(self, x, pad_mask, dense, dense_softmax, classes_mask=None):
        strong = dense(x)  # [bs, frames, nclass]
        strong = self.sigmoid(strong)
        if classes_mask is not None:
            classes_mask = ~classes_mask[:, None].expand_as(strong)

        if self.attention in [True, "legacy"]:
            sof = dense_softmax(x)
            if pad_mask is not None:
                sof = sof.masked_fill(pad_mask.transpose(1, 2), -1e30)
            if classes_mask is not None:
                sof = sof.masked_fill(classes_mask, -1e30)
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)
        else:
            weak = strong.mean(1)

        if classes_mask is not None:
            strong = strong.masked_fill(classes_mask, 0.0)
            weak = weak.masked_fill(classes_mask[:, 0], 0.0)

        return strong.transpose(1, 2), weak

    def forward(self, x, pad_mask=None, embeddings=None, classes_mask=None):
        # x input shape assumption: [Batch, Channels, Time] (Standard PyTorch 1D data)

        x = x.transpose(1, 2).unsqueeze(1)

        x = self.spec_augmenter(x)

        # input size now : (batch_size, 1, n_frames, n_freq/sensors)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # 3. CNN Forward
        x = self.cnn(x)
        # Output: [bs, chan_out, frames, freq_out]

        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        # 4. Flattening for RNN
        if freq != 1:
            x = x.permute(0, 2, 1, 3)  # [bs, frames, chan, freq]
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        if self.use_embeddings and embeddings is not None:
            pass

            # 6. RNN Forward
        x = self.rnn(x)
        x = self.dropout(x)

        return self._get_logits(x, pad_mask, classes_mask)
