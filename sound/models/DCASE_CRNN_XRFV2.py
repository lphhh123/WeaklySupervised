import warnings
import torch
import torch.nn as nn
import random

# 假设这两个文件已经在同级目录下
from .DCASE_CNN import CNN
from .DCASE_RNN import BidirectionalGRU


class IMUSpecAugment(nn.Module):
    """
    针对IMU数据的SpecAugment纯PyTorch实现。
    替代 torchaudio.transforms.TimeMasking 和 FreqMasking
    """

    def __init__(self, time_drop_width, time_stripes_num, sensor_drop_width, sensor_stripes_num):
        super().__init__()
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.sensor_drop_width = sensor_drop_width
        self.sensor_stripes_num = sensor_stripes_num

    def forward(self, x):
        # x shape: [Batch, Channel=1, Time, Sensors]
        # 注意：在CRNN forward中，数据先被reshape成了 [B, 1, T, S] 格式

        if not self.training:
            return x

        x_aug = x.clone()
        batch_size, _, time_dim, sensor_dim = x_aug.shape

        # 1. Time Masking (沿时间轴屏蔽)
        for _ in range(self.time_stripes_num):
            t = random.randint(0, self.time_drop_width)
            t0 = random.randint(0, max(0, time_dim - t))
            x_aug[:, :, t0:t0 + t, :] = 0

        # 2. Sensor Masking (沿传感器轴屏蔽，原FreqMasking)
        # 在IMU中，这意味着随机丢弃某个轴的数据（模拟传感器故障）
        for _ in range(self.sensor_stripes_num):
            f = random.randint(0, self.sensor_drop_width)
            f0 = random.randint(0, max(0, sensor_dim - f))
            x_aug[:, :, :, f0:f0 + f] = 0

        return x_aug


class CRNN(nn.Module):
    def __init__(
            self,
            n_in_channel,  # [修改] 移除默认值，强制要求传入 (例如: 6, 9, 3)
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
            use_embeddings=False,  # [建议] IMU任务通常没有预训练Audio嵌入，建议默认为False
            embedding_size=527,
            embedding_type="global",
            frame_emb_enc_dim=512,
            aggregation_type="global",
            specaugm_t_p=0.2,  # 保留参数名兼容性，但在内部逻辑中可视为 mask probability 或 width
            specaugm_t_l=5,  # Time Mask Width
            specaugm_f_p=0.2,
            specaugm_f_l=2,  # Sensor Mask Width (注意IMU通道少，这个值不能太大，比如6轴数据mask 10就全没了)
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

        # [修改] 初始化自定义的 Augmentation 模块
        self.spec_augmenter = IMUSpecAugment(
            time_drop_width=specaugm_t_l,
            time_stripes_num=2,  # 这里可以参数化，默认设为2条
            sensor_drop_width=specaugm_f_l,
            sensor_stripes_num=1  # 传感器通道很少，通常mask 1条就够了
        )

        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn = 1

        # 注意：DCASE的CNN通常假定输入是单通道图像 (Batch, 1, Time, Freq)
        # 这里的 n_in_cnn 传递给 CNN 类，需确保 CNN 类能处理 input channels
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

        # Embeddings Logic (通常用于AudioSet预训练特征，IMU任务可忽略)
        if self.use_embeddings:
            self._init_embeddings_layers(embedding_size, frame_emb_enc_dim, nb_in)

    def _init_embeddings_layers(self, embedding_size, frame_emb_enc_dim, nb_in):
        # 将Embedding初始化逻辑剥离，保持主函数整洁
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
        # 保持原有的 logits 计算逻辑不变
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
        # 或者 [Batch, Time, Channels]

        # 1. 维度对齐
        # DCASE模型通常内部会将 input 转换为 [Batch, 1, Time, Channels/Freq]
        # 如果你的输入是 [B, C, T]，下面的 transpose(1,2) 变成 [B, T, C]，
        # 然后 unsqueeze(1) 变成 [B, 1, T, C]。
        # 这意味着 CNN 会把 (Time, Sensors) 当作 (Height, Width) 的单通道图像处理。
        x = x.transpose(1, 2).unsqueeze(1)

        # 2. Apply Augmentation (自定义的IMU SpecAugment)
        x = self.spec_augmenter(x)

        # input size now : (batch_size, 1, n_frames, n_freq/sensors)
        if self.cnn_integration:
            bs_in, nc_in = x.size(0), x.size(1)
            x = x.view(bs_in * nc_in, 1, *x.shape[2:])

        # 3. CNN Forward
        x = self.cnn(x)
        # 假设输出 shape: [bs, 64, frames, 4]

        bs, chan, frames, freq = x.size()
        if self.cnn_integration:
            x = x.reshape(bs_in, chan * nc_in, frames, freq)

        # 4. Flattening for RNN [修改此处逻辑]
        # 使用全局平均池化 (Global Average Pooling) 消除 freq 维度
        if freq > 1:
            # 将 [bs, chan, frames, freq] 在最后一个维度取平均
            x = torch.mean(x, dim=-1)  # 变成 [bs, chan, frames]
            x = x.permute(0, 2, 1)  # 变成 [bs, frames, chan]
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # 5. Embedding Concatenation (如果不用Embeddings可跳过)
        if self.use_embeddings and embeddings is not None:
            # ... (此处省略 Embedding 处理逻辑，保持原样即可，若 use_embeddings=False 不会执行)
            pass

            # 6. RNN Forward
        x = self.rnn(x)
        x = self.dropout(x)

        return self._get_logits(x, pad_mask, classes_mask)