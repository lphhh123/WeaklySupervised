# pre_model.py
import math
import torch
import torch.nn as nn


class CNN1DBackbone(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),  # (30,478)→(64,239)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # (64,239)→(64,119)

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # (64,119)→(128,60)
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # (128,60)→(256,30)
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, feat_dim, kernel_size=3, stride=1, padding=1),  # (256,30)→(512,30)
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        return self.layers(x)  # (B, 512, 30)


class CNN1DClassifier(nn.Module):
    def __init__(self, num_classes, task='single', feat_dim=512,in_channels=30,):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = CNN1DBackbone(self.in_channels,feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B,512,30)→(B,512,1)
            nn.Flatten(),  # (B,512)
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)  # (B, num_classes)




class LSTMBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, hidden_dim=256, feat_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.output_proj = nn.Linear(2 * hidden_dim, feat_dim)

    def forward(self, x):
        # x: (B,30,478) → (B,478,30)
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # (B,478,256)
        lstm_out, _ = self.lstm(x)  # (B,478,512)
        feat = self.output_proj(lstm_out)  # (B,478,512)
        return feat.permute(0, 2, 1)  # (B,512,478)


class LSTMClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', hidden_dim=256, feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = LSTMBackbone_7s(self.in_channels,hidden_dim=hidden_dim, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class TransformerBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, d_model=128, nhead=4, num_layers=3, feat_dim=512):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)  # (30,478)→(128,478)
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 478))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,enable_nested_tensor=False,)
        self.output_proj = nn.Conv1d(d_model, feat_dim, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder  # (B,128,478)
        x = x.permute(0, 2, 1)  # (B,478,128)
        x = self.transformer(x)  # (B,478,128)
        x = x.permute(0, 2, 1)  # (B,128,478)
        return self.output_proj(x)  # (B,512,478)


class TransformerClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', d_model=128, nhead=4, num_layers=3, feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = TransformerBackbone_7s(self.in_channels,d_model=d_model, nhead=nhead, num_layers=num_layers, feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class TCNBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, feat_dim, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm1d(feat_dim),
        )

    def forward(self, x):
        return self.layers(x)  # (B,512,478)


class TCNClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = TCNBackbone_7s(self.in_channels,feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)
class ResidualBlock_7s(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3,
            stride=1, padding=dilation, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=dilation, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        return self.relu(out)

class ResTCNBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ResidualBlock_7s(128, 256, dilation=2),
            ResidualBlock_7s(256, 256, dilation=4),
            ResidualBlock_7s(256, feat_dim, dilation=8),
        )

    def forward(self, x):
        x = self.initial(x)  # (B,128,478)
        return self.blocks(x)  # (B,512,478)

class ResTCNClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = ResTCNBackbone_7s(self.in_channels,feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

class MultiScaleBlock_7s(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//3),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels//3),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//3),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)

class MSTCNBackbone_7s(nn.Module):
    def __init__(self, in_channels=30, feat_dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            MultiScaleBlock_7s(in_channels, 128),
            MultiScaleBlock_7s(128, 256),
            MultiScaleBlock_7s(256, feat_dim),
        )

    def forward(self, x):
        return self.layers(x)  # (B,512,478)

class MSTCNClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = MSTCNBackbone_7s(self.in_channels,feat_dim=feat_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)



class AFBlock1D_7s(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return self.relu(out)


class ActionFormerBackbone_7s(nn.Module):
    def __init__(
        self,
        in_channels: int = 30,
        feat_dim: int = 256,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        trans_layers: int = 2,
        trans_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(in_channels, feat_dim, kernel_size=1, bias=False)


        dilations = [1, 2, 4, 8, 16, 32]
        blocks = []
        for i in range(num_blocks):
            d = dilations[i % len(dilations)]
            blocks.append(
                AFBlock1D_7s(
                    feat_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)


        self.trans = TemporalTransformer1D(
            d_model=feat_dim,
            nhead=trans_heads,
            num_layers=trans_layers,
            dim_feedforward=4 * feat_dim,
            dropout=dropout,
            max_len=max_len,
        )


        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=feat_dim)

    def forward(self, x):
        # x: (B, 30, 478)
        x = self.input_proj(x)   # (B, feat_dim, T)
        x = self.blocks(x)
        x = self.trans(x)
        x = self.out_norm(x)
        return x                 # (B, feat_dim, T)


class ActionFormerClassifier_7s(nn.Module):
    def __init__(self, num_classes, task='single', feat_dim=512,in_channels=30):
        super().__init__()
        self.task = task
        self.in_channels = in_channels
        self.backbone = ActionFormerBackbone_7s(
            self.in_channels,
            feat_dim=feat_dim,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)       # (B, C, T)
        pooled = self.pool(feat)      # (B, C, 1)
        flat = pooled.squeeze(-1)     # (B, C)
        logits = self.fc(flat)        # (B, num_classes)
        if return_feat:
            return logits, feat
        return logits


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model


        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        B, T, C = x.shape

        if T <= self.pe.size(1):

            pe = self.pe[:, :T, :]                             # (1, T, C)
        else:

            device = x.device
            dtype = x.dtype
            position = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
                * (-math.log(10000.0) / self.d_model)
            )
            pe = torch.zeros(1, T, self.d_model, device=device, dtype=dtype)     # (1, T, C)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)

        return x + pe


class TemporalTransformer1D(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding1D(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, C, T)
        """
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.encoder(x)       # (B, T, C)
        return x.transpose(1, 2)

