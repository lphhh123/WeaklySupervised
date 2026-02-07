import torch
import torch.nn as nn

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        # x shape: [Batch, Channel, Height, Width]
        # GLU通常作用在 Channel 维度上
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res

class ContextGating(nn.Module):
    def __init__(self, input_num):
        super(ContextGating, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(lin)
        res = x * sig
        return res

class CNN(nn.Module):
    def __init__(
        self,
        n_in_channel,
        activation="Relu",
        conv_dropout=0,
        kernel_size=[3, 3, 3],
        padding=[1, 1, 1],
        stride=[1, 1, 1],
        nb_filters=[64, 64, 64],
        # 【关键修改】默认Pooling策略
        # 原版是 [(1, 4), (1, 4), (1, 4)] -> 传感器维度缩小 64 倍
        # 这对 120 轴可能勉强能行，但对 6 轴/9 轴数据会直接导致维度归零报错
        # 建议设为 None 或保守值，强制外部传入
        pooling=[(1, 2), (1, 2), (1, 2)],
        normalization="batch",
        **transformer_kwargs
    ):
        """
            Initialization of CNN network

        Args:
            n_in_channel: int, number of input channel (对于CRNN结构，这里通常是1)
            activation: str, activation function
            conv_dropout: float, dropout
            kernel_size: list, kernel size
            padding: list, padding
            stride: list, stride
            nb_filters: list, number of filters
            pooling: list of tuples, pooling size for (Time, Sensors)
                     Example: (1, 2) means Time dim stays same, Sensor dim / 2
            normalization: choose between "batch" and "layer".
        """
        super(CNN, self).__init__()

        self.nb_filters = nb_filters
        cnn = nn.Sequential()

        def conv(i, normalization="batch", dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module(
                "conv{0}".format(i),
                nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]),
            )
            if normalization == "batch":
                cnn.add_module(
                    "batchnorm{0}".format(i),
                    nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99),
                )
            elif normalization == "layer":
                cnn.add_module("layernorm{0}".format(i), nn.GroupNorm(1, nOut))

            if activ.lower() == "leakyrelu":
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module("relu{0}".format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module("glu{0}".format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module("cg{0}".format(i), ContextGating(nOut))

            if dropout is not None:
                cnn.add_module("dropout{0}".format(i), nn.Dropout(dropout))

        for i in range(len(nb_filters)):
            conv(i, normalization=normalization, dropout=conv_dropout, activ=activation)
            # Pooling: (Time, Sensors)
            # 这里对应 PyTorch AvgPool2d(kernel_size=(H, W))
            # 在 CRNN forward 中，H=Time, W=Sensors
            cnn.add_module(
                "pooling{0}".format(i), nn.AvgPool2d(pooling[i])
            )

        self.cnn = cnn

    def forward(self, x):
        """
        Forward step of the CNN module

        Args:
            x (Tensor): input batch of size (batch_size, 1, n_frames, n_sensors)
        """
        # conv features
        x = self.cnn(x)
        return x