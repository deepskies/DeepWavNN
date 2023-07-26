import torch.nn as nn
import math


class VanillaCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        out_channels: int,
        hidden_channels_1: int = 1,
        hidden_channels_2: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=int(hidden_channels_1),
            kernel_size=int(kernel_size),
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm_hidden = nn.BatchNorm2d(int(hidden_channels_1))
        conv1_out = math.ceil(in_channels + 2 - int(kernel_size) + 1)

        self.conv2 = nn.Conv2d(
            in_channels=int(hidden_channels_1),
            out_channels=int(hidden_channels_2),
            kernel_size=int(kernel_size),
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm_out = nn.BatchNorm2d(int(hidden_channels_2))
        conv2_out = math.ceil(conv1_out + 2 - int(kernel_size) + 1)

        self.dense_out = nn.Linear(
            in_features=(conv2_out**2) * int(hidden_channels_2),
            out_features=out_channels,
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm_hidden(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.batch_norm_out(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)
        x = self.dense_out(x)
        return x
