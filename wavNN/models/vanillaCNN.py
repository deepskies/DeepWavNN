import torch.nn as nn


class VanillaCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm_hidden = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=False,
        )
        self.batch_norm_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm_hidden(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = self.batch_norm_out(x)
        x = nn.ReLU()(x)
        return x
