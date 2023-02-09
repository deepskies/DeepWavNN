import torch
import torch.nn as nn
import pywt
import numpy as np


class WavMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        level: int,
        tail=False,
    ):
        """
        Simplest version is an MLP that takes a single layer of the wavelet
        And stacks the input channel-wise

        in_channels: Input size, int; should be the x, y of the input image
        out_channels: Out size of the mlp layer
        level: Level of the wavelet used in the MLP
        tail: Bool, If to add an activation at the end of the network to get a class output
        """
        super().__init__()
        assert level != 0, "Level 0 wavelet not supported"

        def linear_in_size(level, in_channels):
            ddim, aa = in_channels, []
            while ddim > 2:
                ddim = int(np.rint(ddim / 2))
                aa.append(ddim)
                if ddim == 2:
                    aa.append(0)
            aa.reverse()
            assert len(aa) > level
            return aa[level]

        # Wavelet transform of input x at a level as defined by the user
        self.wavelet = lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level])

        in_channels = linear_in_size(level, in_channels)

        # Channels for each of the 3 channels of the wavelet (Not including the downscaled original
        self.channel_1_mlp = nn.Linear(in_channels, hidden_size)
        self.channel_2_mlp = nn.Linear(in_channels, hidden_size)
        self.channel_3_mlp = nn.Linear(in_channels, hidden_size)

        # Flatten for when these are stacked
        self.flatten = nn.Flatten()

        # Output of those tied 3 channel layers
        self.output = nn.Linear(hidden_size * 3, out_channels)
        # Activation for a classifier
        self.tail = nn.Softmax(0) if tail else None

    def forward(self, x):
        # forward pass through the network
        # Wavelet at level L
        x = self.wavelet(x)
        # An MLP for each of the transformed levels
        channel_1 = self.channel_1_mlp(x[0])
        channel_2 = self.channel_2_mlp(x[1])
        channel_3 = self.channel_3_mlp(x[2])
        # stack the outputs
        concat = torch.stack([channel_1, channel_2, channel_3], dim=1)
        # Flatten for the output dense
        x = self.flatten(concat)

        x = self.output(x)

        if self.tail is not None:
            x = self.tail(x)

        return x


class VotingWavMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Slightly more complicated version of the WavMLP
        # Builds multiple single transform wavelet networks
        # And builds them into a non-tied voting classifier

    def forward(self):
        pass


class TiedWavMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Slightly more complicated WavMLP
        # Instead to just voting
        # It ties all the outputs of the
        # multiple networks into a dense output

    def forward(self):
        pass


if __name__ == "__main__":
    # simple compile test

    model = WavMLP(in_channels=28, out_channels=2, hidden_size=2, level=1)

    def get_output_shape(model, image_dim):
        return model(torch.rand(*(image_dim))).data.shape

    print(get_output_shape(model, (28, 28)))
