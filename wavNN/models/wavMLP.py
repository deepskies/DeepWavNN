import torch
import torch.nn as nn
import pywt
import numpy as np
from wavNN.utils import voting


class VanillaMLP(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, tail=False):
        super().__init__()

        self.flatten_input = nn.Flatten()
        self.hidden_layer = nn.Linear(in_channels**2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, out_channels)

        if tail:
            self.tail = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.flatten_input(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)

        if hasattr(self, "tail"):
            x = self.tail(x)

        return x


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
            assert len(aa) >= level
            return aa[level]

        # Wavelet transform of input x at a level as defined by the user
        self.wavelet = lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level])

        wav_in_channels = linear_in_size(level, in_channels)

        self.flatten_wavelet = nn.Flatten(start_dim=1, end_dim=-1)
        # Channels for each of the 3 channels of the wavelet (Not including the downscaled original
        self.channel_1_mlp = nn.Linear(wav_in_channels**2, hidden_size)
        self.channel_2_mlp = nn.Linear(wav_in_channels**2, hidden_size)
        self.channel_3_mlp = nn.Linear(wav_in_channels**2, hidden_size)

        # Flatten for when these are stacked
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Output of those tied 3 channel layers, and the flattened concat of those
        self.output = nn.Linear(hidden_size * 3, out_channels)

        # Activation for a classifier
        if tail:
            self.tail = nn.Softmax(dim=0)

    def forward(self, x):
        # forward pass through the network

        x = self.wavelet(x)
        # An MLP for each of the transformed levels
        channel_1 = self.channel_1_mlp(self.flatten_wavelet(x[0]))
        channel_2 = self.channel_2_mlp(self.flatten_wavelet(x[1]))
        channel_3 = self.channel_3_mlp(self.flatten_wavelet(x[2]))
        # stack the outputs
        concat = torch.stack([channel_1, channel_2, channel_3], dim=1)
        # Flatten for the output dense
        x = self.flatten(concat)
        x = self.output(x)

        if hasattr(self, "tail"):
            x = self.tail(x)

        return x


class VotingSingleWavMLP(WavMLP):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        level: int,
        voting_method: str = "hard",
    ):

        super().__init__(in_channels, hidden_size, out_channels, level, tail=False)

        self.vote = {"hard": voting.hard_voting, "soft": voting.soft_voting}[
            voting_method
        ]

    def forward(self, x):

        x = self.wavelet(x)
        # An MLP for each of the transformed levels
        channel_1 = self.channel_1_mlp(self.flatten_wavelet(x[0]))
        channel_2 = self.channel_2_mlp(self.flatten_wavelet(x[1]))
        channel_3 = self.channel_3_mlp(self.flatten_wavelet(x[2]))

        # Let them vote instead of just stacking them
        probabilities = [channel_1, channel_2, channel_3]
        x = self.vote(probabilities)
        return x
