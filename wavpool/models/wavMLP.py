import torch
import torch.nn as nn

from wavpool.utils.levels import Levels
from wavpool.models.wavelet_layer import MicroWav


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

        # Wavelet transform of input x at a level as defined by the user
        self.wav = MicroWav(
            level=int(level), in_channels=in_channels, hidden_size=int(hidden_size)
        )
        # Flatten for when these are stacked
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        # Output of those tied 3 channel layers, and the flattened concat of those
        self.output = nn.Linear(int(hidden_size) * 3, out_channels)

        # Activation for a classifier
        if tail:
            self.tail = nn.Softmax(dim=0)

    def forward(self, x):
        # forward pass through the network

        x = self.wav(x)
        # Flatten for the output dense
        x = self.flatten(x)
        x = self.output(x)

        if hasattr(self, "tail"):
            x = self.tail(x)

        return x
