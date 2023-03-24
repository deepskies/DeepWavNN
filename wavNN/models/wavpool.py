"""
Modifcaiton of the voting mlp that instead of using a normal voting algorithm
Pools the results after reshaping the results from each hidden layer

> need to make a good diagram for this.
"""

import torch
import math
import numpy as np

from wavNN.utils.levels import Levels
from wavNN.models.wavelet_layer import MiniWave


class WavPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        pooling_size: int = None,  # type: ignore
        pooling_mode: str = "average",
        hidden_pooling: int = None,  # type: ignore
        level_pooling: int = None,  # type: ignore
    ) -> None:
        super().__init__()

        possible_levels = Levels.calc_possible_levels(in_channels)
        possible_levels = [level for level in possible_levels if level != 0]

        self.n_levels = len(possible_levels)
        self.input = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.models = torch.nn.ModuleList()
        for level in possible_levels:
            self.models.append(
                MiniWave(level=level, in_channels=in_channels, hidden_size=hidden_size)
            )

        if hidden_pooling is not None:
            assert level_pooling is not None
            pooling_kernel = (hidden_pooling, 1, level_pooling)
        else:
            pooling_kernel = (pooling_size, pooling_size, pooling_size)

        self.pool = torch.nn.ModuleDict(
            {
                "average": torch.nn.AvgPool3d(kernel_size=pooling_kernel),
                "max": torch.nn.MaxPool3d(kernel_size=pooling_kernel),
            }
        )[pooling_mode]

        pool_out_shape = int(
            math.prod(self.pool(torch.rand(1, hidden_size, 3, self.n_levels)).shape)
        )

        self.output = torch.nn.Linear(pool_out_shape, out_features=out_channels)

    def forward(self, x):
        x = torch.stack([model.forward(x) for model in self.models], dim=-1)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.output(x)
