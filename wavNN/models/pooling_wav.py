"""
Modifcaiton of the voting mlp that instead of using a normal voting algorithm
Pools the results after reshaping the results from each hidden layer

> need to make a good diagram for this.
"""

import torch
import math

from wavNN.utils.levels import Levels
from wavNN.models.wavelet_layer import MiniWave


class WavePool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        pooling_size: int,
        pooling_mode: str = "average",
        tail: bool = False,
    ) -> None:
        super().__init__()

        self.tail = tail

        possible_levels = Levels.calc_possible_levels(in_channels)
        possible_levels = [level for level in possible_levels if level != 0]

        self.n_levels = len(possible_levels)
        self.input = torch.nn.Flatten(start_dim=1, end_dim=-1)

        self.models = torch.nn.ModuleList()
        for level in possible_levels:
            self.models.append(
                MiniWave(level=level, in_channels=in_channels, hidden_size=hidden_size)
            )

        self.pool = torch.nn.ModuleDict(
            {
                "average": torch.nn.AvgPool2d(kernel_size=pooling_size),
                "max": torch.nn.MaxPool2d(kernel_size=pooling_size),
            }
        )[pooling_mode]

        expected_pool_shape = [
            math.floor(((in_size - pooling_size) / pooling_size) + 1)
            for in_size in [hidden_size, self.n_levels, 3]
        ]
        pool_out_shape = int(math.prod(expected_pool_shape))
        self.output = torch.nn.Linear(pool_out_shape, out_features=out_channels)

    def forward(self, x):
        x = torch.stack([model.forward(x) for model in self.models], dim=-1)
        x = self.pool(x)
        x = torch.flatten(x)
        return self.output(x)
