"""
Modifcaiton of the voting mlp that instead of using a normal voting algorithm
Pools the results after reshaping the results from each hidden layer

> need to make a good diagram for this.
"""

import torch
import math
import numpy as np

from wavpool.utils.levels import Levels
from wavpool.models.wavelet_layer import MicroWav


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
        hidden_layer_scaling: bool = False,
    ) -> None:
        super().__init__()

        possible_levels = Levels.calc_possible_levels(in_channels)
        possible_levels = [level for level in possible_levels if level != 0]

        self.n_levels = len(possible_levels)

        hidden_sizes = (
            hidden_size
            if type(hidden_size) == list
            else {
                True: [int(hidden_size / (i + 1)) for i in range(self.n_levels)],
                False: [int(hidden_size) for _ in range(self.n_levels)],
            }[hidden_layer_scaling]
        )
        if hidden_layer_scaling:
            hidden_sizes.reverse()

        self.max_hidden = max(hidden_sizes)
        self.models = torch.nn.ModuleList()
        for level, hidden_size in zip(possible_levels, hidden_sizes):
            self.models.append(
                MicroWav(
                    level=int(level),
                    in_channels=in_channels,
                    hidden_size=int(hidden_size),
                )
            )

        if hidden_pooling is not None:
            assert level_pooling is not None
            pooling_kernel = (int(hidden_pooling), 1, int(level_pooling))
        else:
            pooling = int(pooling_size)
            pooling_kernel = (pooling, pooling, pooling)

        self.pool = torch.nn.ModuleDict(
            {
                "average": torch.nn.AvgPool3d(kernel_size=pooling_kernel),
                "max": torch.nn.MaxPool3d(kernel_size=pooling_kernel),
            }
        )[pooling_mode]

        pool_out_shape = int(
            math.prod(self.pool(torch.rand(1, self.max_hidden, 3, self.n_levels)).shape)
        )

        self.output = torch.nn.Linear(pool_out_shape, out_features=out_channels)

    def forward(self, x):
        level_outputs = [model.forward(x) for model in self.models]
        x = [
            torch.nn.functional.pad(
                x, pad=(0, 0, self.max_hidden - x.shape[1], 0), mode="constant", value=0
            )
            for x in level_outputs
        ]
        x = torch.stack(x, dim=-1)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.output(x)
