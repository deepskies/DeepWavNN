import numpy as np
from torch import nn
import torch
from wavNN.models.wavMLP import WavMLP
from wavNN.utils import voting


class VotingMultiWavMLP(nn.Module):
    def __init__(
        self, in_channels, hidden_sizes, out_channels, voting_method="soft", tail=False
    ):
        super().__init__()

        assert voting_method in ["hard", "soft"]
        self.voting_method = voting_method
        self.tail = tail

        def calc_possible_level(input_size):
            ddim, levels = input_size, []
            level = 0
            while ddim > 2:
                ddim = int(np.rint(ddim / 2))
                levels.append(level)
                level += 1
            levels = [level for level in levels if level != 0]
            return levels

        possible_levels = calc_possible_level(in_channels)
        hidden_sizes = (
            hidden_sizes
            if type(hidden_sizes) == list
            else [hidden_sizes for _ in range(len(possible_levels))]
        )

        if len(possible_levels) != len(hidden_sizes):
            Warning(
                "Passed hidden layer sizes and number of levels for the input does match"
            )

        self.models = [
            WavMLP(
                in_channels=in_channels,
                hidden_size=hidden_size,
                out_channels=out_channels,
                level=level,
                tail=False,
            )
            for hidden_size, level in zip(hidden_sizes, possible_levels)
        ]
        self.tail_output = nn.Linear(
            in_features=out_channels, out_features=out_channels
        )

    def vote(self, probabilities):
        return {"hard": voting.hard_voting, "soft": voting.soft_voting}[
            self.voting_method
        ](probabilities)

    def forward(self, x):
        outputs = [model.forward(x) for model in self.models]
        x = self.vote(outputs)

        if self.tail:
            x = self.tail_output(x)
            x = nn.Softmax(dim=0)(x)

        return x


class TiedWavMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Slightly more complicated WavMLP
        # Instead to just voting
        # It ties all the outputs of the
        # multiple networks into a dense output

    def forward(self):
        pass
