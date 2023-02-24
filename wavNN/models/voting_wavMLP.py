import numpy as np
from torch import nn
from wavNN.models.wavMLP import WavMLP
from wavNN.utils import voting


class VotingMultiWavMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, out_size, voting_method="soft"):
        super().__init__()

        assert voting_method in ["hard", "soft"]
        self.voting_method = voting_method

        def calc_possible_level(input_size):
            ddim, levels = input_size, []
            while ddim > 2:
                ddim = int(np.rint(ddim / 2))
                levels.append(ddim)
            return levels

        possible_levels = calc_possible_level(input_size)
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
                in_channels=input_size,
                hidden_size=hidden_size,
                out_channels=out_size,
                level=level,
                tail=False,
            )
            for hidden_size, level in zip(possible_levels, hidden_sizes)
        ]

    def vote(self, probabilities):
        return {"hard": voting.hard_voting, "soft": voting.soft_voting}[
            self.voting_method
        ](probabilities)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return self.vote(outputs)


class TiedWavMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Slightly more complicated WavMLP
        # Instead to just voting
        # It ties all the outputs of the
        # multiple networks into a dense output

    def forward(self):
        pass
