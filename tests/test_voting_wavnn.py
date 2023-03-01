import sys

sys.path.append("..")

import pytest
import numpy as np
from wavNN.models import voting_wavMLP
import torch


def test_out_size_soft_voting():
    in_channels = 28
    out_channels = 10
    hidden_size = 4
    voting_method = "soft"
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


def test_out_size_hard_voting():
    in_channels = 28
    out_channels = 10
    hidden_size = 4
    voting_method = "hard"
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


if __name__ == "__main__":
    pytest.main()
