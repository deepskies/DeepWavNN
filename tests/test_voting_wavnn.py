import pytest
import numpy as np
from WavPool.models import voting_wavMLP
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


def test_out_size_hard_voting_larger_network():
    in_channels = 28
    out_channels = 10
    hidden_size = 256
    voting_method = "hard"
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


def test_voting_multiple_size_networks():
    in_channels = 28
    out_channels = 10
    hidden_size = [64, 128, 256, 512]
    voting_method = "hard"
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


def test_voting_incorrect_num_hidden_sizes():
    in_channels = 28
    out_channels = 10
    hidden_size = [80, 256]
    voting_method = "hard"
    with pytest.raises(AssertionError):
        wavmlp = voting_wavMLP.VotingMultiWavMLP(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_sizes=hidden_size,
            voting_method=voting_method,
        )
        wavmlp(torch.rand(*(1, 28, 28)))


def test_out_size_soft_voting_tail():
    in_channels = 28
    out_channels = 10
    hidden_size = 4
    voting_method = "soft"
    tail = True
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
        tail=tail,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


def test_out_size_hard_voting_tail():
    in_channels = 28
    out_channels = 10
    hidden_size = 4
    voting_method = "hard"
    tail = True
    wavmlp = voting_wavMLP.VotingMultiWavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_sizes=hidden_size,
        voting_method=voting_method,
        tail=tail,
    )
    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[1] == out_channels


if __name__ == "__main__":
    pytest.main()
