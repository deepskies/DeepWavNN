import pytest
import numpy as np
from wavNN.models.wavMLP import WavMLP, TiedWavMLP, VotingWavMLP


def test_mlp_out_size():
    in_channels = 28
    out_channels = 1
    level = 1
    hidden_size = 4
    wavmlp = WavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        hidden_size=hidden_size,
    )

    out_layer = wavmlp[-1].shape()[-1]
    assert out_layer == out_channels


def test_mlp_hidden_size():
    in_channels = 1
    out_channels = 1
    level = 0
    hidden_size = 4
    wavmlp = WavMLP(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        hidden_size=hidden_size,
    )

    hidden_layer = wavmlp[1].shape()[-1]
    assert hidden_layer == hidden_size


def test_mlp_linear_input_size():
    for level, expected_dim in zip([1, 2, 3, 4], [2, 4, 7, 14]):
        hidden_size = 14
        wavmlp = WavMLP(
            in_channels=28, out_channels=1, level=level, hidden_size=hidden_size
        )

        assert wavmlp[0].shape()[0] == expected_dim


def test_non_allowed_levels():
    # random selection of Too Big sizes
    # and one too small

    level = 0
    assert pytest.raises(
        AssertionError,
        WavMLP(in_channels=28, out_channels=1, hidden_size=6, level=level),
    )

    for _ in range(5):
        random_level = np.random.randint(7, 100)
        assert pytest.raises(
            AssertionError,
            WavMLP(in_channels=28, out_channels=1, hidden_size=6, level=random_level),
        )


def test_mlp_forward_pass_tail():
    wavnn = WavMLP(28, 14, 4, level=3, tail=True)
    fake_data = np.random.uniform(0, 1.0, size=(28, 28))
    forward = wavnn(fake_data)

    assert type(forward) == int
    assert forward in [0, 1, 2, 3]


def test_mlp_forward_pass_no_tail():
    wavnn = WavMLP(28, 14, 1, level=3, tail=False)
    fake_data = np.random.uniform(0, 1.0, size=(28, 28))
    forward = wavnn(fake_data)

    assert type(forward) == float
