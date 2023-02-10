import pytest
import numpy as np
import torch
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

    assert wavmlp(torch.rand(*(1, 28, 28))).data.shape[0] == out_channels


def test_mlp_linear_input_size():
    # Todo better test than "don't break"
    for level, expected_dim in zip([1, 2, 3, 4], [2, 4, 7, 14]):
        hidden_size = 14
        WavMLP(in_channels=28, out_channels=1, level=level, hidden_size=hidden_size)(
            torch.rand(*(1, 28, 28))
        )


def test_mlp_linear_multibatch():
    batch_size = 10

    wavnn = WavMLP(28, 14, 1, level=3, tail=False)
    fake_data = torch.rand(batch_size, 28, 28)
    forward = wavnn(fake_data)

    assert len(forward) == batch_size


def test_non_allowed_levels():
    # random selection of Too Big sizes
    # and one too small

    level = 0
    with pytest.raises(AssertionError):
        WavMLP(in_channels=28, out_channels=1, hidden_size=6, level=level)(
            torch.rand(*(1, 28, 28))
        )

    for _ in range(5):
        random_level = np.random.randint(7, 100)
        with pytest.raises(AssertionError):
            WavMLP(in_channels=28, out_channels=1, hidden_size=6, level=random_level)(
                torch.rand(*(1, 28, 28))
            )


def test_mlp_forward_pass_tail():
    wavnn = WavMLP(28, 14, 4, level=3, tail=True)
    fake_data = torch.rand(1, 28, 28)
    forward = wavnn(fake_data)

    assert len(forward) == 1


def test_mlp_forward_pass_no_tail():
    wavnn = WavMLP(28, 14, 1, level=3, tail=False)
    fake_data = torch.rand(1, 28, 28)
    forward = wavnn(fake_data)

    assert len(forward[0]) == 1
