import pytest
import numpy as np
import torch
from wavNN.models.wavCNN import WavCNN


def test_input_layer():
    in_channels = 280
    out_channels = 1
    level = -1
    kernel_size = 2
    stride_distance = 2
    tail = True
    network = WavCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        kernel_size=kernel_size,
        stride_distance=stride_distance,
        tail=tail,
    )
    assert (
        network(torch.rand(*(1, in_channels, in_channels))).data.shape[1]
        == out_channels
    )


def test_multibatch():
    in_channels = 280
    out_channels = 1
    level = 6
    kernel_size = 2
    stride_distance = 2
    tail = True
    network = WavCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        kernel_size=kernel_size,
        stride_distance=stride_distance,
        tail=tail,
    )
    assert network(torch.rand(*(64, in_channels, in_channels))).data.shape[0] == 64


def test_output_no_tail():

    in_channels = 280
    out_channels = 1
    level = 2
    kernel_size = 3
    stride_distance = 2
    tail = False
    network = WavCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        kernel_size=kernel_size,
        stride_distance=stride_distance,
        tail=tail,
    )
    assert (
        network(torch.rand(*(1, in_channels, in_channels))).data.shape[1]
        == out_channels
    )
    assert not hasattr(network, "tail")


def test_allowed_input_levels():
    kernel_sizes = [4, 8]
    allowed_levels = [3, 4, 5, 6, 7]
    for level, kernel_size in zip(allowed_levels, kernel_sizes):
        in_channels = 280
        out_channels = 1
        stride_distance = 2
        tail = True
        network = WavCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            level=level,
            kernel_size=kernel_size,
            stride_distance=stride_distance,
            tail=tail,
        )
        assert (
            network(torch.rand(*(1, in_channels, in_channels))).data.shape[1]
            == out_channels
        )


def test_allowed_strides():
    stride_sizes = [1, 2]
    allowed_levels = [3, 4, 5, 6]
    for level, kernel_size in zip(allowed_levels, stride_sizes):
        in_channels = 280
        out_channels = 1
        kernel_size = 2
        stride_distance = 2
        tail = True
        network = WavCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            level=level,
            kernel_size=kernel_size,
            stride_distance=stride_distance,
            tail=tail,
        )
        assert (
            network(torch.rand(*(1, in_channels, in_channels))).data.shape[1]
            == out_channels
        )


def test_forbidden_input_combos():
    kernel_sizes = [45969, 4304, 2312, 1265]
    allowed_levels = [1, 2, 3, 4]
    for level, kernel_size in zip(allowed_levels, kernel_sizes):
        in_channels = 280
        out_channels = 1
        stride_distance = 4
        tail = True
        with pytest.raises(AssertionError):
            WavCNN(
                in_channels=in_channels,
                out_channels=out_channels,
                level=level,
                kernel_size=kernel_size,
                stride_distance=stride_distance,
                tail=tail,
            )


def test_forbidden_strides():
    stride_sizes = [45969, 4304, 2312, 124322]
    allowed_levels = [4, 4, 5, 6]
    for level, stride_distance in zip(allowed_levels, stride_sizes):
        in_channels = 280
        out_channels = 1
        kernel_size = 2
        tail = True
        with pytest.raises(AssertionError):
            WavCNN(
                in_channels=in_channels,
                out_channels=out_channels,
                level=level,
                kernel_size=kernel_size,
                stride_distance=stride_distance,
                tail=tail,
            )


def test_pooling():
    stride_distance = 1
    level = -1
    in_channels = 280
    out_channels = 1
    kernel_size = 2
    tail = True
    pool = True
    wavcnn = WavCNN(
        in_channels=in_channels,
        out_channels=out_channels,
        level=level,
        kernel_size=kernel_size,
        stride_distance=stride_distance,
        tail=tail,
        pool=pool,
    )
    assert hasattr(wavcnn, "pool")
    assert wavcnn(torch.rand(*(64, in_channels, in_channels))).data.shape[0] == 64
