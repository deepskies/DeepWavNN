# Formula to calculate the levels and their dimensions
import numpy as np


def calc_possible_levels(in_channels):
    _, levels = _input_characterics(in_channels=in_channels)
    return levels


def find_output_size(level, in_channels):
    sizes, levels = _input_characterics(in_channels=in_channels)
    assert len(levels) >= level
    return sizes[levels[level]]


def _input_characterics(in_channels):
    current_dimension, transform_sizes = in_channels, []
    while current_dimension > 2:
        current_dimension = int(np.rint(current_dimension / 2))
        transform_sizes.append(current_dimension)
        if current_dimension == 2:
            transform_sizes.append(0)
    transform_sizes.reverse()
    levels = [i for i in range(len(transform_sizes))]
    return transform_sizes, levels
