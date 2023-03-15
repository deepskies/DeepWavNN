# Formula to calculate the levels and their dimensions
import numpy as np
import pywt
import kymatio


class Levels:
    @staticmethod
    def calc_possible_levels(in_channels, backend="pywt"):
        if backend == "pywt":
            _, levels = Levels._input_characterics(in_channels=in_channels)
        elif backend == "kymatio":
            _, levels = Levels._ky_input_characterics(in_channels=in_channels)
        else:
            raise NotImplementedError
        return levels

    @staticmethod
    def find_output_size(level, in_channels, backend="pywt"):
        sizes, levels = Levels._input_characterics(in_channels=in_channels)
        if backend == "pywt":
            sizes, levels = Levels._input_characterics(in_channels=in_channels)
        elif backend == "kymatio":
            sizes, levels = Levels._ky_input_characterics(in_channels=in_channels)
        else:
            raise NotImplementedError

        assert len(levels) >= level
        return sizes[levels[level]]

    @staticmethod
    def _input_characterics(in_channels, wavelet="haar"):
        transform = np.random.rand(in_channels, in_channels)
        transform = pywt.wavedec2(transform, wavelet)
        transform_sizes = [np.array(x).shape[1] for x in transform]
        levels = [i for i in range(len(transform_sizes))]
        return transform_sizes, levels

    @staticmethod
    def _ky_input_characterics(in_channels):
        raise NotImplementedError
