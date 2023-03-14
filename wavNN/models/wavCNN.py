import torch
import pywt
from wavNN.utils import levels


class WavCNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        level: int,
        kernel_size,
        stride_distance,
        out_channels: int = None,  # type: ignore
        tail=False,
        pool=False,
    ):
        """
        Make a Convolutional NN that performs a wavelet transform on the input
        'out_channel' is only used when the tail is used, otherwise it is calculated such that
        out_n = [
            (wav_in - kernel)/stride+1
        ]
        """

        self.level = level
        assert level != 0

        in_layer_size = levels.find_output_size(level, in_channels)
        self.wavelet = lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level])

        conv_out_channels = (in_channels - kernel_size) / stride_distance

        self.conv = torch.nn.Conv3d(
            in_layer_size,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride_distance,
        )
        self.relu = torch.nn.ReLU()

        if pool:
            pool_kernel_size = 0
            pool_stride = 0
            self.pool = torch.nn.MaxPool3d(
                kernel_size=pool_kernel_size, stride=pool_stride
            )

        if tail:
            assert out_channels is not None

            self.tail_flatten = torch.nn.Flatten()
            if not pool:
                linear_in_size = conv_out_channels**3
            else:
                linear_in_size = (
                    (conv_out_channels - pool_kernel_size) / pool_stride
                ) ** 3

            self.tail = torch.nn.Linear(linear_in_size, out_channels)
            self.tail_activation = torch.nn.Softmax()

    def forward(self, x):
        wavelet = self.wavelet(x)
        x = self.conv(wavelet)
        x = self.relu(x)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "tail"):
            x = self.tail_flatten(x)
            x = self.tail(x)
            x = self.tail_activation(x)

        return x
