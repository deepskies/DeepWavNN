import torch
import pywt
from wavNN.utils.levels import Levels


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
        super().__init__()
        self.level = level
        assert level != 0

        in_layer_size = Levels.find_output_size(level, in_channels)
        self.wavelet = lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level])

        assert (
            kernel_size < in_layer_size
        ), f"Decrease the size of your kernel, kernel size {kernel_size}> input size {in_layer_size}"
        assert (
            stride_distance < in_layer_size
        ), f"Decrease the size of your stride, greater than input size {in_layer_size}"

        conv_out_channels = int(((in_channels - kernel_size) / stride_distance) + 1)

        self.conv = torch.nn.Conv2d(
            in_channels=in_layer_size,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            stride=stride_distance,
        )
        self.relu = torch.nn.ReLU()

        pool_kernel_size = int(conv_out_channels / 1.5)
        pool_stride = int(pool_kernel_size)

        if pool:
            self.pool = torch.nn.MaxPool3d(
                kernel_size=pool_kernel_size, stride=pool_stride
            )

        if tail:
            assert out_channels is not None

            self.tail_flatten = torch.nn.Flatten()
            if not pool:
                linear_in_size = conv_out_channels * 3
            else:
                linear_in_size = int(
                    ((conv_out_channels - pool_kernel_size) / pool_stride) ** 3
                )

            self.tail = torch.nn.Linear(linear_in_size, out_channels)
            self.tail_activation = torch.nn.Softmax()

    def forward(self, x):
        x = self.wavelet(x)
        x = torch.movedim(x, 0, -1)
        print(x.shape)
        print(self.conv)
        x = self.conv(x)
        x = self.relu(x)

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "tail"):
            print(x.shape)
            x = self.tail_flatten(x)
            x = self.tail(x)
            x = self.tail_activation(x)

        return x
