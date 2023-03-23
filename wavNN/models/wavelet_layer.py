from typing import Union
import torch
import pywt
from kymatio.torch import Scattering2D
from wavNN.utils.levels import Levels


class WaveletLayer:
    def __init__(
        self, level: int, input_size: Union[int, None] = None, backend: str = "pywt"
    ) -> None:

        layers = {
            "pywt": lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level]),
            "kymatio": lambda x: self.kymatio_layer(level=level, input_size=input_size)(
                x
            )[level],
        }

        assert backend in layers.keys()
        self.layer = layers[backend]

    def kymatio_layer(self, level, input_size):
        # Ref: This code
        # https://www.kymat.io/gallery_2d/plot_invert_scattering_torch.html#sphx-glr-gallery-2d-plot-invert-scattering-torch-py

        scattering = Scattering2D(J=2, shape=(input_size, input_size), max_order=level)
        return scattering

    def __call__(self, x):
        return self.layer(x)


class MiniWave(torch.nn.Module):
    def __init__(self, level, in_channels, hidden_size) -> None:
        super().__init__()
        self.wavelet = WaveletLayer(level=level)
        wav_in_channels = Levels.find_output_size(level, in_channels)

        self.flatten_wavelet = torch.nn.Flatten(start_dim=1, end_dim=-1)
        # Channels for each of the 3 channels of the wavelet (Not including the downscaled original
        self.channel_1_mlp = torch.nn.Linear(wav_in_channels**2, hidden_size)
        self.channel_2_mlp = torch.nn.Linear(wav_in_channels**2, hidden_size)
        self.channel_3_mlp = torch.nn.Linear(wav_in_channels**2, hidden_size)

    def forward(self, x):
        x = self.wavelet(x)
        # An MLP for each of the transformed levels
        channel_1 = self.channel_1_mlp(self.flatten_wavelet(x[0]))
        channel_2 = self.channel_2_mlp(self.flatten_wavelet(x[1]))
        channel_3 = self.channel_3_mlp(self.flatten_wavelet(x[2]))
        # stack the outputs
        concat = torch.stack([channel_1, channel_2, channel_3], dim=-1)
        return concat
