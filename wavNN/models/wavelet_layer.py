from typing import Union
import torch
import pywt
from kymatio.torch import Scattering2D


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
