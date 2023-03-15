import torch
import pywt
import kymatio


class WaveletLayer:
    def __init__(self, level: int, backend: str = "pywt") -> None:

        layers = {
            "pywt": lambda x: torch.Tensor(pywt.wavedec2(x, "db1")[level]),
            "kymatio": NotImplemented,
        }

        assert backend in layers.keys()
        self.layer = layers[backend]

    def __call__(self, x):
        return self.layer(x)
