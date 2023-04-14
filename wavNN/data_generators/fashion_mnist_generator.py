from wavNN.data_generators.data_generator import DataGenerator
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST


class FashionMNISTGenerator(DataGenerator):
    def __init__(self):
        dataset = FashionMNIST(
            root="wavNN/data/fashionmnist",
            download=True,
            train=True,
            transform=ToTensor(),
        )
        super().__init__(dataset=dataset)
