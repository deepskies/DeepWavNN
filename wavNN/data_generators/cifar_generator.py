from wavNN.data_generators.data_generator import DataGenerator
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10


class CIFARGenerator(DataGenerator):
    def __init__(self):
        dataset = CIFAR10(
            root="WavNN/data/cifar10", download=True, train=True, transform=ToTensor()
        )
        super().__init__(dataset=dataset)
