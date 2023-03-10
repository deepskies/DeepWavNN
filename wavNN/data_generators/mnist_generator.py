from wavNN.data_generators.data_generator import DataGenerator
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class NMISTGenerator(DataGenerator):
    def __init__(self):
        dataset = MNIST(
            root="./data/mnist", download=True, train=True, transform=ToTensor()
        )
        super().__init__(dataset=dataset)
