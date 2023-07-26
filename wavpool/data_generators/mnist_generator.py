from wavpool.data_generators.data_generator import DataGenerator
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST


class MNISTGenerator(DataGenerator):
    def __init__(self):
        dataset = MNIST(
            root="wavNN/data/mnist", download=True, train=True, transform=ToTensor()
        )
        super().__init__(dataset=dataset)
