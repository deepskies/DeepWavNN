from wavNN.data_generators.data_generator import DataGenerator
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFARGenerator(DataGenerator):
    def __init__(self):
        grayscale_transforms = transforms.Compose(
            [transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
        )

        dataset = CIFAR10(
            root="wavNN/data/cifar10",
            download=True,
            train=True,
            transform=grayscale_transforms,
        )
        super().__init__(dataset=dataset)
