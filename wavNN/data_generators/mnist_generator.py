from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np


class NMISTGenerator:
    def __init__(self):
        self.dataset = MNIST(
            root="./data/mnist", download=True, train=True, transform=ToTensor()
        )

    def __call__(self, *args, **kwargs):
        sample_size = kwargs["sample_size"]
        split = kwargs["split"]
        batch_size = 64 if "batch_size" not in kwargs else kwargs["batch_size"]
        shuffle = False if "shuffle" not in kwargs else kwargs["shuffle"]

        if split:
            if type(sample_size) == int:
                sample_size = [sample_size for _ in range(3)]

            assert sum(sample_size) <= len(self.dataset), (
                f""
                f"Too many requested samples, "
                f"decreases your sample size to less "
                f"than {len(self.dataset)}"
            )

            assert len(sample_size) == 3, (
                "The sample size of validation "
                "and test must be individually specified"
            )

            # this is quick and dirty. Ideally I'd be shuffling when i load in.
            # But I'm chronically lazy and this is what I'm doing
            # I can change it later.

            samples = np.cumsum(sample_size)
            training_data = Subset(self.dataset, list(range(0, samples[0])))
            val_data = Subset(self.dataset, list(range(samples[0], samples[1])))
            test_data = Subset(self.dataset, list(range(samples[1], samples[2])))

            training = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
            validation = DataLoader(val_data, batch_size=batch_size, shuffle=shuffle)
            test = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        else:
            assert sample_size <= len(self.dataset), (
                f"Too many requested samples, decreases your"
                f" sample size to less than {len(self.dataset)}"
            )

            training_data = Subset(self.dataset, list(range(0, sample_size)))

            training = DataLoader(training_data, batch_size=batch_size, shuffle=shuffle)
            validation = None
            test = None

        return {"training": training, "validation": validation, "test": test}
