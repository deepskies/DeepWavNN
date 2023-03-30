"""
Basic training loop for the selected model
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import tqdm

from torcheval.metrics.functional import multiclass_f1_score


class TrainingLoop:
    def __init__(
        self,
        model_class,
        model_params,
        data_class,
        data_params,
        optimizer_class,
        optimizer_config,
        loss,
        **training_configs,
    ):

        self.model = model_class(**model_params)
        self.data_loader = data_class()(**data_params)

        # Todo lr and momentum params
        self.loss = loss()

        self.epochs = (
            300 if "epochs" not in training_configs else training_configs["epochs"]
        )
        self.early_stopping_tolerence = (
            3
            if "early_stopping_tolerence" not in training_configs
            else training_configs["early_stopping_tolerence"]
        )

        self.extra_metrics = (
            [self.f1, self.accuracy]
            if "extra_metrics" not in training_configs
            else training_configs["extra_metrics"]
        )

        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_config)

        self.history = {
            "train_loss": [],
            "val_loss": [],
        }
        for metric in self.extra_metrics:
            self.history[f"train_{metric.__name__}"] = []
            self.history[f"val_{metric.__name__}"] = []

        self.early_stopping_critreon = 0
        self.current_epoch = 0

    def f1(self, prediction, label):
        return multiclass_f1_score(target=label, input=prediction).detach().numpy()

    def accuracy(self, prediction, label):
        _, predicted_class = torch.max(prediction, 1)
        return (label == predicted_class).sum().item() / label.size(0)

    def train_one_epoch(self):
        self.model.train(True)
        running_loss = 0
        running_metrics = [0 for _ in range(len(self.extra_metrics))]
        i = 0

        for i, batch in enumerate(
            tqdm.tqdm(self.data_loader["training"], desc="Training....")
        ):
            data_input, label = batch
            self.optimizer.zero_grad()

            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            for index in range(len(self.extra_metrics)):
                running_metrics[index] += self.extra_metrics[index](
                    model_prediction, label
                )

            loss.backward()
            self.optimizer.step()
            running_loss += loss

        loss = running_loss / (i + 1)
        extra_metrics = [metric / (i + 1) for metric in running_metrics]

        return loss, extra_metrics

    def is_still_training(self):
        # Monitor val loss
        if self.current_epoch >= 5:
            if self.history["val_loss"][-2] < self.history["val_loss"][-1]:
                self.early_stopping_critreon += 1
            if self.early_stopping_tolerence <= self.early_stopping_critreon:
                return False

        self.current_epoch += 1
        if self.current_epoch >= self.epochs:
            return False

        return True

    def train(self, plot=False):
        not_stopping = True
        while not_stopping:
            train_loss, train_metrics = self.train_one_epoch()
            val_loss, val_metrics = self.validate()

            self.history["train_loss"].append(train_loss.detach().numpy())  # type: ignore
            self.history["val_loss"].append(val_loss.detach().numpy())  # type: ignore

            for metric_index, metric in enumerate(self.extra_metrics):
                self.history[f"train_{metric.__name__}"].append(
                    train_metrics[metric_index]
                )
                self.history[f"val_{metric.__name__}"].append(val_metrics[metric_index])

            not_stopping = self.is_still_training()

        if plot:
            self.plot_history()

    def validate(self):
        self.model.train(False)

        running_loss = 0
        running_metrics = [0 for _ in range(len(self.extra_metrics))]
        i = 0
        for i, batch in enumerate(self.data_loader["validation"]):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            for index in range(len(self.extra_metrics)):
                running_metrics[index] += self.extra_metrics[index](
                    model_prediction, label
                )

            running_loss += loss

        loss = running_loss / (i + 1)

        extra_metrics = [metric / (i + 1) for metric in running_metrics]
        return loss, extra_metrics

    def test(self):
        self.model.train(False)
        running_loss = 0
        i = 0
        for i, batch in enumerate(self.data_loader["test"]):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            running_loss += loss

        return running_loss / (i + 1)

    def plot_history(self, save_path=None):
        history = pd.DataFrame(self.history)
        train, val = history["train_loss"], history["val_loss"]

        epochs = range(len(history))

        plt.scatter(epochs, train, label="Train Loss", alpha=0.8, marker="o")  # type: ignore
        plt.scatter(epochs, val, label="Validation Loss", alpha=0.8, marker="x")  # type: ignore
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        if save_path is not None:
            plt.savefig(f"{save_path}/history.png")

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = f"{save_path}/model.pt"
        torch.save(self.model.state_dict(), model_path)

        history_path = f"{save_path}/history.csv"
        pd.DataFrame(self.history).to_csv(history_path)

        self.plot_history(save_path)

    def __call__(self, save=False, save_path="./model_results/"):
        self.train()

        if save:
            self.save(save_path)


if __name__ == "__main__":
    import sys

    sys.path.append("..")

    from wavNN.models.wavpool import WavPool
    from wavNN.data_generators.fashion_mnist_generator import FashionMNISTGenerator

    model_params = {
        "in_channels": 28,
        "hidden_size": 256,
        "out_channels": 10,
        "pooling_size": 3,
        "pooling_mode": "average",
    }

    wavepool_history = {}
    num_tests = 3

    data_params = {"sample_size": [4000, 2000, 2000], "split": True}
    optimizer_config = {"lr": 0.1, "momentum": False}

    loop = TrainingLoop(
        model_class=WavPool,
        model_params=model_params,
        data_class=FashionMNISTGenerator,
        data_params=data_params,
        optimizer_class=torch.optim.SGD,
        optimizer_config=optimizer_config,
        loss=torch.nn.CrossEntropyLoss,
    )
    loop()
    history = loop.history
    pd.DataFrame(history).to_csv("wavpool_test.csv")
