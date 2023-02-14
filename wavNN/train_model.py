"""
Basic training loop for the selected model
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import torch


class TrainingLoop:
    def __init__(
        self, model_class, model_params, data_class, data_params, **training_configs
    ):

        self.model = model_class(**model_params)
        self.data_loader = data_class()(**data_params)

        # Todo lr and momentum params
        self.loss = training_configs["loss"]()
        self.optimizer = training_configs["optimizer"](
            self.model.parameters(), lr=0.001, momentum=0.9
        )
        self.epochs = training_configs["epochs"]

        self.history = {}

    def train_one_epoch(self):
        self.model.train(True)
        running_loss = 0

        for i, batch in enumerate(self.data_loader["train"]):
            data_input, label = batch
            self.optimizer.zero_grad()

            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            loss.backwards()
            self.optimizer.step()
            running_loss += loss

        loss = running_loss / (i + 1)
        return loss

    def train(self, plot=False):
        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            self.history[epoch] = {"train_loss": train_loss, "val_loss": val_loss}

        if plot:
            self.plot_history()

    def validate(self):
        self.model.train(False)
        running_loss = 0

        for i, batch in enumerate(self.data_loader["validate"]):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            running_loss += loss

        loss = running_loss / (i + 1)
        return loss

    def test(self):
        self.model.train(False)
        running_loss = 0

        for i, batch in enumerate(self.data_loader["validate"]):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            running_loss += loss

        return running_loss / (i + 1)

    def plot_history(self, save_path=None):
        history = pd.DataFrame(self.history)
        train, val = history["training_loss"], history["val_loss"]

        epochs = range(len(history))

        plt.scatter(epochs, train, label="Train Loss", alpha=0.8, marker="o")
        plt.scatter(epochs, val, label="Validation Loss", alpha=0.8, marker="x")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()
        if save_path is not None:
            plt.savefig(save_path)

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = f"{save_path}/model"
        torch.save(self.model.state_dict(), model_path)

        history_path = f"{save_path}/history.csv"
        pd.DataFrame(self.history).to_csv(history_path)

    def __call__(self, save=False, save_path="./model_results/"):
        self.train()
        self.test()

        if save:
            self.save(save_path)


if __name__ == "__main__":
    from models.wavMLP import *
    from data.nmist_generator import *

    model_params = {}

    data_params = {
        "sample_size": [2000, 1000, 1000],
    }

    training = TrainingLoop(
        model_class=WavMLP,
        model_params=model_params,
        data_class=NMISTGenerator,
        data_params=data_params,
        optimizer=torch.optim.SGD,
        loss=torch.nn.CrossEntropyLoss,
        epochs=20,
    )

    training()
    training.save("../results/test/")
