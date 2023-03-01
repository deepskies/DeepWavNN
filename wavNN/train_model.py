"""
Basic training loop for the selected model
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
import torch


class TrainingLoop:
    def __init__(
        self,
        model_class,
        model_params,
        data_class,
        data_params,
        optimizer_class,
        optimizer_config,
        **training_configs,
    ):

        self.model = model_class(**model_params)
        self.data_loader = data_class()(**data_params)

        # Todo lr and momentum params
        self.loss = training_configs["loss"]()
        self.epochs = training_configs["epochs"]

        if "tied_model" in training_configs:
            from itertools import chain

            self.optimizer = optimizer_class(
                chain(*[model.parameters() for model in self.model.models]),
                lr=optimizer_config["lr"],
                momentum=optimizer_config["momentum"],
            )
        else:
            self.optimizer = optimizer_config(
                self.model.parameters(), **optimizer_config
            )

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
        }

    def train_one_epoch(self):
        self.model.train(True)
        running_loss = 0
        running_accuracy = 0

        for i, batch in enumerate(self.data_loader["training"]):
            data_input, label = batch
            self.optimizer.zero_grad()

            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            _, predicted_class = torch.max(model_prediction, 1)
            running_accuracy += (label == predicted_class).sum().item() / label.size(0)
            loss.backward()
            self.optimizer.step()
            running_loss += loss

        loss = running_loss / (i + 1)
        accuracy = running_accuracy / (i + 1)

        return loss.item(), accuracy

    def train(self, plot=False):
        for epoch in range(self.epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)

        if plot:
            self.plot_history()

    def validate(self):
        self.model.train(False)
        running_loss = 0
        running_accuracy = 0

        for i, batch in enumerate(self.data_loader["validation"]):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            _, predicted_class = torch.max(model_prediction, 1)

            running_accuracy += (label == predicted_class).sum().item() / label.size(0)
            running_loss += loss

        loss = running_loss / (i + 1)
        accuracy = running_accuracy / (i + 1)
        return loss.item(), accuracy

    def test(self):
        self.model.train(False)
        running_loss = 0

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

        plt.scatter(epochs, train, label="Train Loss", alpha=0.8, marker="o")
        plt.scatter(epochs, val, label="Validation Loss", alpha=0.8, marker="x")
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
        self.test()

        if save:
            self.save(save_path)
