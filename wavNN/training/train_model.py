"""
Basic training loop for the selected model
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt

from wavNN.training.training_metrics import TrainingMetrics


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
            [TrainingMetrics.f1, TrainingMetrics.accuracy]
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
        self.n_classes = None

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
        loss, extra_metrics, _ = self.test_single_epoch(
            data_loader=self.data_loader["validation"]
        )
        return loss, extra_metrics

    def test_single_epoch(self, data_loader):
        self.model.train(False)
        running_loss = 0
        running_metrics = [0 for _ in range(len(self.extra_metrics))]
        i = 0

        predictions = torch.tensor([])
        for i, batch in enumerate(data_loader):
            data_input, label = batch
            model_prediction = self.model(data_input)
            loss = self.loss(model_prediction, label)
            for index in range(len(self.extra_metrics)):
                running_metrics[index] += self.extra_metrics[index](
                    model_prediction, label
                )

            predictions = torch.concat((predictions, model_prediction))
            running_loss += loss

        loss = running_loss / (i + 1)

        extra_metrics = [metric / (i + 1) for metric in running_metrics]

        return loss, extra_metrics, predictions

    def test(self, save_path):
        _, _, predictions = self.test_single_epoch(self.data_loader["test"])
        labels = torch.Tensor([])
        for batch in self.data_loader["test"]:
            _, label = batch
            labels = torch.concat((labels, label))

        self.plot_test_results(predictions, labels.detach().numpy(), save_path)

    def plot_test_results(self, predictions, labels, save_path=None):
        roc_curve = TrainingMetrics.auc_curve(predictions, labels)
        confusion = TrainingMetrics.confusion_matrix(predictions, labels)

        plt.plot(roc_curve[0], roc_curve[1])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC AUC Curve")
        plt.show()
        if save_path is not None:
            plt.savefig(f"{save_path}/roc.png")
        plt.close("all")

        plt.imshow(confusion)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        if save_path is not None:
            plt.savefig(f"{save_path}/confusion.png")
        else:
            plt.show()
        plt.close("all")

    def plot_history(self, save_path=None):
        history = pd.DataFrame(self.history)
        epochs = range(len(history))
        n_subplots = len(self.extra_metrics) + 1
        fig, subplots = plt.subplots(nrows=n_subplots, ncols=1)

        for metric_index, metrics in enumerate(self.extra_metrics):
            training = history[f"train_{metrics.__name__}"]
            val = history[f"val_{metrics.__name__}"]

            subplots[metric_index].plot(epochs, training, label="Train")
            subplots[metric_index].plot(epochs, val, label="Validation")
            subplots[metric_index].set_xticks([])
            subplots[metric_index].set_ylabel(metrics.__name__)
            subplots[metric_index].legend()

        metric_index = -1
        subplots[metric_index].plot(epochs, history["train_loss"], label="Train")
        subplots[metric_index].plot(epochs, history["val_loss"], label="Validation")
        subplots[metric_index].set_xticks(epochs)
        subplots[metric_index].set_ylabel("Loss")
        subplots[metric_index].legend()

        plt.xlabel("epoch")

        if save_path is not None:
            fig.savefig(f"{save_path}/history.png")
        else:
            plt.show()
        plt.close("all")

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = f"{save_path}/model.pt"
        torch.save(self.model.state_dict(), model_path)

        history_path = f"{save_path}/history.csv"
        pd.DataFrame(self.history).to_csv(history_path)

    def __call__(self, save=False, save_path="./model_results/"):
        self.train()

        if save:
            self.test(save_path)
            self.save(save_path)
            self.plot_history(save_path)
        else:
            self.test(save_path=None)
            self.plot_history(save_path=None)
