"""
Generate parameters for a network using a guassian processor optimizer
(Using https://github.com/fmfn/BayesianOptimization)
Lightweight wrapper on the train_model

"""
import json
from bayes_opt import BayesianOptimization
import torch
import os
import numpy as np
import math

from wavNN.train_model import TrainingLoop
from wavNN.models.wavMLP import WavMLP
from wavNN.models.wavpool import WavPool

from wavNN.models.vanillaMLP import VanillaMLP, BananaSplitMLP

from wavNN.data_generators.mnist_generator import NMISTGenerator


def run_vanilla(split=False):
    def training_function_vanilla(
        hidden_size, loss_id, optimizer_class_id, optimizer_lr, optimizer_momentum_id
    ):

        model_class = VanillaMLP if not split else BananaSplitMLP

        optimizer_class = (
            torch.optim.SGD if optimizer_class_id < 0.5 else torch.optim.Adam
        )
        loss = torch.nn.CrossEntropyLoss if loss_id < 0.5 else torch.nn.MultiMarginLoss

        optimizer_config = {"lr": optimizer_lr}
        if optimizer_class == torch.optim.SGD:
            optimizer_config["momentum"] = optimizer_momentum_id < 0.5

        model_params = {
            "in_channels": 28,
            "hidden_size": math.ceil(hidden_size),
            "out_channels": 10,
        }

        data_params = {"sample_size": [4000, 2000, 2000], "split": True}

        training = TrainingLoop(
            model_class=model_class,
            model_params=model_params,
            data_class=NMISTGenerator,
            data_params=data_params,
            loss=loss,
            epochs=50,
            optimizer_class=optimizer_class,
            optimizer_config=optimizer_config,
        )
        training()
        history = training.history
        accuracy = np.max(np.asarray(history["val_accuracy"]))

        return accuracy

    parameter_space_vanilla = {
        "hidden_size": (10, 750),
        "loss_id": (0, 1),
        "optimizer_class_id": (0, 1),
        "optimizer_lr": (0.000001, 0.1),
        "optimizer_momentum_id": (0, 1),
    }

    optimizer_vanilla = BayesianOptimization(
        f=training_function_vanilla,
        pbounds=parameter_space_vanilla,
        verbose=0,
        random_state=1,
    )

    optimizer_vanilla.maximize(init_points=5, n_iter=30)

    history = optimizer_vanilla.res

    outpath = (
        "results/optimization/vanilla_baysianopt.json"
        if not split
        else "results/optimization/vanilla_split_baysianopt.json"
    )
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    with open(outpath, "w") as f:
        json.dump(history, fp=f, default=str)


def run_wavpool():
    def training_function_wavpool(
        hidden_size,
        hidden_pooling_size,
        level_pooling_size,
        pooling_mode_id,
        loss_id,
        optimizer_class_id,
        optimizer_lr,
        optimizer_momentum_id,
    ):

        optimizer_class = (
            torch.optim.SGD if optimizer_class_id < 0.5 else torch.optim.Adam
        )
        loss = torch.nn.CrossEntropyLoss if loss_id < 0.5 else torch.nn.MultiMarginLoss

        optimizer_config = {"lr": optimizer_lr}
        if optimizer_class == torch.optim.SGD:
            optimizer_config["momentum"] = optimizer_momentum_id < 0.5

        pooling_mode = "average" if pooling_mode_id < 0.5 else "max"

        model_params = {
            "in_channels": 28,
            "hidden_size": math.ceil(hidden_size),
            "hidden_pooling": math.ceil(hidden_size / hidden_pooling_size),
            "level_pooling": math.ceil(level_pooling_size),
            "out_channels": 10,
            "pooling_mode": pooling_mode,
        }

        data_params = {"sample_size": [4000, 2000, 2000], "split": True}
        try:
            training = TrainingLoop(
                model_class=WavPool,
                model_params=model_params,
                data_class=NMISTGenerator,
                data_params=data_params,
                loss=loss,
                epochs=50,
                optimizer_class=optimizer_class,
                optimizer_config=optimizer_config,
            )
            training()
            history = training.history
            accuracy = np.max(np.asarray(history["val_accuracy"]))

        except RuntimeError:
            accuracy = 0
        return accuracy

    parameter_space_wav = {
        "hidden_size": (10, 750),
        "loss_id": (0, 1),
        "optimizer_class_id": (0, 1),
        "optimizer_lr": (0.000001, 0.1),
        "optimizer_momentum_id": (0, 1),
        "hidden_pooling_size": (0.1, 4),
        "level_pooling_size": (1, 3),
        "pooling_mode_id": (0, 1),
    }

    optimizer_wav = BayesianOptimization(
        f=training_function_wavpool,
        pbounds=parameter_space_wav,
        verbose=1,
        random_state=1,
    )

    optimizer_wav.maximize(init_points=5, n_iter=30)

    history = optimizer_wav.res
    outpath = "results/optimization/wavpool_baysianopt.json"
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    with open(outpath, "w") as f:
        json.dump(history, fp=f, default=str)


def run_wavmlp():
    def training_function_wavmlp(
        hidden_size,
        loss_id,
        level,
        optimizer_class_id,
        optimizer_lr,
        optimizer_momentum_id,
    ):

        optimizer_class = (
            torch.optim.SGD if optimizer_class_id < 0.5 else torch.optim.Adam
        )
        loss = torch.nn.CrossEntropyLoss if loss_id < 0.5 else torch.nn.MultiMarginLoss

        optimizer_config = {"lr": optimizer_lr}
        if optimizer_class == torch.optim.SGD:
            optimizer_config["momentum"] = optimizer_momentum_id < 0.5

        model_params = {
            "in_channels": 28,
            "hidden_size": math.ceil(hidden_size),
            "level": math.ceil(level),
            "out_channels": 10,
        }

        data_params = {"sample_size": [4000, 2000, 2000], "split": True}

        training = TrainingLoop(
            model_class=WavPool,
            model_params=model_params,
            data_class=NMISTGenerator,
            data_params=data_params,
            loss=loss,
            epochs=50,
            optimizer_class=optimizer_class,
            optimizer_config=optimizer_config,
        )
        training()
        history = training.history
        accuracy = np.max(np.asarray(history["val_accuracy"]))
        return accuracy

    parameter_space_wav = {
        "hidden_size": (10, 750),
        "loss_id": (0, 1),
        "optimizer_class_id": (0, 1),
        "optimizer_lr": (0.000001, 0.1),
        "level": (0.1, 3),
        "optimizer_momentum_id": (0, 1),
    }

    optimizer_wav = BayesianOptimization(
        f=training_function_wavmlp,
        pbounds=parameter_space_wav,
        verbose=1,
        random_state=1,
    )

    optimizer_wav.maximize(init_points=5, n_iter=30)

    history = optimizer_wav.res
    outpath = "results/optimization/wavmlp_baysianopt.json"
    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    with open(outpath, "w") as f:
        json.dump(history, fp=f, default=str)


if __name__ == "__main__":
    # run_vanilla(split=True)
    # run_wavmlp()
    run_wavpool()
