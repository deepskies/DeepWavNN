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
import pandas as pd

from wavNN.training.train_model import TrainingLoop
from wavNN.models.wavMLP import WavMLP
from wavNN.models.wavpool import WavPool

from wavNN.models.vanillaMLP import VanillaMLP, BananaSplitMLP

from wavNN.data_generators.mnist_generator import NMISTGenerator

import wavNN


class Optimize:
    def __init__(
        self,
        model,
        parameter_space,
        parameter_selection_function,
        data_class,
        data_params,
        monitor_metric="val_loss",
        epochs=80,
        n_optimizizer_iters=40,
        save=False,
        save_path="",
    ):
        self.model = model
        self.parameter_space = parameter_space
        self.parameter_selection = parameter_selection_function
        self.monitor_metric = monitor_metric
        self.data_class = data_class
        self.data_params = data_params
        self.epochs = epochs
        self.opt_iters = n_optimizizer_iters
        self.save = save

        if self.save:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_path = save_path

    def training_loop(self, **model_parameters):

        (
            model_params,
            optimizer,
            optimizer_params,
            loss_function,
        ) = self.parameter_selection(**model_parameters)

        training = TrainingLoop(
            model_class=self.model,
            model_params=model_params,
            data_class=self.data_class,
            data_params=self.data_params,
            loss=loss_function,
            epochs=self.epochs,
            optimizer_class=optimizer,
            optimizer_config=optimizer_params,
        )
        training()
        history = training.history
        quality = np.max(np.asarray(history[self.monitor_metric]))
        return quality

    def run_optimization(self):
        optimizer = BayesianOptimization(
            f=self.training_loop,
            pbounds=self.parameter_space,
            verbose=1,
            random_state=1,
        )

        optimizer.maximize(init_points=5, n_iter=self.opt_iters)

        history = optimizer.res
        return history

    def __call__(self):
        optimizer_results = self.run_optimization()
        if self.save:
            with open(self.save_path, "w") as f:
                json.dump(optimizer_results, fp=f, default=str)
        results = pd.DataFrame(optimizer_results)
        return results.iloc[results["target"].idxmax()]["params"]


class OptimizeFromConfig(Optimize):
    def __init__(self, config: dict):
        optimizer_kwargs = self.read_config(config)

        super().__init__(**optimizer_kwargs)

    def read_config(self, config_file):

        model = eval(config_file["model"])
        data_class = eval(config_file["data_class"])
        data_params = config_file["data_configs"]
        monitor_metric = config_file["monitor"]
        epochs = int(config_file["epochs"])
        n_optimizer_iters = int(config_file["n_optimizer_iters"])
        save = bool(config_file["save"])
        save_path = config_file["save_path"]

        parameter_space = self.build_parameter_space(config_file)
        parameter_function = self.build_selection_function(config_file)

        return {
            "model": model,
            "parameter_space": parameter_space,
            "parameter_selection_function": parameter_function,
            "data_class": data_class,
            "data_params": data_params,
            "monitor_metric": monitor_metric,
            "epochs": epochs,
            "n_optimizizer_iters": n_optimizer_iters,
            "save": save,
            "save_path": save_path,
        }

    def build_parameter_space(self, config_file):
        parameter_space = {}

        for cateogry in ["model", "optimizer"]:
            for field in config_file[cateogry]:
                continious = ""
                parameter_space[f"{cateogry}_{field}"] = (
                    config_file[cateogry][field]
                    if continious
                    else (0, len(config_file[cateogry][field]) - 1)
                )

        parameter_space["loss_id"] = (0, len(config_file["loss"]) - 1)

        return parameter_space

    def build_selection_function(self, config_file):
        def selection_function(param_dict):
            model_params = {}
            for parameter in config_file["model"]:
                parameter_type = ""
                parameter_name = f"model_{parameter}"
                model_params[parameter] = (
                    param_dict[parameter_name]
                    if parameter_type == "continious"
                    else math.floor(param_dict[parameter_name])
                )

            optimizer_params = {}
            for parameter in config_file["optimizer"]:
                """"""
            optimizer_id = math.floor(param_dict["optimizer_id"])
            optimizer = eval(config_file["optimizer"][optimizer_id])

            loss_id = math.floor(param_dict["loss_id"])
            loss_function = eval(config_file["loss"][loss_id])

            return model_params, optimizer, optimizer_params, loss_function

        return selection_function


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
