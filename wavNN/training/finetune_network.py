"""
Generate parameters for a network using a guassian processor optimizer
(Using https://github.com/fmfn/BayesianOptimization)
Lightweight wrapper on the train_model

"""
import json
from typing import Union
from bayes_opt import BayesianOptimization
import os
import numpy as np
import math
import pandas as pd

from wavNN.training.train_model import TrainingLoop
import wavNN


class Optimize:
    def __init__(
        self,
        model,
        parameter_space,
        parameter_selection_function,
        data_class,
        data_params,
        monitor_metric="val_accuracy",
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
        return results.iloc[results["target"].idxmax()]["params"]  # type: ignore


# TODO Cleo to build params from scratch
class OptimizeFromConfig(Optimize):
    def __init__(self, config: Union[dict, str]):

        if type(config) == str:
            assert os.path.exists(config)
            with open(config, "rb") as f:
                config = json.load(f)

        optimizer_kwargs = self.read_config(config)

        super().__init__(**optimizer_kwargs)

    def add_config_params(self, config_file):
        default_config = {
            "data_config": {},
            "monitor": "val_accuracy",
            "epochs": 20,
            "n_optimizer_iters": 40,
            "save": False,
            "save_path": "",
            "parameters_space": {},
            "parameter_function": {},
        }
        for field in default_config:
            if field not in config_file.keys():
                config_file[field] = default_config[field]

    def read_config(self, config_file):

        model = config_file["model"]
        data_class = config_file["data_class"]
        data_params = config_file["data_config"]
        monitor_metric = config_file["monitor"]
        epochs = int(config_file["epochs"])
        n_optimizer_iters = int(config_file["n_optimizer_iters"])
        save = bool(config_file["save"])
        save_path = config_file["save_path"]

        parameter_space = self.build_parameter_space(config_file)
        parameter_function = self.build_selection_function(config_file)

        optimizer_config = {
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

        return optimizer_config

    def build_parameter_space(self, config_file):
        parameter_space = {}

        for cateogry in ["model_config", "optimizer"]:
            for field in config_file[cateogry]:
                optimizer_param = config_file[cateogry][field]
                continious = type(optimizer_param) == tuple
                if type(optimizer_param) == int:
                    optimizer_param = [optimizer_param]

                parameter_space[f"{cateogry}_{field}"] = (
                    optimizer_param if continious else (0, len(optimizer_param) - 1)
                )

        parameter_space["loss_id"] = (0, len(config_file["loss"]) - 1)

        return parameter_space

    def build_selection_function(self, config_file):
        def selection_function(**param_dict):
            model_params = {}
            optimizer_params = {}

            for category, parameters in zip(
                ["model_config", "optimizer"], [model_params, optimizer_params]
            ):
                for parameter in config_file[category]:
                    continious = type(config_file[category][parameter]) == tuple

                    if type(config_file[category][parameter]) == int:
                        config_file[category][parameter] = [
                            config_file[category][parameter]
                        ]

                    parameter_name = f"{category}_{parameter}"
                    parameters[parameter] = (
                        param_dict[parameter_name]
                        if continious
                        else config_file[category][parameter][
                            math.floor(param_dict[parameter_name])
                        ]
                    )

            optimizer_id = math.floor(param_dict["optimizer_id"])
            optimizer = config_file["optimizer"]["id"][optimizer_id]
            optimizer_params.pop("id")
            loss_id = math.floor(param_dict["loss_id"])
            loss_function = config_file["loss"][loss_id]

            # for param in config_file["training_configs"]:
            #     model_params[param] = config_file[
            #         "training_configs"
            #     ]  # Untouched params

            return model_params, optimizer, optimizer_params, loss_function

        return selection_function
