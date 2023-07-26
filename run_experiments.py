import wavpool.models as models
import wavpool.data_generators as datagens

from wavpool.training.train_model import TrainingLoop
from wavpool.training.training_metrics import TrainingMetrics

import torch
import datetime
import os
import json


class RunExperiment:
    def __init__(self, experiment_config) -> None:
        self.experiment_config = RunExperiment.add_experiment_defaults(
            experiment_config
        )
        self.model_parameters = {}
        self.training_history = {}

        self.model = self.locate_model()
        self.datagen = self.locate_datagen()
        self.optimizer = self.locate_optimizer()
        self.loss = self.locate_loss()

    @staticmethod
    def add_experiment_defaults(experiment_config):
        # Check that the required params with no defaults are filled
        assert "model" in experiment_config.keys()
        assert "save_path" in experiment_config.keys()
        assert "data" in experiment_config.keys()

        default_config = {
            "model_kwargs": {"in_channels": 28, "hidden_size": 256, "out_channels": 10},
            "data_kwargs": {
                "sample_size": [10000, 4000, 4000],
                "split": True,
                "batch_size": 640,
            },
            "optimizer_kwargs": {"lr": 0.01, "momentum": False},
            "optimizer": "SGD",
            "loss": "CrossEntropyLoss",
            "epochs": 20,
            "training_metrics": [
                TrainingMetrics.accuracy,
                TrainingMetrics.auc_roc,
                TrainingMetrics.f1,
            ],
            "num_tests": 3,
            "num_inference_tests": 50,
            "experiment_name": f"{experiment_config['model']}_{experiment_config['data']}_{datetime.datetime.now().date()}",
        }
        for config_param in default_config:
            if config_param not in experiment_config:
                experiment_config[config_param] = default_config[config_param]

        return experiment_config

    def locate_model(self):
        model_locations = models.__dict__

        if not self.experiment_config["model"] in model_locations.keys():
            raise NotImplementedError

        return model_locations[self.experiment_config["model"]]

    def locate_datagen(self):
        model_locations = datagens.__dict__

        if not self.experiment_config["data"] in model_locations.keys():
            raise NotImplementedError

        return model_locations[self.experiment_config["data"]]

    def locate_optimizer(self):
        optimizers = {opt[0]: opt[1] for opt in torch.optim.__dict__.items()}
        if not self.experiment_config["optimizer"] in optimizers.keys():
            raise NotImplementedError

        return optimizers[self.experiment_config["optimizer"]]

    def locate_loss(self):
        losses = {
            loss[0]: loss[1]
            for loss in torch.nn.__dict__.items()
            if "loss" in loss[0].lower()
        }
        if not self.experiment_config["loss"] in losses.keys():
            raise NotImplementedError

        return losses[self.experiment_config["loss"]]

    def time_inference(self):
        model = self.model(**self.experiment_config["model_kwargs"])
        in_size = self.experiment_config["model_kwargs"]["in_channels"]
        random_input = torch.rand(1, 1, in_size, in_size)

        start_time = datetime.datetime.now()
        model(random_input)
        inference_time = abs(start_time - datetime.datetime.now()).total_seconds()

        return inference_time

    def count_params(self):
        model = self.model(**self.experiment_config["model_kwargs"])
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return total_trainable_params

    def run_experiment(self):
        loop = TrainingLoop(
            model_class=self.model,
            model_params=self.experiment_config["model_kwargs"],
            data_class=self.datagen,
            data_params=self.experiment_config["data_kwargs"],
            optimizer_class=self.optimizer,
            optimizer_config=self.experiment_config["optimizer_kwargs"],
            loss=self.loss,
            epochs=self.experiment_config["epochs"],
            extra_metrics=self.experiment_config["training_metrics"],
        )
        loop()
        history = loop.history
        return history

    def save_results(self):
        save_path = f"{self.experiment_config['save_path']}/{self.experiment_config['experiment_name']}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        history_path = f"{save_path}/history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, default=str)

        parameter_path = f"{save_path}/parameter_history.json"
        with open(parameter_path, "w") as f:
            json.dump(self.model_parameters, f, default=str)

        config_path = f"{save_path}/experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.experiment_config, f, default=str)

    def __call__(self):
        training_timing = []
        for experiment_iteration in range(self.experiment_config["num_tests"]):
            start_time = datetime.datetime.now()
            self.training_history[experiment_iteration] = self.run_experiment()

            training_timing.append(
                abs(start_time - datetime.datetime.now()).total_seconds()
            )

        self.model_parameters = {
            "num_parameters": self.count_params(),
            "inference_timing": [
                self.time_inference()
                for _ in range(self.experiment_config["num_inference_tests"])
            ],
            "training_timing": training_timing,
        }

        self.save_results()


if __name__ == "__main__":
    from wavNN.training.finetune_network import OptimizeFromConfig

    models_to_test = [
        (
            "VanillaCNN",
            {
                "kernel_size": (2, 4),
                "out_channels": 10,
                "hidden_channels_1": (1, 20),
                "hidden_channels_2": (1, 20),
            },
        ),
    ]
    data_to_test = ["CIFARGenerator", "MNISTGenerator", "FashionMNISTGenerator"]
    data_sizes = [32, 28, 28]

    for model in models_to_test:
        for data, data_size in zip(data_to_test, data_sizes):

            model_class = models.__dict__[model[0]]
            data_class = datagens.__dict__[data]

            model[1]["in_channels"] = data_size
            opt_config = {
                "model": model_class,
                "model_config": model[1],
                "data_class": data_class,
                "data_config": {"sample_size": [4000, 2000, 2000], "split": True},
                "optimizer": {
                    "id": [torch.optim.SGD, torch.optim.Adam],
                    "lr": (0.000001, 0.8),
                },
                "loss": [torch.nn.CrossEntropyLoss],
                "monitor": "val_f1",
                "epochs": 20,
                "n_optimizer_iters": 30,
                "save": False,
                "save_path": "",
            }
            optimizer_engine = OptimizeFromConfig(opt_config)
            best_parms = optimizer_engine()

            selection_function = optimizer_engine.build_selection_function(opt_config)
            (
                model_params,
                optimizer,
                optimizer_params,
                _,
            ) = selection_function(**best_parms)

            optimizer = {torch.optim.SGD: "SGD", torch.optim.Adam: "Adam"}[optimizer]

            model_params["in_channels"] = data_size
            experiment_config = {
                "model": model[0],
                "model_kwargs": model_params,
                "data": data,
                "epochs": 120,
                "save_path": "./results/optimize_params",
                "optimizer": optimizer,
                "optimizer_kwargs": optimizer_params,
            }

            experiment = RunExperiment(experiment_config=experiment_config)
            experiment()
