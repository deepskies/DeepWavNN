import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from WavPool.training.training_metrics import TrainingMetrics


def plot_test_results(predictions, labels, save_path=None):
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


def plot_history(history, extra_metric_names, save_path=None):
    history = pd.DataFrame(history)
    epochs = range(len(history))
    n_subplots = len(extra_metric_names) + 1
    fig, subplots = plt.subplots(nrows=n_subplots, ncols=1)

    for metric_index, metrics in enumerate(extra_metric_names):
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


def plot_history_errorbar(
    subplots: list,
    histories: list,
    extra_metrics: list,
    labels: list,
    colorway: list,
    save_path=None,
    title="",
    show=False,
    clear=False,
):
    for label, color in zip(labels, colorway):

        for metric_index, metrics in enumerate(extra_metrics):

            training = [history[f"train_{metrics.__name__}"] for history in histories]
            val = [history[f"val_{metrics.__name__}"] for history in histories]

            mean_training = pd.DataFrame(training).mean(axis=0)
            std_training = pd.DataFrame(training).std(axis=0)

            epochs = range(len(mean_training))

            mean_val = pd.DataFrame(val).mean(axis=0)

            subplots[metric_index].grid(
                color="grey", linestyle="--", linewidth=0.5, alpha=0.6
            )

            subplots[metric_index].plot(epochs, mean_training, label=label, color=color)
            subplots[metric_index].fill_between(
                epochs,
                mean_training - std_training,
                mean_training + std_training,
                alpha=0.3,
                color=color,
            )
            subplots[metric_index].plot(
                epochs, mean_val, linestyle="dotted", color=color
            )
            subplots[metric_index].set_ylabel(metrics.__name__)

            training = [history[f"train_loss"] for history in histories]
            val = [history[f"val_loss"] for history in histories]

            mean_training = pd.DataFrame(training).mean(axis=0)
            std_training = pd.DataFrame(training).std(axis=0)
            mean_val = pd.DataFrame(val).mean(axis=0)

            epochs = range(len(mean_training))

            metric_index = -1

            subplots[metric_index].plot(epochs, mean_training, label=label, color=color)
            subplots[metric_index].fill_between(
                epochs,
                mean_training - std_training,
                mean_training + std_training,
                alpha=0.3,
                color=color,
            )
            subplots[metric_index].plot(
                epochs, mean_val, linestyle="dotted", color=color
            )
            subplots[metric_index].set_ylabel("Loss")
            subplots[metric_index].grid(
                color="grey", linestyle="--", linewidth=0.5, alpha=0.6
            )

            subplots[0].set_title(title.strip("_"))

    if show:
        plt.legend()
        plt.show()

    if clear:
        plt.close("all")

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}_history_errorbar.png")


def plot_model_parameter_comparison(
    num_params,
    inference_time,
    training_time,
    labels,
    colors,
    save_path=None,
    title="",
    show=False,
    clear=False,
):

    _, subplots = plt.subplots(nrows=3, ncols=1, figsize=(6, 10))

    bar_x = [i + 1 for i in range(len(num_params))]

    subplots[0].barh(y=bar_x, width=num_params, color=colors)
    subplots[0].set_xlabel("Number Parameters")
    subplots[0].grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

    for model, color, inference, training in zip(
        labels, colors, inference_time, training_time
    ):
        w = 0.00005
        subplots[1].hist(
            inference,
            bins=np.arange(min(inference), max(inference) + w, w),
            label=model,
            color=color,
        )
        subplots[1].set_xlabel("Mean Single Inference Time (s)")
        subplots[1].grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

        w = 3.0
        subplots[2].hist(
            training,
            bins=np.arange(min(training), max(training) + w, w),
            label=model,
            color=color,
        )

        subplots[2].set_xlabel("Mean Full Training Time (s)")
        subplots[2].grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

    subplots[0].set_title(title.strip("_"))
    if show:
        plt.legend()
        plt.show()

    if clear:
        plt.close("all")

    if save_path is not None:
        plt.savefig(f"{save_path}/{title}_history_errorbar.png")
