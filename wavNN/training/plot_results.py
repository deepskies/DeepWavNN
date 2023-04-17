import matplotlib.pyplot as plt
import pandas as pd


from wavNN.training.training_metrics import TrainingMetrics


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
