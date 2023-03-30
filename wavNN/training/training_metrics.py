from torcheval.metrics.functional import (
    multiclass_f1_score,
    multiclass_confusion_matrix,
)
from sklearn.metrics import roc_auc_score, roc_curve

import torch


class TrainingMetrics:
    @staticmethod
    def f1(prediction, label):
        return multiclass_f1_score(target=label, input=prediction).detach().numpy()

    @staticmethod
    def accuracy(prediction, label):
        _, predicted_class = torch.max(prediction, 1)
        return (label == predicted_class).sum().item() / label.size(0)

    @staticmethod
    def auc_roc(prediction: torch.Tensor, label: torch.Tensor):
        _, predicted_class = torch.max(prediction, 1)
        return roc_auc_score(label, predicted_class).detach().numpy()

    @staticmethod
    def auc_curve(prediction, label):
        _, predicted_class = torch.max(prediction, 1)
        score_fpr, score_tpr, _ = roc_curve(label, predicted_class)
        return score_fpr, score_tpr

    @staticmethod
    def confusion_matrix(prediction, label):
        _, predicted_class = torch.max(prediction, 1)
        ## Assume all the classes are presentated in the label
        num_classes = torch.unique(label)
        return (
            multiclass_confusion_matrix(
                input=predicted_class, target=label, num_classes=num_classes
            )
            .detach()
            .numpy()
        )
