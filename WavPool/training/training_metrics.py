from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import numpy as np

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
        n_classes = prediction.shape[1]
        eye = np.eye(n_classes)

        _, predicted_class = torch.max(prediction, 1)
        predicted_class = predicted_class.detach().numpy()
        return roc_auc_score(eye[label], eye[predicted_class], multi_class="ovo")

    @staticmethod
    def auc_curve(prediction, label):
        n_classes = prediction.shape[1]
        eye = np.eye(n_classes)

        _, predicted_class = torch.max(prediction, 1)
        predicted_class = predicted_class.detach().numpy()
        print(np.array(label))
        label = eye[np.array(label).astype(int)].ravel()
        score_fpr, score_tpr, _ = roc_curve(label, eye[predicted_class].ravel())
        return score_fpr, score_tpr

    @staticmethod
    def confusion_matrix(prediction, label):
        num_classes = [i + 1 for i in range(prediction.shape[1])]
        _, predicted_class = torch.max(prediction, 1)
        return confusion_matrix(
            label.ravel(), predicted_class.ravel(), labels=num_classes
        )
