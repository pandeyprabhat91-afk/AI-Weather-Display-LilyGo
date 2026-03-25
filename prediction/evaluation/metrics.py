import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)
from features.config import N_CLASSES


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES))),
        "roc_auc_ovr": float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                          labels=list(range(N_CLASSES)))
        ) if len(np.unique(y_true)) > 1 else 0.0,
    }
