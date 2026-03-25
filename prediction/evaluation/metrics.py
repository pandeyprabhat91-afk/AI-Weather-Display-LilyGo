import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)
from features.config import N_CLASSES


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC-AUC robust to missing classes (model trained on fewer classes than N_CLASSES)."""
    unique_true = np.unique(y_true)
    if len(unique_true) < 2:
        return 0.0
    # Use only the columns that the model actually produced
    n_model_classes = y_proba.shape[1]
    present = [c for c in unique_true if c < n_model_classes]
    if len(present) < 2:
        return 0.0
    try:
        return float(roc_auc_score(
            y_true, y_proba[:, :n_model_classes],
            multi_class="ovr", average="macro",
            labels=list(range(n_model_classes)),
        ))
    except ValueError:
        return 0.0


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
        "roc_auc_ovr": _safe_roc_auc(y_true, y_proba),
    }
