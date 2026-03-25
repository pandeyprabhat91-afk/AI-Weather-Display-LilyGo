import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from features.config import CLASS_NAMES, N_CLASSES


def confusion_matrix_fig(cm: np.ndarray, title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    fig.tight_layout()
    return fig


def roc_fig(y_true: np.ndarray, models_proba: dict) -> plt.Figure:
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "green"]
    for (name, proba), color in zip(models_proba.items(), colors):
        # Pad proba to N_CLASSES columns if model was trained on fewer classes
        if proba.shape[1] < N_CLASSES:
            pad = np.zeros((proba.shape[0], N_CLASSES - proba.shape[1]))
            proba = np.hstack([proba, pad])
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (macro one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def feature_importance_fig(importance: np.ndarray, feature_names: list,
                            model_type: str, title: str = "", top_n: int = 20) -> plt.Figure:
    scores = np.mean(np.abs(importance), axis=0) if importance.ndim > 1 else importance
    idx = np.argsort(scores)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], scores[idx], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title or "Feature Importance")
    fig.tight_layout()
    return fig
