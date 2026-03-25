import numpy as np
import pytest
from evaluation.metrics import compute_all_metrics
from features.config import N_CLASSES


@pytest.fixture
def perfect_preds():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = y_true.copy()
    y_proba = np.eye(N_CLASSES)[y_true]
    return y_true, y_pred, y_proba


@pytest.fixture
def random_preds():
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.integers(0, N_CLASSES, n)
    y_pred = rng.integers(0, N_CLASSES, n)
    y_proba = rng.dirichlet(np.ones(N_CLASSES), n)
    return y_true, y_pred, y_proba


def test_perfect_accuracy(perfect_preds):
    y_true, y_pred, y_proba = perfect_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == pytest.approx(1.0)

def test_perfect_macro_f1(perfect_preds):
    y_true, y_pred, y_proba = perfect_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert metrics["macro_f1"] == pytest.approx(1.0)

def test_metrics_keys_present(random_preds):
    y_true, y_pred, y_proba = random_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    for key in ["accuracy", "macro_f1", "per_class_precision", "per_class_recall",
                "per_class_f1", "confusion_matrix", "roc_auc_ovr"]:
        assert key in metrics

def test_per_class_metrics_length(random_preds):
    y_true, y_pred, y_proba = random_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert len(metrics["per_class_precision"]) == N_CLASSES
    assert len(metrics["per_class_recall"]) == N_CLASSES
    assert len(metrics["per_class_f1"]) == N_CLASSES

def test_confusion_matrix_shape(random_preds):
    y_true, y_pred, y_proba = random_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert metrics["confusion_matrix"].shape == (N_CLASSES, N_CLASSES)
