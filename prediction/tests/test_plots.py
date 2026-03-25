import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evaluation.plots import confusion_matrix_fig, roc_fig, feature_importance_fig
from features.config import N_CLASSES, CLASS_NAMES, TOTAL_FEATURE_COUNT


@pytest.fixture
def sample_metrics():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, N_CLASSES, 50)
    y_proba = rng.dirichlet(np.ones(N_CLASSES), 50)
    y_pred = y_proba.argmax(axis=1)
    from evaluation.metrics import compute_all_metrics
    return compute_all_metrics(y_true, y_pred, y_proba)


def test_confusion_matrix_fig_returns_figure(sample_metrics):
    fig = confusion_matrix_fig(sample_metrics["confusion_matrix"], title="Test")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_roc_fig_accepts_multiple_models():
    rng = np.random.default_rng(0)
    n = 100
    y_true = rng.integers(0, N_CLASSES, n)
    models_proba = {
        "LR": rng.dirichlet(np.ones(N_CLASSES), n),
        "RF": rng.dirichlet(np.ones(N_CLASSES), n),
        "NN": rng.dirichlet(np.ones(N_CLASSES), n),
    }
    fig = roc_fig(y_true, models_proba)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_feature_importance_fig_lr_shape():
    coef = np.random.randn(N_CLASSES, TOTAL_FEATURE_COUNT)
    names = [f"feat_{i}" for i in range(TOTAL_FEATURE_COUNT)]
    fig = feature_importance_fig(coef, names, model_type="lr", title="LR Coef")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_feature_importance_fig_rf_shape():
    imp = np.abs(np.random.randn(TOTAL_FEATURE_COUNT))
    imp /= imp.sum()
    names = [f"feat_{i}" for i in range(TOTAL_FEATURE_COUNT)]
    fig = feature_importance_fig(imp, names, model_type="rf", title="RF Importance")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
