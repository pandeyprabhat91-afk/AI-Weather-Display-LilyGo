import numpy as np
import pytest
from models.neural_network import NeuralNetworkModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT, RANDOM_SEED


@pytest.fixture
def trained_nn(small_Xy):
    X, y = small_Xy
    model = NeuralNetworkModel()
    model.fit(X, y, X_val=X, y_val=y, epochs=3)
    return model, X, y


def test_fit_predict_shape(trained_nn):
    model, X, y = trained_nn
    assert model.predict(X).shape == (X.shape[0],)

def test_predict_proba_shape(trained_nn):
    model, X, y = trained_nn
    assert model.predict_proba(X).shape == (X.shape[0], N_CLASSES)

def test_predict_proba_sums_to_one(trained_nn):
    model, X, y = trained_nn
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)

def test_save_load_roundtrip(trained_nn, tmp_path):
    model, X, y = trained_nn
    path = str(tmp_path / "nn_model")
    model.save(path)
    from models.neural_network import NeuralNetworkModel
    loaded = NeuralNetworkModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
