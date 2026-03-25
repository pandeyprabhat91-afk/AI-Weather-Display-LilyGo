import numpy as np
import pytest
from models.logistic_regression import LogisticRegressionModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT


def test_fit_predict_shape(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)

def test_predict_classes_in_range(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.min() >= 0
    assert preds.max() < N_CLASSES

def test_predict_proba_shape(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    assert model.predict_proba(X).shape == (X.shape[0], N_CLASSES)

def test_predict_proba_sums_to_one(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1.0, atol=1e-5)

def test_feature_importance_shape(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    assert model.get_feature_importances().shape == (N_CLASSES, TOTAL_FEATURE_COUNT)

def test_save_load_roundtrip(small_Xy, tmp_path):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    path = str(tmp_path / "lr.pkl")
    model.save(path)
    loaded = LogisticRegressionModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
