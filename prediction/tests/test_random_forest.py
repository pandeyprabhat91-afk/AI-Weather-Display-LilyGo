import numpy as np
import pytest
from models.random_forest import RandomForestModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT


def test_fit_predict_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel(n_estimators=5, max_depth=3)
    model.fit(X, y)
    assert model.predict(X).shape == (X.shape[0],)

def test_predict_proba_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel(n_estimators=5, max_depth=3)
    model.fit(X, y)
    assert model.predict_proba(X).shape == (X.shape[0], N_CLASSES)

def test_feature_importance_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel(n_estimators=5, max_depth=3)
    model.fit(X, y)
    imp = model.get_feature_importances()
    assert imp.shape == (TOTAL_FEATURE_COUNT,)
    assert abs(imp.sum() - 1.0) < 1e-5

def test_save_load_roundtrip(small_Xy, tmp_path):
    X, y = small_Xy
    model = RandomForestModel(n_estimators=5, max_depth=3)
    model.fit(X, y)
    path = str(tmp_path / "rf.pkl")
    model.save(path)
    loaded = RandomForestModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
