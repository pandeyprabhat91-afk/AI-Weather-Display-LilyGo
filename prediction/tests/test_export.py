import os
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from deployment.export import export_scaler_header, export_lr_header, export_rf_c, export_tflite
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from features.config import TOTAL_FEATURE_COUNT, N_CLASSES


@pytest.fixture
def fitted_lr(small_Xy):
    X, y = small_Xy
    m = LogisticRegressionModel()
    m.fit(X, y)
    return m, X

@pytest.fixture
def fitted_rf(small_Xy):
    X, y = small_Xy
    m = RandomForestModel(n_estimators=5, max_depth=3)
    m.fit(X, y)
    return m, X

@pytest.fixture
def fitted_nn(small_Xy):
    X, y = small_Xy
    m = NeuralNetworkModel()
    m.fit(X, y, X_val=X, y_val=y, epochs=2)
    return m, X


def test_export_scaler_header(fitted_scaler, tmp_path):
    path = str(tmp_path / "scaler_params.h")
    export_scaler_header(fitted_scaler, path)
    assert os.path.exists(path)
    content = open(path).read()
    assert "scaler_mean" in content
    assert "scaler_std" in content

def test_export_lr_header(fitted_lr, tmp_path):
    model, X = fitted_lr
    path = str(tmp_path / "lr_coefficients.h")
    export_lr_header(model, path)
    assert os.path.exists(path)
    content = open(path).read()
    assert "lr_coef" in content
    assert "lr_intercept" in content

def test_export_rf_c(fitted_rf, tmp_path):
    model, X = fitted_rf
    path = str(tmp_path / "rf_model.c")
    export_rf_c(model, path)
    assert os.path.exists(path)
    assert len(open(path).read()) > 100

def test_export_tflite_creates_file(fitted_nn, tmp_path):
    model, X = fitted_nn
    path = str(tmp_path / "model.tflite")
    export_tflite(model.get_keras_model(), X[:100], path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

def test_export_tflite_creates_header(fitted_nn, tmp_path):
    model, X = fitted_nn
    tflite_path = str(tmp_path / "model.tflite")
    header_path = str(tmp_path / "model_data.h")
    export_tflite(model.get_keras_model(), X[:100], tflite_path, header_path=header_path)
    assert os.path.exists(header_path)
    assert "model_data" in open(header_path).read()
