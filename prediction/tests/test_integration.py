"""End-to-end smoke test: features → scale → train all 3 → evaluate → export artifacts."""
import os
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from features.engineering import build_features
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from evaluation.metrics import compute_all_metrics
from features.config import RANDOM_SEED


@pytest.mark.slow
def test_full_pipeline_smoke(synthetic_weather_df, tmp_path):
    X, y = build_features(synthetic_weather_df)
    assert X.shape[1] == 128

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)

    lr = LogisticRegressionModel(random_seed=RANDOM_SEED)
    lr.fit(X_s, y)
    lr_m = compute_all_metrics(y, lr.predict(X_s), lr.predict_proba(X_s))
    assert "accuracy" in lr_m

    rf = RandomForestModel(n_estimators=10, max_depth=3, random_seed=RANDOM_SEED)
    rf.fit(X_s, y)
    rf_m = compute_all_metrics(y, rf.predict(X_s), rf.predict_proba(X_s))
    assert "accuracy" in rf_m

    nn = NeuralNetworkModel(random_seed=RANDOM_SEED)
    nn.fit(X_s, y, X_val=X_s, y_val=y, epochs=2)
    nn_m = compute_all_metrics(y, nn.predict(X_s), nn.predict_proba(X_s))
    assert "accuracy" in nn_m

    from deployment.export import (export_scaler_header, export_lr_header,
                                    export_rf_c, export_tflite)
    export_scaler_header(scaler, str(tmp_path / "scaler_params.h"))
    export_lr_header(lr, str(tmp_path / "lr_coefficients.h"))
    export_rf_c(rf, str(tmp_path / "rf_model.c"))
    export_tflite(nn.get_keras_model(), X_s[:50],
                  str(tmp_path / "model.tflite"),
                  header_path=str(tmp_path / "model_data.h"))

    for fname in ["scaler_params.h", "lr_coefficients.h", "rf_model.c",
                  "model.tflite", "model_data.h"]:
        assert os.path.exists(str(tmp_path / fname)), f"Missing: {fname}"

    print(f"\nLR  accuracy: {lr_m['accuracy']:.3f}  F1: {lr_m['macro_f1']:.3f}")
    print(f"RF  accuracy: {rf_m['accuracy']:.3f}  F1: {rf_m['macro_f1']:.3f}")
    print(f"NN  accuracy: {nn_m['accuracy']:.3f}  F1: {nn_m['macro_f1']:.3f}")
