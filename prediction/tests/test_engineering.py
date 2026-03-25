import numpy as np
import pytest
from features.engineering import build_features, compute_dew_point, compute_abs_humidity
from features.config import TOTAL_FEATURE_COUNT, N_CLASSES


def test_build_features_output_shapes(synthetic_weather_df):
    X, y = build_features(synthetic_weather_df)
    assert X.ndim == 2
    assert X.shape[1] == TOTAL_FEATURE_COUNT
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]

def test_build_features_min_samples(synthetic_weather_df):
    X, y = build_features(synthetic_weather_df)
    assert X.shape[0] > 0

def test_build_features_labels_in_range(synthetic_weather_df):
    _, y = build_features(synthetic_weather_df)
    assert y.min() >= 0
    assert y.max() < N_CLASSES

def test_build_features_no_nans(synthetic_weather_df):
    X, y = build_features(synthetic_weather_df)
    assert not np.isnan(X).any()
    assert not np.isnan(y.astype(float)).any()

def test_build_features_dtype(synthetic_weather_df):
    X, y = build_features(synthetic_weather_df)
    assert X.dtype == np.float32
    assert y.dtype == np.int32

def test_build_features_bme688_extras_zero_padded(synthetic_weather_df):
    X, _ = build_features(synthetic_weather_df)
    assert (X[:, -4:] == 0.0).all()

def test_build_features_with_bme688_extras(synthetic_weather_df):
    df = synthetic_weather_df.copy()
    rng = np.random.default_rng(0)
    df["gas_resistance"] = rng.uniform(1000, 50000, len(df))
    df["iaq"]  = rng.uniform(0, 500, len(df))
    df["eco2"] = rng.uniform(400, 2000, len(df))
    df["bvoc"] = rng.uniform(0, 10, len(df))
    X, _ = build_features(df)
    assert not (X[:, -4:] == 0.0).all()

def test_dew_point_formula():
    T_d = compute_dew_point(T=20.0, RH=50.0)
    assert abs(T_d - 9.3) < 0.5

def test_abs_humidity_formula():
    AH = compute_abs_humidity(T=25.0, RH=60.0)
    assert 10.0 < AH < 20.0

def test_build_features_too_few_rows():
    import pandas as pd
    tiny_df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=5, freq="h"),
        "temperature": [10.0] * 5,
        "humidity": [60.0] * 5,
        "pressure": [1013.0] * 5,
        "label": [0] * 5,
    })
    X, y = build_features(tiny_df)
    assert X.shape[0] == 0
