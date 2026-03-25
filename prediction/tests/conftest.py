import numpy as np
import pandas as pd
import pytest
from features.config import TOTAL_FEATURE_COUNT, RANDOM_SEED


@pytest.fixture
def synthetic_weather_df():
    rng = np.random.default_rng(RANDOM_SEED)
    n = 500
    times = pd.date_range("2020-06-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times,
        "temperature": rng.uniform(-5, 35, n),
        "humidity": rng.uniform(20, 100, n),
        "pressure": rng.uniform(980, 1030, n),
        "label": rng.integers(0, 5, n),
    })


@pytest.fixture
def small_Xy(synthetic_weather_df):
    from features.engineering import build_features
    X, y = build_features(synthetic_weather_df)
    return X, y


@pytest.fixture
def fitted_scaler(small_Xy):
    from sklearn.preprocessing import StandardScaler
    X, _ = small_Xy
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler
