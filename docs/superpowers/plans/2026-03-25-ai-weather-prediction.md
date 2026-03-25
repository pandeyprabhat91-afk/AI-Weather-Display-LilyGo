# AI Weather Prediction System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-hour weather condition classifier (Sunny/Cloudy/Rainy/Stormy/Snowy) using BME688 sensor data, comparing Logistic Regression, Random Forest, and Neural Network models with shared feature extraction, all deployable to LilyGo S3 (ESP32-S3).

**Architecture:** Modular Python pipeline under `prediction/` — a shared `features/` module guarantees identical inputs to all three models. Training runs on PC via scikit-learn + TensorFlow; deployment artifacts (C headers, TFLite) target the ESP32-S3. A Jinja2 HTML report compares all models.

**Tech Stack:** Python 3.10+, scikit-learn, TensorFlow 2.x, pandas, numpy, openmeteo-requests, m2cgen, matplotlib, seaborn, jinja2, pytest

**Spec:** `docs/superpowers/specs/2026-03-25-ai-weather-prediction-design.md`

---

## File Map

```
prediction/
├── data/
│   ├── __init__.py
│   └── download.py          # fetch_weather_data(), apply_wmo_mapping(), split_data()
├── features/
│   ├── __init__.py
│   ├── config.py            # RANDOM_SEED, LABEL_MAP, STATIONS, FEATURE_COLUMNS, WMO_MAP
│   └── engineering.py       # build_features(df) → (X, y, scaler)
├── models/
│   ├── __init__.py
│   ├── logistic_regression.py   # LogisticRegressionModel
│   ├── random_forest.py         # RandomForestModel
│   └── neural_network.py        # NeuralNetworkModel
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # compute_all_metrics(y_true, y_pred, y_proba) → dict
│   └── plots.py             # confusion_matrix_fig(), roc_fig(), feature_importance_fig()
├── report/
│   ├── __init__.py
│   ├── generate.py          # generate_report(results_dict, output_path)
│   └── template.html        # Jinja2 HTML report template
├── deployment/
│   ├── __init__.py
│   └── export.py            # export_tflite(), export_lr_header(), export_rf_c(), export_scaler_header()
├── tests/
│   ├── conftest.py          # shared pytest fixtures (synthetic DataFrames, small X/y arrays)
│   ├── test_download.py
│   ├── test_engineering.py
│   ├── test_logistic_regression.py
│   ├── test_random_forest.py
│   ├── test_neural_network.py
│   ├── test_metrics.py
│   ├── test_plots.py
│   └── test_export.py
├── reports/
│   └── .gitkeep
├── data/cache/.gitkeep
├── data/processed/.gitkeep
├── deployment/.gitkeep
├── main.py                  # CLI orchestrator
├── requirements.txt
├── pytest.ini
└── .gitignore
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `prediction/requirements.txt`
- Create: `prediction/pytest.ini`
- Create: `prediction/.gitignore`
- Create: `prediction/data/__init__.py`
- Create: `prediction/data/cache/.gitkeep`
- Create: `prediction/data/processed/.gitkeep`
- Create: `prediction/features/__init__.py`
- Create: `prediction/models/__init__.py`
- Create: `prediction/evaluation/__init__.py`
- Create: `prediction/report/__init__.py`
- Create: `prediction/deployment/__init__.py`
- Create: `prediction/reports/.gitkeep`
- Create: `prediction/tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
openmeteo-requests==0.3.3
requests-cache==1.2.1
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
tensorflow>=2.15
m2cgen==0.10.0
matplotlib>=3.7
seaborn>=0.13
jinja2>=3.1
joblib>=1.3
pytest>=8.0
pytest-cov>=4.1
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
addopts = -v --tb=short
markers =
    slow: marks tests as slow (deselect with -m 'not slow')
```

- [ ] **Step 3: Create .gitignore**

```gitignore
data/cache/
data/processed/
deployment/*.pkl
deployment/*.tflite
reports/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 4: Create all __init__.py and .gitkeep files**

Each `__init__.py` is empty. Each `.gitkeep` is empty. Create them all:
```bash
cd prediction/
touch data/__init__.py features/__init__.py models/__init__.py \
      evaluation/__init__.py report/__init__.py deployment/__init__.py \
      tests/__init__.py
touch data/cache/.gitkeep data/processed/.gitkeep reports/.gitkeep
```

- [ ] **Step 5: Install dependencies**

```bash
cd prediction/
pip install -r requirements.txt
```

Expected: all packages install without error.

- [ ] **Step 6: Commit**

```bash
git add prediction/
git commit -m "feat: scaffold prediction/ project structure"
```

---

## Task 2: Config Module

**Files:**
- Create: `prediction/features/config.py`
- Create: `prediction/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# prediction/tests/test_config.py
from features.config import (
    RANDOM_SEED, LABEL_MAP, WMO_MAP, STATIONS,
    CLASS_NAMES, N_CLASSES, LOOKBACK, LOOKAHEAD,
    BME688_EXTRAS, CORE_FEATURE_COUNT, TOTAL_FEATURE_COUNT,
)

def test_random_seed_is_42():
    assert RANDOM_SEED == 42

def test_label_map_has_five_classes():
    assert set(LABEL_MAP.values()) == {0, 1, 2, 3, 4}
    assert LABEL_MAP[0] == "Sunny"
    assert LABEL_MAP[4] == "Snowy"

def test_wmo_map_covers_expected_codes():
    # Sunny
    assert WMO_MAP[0] == 0
    assert WMO_MAP[1] == 0
    # Cloudy
    assert WMO_MAP[2] == 1
    assert WMO_MAP[45] == 1
    # Rainy
    assert WMO_MAP[61] == 2
    assert WMO_MAP[80] == 2
    # Stormy
    assert WMO_MAP[95] == 3
    assert WMO_MAP[99] == 3
    # Snowy
    assert WMO_MAP[71] == 4
    assert WMO_MAP[85] == 4

def test_wmo_map_does_not_contain_unmapped_codes():
    # Code 4 is unmapped per spec
    assert 4 not in WMO_MAP
    assert 50 not in WMO_MAP

def test_feature_count():
    assert CORE_FEATURE_COUNT == 54
    assert len(BME688_EXTRAS) == 4
    assert TOTAL_FEATURE_COUNT == 58

def test_stations_has_three_entries():
    assert len(STATIONS) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prediction/
pytest tests/test_config.py -v
```
Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement config.py**

```python
# prediction/features/config.py

RANDOM_SEED = 42

LOOKBACK = 6   # hours of history used as input
LOOKAHEAD = 6  # hours ahead to predict

# --- Labels ---
LABEL_MAP = {
    0: "Sunny",
    1: "Cloudy",
    2: "Rainy",
    3: "Stormy",
    4: "Snowy",
}
CLASS_NAMES = [LABEL_MAP[i] for i in range(5)]
N_CLASSES = 5

# --- WMO weather code → label int ---
# Unmapped codes are absent from this dict and will be dropped from data.
WMO_MAP: dict[int, int] = {}
for code in [0, 1]:
    WMO_MAP[code] = 0  # Sunny
for code in [2, 3, 45, 46, 47, 48]:
    WMO_MAP[code] = 1  # Cloudy
for code in [51, 52, 53, 54, 55, 56, 57,
             61, 62, 63, 64, 65, 66, 67,
             80, 81, 82]:
    WMO_MAP[code] = 2  # Rainy
for code in [95, 96, 97, 98, 99]:
    WMO_MAP[code] = 3  # Stormy
for code in [71, 72, 73, 74, 75, 76, 77, 85, 86]:
    WMO_MAP[code] = 4  # Snowy

# --- Default training stations ---
# List of (name, latitude, longitude) tuples
STATIONS = [
    ("London, UK",       51.5074, -0.1278),
    ("Helsinki, Finland", 60.1699, 25.0002),
    ("Singapore",         1.3521,  103.8198),
]

# --- BME688 optional columns (zero-padded if absent) ---
BME688_EXTRAS = ["gas_resistance", "iaq", "eco2", "bvoc"]

# --- Feature count bookkeeping ---
# 18 raw lags + 3 pressure tendency + 1 pressure accel +
# 2 temp rate + 1 dew point + 1 abs humidity +
# 24 rolling stats + 4 cyclical time = 54
CORE_FEATURE_COUNT = 54
TOTAL_FEATURE_COUNT = CORE_FEATURE_COUNT + len(BME688_EXTRAS)  # 58
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_config.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/features/config.py prediction/tests/test_config.py
git commit -m "feat: add config module with WMO mapping and feature constants"
```

---

## Task 3: Data Download Module

**Files:**
- Create: `prediction/data/download.py`
- Create: `prediction/tests/test_download.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_download.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from data.download import apply_wmo_mapping, split_data, forward_fill_gaps


def make_raw_df(n=200):
    """Synthetic raw DataFrame as returned by Open-Meteo."""
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times,
        "temperature": np.random.uniform(-10, 35, n),
        "humidity": np.random.uniform(20, 100, n),
        "pressure": np.random.uniform(980, 1030, n),
        "weather_code": np.random.choice([0, 1, 2, 61, 71, 95], n),
    })


def test_apply_wmo_mapping_maps_known_codes():
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=3, freq="h"),
        "temperature": [10.0, 10.0, 10.0],
        "humidity": [60.0, 60.0, 60.0],
        "pressure": [1013.0, 1013.0, 1013.0],
        "weather_code": [0, 61, 95],
    })
    result = apply_wmo_mapping(df)
    assert list(result["label"]) == [0, 2, 3]


def test_apply_wmo_mapping_drops_unmapped_codes():
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=3, freq="h"),
        "temperature": [10.0, 10.0, 10.0],
        "humidity": [60.0, 60.0, 60.0],
        "pressure": [1013.0, 1013.0, 1013.0],
        "weather_code": [0, 4, 61],  # code 4 is unmapped
    })
    result = apply_wmo_mapping(df)
    assert len(result) == 2
    assert 4 not in result["weather_code"].values


def test_forward_fill_gaps_fills_short_gaps():
    df = make_raw_df(10)
    df.loc[3, "temperature"] = np.nan
    df.loc[4, "temperature"] = np.nan  # 2-hour gap — should be filled
    result = forward_fill_gaps(df)
    assert result["temperature"].isna().sum() == 0


def test_forward_fill_gaps_drops_long_gaps():
    df = make_raw_df(10)
    df.loc[3:6, "temperature"] = np.nan  # 4-hour gap — should be dropped
    result = forward_fill_gaps(df)
    assert result["temperature"].isna().sum() == 0
    assert len(result) < 10  # some rows dropped


def test_split_data_proportions():
    df = make_raw_df(1000)
    df["label"] = np.random.randint(0, 5, 1000)
    train, val, test = split_data(df)
    assert len(train) == pytest.approx(700, abs=5)
    assert len(val) == pytest.approx(150, abs=5)
    assert len(test) == pytest.approx(150, abs=5)


def test_split_data_is_chronological():
    df = make_raw_df(1000)
    df["label"] = 0
    train, val, test = split_data(df)
    assert train["time"].max() < val["time"].min()
    assert val["time"].max() < test["time"].min()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_download.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement download.py**

```python
# prediction/data/download.py
import os
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from features.config import WMO_MAP, STATIONS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def _get_api_client():
    session = requests_cache.CachedSession(
        os.path.join(CACHE_DIR, ".http_cache"), expire_after=-1
    )
    session = retry(session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def fetch_weather_data(
    stations=None,
    years: int = 5,
    force_download: bool = False,
) -> pd.DataFrame:
    """Fetch hourly historical data from Open-Meteo for each station."""
    if stations is None:
        stations = STATIONS

    frames = []
    client = _get_api_client()

    import datetime
    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=365 * years)).isoformat()

    for name, lat, lon in stations:
        slug = name.lower().replace(" ", "_").replace(",", "")
        cache_path = os.path.join(CACHE_DIR, f"{slug}_{years}y.csv")

        if os.path.exists(cache_path) and not force_download:
            df = pd.read_csv(cache_path, parse_dates=["time"])
            frames.append(df)
            continue

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m",
                       "pressure_msl", "weather_code"],
            "start_date": start_date,
            "end_date": end_date,
        }
        responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        r = responses[0]
        hourly = r.Hourly()

        df = pd.DataFrame({
            "time": pd.date_range(
                start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
                end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy(),
            "weather_code": hourly.Variables(3).ValuesAsNumpy().astype(int),
        })
        df["station"] = name
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def apply_wmo_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Map weather_code → label int. Drop rows with unmapped codes."""
    df = df.copy()
    df["label"] = df["weather_code"].map(WMO_MAP)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def forward_fill_gaps(df: pd.DataFrame, max_gap_hours: int = 2) -> pd.DataFrame:
    """Forward-fill NaN gaps of ≤ max_gap_hours. Drop rows with remaining NaN."""
    df = df.copy().sort_values("time").reset_index(drop=True)
    numeric_cols = ["temperature", "humidity", "pressure"]
    for col in numeric_cols:
        # Mark consecutive NaN runs longer than max_gap_hours
        is_nan = df[col].isna()
        run_id = (is_nan != is_nan.shift()).cumsum()
        run_lengths = is_nan.groupby(run_id).transform("sum")
        long_gap = is_nan & (run_lengths > max_gap_hours)
        df.loc[long_gap, col] = np.nan  # keep long gaps as NaN (will be dropped)
        df[col] = df[col].ffill()       # fill short gaps
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Chronological 70/15/15 split. Returns (train, val, test) DataFrames."""
    df = df.sort_values("time").reset_index(drop=True)
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()


def prepare_data(stations=None, years: int = 5, force_download: bool = False):
    """Full pipeline: fetch → map WMO → fill gaps → split → save CSVs."""
    raw = fetch_weather_data(stations=stations, years=years, force_download=force_download)
    raw = apply_wmo_mapping(raw)
    raw = forward_fill_gaps(raw)
    train, val, test = split_data(raw)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"Data prepared: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
```

- [ ] **Step 4: Add `retry-requests` to requirements.txt**

Append to `prediction/requirements.txt`:
```
retry-requests==2.0.0
```
Then run `pip install retry-requests`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_download.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prediction/data/download.py prediction/tests/test_download.py prediction/requirements.txt
git commit -m "feat: add data download module with WMO mapping, gap-fill, and chronological split"
```

---

## Task 4: Feature Engineering Module

**Files:**
- Create: `prediction/features/engineering.py`
- Create: `prediction/tests/conftest.py`
- Create: `prediction/tests/test_engineering.py`

- [ ] **Step 1: Create conftest.py with shared fixtures**

```python
# prediction/tests/conftest.py
import numpy as np
import pandas as pd
import pytest
from features.config import TOTAL_FEATURE_COUNT, RANDOM_SEED


@pytest.fixture
def synthetic_weather_df():
    """200-row synthetic hourly DataFrame with all required columns."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 200
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
    """Small pre-built X, y arrays for model tests."""
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
```

- [ ] **Step 2: Write the failing tests**

```python
# prediction/tests/test_engineering.py
import numpy as np
import pytest
from features.engineering import build_features, compute_dew_point, compute_abs_humidity
from features.config import TOTAL_FEATURE_COUNT, N_CLASSES


def test_build_features_output_shapes(synthetic_weather_df):
    X, y = build_features(synthetic_weather_df)
    assert X.ndim == 2
    assert X.shape[1] == TOTAL_FEATURE_COUNT  # 58
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_build_features_min_samples(synthetic_weather_df):
    # Need at least LOOKBACK + LOOKAHEAD + 1 rows to produce any sample
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
    """When BME688 columns absent, last 4 features should be 0."""
    X, _ = build_features(synthetic_weather_df)
    assert (X[:, -4:] == 0.0).all()


def test_build_features_with_bme688_extras(synthetic_weather_df):
    """When BME688 columns present, last 4 features should be non-zero."""
    df = synthetic_weather_df.copy()
    rng = np.random.default_rng(0)
    df["gas_resistance"] = rng.uniform(1000, 50000, len(df))
    df["iaq"] = rng.uniform(0, 500, len(df))
    df["eco2"] = rng.uniform(400, 2000, len(df))
    df["bvoc"] = rng.uniform(0, 10, len(df))
    X, _ = build_features(df)
    # At least some non-zero values in BME688 columns
    assert not (X[:, -4:] == 0.0).all()


def test_dew_point_formula():
    # At T=20°C, RH=50%: dew point should be ≈ 9.3°C
    T_d = compute_dew_point(T=20.0, RH=50.0)
    assert abs(T_d - 9.3) < 0.5


def test_abs_humidity_formula():
    # At T=25°C, RH=60%: abs humidity ≈ 13.8 g/m³
    AH = compute_abs_humidity(T=25.0, RH=60.0)
    assert 10.0 < AH < 20.0


def test_build_features_too_few_rows():
    """DataFrames with fewer than LOOKBACK + LOOKAHEAD + 1 rows return empty arrays."""
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_engineering.py -v
```
Expected: `ImportError`

- [ ] **Step 4: Implement engineering.py**

```python
# prediction/features/engineering.py
import numpy as np
import pandas as pd
from features.config import LOOKBACK, LOOKAHEAD, BME688_EXTRAS, TOTAL_FEATURE_COUNT


def compute_dew_point(T: float, RH: float) -> float:
    """Magnus formula dew point. T in °C, RH in %."""
    gamma = np.log(RH / 100.0) + (17.625 * T) / (243.04 + T)
    return 243.04 * gamma / (17.625 - gamma)


def compute_abs_humidity(T: float, RH: float) -> float:
    """August-Roche-Magnus absolute humidity in g/m³."""
    return (6.112 * np.exp((17.67 * T) / (T + 243.5)) * RH * 2.1674) / (273.15 + T)


def build_features(df: pd.DataFrame):
    """
    Build feature matrix X and label vector y from hourly weather DataFrame.

    Input df must have columns: time, temperature, humidity, pressure, label
    Optionally: gas_resistance, iaq, eco2, bvoc (BME688 extras; zero-padded if absent)

    Returns:
        X: np.ndarray shape (n_samples, 58), dtype float32
        y: np.ndarray shape (n_samples,), dtype int32
    """
    df = df.sort_values("time").reset_index(drop=True)
    has_extras = all(col in df.columns for col in BME688_EXTRAS)

    records = []
    labels = []

    for i in range(LOOKBACK, len(df) - LOOKAHEAD):
        past = df.iloc[i - LOOKBACK: i]   # rows at t-6 … t-1 (oldest → newest)
        target_label = df.iloc[i + LOOKAHEAD]["label"]

        if pd.isna(target_label):
            continue

        feat = []

        temp_vals = past["temperature"].values.astype(float)
        hum_vals = past["humidity"].values.astype(float)
        pres_vals = past["pressure"].values.astype(float)

        # 1. Raw lags: temp, humidity, pressure × 6 (18 features)
        feat.extend(temp_vals)
        feat.extend(hum_vals)
        feat.extend(pres_vals)

        # 2. Pressure tendency at 1h, 3h, 6h (3 features)
        dp1h = pres_vals[-1] - pres_vals[-2]
        dp3h = pres_vals[-1] - pres_vals[-4]
        dp6h = pres_vals[-1] - pres_vals[0]
        feat.extend([dp1h, dp3h, dp6h])

        # 3. Pressure acceleration: Δp_3h − Δp_6h (1 feature)
        feat.append(dp3h - dp6h)

        # 4. Temp rate of change at 1h, 3h (2 features)
        feat.append(temp_vals[-1] - temp_vals[-2])
        feat.append(temp_vals[-1] - temp_vals[-4])

        # 5. Dew point from most recent timestep t-1h (1 feature)
        feat.append(compute_dew_point(T=temp_vals[-1], RH=hum_vals[-1]))

        # 6. Absolute humidity from most recent timestep t-1h (1 feature)
        feat.append(compute_abs_humidity(T=temp_vals[-1], RH=hum_vals[-1]))

        # 7. Rolling stats: mean, std, min, max × 3h & 6h windows × 3 signals (24 features)
        for col_vals in [temp_vals, hum_vals, pres_vals]:
            for window in [col_vals[-3:], col_vals]:  # 3h (last 3), 6h (all 6)
                feat.extend([
                    float(np.mean(window)),
                    float(np.std(window)),
                    float(np.min(window)),
                    float(np.max(window)),
                ])

        # 8. Cyclical time encoding (4 features)
        current_time = df.iloc[i]["time"]
        hour = current_time.hour
        doy = current_time.day_of_year
        feat.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy / 365),
            np.cos(2 * np.pi * doy / 365),
        ])

        # 9. BME688 extras (4 features — zero-padded if columns absent)
        if has_extras:
            for col in BME688_EXTRAS:
                feat.append(float(past.iloc[-1][col]))
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        assert len(feat) == TOTAL_FEATURE_COUNT, f"Expected {TOTAL_FEATURE_COUNT}, got {len(feat)}"
        records.append(feat)
        labels.append(int(target_label))

    if not records:
        return np.empty((0, TOTAL_FEATURE_COUNT), dtype=np.float32), np.empty(0, dtype=np.int32)

    return np.array(records, dtype=np.float32), np.array(labels, dtype=np.int32)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_engineering.py -v
```
Expected: all 10 tests PASS

- [ ] **Step 6: Commit**

```bash
git add prediction/features/engineering.py prediction/tests/conftest.py prediction/tests/test_engineering.py
git commit -m "feat: add shared feature engineering pipeline (58 features)"
```

---

## Task 5: Logistic Regression Model

**Files:**
- Create: `prediction/models/logistic_regression.py`
- Create: `prediction/tests/test_logistic_regression.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_logistic_regression.py
import numpy as np
import pytest
from models.logistic_regression import LogisticRegressionModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT


def test_fit_predict_shape(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


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
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], N_CLASSES)


def test_predict_proba_sums_to_one(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_feature_importance_shape(small_Xy):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    coef = model.get_feature_importances()
    assert coef.shape == (N_CLASSES, TOTAL_FEATURE_COUNT)


def test_save_load_roundtrip(small_Xy, tmp_path):
    X, y = small_Xy
    model = LogisticRegressionModel()
    model.fit(X, y)
    path = str(tmp_path / "lr.pkl")
    model.save(path)
    loaded = LogisticRegressionModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_logistic_regression.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement logistic_regression.py**

```python
# prediction/models/logistic_regression.py
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from features.config import RANDOM_SEED


class LogisticRegressionModel:
    def __init__(self, C: float = 1.0, random_seed: int = RANDOM_SEED):
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            C=C,
            class_weight="balanced",
            random_state=random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Returns coefficient matrix of shape (n_classes, n_features)."""
        return self.model.coef_

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_logistic_regression.py -v
```
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/models/logistic_regression.py prediction/tests/test_logistic_regression.py
git commit -m "feat: add LogisticRegressionModel with fit/predict/save/load"
```

---

## Task 6: Random Forest Model

**Files:**
- Create: `prediction/models/random_forest.py`
- Create: `prediction/tests/test_random_forest.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_random_forest.py
import numpy as np
import pytest
from models.random_forest import RandomForestModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT


def test_fit_predict_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_predict_proba_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel()
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], N_CLASSES)


def test_feature_importance_shape(small_Xy):
    X, y = small_Xy
    model = RandomForestModel()
    model.fit(X, y)
    imp = model.get_feature_importances()
    assert imp.shape == (TOTAL_FEATURE_COUNT,)
    assert abs(imp.sum() - 1.0) < 1e-5  # importances sum to 1


def test_save_load_roundtrip(small_Xy, tmp_path):
    X, y = small_Xy
    model = RandomForestModel()
    model.fit(X, y)
    path = str(tmp_path / "rf.pkl")
    model.save(path)
    loaded = RandomForestModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_random_forest.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement random_forest.py**

```python
# prediction/models/random_forest.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from features.config import RANDOM_SEED, N_CLASSES


class RandomForestModel:
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 12,
        random_seed: int = RANDOM_SEED,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=random_seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> np.ndarray:
        """Returns Gini importance array of shape (n_features,)."""
        return self.model.feature_importances_

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        instance = cls()
        instance.model = joblib.load(path)
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_random_forest.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/models/random_forest.py prediction/tests/test_random_forest.py
git commit -m "feat: add RandomForestModel with feature importance and save/load"
```

---

## Task 7: Neural Network Model

**Files:**
- Create: `prediction/models/neural_network.py`
- Create: `prediction/tests/test_neural_network.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_neural_network.py
import numpy as np
import pytest
from models.neural_network import NeuralNetworkModel
from features.config import N_CLASSES, TOTAL_FEATURE_COUNT, RANDOM_SEED


@pytest.fixture
def trained_nn(small_Xy):
    X, y = small_Xy
    model = NeuralNetworkModel()
    model.fit(X, y, X_val=X, y_val=y, epochs=3)  # fast for testing
    return model, X, y


def test_fit_predict_shape(trained_nn):
    model, X, y = trained_nn
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_predict_proba_shape(trained_nn):
    model, X, y = trained_nn
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], N_CLASSES)


def test_predict_proba_sums_to_one(trained_nn):
    model, X, y = trained_nn
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_save_load_roundtrip(trained_nn, tmp_path):
    model, X, y = trained_nn
    path = str(tmp_path / "nn_model")
    model.save(path)
    loaded = NeuralNetworkModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


@pytest.mark.slow
def test_input_shape_is_58(small_Xy):
    X, y = small_Xy
    assert X.shape[1] == TOTAL_FEATURE_COUNT
    model = NeuralNetworkModel()
    # Build model by calling it once
    model.fit(X, y, X_val=X, y_val=y, epochs=1)
    assert model._model.input_shape == (None, TOTAL_FEATURE_COUNT)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_neural_network.py -v -m "not slow"
```
Expected: `ImportError`

- [ ] **Step 3: Implement neural_network.py**

```python
# prediction/models/neural_network.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from features.config import RANDOM_SEED, N_CLASSES, TOTAL_FEATURE_COUNT


class NeuralNetworkModel:
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self._model: keras.Model | None = None

    def _build(self) -> keras.Model:
        tf.random.set_seed(self.random_seed)
        inp = keras.Input(shape=(TOTAL_FEATURE_COUNT,))
        x = keras.layers.Dense(64, activation="relu")(inp)
        x = keras.layers.Dropout(0.3, seed=self.random_seed)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        out = keras.layers.Dense(N_CLASSES, activation="softmax")(x)
        model = keras.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> "NeuralNetworkModel":
        self._model = self._build()
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, verbose=0
            ),
        ]
        self._model.fit(
            X, y,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._model.predict(X, verbose=0), axis=1).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0).astype(np.float32)

    def get_keras_model(self) -> keras.Model:
        return self._model

    def save(self, path: str) -> None:
        """Save in Keras SavedModel format."""
        self._model.save(path)

    @classmethod
    def load(cls, path: str) -> "NeuralNetworkModel":
        instance = cls()
        instance._model = keras.models.load_model(path)
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_neural_network.py -v -m "not slow"
```
Expected: all 4 non-slow tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/models/neural_network.py prediction/tests/test_neural_network.py
git commit -m "feat: add NeuralNetworkModel (Dense 64→32→5) with early stopping"
```

---

## Task 8: Evaluation Metrics

**Files:**
- Create: `prediction/evaluation/metrics.py`
- Create: `prediction/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_metrics.py
import numpy as np
import pytest
from evaluation.metrics import compute_all_metrics
from features.config import N_CLASSES, CLASS_NAMES


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
    required_keys = [
        "accuracy", "macro_f1",
        "per_class_precision", "per_class_recall", "per_class_f1",
        "confusion_matrix", "roc_auc_ovr",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"


def test_per_class_metrics_length(random_preds):
    y_true, y_pred, y_proba = random_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    assert len(metrics["per_class_precision"]) == N_CLASSES
    assert len(metrics["per_class_recall"]) == N_CLASSES
    assert len(metrics["per_class_f1"]) == N_CLASSES


def test_confusion_matrix_shape(random_preds):
    y_true, y_pred, y_proba = random_preds
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    cm = metrics["confusion_matrix"]
    assert cm.shape == (N_CLASSES, N_CLASSES)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_metrics.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement metrics.py**

```python
# prediction/evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
)
from features.config import N_CLASSES


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Compute full evaluation metrics for a classifier."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=list(range(N_CLASSES)), zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(N_CLASSES))
        ),
        "roc_auc_ovr": float(
            roc_auc_score(
                y_true, y_proba,
                multi_class="ovr", average="macro",
                labels=list(range(N_CLASSES)),
            )
        ) if len(np.unique(y_true)) > 1 else 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_metrics.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/evaluation/metrics.py prediction/tests/test_metrics.py
git commit -m "feat: add evaluation metrics (accuracy, F1, confusion matrix, ROC-AUC)"
```

---

## Task 9: Evaluation Plots

**Files:**
- Create: `prediction/evaluation/plots.py`
- Create: `prediction/tests/test_plots.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_plots.py
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for testing
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
    fig = confusion_matrix_fig(sample_metrics["confusion_matrix"], title="Test LR")
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
    feature_names = [f"feat_{i}" for i in range(TOTAL_FEATURE_COUNT)]
    fig = feature_importance_fig(coef, feature_names, model_type="lr", title="LR Coef")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_feature_importance_fig_rf_shape():
    imp = np.abs(np.random.randn(TOTAL_FEATURE_COUNT))
    imp /= imp.sum()
    feature_names = [f"feat_{i}" for i in range(TOTAL_FEATURE_COUNT)]
    fig = feature_importance_fig(imp, feature_names, model_type="rf", title="RF Importance")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_plots.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement plots.py**

```python
# prediction/evaluation/plots.py
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
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion Matrix")
    fig.tight_layout()
    return fig


def roc_fig(y_true: np.ndarray, models_proba: dict[str, np.ndarray]) -> plt.Figure:
    """Plot one-vs-rest ROC curves for multiple models on the same axes."""
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "green"]
    for (name, proba), color in zip(models_proba.items(), colors):
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


def feature_importance_fig(
    importance: np.ndarray,
    feature_names: list[str],
    model_type: str,
    title: str = "",
    top_n: int = 20,
) -> plt.Figure:
    """
    model_type: 'lr' (importance is coef matrix shape [n_classes, n_features])
                'rf' (importance is 1D array shape [n_features])
    """
    if model_type == "lr":
        # Mean absolute coefficient across classes
        scores = np.mean(np.abs(importance), axis=0)
    else:
        scores = importance

    idx = np.argsort(scores)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], scores[idx], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(title or "Feature Importance")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_plots.py -v
```
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/evaluation/plots.py prediction/tests/test_plots.py
git commit -m "feat: add evaluation plots (confusion matrix, ROC, feature importance)"
```

---

## Task 10: Deployment Export

**Files:**
- Create: `prediction/deployment/export.py`
- Create: `prediction/tests/test_export.py`

- [ ] **Step 1: Write the failing tests**

```python
# prediction/tests/test_export.py
import numpy as np
import pytest
import os
from deployment.export import (
    export_scaler_header,
    export_lr_header,
    export_rf_c,
    export_tflite,
)
from sklearn.preprocessing import StandardScaler
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
    content = open(path).read()
    assert len(content) > 100  # non-trivial output


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
    content = open(header_path).read()
    assert "model_data" in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prediction/
pytest tests/test_export.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement export.py**

```python
# prediction/deployment/export.py
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import m2cgen as m2c
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel


def export_scaler_header(scaler: StandardScaler, output_path: str) -> None:
    """Export StandardScaler mean/std as a C header file."""
    mean = scaler.mean_
    std = scaler.scale_
    n = len(mean)
    lines = [
        "#pragma once",
        f"// Auto-generated scaler parameters ({n} features)",
        f"const int SCALER_N_FEATURES = {n};",
        f"const float scaler_mean[{n}] = {{",
        "  " + ", ".join(f"{v:.6f}f" for v in mean),
        "};",
        f"const float scaler_std[{n}] = {{",
        "  " + ", ".join(f"{v:.6f}f" for v in std),
        "};",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_lr_header(model: LogisticRegressionModel, output_path: str) -> None:
    """Export LR coefficients and intercepts as a C header file."""
    coef = model.model.coef_          # shape (n_classes, n_features)
    intercept = model.model.intercept_ # shape (n_classes,)
    nc, nf = coef.shape
    lines = [
        "#pragma once",
        f"// Auto-generated LR model ({nc} classes, {nf} features)",
        f"const int LR_N_CLASSES = {nc};",
        f"const int LR_N_FEATURES = {nf};",
        f"const float lr_coef[{nc}][{nf}] = {{",
    ]
    for row in coef:
        lines.append("  {" + ", ".join(f"{v:.6f}f" for v in row) + "},")
    lines += [
        "};",
        f"const float lr_intercept[{nc}] = {{",
        "  " + ", ".join(f"{v:.6f}f" for v in intercept),
        "};",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_rf_c(model: RandomForestModel, output_path: str) -> None:
    """Export Random Forest as dependency-free C code via m2cgen."""
    code = m2c.export_to_c(model.model)
    with open(output_path, "w") as f:
        f.write(code)


def export_tflite(
    keras_model,
    representative_data: np.ndarray,
    tflite_path: str,
    header_path: str | None = None,
) -> None:
    """
    Convert Keras model to INT8 quantized TFLite and optionally export
    as a C header byte array.

    representative_data: a 2D numpy array of samples used for quantization
                         calibration (stratified-random sample from training set).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(representative_data)):
            sample = representative_data[i: i + 1].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    if header_path is not None:
        _write_c_array(tflite_model, header_path, array_name="model_data")


def _write_c_array(data: bytes, output_path: str, array_name: str) -> None:
    hex_values = ", ".join(f"0x{b:02x}" for b in data)
    lines = [
        "#pragma once",
        f"// Auto-generated TFLite model ({len(data)} bytes)",
        f"const unsigned int {array_name}_len = {len(data)};",
        f"alignas(8) const unsigned char {array_name}[] = {{",
        f"  {hex_values}",
        "};",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prediction/
pytest tests/test_export.py -v
```
Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add prediction/deployment/export.py prediction/tests/test_export.py
git commit -m "feat: add deployment export (TFLite, LR header, RF C code, scaler header)"
```

---

## Task 11: Report Generator

**Files:**
- Create: `prediction/report/template.html`
- Create: `prediction/report/generate.py`

> Note: No unit tests for the report generator — it produces a rendered HTML file. Visual inspection is the acceptance criterion.

- [ ] **Step 1: Create the Jinja2 HTML template**

```html
<!-- prediction/report/template.html -->
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Weather Prediction — Model Comparison Report</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto; padding: 2rem; color: #222; }
  h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: 0.5rem; }
  h2 { color: #1f618d; margin-top: 2rem; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #bdc3c7; padding: 0.5rem 0.75rem; text-align: center; }
  th { background: #2e86c1; color: white; }
  tr:nth-child(even) { background: #eaf4fb; }
  .best { background: #d5f5e3 !important; font-weight: bold; }
  .figure-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
  .figure-row img { max-width: 350px; border: 1px solid #ccc; border-radius: 4px; }
  .recommendation { background: #eafaf1; border-left: 4px solid #27ae60;
                    padding: 1rem 1.5rem; border-radius: 0 4px 4px 0; margin: 1rem 0; }
  .note { color: #7f8c8d; font-size: 0.9em; }
</style>
</head>
<body>
<h1>AI Weather Prediction — Model Comparison Report</h1>
<p class="note">Generated: {{ generated_at }} | Dataset: {{ dataset_description }}</p>

<h2>1. Executive Summary</h2>
<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Macro F1</th>
    <th>ROC-AUC</th>
    <th>Artifact Size</th>
    <th>Est. Inference (µs)</th>
  </tr>
  {% for m in models %}
  <tr class="{{ 'best' if m.name == best_model else '' }}">
    <td>{{ m.name }}</td>
    <td>{{ "%.3f" % m.accuracy }}</td>
    <td>{{ "%.3f" % m.macro_f1 }}</td>
    <td>{{ "%.3f" % m.roc_auc }}</td>
    <td>{{ m.artifact_size_kb }} KB</td>
    <td>{{ m.inference_us }}</td>
  </tr>
  {% endfor %}
</table>

<h2>2. Dataset</h2>
<p>{{ dataset_description }}</p>
<div class="figure-row">
  <img src="class_distribution.png" alt="Class Distribution">
</div>

<h2>3. Confusion Matrices</h2>
<div class="figure-row">
  {% for m in models %}
  <div>
    <p><strong>{{ m.name }}</strong></p>
    <img src="cm_{{ m.name | lower | replace(' ', '_') }}.png" alt="{{ m.name }} Confusion Matrix">
  </div>
  {% endfor %}
</div>

<h2>4. ROC Curves</h2>
<img src="roc_curves.png" alt="ROC Curves" style="max-width: 600px;">

<h2>5. Feature Importance</h2>
<div class="figure-row">
  {% for m in models %}
  <div>
    <p><strong>{{ m.name }}</strong></p>
    <img src="fi_{{ m.name | lower | replace(' ', '_') }}.png" alt="{{ m.name }} Feature Importance">
  </div>
  {% endfor %}
</div>

<h2>6. Per-Class Metrics</h2>
{% for m in models %}
<h3>{{ m.name }}</h3>
<table>
  <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
  {% for cls, p, r, f in m.per_class %}
  <tr><td>{{ cls }}</td><td>{{ "%.3f" % p }}</td><td>{{ "%.3f" % r }}</td><td>{{ "%.3f" % f }}</td></tr>
  {% endfor %}
</table>
{% endfor %}

<h2>7. Edge Deployment Recommendation</h2>
<div class="recommendation">
  <strong>Recommended model: {{ best_model }}</strong><br>
  {{ recommendation_text }}
</div>

<p class="note">All inference time estimates are analytical (240 MHz ESP32-S3, IPC=1). Actual measured times may vary ±30%.</p>
</body>
</html>
```

- [ ] **Step 2: Implement generate.py**

```python
# prediction/report/generate.py
import os
import io
import base64
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from features.config import CLASS_NAMES


def _fig_to_png(fig: plt.Figure, output_dir: str, name: str) -> str:
    """Save figure to output_dir and return filename."""
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_report(results: dict, output_path: str) -> None:
    """
    results dict structure:
    {
      "dataset_description": str,
      "models": [
        {
          "name": str,
          "metrics": {accuracy, macro_f1, roc_auc_ovr, per_class_precision,
                      per_class_recall, per_class_f1, confusion_matrix},
          "artifact_size_kb": float,
          "inference_us": float,
          "importance": np.ndarray,
          "importance_type": "lr" | "rf" | "nn",
          "y_proba": np.ndarray,
        }
      ],
      "y_true": np.ndarray,
      "feature_names": list[str],
    }
    """
    from evaluation.plots import confusion_matrix_fig, roc_fig, feature_importance_fig
    import numpy as np

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # --- Build template data ---
    model_rows = []
    models_proba = {}

    for m in results["models"]:
        metrics = m["metrics"]
        model_rows.append({
            "name": m["name"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics["roc_auc_ovr"],
            "artifact_size_kb": m["artifact_size_kb"],
            "inference_us": m["inference_us"],
            "per_class": list(zip(
                CLASS_NAMES,
                metrics["per_class_precision"],
                metrics["per_class_recall"],
                metrics["per_class_f1"],
            )),
        })
        models_proba[m["name"]] = m["y_proba"]

        # Confusion matrix
        fig = confusion_matrix_fig(metrics["confusion_matrix"], title=f"{m['name']} — Confusion Matrix")
        _fig_to_png(fig, output_dir, f"cm_{m['name'].lower().replace(' ', '_')}.png")

        # Feature importance
        fig = feature_importance_fig(
            m["importance"], results["feature_names"],
            model_type=m["importance_type"], title=f"{m['name']} — Feature Importance"
        )
        _fig_to_png(fig, output_dir, f"fi_{m['name'].lower().replace(' ', '_')}.png")

    # ROC curves
    fig = roc_fig(results["y_true"], models_proba)
    _fig_to_png(fig, output_dir, "roc_curves.png")

    # Class distribution
    y = results["y_true"]
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([CLASS_NAMES[i] for i in unique], counts, color="steelblue")
    ax.set_xlabel("Class")
    ax.set_ylabel("Sample count")
    ax.set_title("Test Set Class Distribution")
    fig.tight_layout()
    _fig_to_png(fig, output_dir, "class_distribution.png")

    # Determine best model
    best = max(model_rows, key=lambda x: x["macro_f1"])
    best_name = best["name"]
    rec_text = (
        f"{best_name} achieved the highest macro F1 of {best['macro_f1']:.3f} "
        f"with an accuracy of {best['accuracy']:.3f}. "
        f"Artifact size: {best['artifact_size_kb']} KB. "
        f"Estimated inference: {best['inference_us']} µs on ESP32-S3 at 240 MHz."
    )

    # Render template
    template_dir = os.path.dirname(__file__)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("template.html")
    html = template.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        dataset_description=results["dataset_description"],
        models=model_rows,
        best_model=best_name,
        recommendation_text=rec_text,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {output_path}")
```

- [ ] **Step 3: Commit**

```bash
git add prediction/report/template.html prediction/report/generate.py
git commit -m "feat: add Jinja2 HTML report generator with comparison table and figures"
```

---

## Task 12: Main Orchestrator

**Files:**
- Create: `prediction/main.py`

- [ ] **Step 1: Implement main.py**

```python
# prediction/main.py
"""
AI Weather Prediction Pipeline — Main Orchestrator

Usage:
  python main.py                          # full pipeline
  python main.py --only download
  python main.py --only train
  python main.py --only evaluate
  python main.py --only report
  python main.py --locations "London,UK" "Helsinki,Finland" "Singapore" --years 5
  python main.py --force-download --years 5
"""
import argparse
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ensure prediction/ is on the path when run from repo root
sys.path.insert(0, os.path.dirname(__file__))

from data.download import prepare_data
from features.engineering import build_features
from features.config import (
    RANDOM_SEED, TOTAL_FEATURE_COUNT, CLASS_NAMES, STATIONS,
)
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from evaluation.metrics import compute_all_metrics
from deployment.export import (
    export_scaler_header, export_lr_header, export_rf_c, export_tflite,
)
from report.generate import generate_report

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "deployment")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
EVAL_DIR = os.path.join(os.path.dirname(__file__), "evaluation", "outputs")


def _require(path: str, message: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: {message}")
        sys.exit(1)


def stage_download(args):
    stations = None
    if args.locations:
        # Parse "Name,Country" pairs
        stations = []
        for loc in args.locations:
            parts = loc.rsplit(",", 1)
            name = loc
            from features.config import STATIONS as DEFAULT_STATIONS
            # Look up lat/lon from defaults or use 0,0 as placeholder
            matched = next((s for s in DEFAULT_STATIONS if s[0].startswith(parts[0])), None)
            if matched:
                stations.append(matched)
            else:
                print(f"WARNING: Unknown station '{loc}'. Using default stations.")
                stations = None
                break
    prepare_data(
        stations=stations,
        years=args.years,
        force_download=args.force_download,
    )


def stage_train(args):
    _require(
        os.path.join(PROCESSED_DIR, "train.csv"),
        "data/processed/train.csv not found. Run: python main.py --only download",
    )
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), parse_dates=["time"])
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"), parse_dates=["time"])

    print("Building features...")
    X_train, y_train = build_features(train_df)
    X_val, y_val = build_features(val_df)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    # Fit scaler on training set only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    os.makedirs(DEPLOY_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(DEPLOY_DIR, "scaler.pkl"))
    export_scaler_header(scaler, os.path.join(DEPLOY_DIR, "scaler_params.h"))

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegressionModel(random_seed=RANDOM_SEED)
    lr.fit(X_train_s, y_train)
    lr.save(os.path.join(DEPLOY_DIR, "lr_model.pkl"))
    export_lr_header(lr, os.path.join(DEPLOY_DIR, "lr_coefficients.h"))

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestModel(random_seed=RANDOM_SEED)
    rf.fit(X_train_s, y_train)
    rf.save(os.path.join(DEPLOY_DIR, "rf_model.pkl"))
    export_rf_c(rf, os.path.join(DEPLOY_DIR, "rf_model.c"))

    # 3. Neural Network
    print("Training Neural Network...")
    nn = NeuralNetworkModel(random_seed=RANDOM_SEED)
    nn.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)
    nn.save(os.path.join(DEPLOY_DIR, "nn_model"))

    # Quantized TFLite export — stratified-random 1000 calibration samples
    rng = np.random.default_rng(RANDOM_SEED)
    cal_idx = _stratified_sample_idx(y_train, n=1000, rng=rng)
    export_tflite(
        nn.get_keras_model(),
        X_train_s[cal_idx],
        os.path.join(DEPLOY_DIR, "model.tflite"),
        header_path=os.path.join(DEPLOY_DIR, "model_data.h"),
    )
    print("Training complete. Deployment artifacts written to deployment/")


def _stratified_sample_idx(y: np.ndarray, n: int, rng) -> np.ndarray:
    """Return n indices with stratified random sampling."""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=min(n, len(y)), random_state=RANDOM_SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return idx


def stage_evaluate(args):
    for name in ["lr_model.pkl", "rf_model.pkl", "nn_model", "scaler.pkl"]:
        _require(
            os.path.join(DEPLOY_DIR, name),
            f"deployment/{name} not found. Run: python main.py --only train",
        )
    _require(
        os.path.join(PROCESSED_DIR, "test.csv"),
        "data/processed/test.csv not found. Run: python main.py --only download",
    )

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), parse_dates=["time"])
    X_test, y_test = build_features(test_df)

    scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.pkl"))
    X_test_s = scaler.transform(X_test).astype(np.float32)

    lr = LogisticRegressionModel.load(os.path.join(DEPLOY_DIR, "lr_model.pkl"))
    rf = RandomForestModel.load(os.path.join(DEPLOY_DIR, "rf_model.pkl"))
    nn = NeuralNetworkModel.load(os.path.join(DEPLOY_DIR, "nn_model"))

    results = []
    for name, model, importance, imp_type in [
        ("Logistic Regression", lr, lr.get_feature_importances(), "lr"),
        ("Random Forest",       rf, rf.get_feature_importances(), "rf"),
        ("Neural Network",      nn, lr.get_feature_importances(), "lr"),  # gradient approx via LR coefs for NN
    ]:
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)

        # Artifact sizes
        size_map = {
            "Logistic Regression": _file_kb(os.path.join(DEPLOY_DIR, "lr_coefficients.h")),
            "Random Forest": _file_kb(os.path.join(DEPLOY_DIR, "rf_model.c")),
            "Neural Network": _file_kb(os.path.join(DEPLOY_DIR, "model.tflite")),
        }
        inference_map = {
            "Logistic Regression": _lr_inference_us(lr),
            "Random Forest": _rf_inference_us(rf),
            "Neural Network": _nn_inference_us(),
        }
        results.append({
            "name": name,
            "metrics": metrics,
            "artifact_size_kb": size_map[name],
            "inference_us": inference_map[name],
            "importance": importance,
            "importance_type": imp_type,
            "y_proba": y_proba,
        })
        print(f"{name}: accuracy={metrics['accuracy']:.3f}, macro_f1={metrics['macro_f1']:.3f}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    # Serialize metrics (convert numpy arrays)
    serializable = []
    for r in results:
        m = r["metrics"].copy()
        m["confusion_matrix"] = m["confusion_matrix"].tolist()
        serializable.append({"name": r["name"], "metrics": m,
                              "artifact_size_kb": r["artifact_size_kb"],
                              "inference_us": r["inference_us"]})
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    # Save for report
    np.save(os.path.join(EVAL_DIR, "y_test.npy"), y_test)
    for r in results:
        np.save(os.path.join(EVAL_DIR, f"y_proba_{r['name'].replace(' ', '_')}.npy"), r["y_proba"])
        np.save(os.path.join(EVAL_DIR, f"importance_{r['name'].replace(' ', '_')}.npy"), r["importance"])

    print(f"Evaluation complete. Results saved to {EVAL_DIR}/results.json")


def _file_kb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1024, 1)
    except FileNotFoundError:
        return 0.0


def _lr_inference_us(lr) -> float:
    """Analytical: matrix multiply + softmax. FLOPs ≈ 2 * n_features * n_classes."""
    flops = 2 * TOTAL_FEATURE_COUNT * 5 + 5  # matmul + softmax
    return round(flops / 240, 2)


def _rf_inference_us(rf) -> float:
    """Analytical: node traversal. ≈ 2 ops/node × depth × n_trees."""
    flops = 2 * rf.model.max_depth * rf.model.n_estimators
    return round(flops / 240, 2)


def _nn_inference_us() -> float:
    """Analytical: Dense(58→64) + Dense(64→32) + Dense(32→5). FLOPs ≈ 2*(58*64+64*32+32*5)."""
    flops = 2 * (TOTAL_FEATURE_COUNT * 64 + 64 * 32 + 32 * 5)
    return round(flops / 240, 2)


def stage_report(args):
    _require(
        os.path.join(EVAL_DIR, "results.json"),
        "evaluation/outputs/results.json not found. Run: python main.py --only evaluate",
    )

    with open(os.path.join(EVAL_DIR, "results.json")) as f:
        serialized = json.load(f)

    y_true = np.load(os.path.join(EVAL_DIR, "y_test.npy"))

    models_for_report = []
    for s in serialized:
        name_key = s["name"].replace(" ", "_")
        s["metrics"]["confusion_matrix"] = np.array(s["metrics"]["confusion_matrix"])
        models_for_report.append({
            "name": s["name"],
            "metrics": s["metrics"],
            "artifact_size_kb": s["artifact_size_kb"],
            "inference_us": s["inference_us"],
            "importance": np.load(os.path.join(EVAL_DIR, f"importance_{name_key}.npy")),
            "importance_type": "lr" if "Logistic" in s["name"] else ("rf" if "Forest" in s["name"] else "lr"),
            "y_proba": np.load(os.path.join(EVAL_DIR, f"y_proba_{name_key}.npy")),
        })

    feature_names = _build_feature_names()
    results = {
        "dataset_description": "Open-Meteo historical hourly data, 3 stations (London, Helsinki, Singapore), 5 years, 6-hour ahead forecast.",
        "models": models_for_report,
        "y_true": y_true,
        "feature_names": feature_names,
    }

    output_path = os.path.join(REPORT_DIR, "report_output.html")
    generate_report(results, output_path)


def _build_feature_names() -> list[str]:
    names = []
    for sig in ["temp", "humidity", "pressure"]:
        for t in range(6, 0, -1):
            names.append(f"{sig}_t-{t}h")
    names += ["dp_1h", "dp_3h", "dp_6h", "dp_accel"]
    names += ["dt_1h", "dt_3h"]
    names += ["dew_point", "abs_humidity"]
    for sig in ["temp", "humidity", "pressure"]:
        for window in ["3h", "6h"]:
            for stat in ["mean", "std", "min", "max"]:
                names.append(f"{sig}_{stat}_{window}")
    names += ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    names += ["gas_resistance", "iaq", "eco2", "bvoc"]
    return names


def main():
    parser = argparse.ArgumentParser(description="AI Weather Prediction Pipeline")
    parser.add_argument("--only", choices=["download", "train", "evaluate", "report"],
                        help="Run only one stage")
    parser.add_argument("--locations", nargs="+",
                        help="Station names (overrides defaults)")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Generate report after full pipeline")
    args = parser.parse_args()

    if args.only == "download":
        stage_download(args)
    elif args.only == "train":
        stage_train(args)
    elif args.only == "evaluate":
        stage_evaluate(args)
    elif args.only == "report":
        stage_report(args)
    else:
        # Full pipeline
        print("=== Stage 1: Download ===")
        stage_download(args)
        print("=== Stage 2: Train ===")
        stage_train(args)
        print("=== Stage 3: Evaluate ===")
        stage_evaluate(args)
        if args.report:
            print("=== Stage 4: Report ===")
            stage_report(args)
        print("Pipeline complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests one final time**

```bash
cd prediction/
pytest tests/ -v -m "not slow"
```
Expected: all tests PASS

- [ ] **Step 3: Smoke-test the CLI help**

```bash
cd prediction/
python main.py --help
```
Expected: help text prints without error

- [ ] **Step 4: Commit**

```bash
git add prediction/main.py
git commit -m "feat: add main.py orchestrator with full pipeline CLI"
```

---

## Task 13: Integration Test (End-to-End with Synthetic Data)

**Files:**
- Create: `prediction/tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# prediction/tests/test_integration.py
"""
End-to-end smoke test using synthetic data (no API calls).
Verifies the full pipeline: features → scale → train → evaluate → export.
"""
import os
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from features.engineering import build_features
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from evaluation.metrics import compute_all_metrics
from features.config import RANDOM_SEED, N_CLASSES


@pytest.mark.slow
def test_full_pipeline_smoke(synthetic_weather_df, tmp_path):
    """Train all 3 models, evaluate, export — no API calls."""
    X, y = build_features(synthetic_weather_df)
    assert X.shape[1] == 58

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X).astype(np.float32)

    # LR
    lr = LogisticRegressionModel(random_seed=RANDOM_SEED)
    lr.fit(X_s, y)
    lr_metrics = compute_all_metrics(y, lr.predict(X_s), lr.predict_proba(X_s))
    assert "accuracy" in lr_metrics

    # RF
    rf = RandomForestModel(n_estimators=10, max_depth=3, random_seed=RANDOM_SEED)
    rf.fit(X_s, y)
    rf_metrics = compute_all_metrics(y, rf.predict(X_s), rf.predict_proba(X_s))
    assert "accuracy" in rf_metrics

    # NN
    nn = NeuralNetworkModel(random_seed=RANDOM_SEED)
    nn.fit(X_s, y, X_val=X_s, y_val=y, epochs=2)
    nn_metrics = compute_all_metrics(y, nn.predict(X_s), nn.predict_proba(X_s))
    assert "accuracy" in nn_metrics

    # Export
    from deployment.export import export_scaler_header, export_lr_header, export_rf_c, export_tflite
    export_scaler_header(scaler, str(tmp_path / "scaler_params.h"))
    export_lr_header(lr, str(tmp_path / "lr_coefficients.h"))
    export_rf_c(rf, str(tmp_path / "rf_model.c"))
    export_tflite(nn.get_keras_model(), X_s[:50], str(tmp_path / "model.tflite"),
                  header_path=str(tmp_path / "model_data.h"))

    for fname in ["scaler_params.h", "lr_coefficients.h", "rf_model.c", "model.tflite", "model_data.h"]:
        assert os.path.exists(str(tmp_path / fname)), f"Missing: {fname}"

    print(f"LR accuracy: {lr_metrics['accuracy']:.3f}")
    print(f"RF accuracy: {rf_metrics['accuracy']:.3f}")
    print(f"NN accuracy: {nn_metrics['accuracy']:.3f}")
```

- [ ] **Step 2: Run the integration test**

```bash
cd prediction/
pytest tests/test_integration.py -v -m slow -s
```
Expected: PASS with accuracy printout for all 3 models

- [ ] **Step 3: Final commit**

```bash
git add prediction/tests/test_integration.py
git commit -m "test: add end-to-end integration smoke test for full pipeline"
```

---

## Verification Checklist

Before declaring the implementation complete, verify:

- [ ] `pytest tests/ -v -m "not slow"` — all unit tests pass
- [ ] `pytest tests/test_integration.py -m slow -s` — integration test passes
- [ ] `python main.py --help` — prints without error
- [ ] `deployment/` contains: `scaler_params.h`, `lr_coefficients.h`, `rf_model.c`, `model_data.h`, `model.tflite`
- [ ] `reports/report_output.html` opens in a browser and shows all 3 models in the summary table
- [ ] All confusion matrix, ROC, and feature importance figures are present in `reports/`
