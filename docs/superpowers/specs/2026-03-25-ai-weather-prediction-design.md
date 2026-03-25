# AI Weather Prediction System — Design Spec
**Date:** 2026-03-25
**Project:** AI-Weather-Display-LilyGo
**Status:** Approved (rev 2)

---

## 1. Overview

An AI weather prediction system for edge devices that uses a **Bosch BME688** sensor as its live data source. Three machine learning models — Logistic Regression, Random Forest, and Neural Network — are trained on a public open-source dataset (Open-Meteo), share an identical feature extraction pipeline, and are evaluated comparatively. All three deployment artifacts are generated. A final HTML report compares all three models and recommends the best one for edge deployment based on accuracy vs. size tradeoff.

### Goals
- Predict 6-hour-ahead weather condition: **Sunny / Cloudy / Rainy / Stormy / Snowy**
- Train offline in Python (scikit-learn + TensorFlow/Keras)
- Produce deployment artifacts for all three models targeting LilyGo S3 (ESP32-S3)
- Produce an auto-generated comparative HTML report with a deployment recommendation

### Non-Goals
- Real-time retraining on-device
- Multi-day forecasting
- Server/cloud inference

---

## 2. Sensor — Bosch BME688

The BME688 provides the following live readings used at inference time:

| Signal | Unit | Role |
|--------|------|------|
| Temperature | °C | Primary feature |
| Relative Humidity | % | Primary feature |
| Barometric Pressure | hPa | Primary feature (most predictive) |
| Gas Resistance | Ω | Optional — not in training data |
| IAQ Index | 0–500 | Optional — not in training data |
| eCO2 | ppm equivalent | Optional — not in training data |
| Breath VOC | ppm | Optional — not in training data |

**Feature gap note:** Public training datasets do not include gas/IAQ/eCO2/bVOC signals. The training pipeline uses temperature, humidity, and pressure as primary inputs. The feature extraction module accepts BME688 extras as optional columns (zero-padded when absent) so the live inference pipeline can include them. **The Neural Network is always trained with exactly 58 features** (54 core + 4 BME688 extras zero-padded during training on public data) so the inference input shape is stable.

---

## 3. Dataset

**Source:** [Open-Meteo](https://open-meteo.com/) Historical Weather API
**Client library:** `openmeteo-requests` + `requests-cache`
**No API key required.** Fully open-source.

### 3.1 Fields Fetched (hourly)
- `temperature_2m` → Temperature (°C)
- `relative_humidity_2m` → Humidity (%)
- `pressure_msl` → Pressure (hPa)
- `weather_code` → WMO code → target label

### 3.2 Training Stations (hardcoded defaults)
Three geographically diverse stations to ensure Snow and Stormy classes are well-represented:

| Station | Location | Climate |
|---------|----------|---------|
| `London, UK` | 51.5°N, 0.1°W | Temperate |
| `Helsinki, Finland` | 60.2°N, 25.0°E | Cold / Snowy |
| `Singapore` | 1.4°N, 103.8°E | Tropical / Stormy |

The CLI `--locations` flag overrides these defaults (see Section 9).

### 3.3 WMO Weather Code → Label Mapping

| WMO Codes | Label | Int |
|-----------|-------|-----|
| 0, 1 | Sunny | 0 |
| 2, 3, 45, 46, 47, 48 | Cloudy | 1 |
| 51, 52, 53, 54, 55, 56, 57, 61, 62, 63, 64, 65, 66, 67, 80, 81, 82 | Rainy | 2 |
| 95, 96, 97, 98, 99 | Stormy | 3 |
| 71, 72, 73, 74, 75, 76, 77, 85, 86 | Snowy | 4 |
| All other codes (4–44, 49–50, 68–70, 78–79, 83–84, 87–94) | **Dropped** | — |

Rows with unmapped WMO codes are **dropped from the dataset**. These codes represent rare transitional/obscure phenomena (e.g., dust devils, volcanic ash) that are not present in the three selected stations. Expected drop rate: < 0.5% of rows.

### 3.4 Train / Val / Test Split
Split is **strictly chronological** — no shuffling, no stratification — to prevent temporal leakage:
- Train: first 70% of the combined timeline
- Val: next 15%
- Test: final 15% (most recent data)

Data from all three stations is merged and sorted by timestamp before splitting. Each station contributes to all splits proportionally by its data volume.

**Data volume:** ~5 years × 3 stations ≈ 130,000 hourly samples after dropping unmapped codes.

### 3.5 Missing Data Handling
Open-Meteo occasionally returns `NaN` for individual hourly readings:
1. **Forward-fill** gaps of ≤ 2 consecutive hours (short outages / interpolation artifacts).
2. **Drop** any remaining rows with `NaN` in temperature, humidity, or pressure after forward-fill.
3. **Drop** any sliding window (sample) where one or more of its 6 input timesteps was dropped in step 2.

This policy is applied before feature engineering. Expected effective loss: < 1% of samples.

### 3.6 Caching
- Raw API responses are cached to `data/cache/<station_slug>_<year_range>.csv`.
- Cache is considered valid indefinitely (historical data does not change).
- Running `--only download` skips fetch if the cache file exists. Use `--force-download` to re-fetch.
- If a fetch fails mid-way, the partial file is deleted and the next run re-fetches from scratch.

---

## 4. Shared Feature Extraction

All three models import from `features/engineering.py` exclusively. A single `FEATURE_COLUMNS` constant in `features/config.py` governs the feature list — changing it once changes it everywhere.

**Global seed:** `RANDOM_SEED = 42` is defined in `features/config.py` and applied to all stochastic operations (train/val/test split, RF, NN weight init, Dropout).

**Input window:** The last 6 hours of readings (t-1h through t-6h) used to predict the condition at t+6h.

**Scaling:** A `sklearn.preprocessing.StandardScaler` is fit on the training set only and applied to val and test sets. The fitted scaler is serialized to `deployment/scaler.pkl`. At edge inference time, raw BME688 readings must be scaled using the same parameters (mean/std stored as C arrays in `deployment/scaler_params.h`).

### 4.1 Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| Raw lags | temp, humidity, pressure × 6 timesteps | 18 |
| Pressure tendency | Δpressure at 1h, 3h, 6h intervals | 3 |
| Pressure acceleration | 2nd derivative of pressure (Δp_3h − Δp_6h) | 1 |
| Temp rate of change | Δtemp at 1h, 3h intervals | 2 |
| Dew point | Magnus formula (see §4.2) | 1 |
| Absolute humidity | August-Roche-Magnus formula (see §4.2) | 1 |
| Rolling stats | mean, std, min, max × 3h & 6h windows × 3 signals | 24 |
| Cyclical time | sin/cos of hour-of-day, sin/cos of day-of-year | 4 |
| BME688 extras | gas resistance, IAQ, eCO2, bVOC (zero-padded if absent) | 4 |
| **Total** | | **58** |

Rolling stats are computed over the same 6-step input window (3h rolling = most recent 3 of the 6 readings; 6h rolling = all 6 readings). These are aggregations of the raw lag values and are distinct features from the individual lags.

### 4.2 Derived Feature Formulas

**Dew point (Magnus formula):**
```
γ(T, RH) = ln(RH/100) + (17.625 × T) / (243.04 + T)
T_d = 243.04 × γ / (17.625 - γ)
```
where T is in °C and RH is in %.
Both formulas are computed from the **most recent timestep only (t-1h)**, i.e., the last entry in the 6-step input window. The same choice must be made in the embedded C inference code to avoid training/inference skew.

**Absolute humidity (August-Roche-Magnus, g/m³):**
```
AH = (6.112 × exp((17.67 × T) / (T + 243.5)) × RH × 2.1674) / (273.15 + T)
```

### 4.3 Key Engineering Notes
- **Cyclical encoding** (`sin`/`cos`) for time features prevents ordinal distance artifacts (23:00 and 00:00 are adjacent, not distant).
- **Pressure tendency** and **pressure acceleration** are among the most predictive features for short-horizon forecasting — they distinguish a steady low from a developing storm.
- **BME688 extras** are zero-padded during training on public data. At live inference, actual sensor readings replace the zeros. This trains the model to treat zero gas/IAQ as "no signal available."

---

## 5. Models

### 5.1 Logistic Regression (Linear Baseline)

> Note: "Linear regression" for classification = **Logistic Regression** — a linear model with softmax output. This is the standard linear baseline in multi-class classification studies.

- **Type:** Multinomial Logistic Regression
- **Solver:** `lbfgs`, `max_iter=1000`
- **Regularization:** L2 (Ridge), `C=1.0` (tunable via config)
- **Class weights:** `balanced`
- **Random state:** `RANDOM_SEED`
- **Interpretability:** Coefficient matrix (shape 5×58) exported for per-class feature importance
- **Edge deployment:** Coefficients serialized as JSON → `deployment/lr_coefficients.h` (C float array). Inference = matrix multiply + softmax, ~microseconds on ESP32-S3.

### 5.2 Random Forest Classifier

- **Trees:** 200 estimators
- **Max depth:** 12 (limits `m2cgen` C output to approximately 30–50 KB; validated at this depth/tree-count combination)
- **Class weights:** `balanced_subsample`
- **Random state:** `RANDOM_SEED`
- **Feature importance:** Gini-based importances (shape 58) exported for report
- **Edge deployment:** Converted to dependency-free C inference code via `m2cgen`. Output file: `deployment/rf_model.c`. Target size: < 50 KB.

### 5.3 Neural Network (TFLite)

- **Architecture:** `Input(58) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(5, Softmax)`
- **Optimizer:** Adam, lr=0.001
- **Loss:** Categorical crossentropy
- **Random seed:** `tf.random.set_seed(RANDOM_SEED)`
- **Training:** Early stopping on `val_loss` (patience=10), `ReduceLROnPlateau` (factor=0.5, patience=5)
- **Quantization:** INT8 post-training quantization using 1000 **stratified-random** samples drawn from the training set (to avoid seasonal bias from chronological selection). Target model size: ~10–20 KB.
- **Edge deployment:** `.tflite` → `deployment/model_data.h` (C array via `xxd`). Run via TFLite Micro runtime on ESP32-S3.
- **BME688 optional features:** Input is always shape `(58,)`. At live inference, BME688 extras replace the zero-padded positions.

---

## 6. Evaluation

All metrics computed identically for all 3 models on the held-out test set.

| Metric | Details |
|--------|---------|
| Accuracy | Overall |
| Macro F1 | Fair across imbalanced classes |
| Per-class Precision / Recall / F1 | Per label (5 classes) |
| Confusion matrix | 5×5 heatmap |
| ROC-AUC (one-vs-rest) | Probabilistic discrimination |
| Model artifact size | Serialized deployment file size in KB |
| Inference time estimate | Analytical: FLOPs / (240 MHz × IPC), IPC=1 for ESP32-S3. For RF: decision node count × 2 ops / 240 MHz. Reported in microseconds. |

**Inference time methodology:** All estimates are analytical (static analysis of the model graph), not measured on hardware. Clock assumption: 240 MHz, IPC = 1. This gives a conservative lower-bound estimate. Actual measured time will vary by ±30% depending on cache, pipeline, and data alignment.

---

## 7. Auto-Generated HTML Report

`report/generate.py` produces `reports/report_output.html` (in `reports/`, not `docs/`, so it can be gitignored cleanly) containing:

1. **Executive summary table** — all 3 models side-by-side on all metrics
2. **Dataset description** — class distribution chart, station locations, data volume
3. **Feature engineering** — feature list, feature importance per model (coefficient heatmap for LR, bar chart for RF, gradient-based for NN)
4. **Confusion matrices** — 3 heatmaps, one per model
5. **ROC curves** — all models overlaid on one chart per class
6. **Edge deployment section** — model artifact sizes, quantization results, analytical inference time estimates
7. **Deployment recommendation** — the recommended model for LilyGo deployment, with reasoning (accuracy vs. size vs. latency tradeoff)

---

## 8. Project Structure

```
ai-weather-prediction/
├── data/
│   ├── download.py          ← Open-Meteo fetch; caches to data/cache/
│   ├── cache/               ← raw CSV files per station (gitignored)
│   └── processed/           ← train.csv, val.csv, test.csv (gitignored)
├── features/
│   ├── __init__.py
│   ├── engineering.py       ← ALL feature engineering (shared by all models)
│   └── config.py            ← FEATURE_COLUMNS, LABEL_MAP, RANDOM_SEED constants
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── neural_network.py
├── evaluation/
│   ├── metrics.py           ← compute_all_metrics(y_true, y_pred, y_proba)
│   └── plots.py             ← confusion_matrix_plot(), roc_plot(), feature_importance_plot()
├── report/
│   ├── generate.py          ← assembles HTML report from evaluation outputs
│   └── template.html        ← Jinja2 template
├── deployment/
│   ├── export_tflite.py     ← TFLite conversion + INT8 quantization
│   ├── scaler.pkl           ← fitted StandardScaler (gitignored)
│   ├── scaler_params.h      ← scaler mean/std as C arrays
│   ├── lr_coefficients.h    ← LR model as C float array
│   ├── rf_model.c           ← RF model as C inference code (m2cgen)
│   └── model_data.h         ← TFLite model as C byte array
├── reports/
│   └── report_output.html   ← generated report (gitignored)
├── main.py                  ← orchestrator (single entry point)
├── requirements.txt
└── docs/
    └── superpowers/specs/
        └── 2026-03-25-ai-weather-prediction-design.md
```

---

## 9. CLI Entry Point

```bash
# Full pipeline: download → features → train all 3 → evaluate → report
python main.py --years 5 --report

# Override default stations
python main.py --locations "London,UK" "Helsinki,Finland" "Singapore" --years 5 --report

# Individual stages (each checks that prerequisites exist)
python main.py --only download
python main.py --only train      # trains LR, RF, NN sequentially in that order
python main.py --only evaluate
python main.py --only report

# Force re-download even if cache exists
python main.py --force-download --years 5
```

**Orchestration order for `--only train`:** LR → RF → NN (sequential). All three models are always trained. Training does not proceed if `data/processed/` is absent (fails with a clear error message directing the user to run `--only download` first).

**Prerequisite checks for all stages:**
- `--only train` → requires `data/processed/` to exist
- `--only evaluate` → requires trained model files in `deployment/` (lr, rf, nn artifacts)
- `--only report` → requires evaluation outputs in `evaluation/` (metrics JSON + plot files)

Each stage fails with a clear error message if prerequisites are missing.

---

## 10. Dependencies

```
openmeteo-requests      # Open-Meteo API client
requests-cache          # HTTP caching
pandas
numpy
scikit-learn            # LR, RF, StandardScaler, metrics
tensorflow              # Neural network + TFLite export
m2cgen                  # RF → C code conversion
matplotlib
seaborn
jinja2                  # Report templating
```

---

## 11. Deployment Artifacts

All three artifacts are always generated regardless of which model scores best. The report provides a recommendation; deployment choice is left to the user.

| Model | Artifact | Target size |
|-------|----------|-------------|
| Logistic Regression | `deployment/lr_coefficients.h` | < 5 KB |
| Logistic Regression | `deployment/scaler_params.h` | < 2 KB |
| Random Forest | `deployment/rf_model.c` | < 50 KB |
| Neural Network | `deployment/model_data.h` | < 20 KB |

All artifacts are self-contained C/C++ files requiring no external libraries beyond TFLite Micro (NN only). The scaler parameters apply to all three models at inference time.
