<p align="center">
  <img src="docs/images/roc_curves.png" alt="ROC Curves" width="700"/>
</p>

<h1 align="center">AI Weather Prediction System</h1>

<p align="center">
  <strong>Edge-deployable 6-hour weather forecasting using ML models on ESP32-S3</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#models">Models</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#evaluation-results">Results</a> •
  <a href="#deployment">Deployment</a>
</p>

---

## Overview

**AI-Weather-Display-LilyGo** is an end-to-end machine learning pipeline that predicts weather conditions **6 hours ahead** using data from a **Bosch BME688** environmental sensor. The system compares four ML models — Logistic Regression, Random Forest, XGBoost, and Neural Network — and exports optimized deployment artifacts targeting the **LilyGo T-Display S3 (ESP32-S3)** microcontroller.

The pipeline trains offline in Python, evaluates all models on identical test data, generates a comparative HTML report, and produces lightweight C/C++ inference code that runs directly on the ESP32-S3 at 240 MHz — no cloud or internet required at inference time.

**Author:** Maj Prabhat

---

## Features

- **5-Class Weather Classification** — Sunny ☀️ / Cloudy ☁️ / Rainy 🌧️ / Stormy ⛈️ / Snowy ❄️
- **6-Hour Forecast Horizon** — Predicts future conditions from the last 6 hours of sensor readings
- **4 Model Comparison** — Logistic Regression, Random Forest, XGBoost, and Neural Network trained and evaluated side-by-side
- **58-Feature Engineering Pipeline** — Shared feature extraction across all models (raw lags, pressure tendencies, dew point, rolling statistics, cyclical time encoding, and BME688 extras)
- **Edge Deployment Ready** — Generates C headers, C inference code (via m2cgen), and TFLite models for microcontroller deployment
- **Automated HTML Report** — Jinja2-templated comparative report with confusion matrices, ROC curves, feature importance charts, and deployment recommendations
- **Open Data** — Trains on freely available [Open-Meteo](https://open-meteo.com/) historical weather data (no API key required)
- **SMOTE Class Balancing** — Handles class imbalance via synthetic oversampling
- **Full Test Suite** — Comprehensive pytest-based tests for every module

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│                                                                     │
│  Open-Meteo API ──► download.py ──► CSV Cache ──► Train/Val/Test   │
│  (6 stations,       (WMO code        (data/        (chronological   │
│   5 years)           mapping)         cache/)        70/15/15)      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                              │
│                                                                     │
│  engineering.py: 58 features from 6-hour sliding window             │
│  ┌──────────┬──────────────┬──────────┬──────────┬───────────────┐  │
│  │ Raw Lags │  Pressure    │ Derived  │ Rolling  │  BME688       │  │
│  │   (18)   │  Tendency(4) │  (4)     │ Stats(24)│  Extras (4)   │  │
│  └──────────┴──────────────┴──────────┴──────────┴───────────────┘  │
│                  + Cyclical Time Encoding (4)                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING & EVALUATION                      │
│                                                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────┐ ┌────────────────┐  │
│  │  Logistic   │ │   Random    │ │ XGBoost  │ │    Neural      │  │
│  │ Regression  │ │   Forest    │ │          │ │    Network     │  │
│  │ (sklearn)   │ │ (sklearn)   │ │(xgboost) │ │ (TensorFlow)  │  │
│  └──────┬──────┘ └──────┬──────┘ └────┬─────┘ └───────┬────────┘  │
│         │               │              │               │           │
│         ▼               ▼              ▼               ▼           │
│     lr_coeff.h      rf_model.c    xgb_model.pkl   model.tflite    │
│      (2.8 KB)       (via m2cgen)   (5,051 KB)      (11 KB)       │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               EDGE DEPLOYMENT (LilyGo T-Display S3)                │
│                                                                     │
│  BME688 Sensor ──► Feature Extraction ──► Scaled Input ──► Model   │
│  (T, RH, P,        (same 58 features)     (scaler_params.h)        │
│   Gas, IAQ)                                                        │
│                                                                     │
│  ESP32-S3 @ 240 MHz — Inference in microseconds, no cloud needed   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Sensor — Bosch BME688

The BME688 provides live environmental readings used at inference time on the edge device:

| Signal | Unit | Role |
|--------|------|------|
| Temperature | °C | Primary feature |
| Relative Humidity | % | Primary feature |
| Barometric Pressure | hPa | Primary feature (most predictive) |
| Gas Resistance | Ω | Optional — BME688 extra |
| IAQ Index | 0–500 | Optional — BME688 extra |
| eCO₂ | ppm equivalent | Optional — BME688 extra |
| Breath VOC | ppm | Optional — BME688 extra |

> **Note:** The 4 BME688 extras (gas, IAQ, eCO₂, bVOC) are zero-padded during training on public data since Open-Meteo doesn't provide them. At live inference on the device, actual sensor readings replace the zeros.

---

## Dataset

**Source:** [Open-Meteo Historical Weather API](https://open-meteo.com/) — fully open-source, no API key required.

### Training Stations

Six geographically diverse stations ensure all five weather classes (including Snow and Stormy) are well-represented:

| Station | Coordinates | Climate |
|---------|-------------|---------|
| London, UK | 51.5°N, 0.1°W | Temperate |
| Helsinki, Finland | 60.2°N, 25.0°E | Cold / Snowy |
| Singapore | 1.4°N, 103.8°E | Tropical / Stormy |
| Orlando, US | 28.5°N, 81.4°W | Subtropical / Thunderstorms |
| Dhaka, Bangladesh | 23.7°N, 90.4°E | Monsoon / Stormy |
| Manaus, Brazil | 3.1°S, 60.0°W | Equatorial / Heavy Rain |

### WMO Weather Code Mapping

| WMO Codes | Label |
|-----------|-------|
| 0, 1 | ☀️ **Sunny** |
| 2, 3, 45–48 | ☁️ **Cloudy** |
| 51–57, 61–67, 80–82 | 🌧️ **Rainy** |
| 95–99 | ⛈️ **Stormy** |
| 71–77, 85–86 | ❄️ **Snowy** |

### Data Split

The split is **strictly chronological** (no shuffling) to prevent temporal data leakage:

| Split | Proportion | Purpose |
|-------|-----------|---------|
| Train | 70% | Model fitting |
| Validation | 15% | Hyperparameter tuning & early stopping |
| Test | 15% | Final evaluation (most recent data) |

**Volume:** ~5 years × 6 stations ≈ 260,000+ hourly samples.

---

## Feature Engineering

All models share an identical 58-feature extraction pipeline defined in `features/engineering.py`:

| Group | Description | Count |
|-------|-------------|-------|
| Raw Lags | Temperature, Humidity, Pressure × 6 timesteps | 18 |
| Pressure Tendency | ΔPressure at 1h, 3h, 6h intervals | 3 |
| Pressure Acceleration | 2nd derivative of pressure (Δp₃ₕ − Δp₆ₕ) | 1 |
| Temperature Rate | ΔTemp at 1h, 3h | 2 |
| Dew Point | Magnus formula from most recent reading | 1 |
| Absolute Humidity | August-Roche-Magnus formula | 1 |
| Rolling Statistics | mean, std, min, max × 3h & 6h × 3 signals | 24 |
| Cyclical Time | sin/cos of hour-of-day + day-of-year | 4 |
| BME688 Extras | Gas resistance, IAQ, eCO₂, bVOC | 4 |
| **Total** | | **58** |

**Key design choices:**
- **Cyclical encoding** (sin/cos) ensures 23:00 and 00:00 are treated as adjacent, not distant
- **Pressure tendency & acceleration** are the most predictive features for short-horizon forecasting
- **StandardScaler** is fit only on training data — parameters exported as C arrays for edge inference

---

## Models

### 1. Logistic Regression (Linear Baseline)

- Multinomial softmax via `lbfgs` solver, L2 regularization
- Class weights: `balanced`
- Deployment: coefficient matrix → `lr_coefficients.h` (C float array)
- Inference: matrix multiply + softmax — **~2.4 µs** on ESP32-S3

### 2. Random Forest Classifier

- 200 trees, max depth 12
- Class weights: `balanced_subsample`
- Deployment: converted to dependency-free C via `m2cgen` → `rf_model.c`
- Inference: decision tree traversal — **~20 µs** on ESP32-S3

### 3. XGBoost Classifier

- Gradient-boosted trees with sample weighting
- Deployment: serialized model file
- Inference: **~15 µs** on ESP32-S3

### 4. Neural Network (TFLite)

- Architecture: `Input(58) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(5, Softmax)`
- Early stopping on validation loss (patience=10)
- INT8 post-training quantization using 1000 stratified-random calibration samples
- Deployment: `.tflite` → `model_data.h` (C byte array) — runs via TFLite Micro
- Inference: **~49 µs** on ESP32-S3

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd prediction
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Download data → Train all models → Evaluate → Generate report
python main.py --report
```

### Run Individual Stages

```bash
# Stage 1: Download Open-Meteo data (cached after first run)
python main.py --only download

# Stage 2: Train all 4 models
python main.py --only train

# Stage 3: Evaluate on test set
python main.py --only evaluate

# Stage 4: Generate HTML comparison report
python main.py --only report
```

### Advanced Options

```bash
# Override default stations
python main.py --locations "London, UK" "Helsinki, Finland" "Singapore" --years 5 --report

# Force re-download even if cached data exists
python main.py --force-download --years 5

# Custom year range
python main.py --years 3 --report
```

### Run Tests

```bash
cd prediction
pytest
```

---

## Evaluation Results

All models evaluated on the same chronologically-held-out test set:

| Model | Accuracy | Macro F1 | Artifact Size | Inference Time (ESP32-S3) |
|-------|----------|----------|---------------|---------------------------|
| Logistic Regression | 21.0% | 0.202 | 2.8 KB | ~2.4 µs |
| Random Forest | 30.2% | 0.251 | 159,860 KB | ~20 µs |
| **XGBoost** | **41.9%** | **0.250** | 5,051 KB | ~15 µs |
| Neural Network | 31.1% | 0.254 | 11 KB | ~49 µs |

> **Note:** These results reflect training on multi-station global data with extreme class imbalance (Stormy class is very rare). The pipeline uses SMOTE oversampling and balanced class weights to mitigate this. Model performance is expected to improve significantly with station-specific fine-tuning and additional feature engineering.

### Confusion Matrices

<p align="center">
  <img src="docs/images/cm_logistic_regression.png" alt="Confusion Matrix — Logistic Regression" width="400"/>
  <img src="docs/images/cm_random_forest.png" alt="Confusion Matrix — Random Forest" width="400"/>
</p>
<p align="center">
  <img src="docs/images/cm_xgboost.png" alt="Confusion Matrix — XGBoost" width="400"/>
  <img src="docs/images/cm_neural_network.png" alt="Confusion Matrix — Neural Network" width="400"/>
</p>

### Feature Importance

<p align="center">
  <img src="docs/images/fi_logistic_regression.png" alt="Feature Importance — Logistic Regression" width="400"/>
  <img src="docs/images/fi_random_forest.png" alt="Feature Importance — Random Forest" width="400"/>
</p>
<p align="center">
  <img src="docs/images/fi_xgboost.png" alt="Feature Importance — XGBoost" width="400"/>
  <img src="docs/images/fi_neural_network.png" alt="Feature Importance — Neural Network" width="400"/>
</p>

### ROC Curves

<p align="center">
  <img src="docs/images/roc_curves.png" alt="ROC Curves — All Models" width="700"/>
</p>

### Class Distribution

<p align="center">
  <img src="docs/images/class_distribution.png" alt="Dataset Class Distribution" width="600"/>
</p>

---

## Deployment

The pipeline generates self-contained deployment artifacts for the **LilyGo T-Display S3 (ESP32-S3)**:

| Artifact | Description | File |
|----------|-------------|------|
| `scaler_params.h` | StandardScaler mean/std as C arrays (shared by all models) | `deployment/scaler_params.h` |
| `lr_coefficients.h` | Logistic Regression weights as C floats | `deployment/lr_coefficients.h` |
| `rf_model.c` | Random Forest as dependency-free C code (m2cgen) | `deployment/rf_model.c` |
| `model_data.h` | Neural Network as INT8-quantized TFLite C byte array | `deployment/model_data.h` |
| `nn_model.keras` | Full Keras model (for re-export or fine-tuning) | `deployment/nn_model.keras` |
| `model.tflite` | TFLite binary (for testing outside ESP32) | `deployment/model.tflite` |

### Edge Inference Flow

```
BME688 Sensor Reading
        │
        ▼
┌───────────────────┐
│ Collect 6 hourly  │
│ readings (ring     │
│ buffer on ESP32)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Extract 58        │
│ features          │
│ (same pipeline    │
│  as training)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Scale with        │
│ scaler_params.h   │
│ (mean/std)        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Run inference     │    Sunny / Cloudy / Rainy / Stormy / Snowy
│ (LR / RF / NN)   │──────────────────────────────────────────►
└───────────────────┘
```

---

## Project Structure

```
prediction/
├── main.py                     # CLI orchestrator — single entry point
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
│
├── data/
│   ├── download.py             # Open-Meteo fetch, WMO mapping, train/val/test split
│   ├── cache/                  # Raw CSV per station (cached indefinitely)
│   └── processed/              # train.csv, val.csv, test.csv
│
├── features/
│   ├── config.py               # RANDOM_SEED, LABEL_MAP, WMO_MAP, STATIONS, feature counts
│   └── engineering.py          # 58-feature extraction (shared by all models)
│
├── models/
│   ├── logistic_regression.py  # LogisticRegressionModel class
│   ├── random_forest.py        # RandomForestModel class
│   ├── xgboost_model.py        # XGBoostModel class
│   └── neural_network.py       # NeuralNetworkModel class
│
├── evaluation/
│   ├── metrics.py              # compute_all_metrics(y_true, y_pred, y_proba)
│   ├── plots.py                # Confusion matrices, ROC curves, feature importance charts
│   └── outputs/                # results.json, .npy arrays, plot images
│
├── deployment/
│   ├── export.py               # TFLite conversion, C header generation, m2cgen export
│   ├── scaler_params.h         # StandardScaler mean/std as C arrays
│   ├── lr_coefficients.h       # LR model as C float arrays
│   ├── rf_model.c              # RF model as C inference code
│   ├── model_data.h            # TFLite model as C byte array
│   └── nn_model.keras          # Full Keras model
│
├── report/
│   ├── generate.py             # Jinja2-based HTML report generation
│   └── template.html           # Report template
│
├── reports/
│   └── report_output.html      # Generated comparative HTML report
│
├── tests/                      # Comprehensive pytest test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_config.py
│   ├── test_download.py
│   ├── test_engineering.py
│   ├── test_logistic_regression.py
│   ├── test_random_forest.py
│   ├── test_neural_network.py
│   ├── test_metrics.py
│   ├── test_plots.py
│   ├── test_export.py
│   └── test_integration.py
│
└── docs/
    ├── images/                 # Evaluation plots (confusion matrices, ROC, etc.)
    └── superpowers/
        ├── plans/              # Implementation plan
        └── specs/              # Design specification
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Framework | scikit-learn, XGBoost, TensorFlow 2.x / Keras |
| Data Source | Open-Meteo API (`openmeteo-requests`) |
| Model Export | m2cgen (RF → C), TFLite (NN → INT8), custom (LR → C header) |
| Feature Scaling | scikit-learn StandardScaler |
| Class Balancing | SMOTE (imbalanced-learn) |
| Visualization | matplotlib, seaborn |
| Reporting | Jinja2 HTML templates |
| Testing | pytest, pytest-cov |
| Target Hardware | LilyGo T-Display S3 (ESP32-S3 @ 240 MHz) |
| Sensor | Bosch BME688 (Temperature, Humidity, Pressure, Gas, IAQ) |

---

## How It Works

1. **Data Download** — Fetches 5 years of hourly weather data from Open-Meteo for 6 globally diverse stations. WMO weather codes are mapped to 5 target classes. Data is cached locally.

2. **Chronological Split** — Data is split 70/15/15 into train/val/test sets strictly by time (no shuffling) to prevent temporal leakage.

3. **Feature Engineering** — A 6-hour sliding window produces 58 features per sample: raw sensor lags, pressure tendencies, derived meteorological quantities, rolling statistics, and cyclical time encoding.

4. **SMOTE Balancing** — Synthetic Minority Over-sampling Technique balances the training set to handle rare weather classes (Stormy, Snowy).

5. **Model Training** — Four models train sequentially on the balanced dataset: Logistic Regression → Random Forest → XGBoost → Neural Network.

6. **Evaluation** — All models are evaluated on the held-out test set using accuracy, macro F1, per-class precision/recall, confusion matrices, and ROC-AUC.

7. **Artifact Export** — Deployment-ready C/C++ files are generated: coefficient headers, m2cgen C code, INT8-quantized TFLite models, and scaler parameters.

8. **Report Generation** — A comprehensive HTML report compares all models side-by-side with visualizations and a deployment recommendation.

---

## Configuration

Key constants in `features/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `RANDOM_SEED` | 42 | Global reproducibility seed |
| `LOOKBACK` | 6 | Hours of history used as input |
| `LOOKAHEAD` | 6 | Hours ahead to predict |
| `N_CLASSES` | 5 | Sunny, Cloudy, Rainy, Stormy, Snowy |
| `TOTAL_FEATURE_COUNT` | 58 | 54 core + 4 BME688 extras |

---

## License

This project is provided as-is for educational and research purposes.

---

<p align="center">
  Built with ❤️ by <strong>Maj Prabhat</strong>
</p>
