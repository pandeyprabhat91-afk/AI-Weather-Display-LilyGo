# AI Weather Prediction System — Design Spec
**Date:** 2026-03-25
**Project:** AI-Weather-Display-LilyGo
**Status:** Approved

---

## 1. Overview

An AI weather prediction system for edge devices that uses a **Bosch BME688** sensor as its live data source. Three machine learning models — Logistic Regression, Random Forest, and Neural Network — are trained on a public open-source dataset, share an identical feature extraction pipeline, and are evaluated comparatively. The Neural Network is deployed to a **LilyGo S3** microcontroller via TFLite INT8 quantization. A final HTML report compares all three models.

### Goals
- Predict 6-hour-ahead weather condition: **Sunny / Cloudy / Rainy / Stormy / Snowy**
- Train offline in Python (scikit-learn + TensorFlow/Keras)
- Deploy best model to LilyGo S3 (ESP32-S3) for real-time edge inference
- Produce an auto-generated comparative report

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
| Gas Resistance | Ω | Secondary (not in training data) |
| IAQ Index | 0–500 | Secondary (not in training data) |
| eCO2 | ppm equivalent | Secondary (not in training data) |
| Breath VOC | ppm | Secondary (not in training data) |

**Feature gap note:** Public training datasets do not include gas/IAQ/eCO2/bVOC signals. The training pipeline uses temperature, humidity, and pressure as primary inputs. The feature extraction module accepts BME688 extras as optional columns so the live inference pipeline can include them without breaking the training pipeline.

---

## 3. Dataset

**Source:** [Open-Meteo](https://open-meteo.com/) Historical Weather API
**Client library:** `openmeteo-requests`
**No API key required.** Fully open-source.

**Fields fetched (hourly):**
- `temperature_2m` → Temperature
- `relative_humidity_2m` → Humidity
- `pressure_msl` → Pressure
- `weather_code` → WMO code → target label

**WMO Weather Code → Label Mapping:**

| WMO Codes | Label | Int |
|-----------|-------|-----|
| 0–1 | Sunny | 0 |
| 2–3, 45–48 | Cloudy | 1 |
| 51–67, 80–82 | Rainy | 2 |
| 95–99 | Stormy | 3 |
| 71–77, 85–86 | Snowy | 4 |

**Data volume:** ~5 years × 3 geographically diverse stations (temperate, cold/snowy, tropical) ≈ 130,000 hourly samples. Diversity ensures Snow and Stormy classes are well-represented.

**Train / Val / Test split:** 70% / 15% / 15%, stratified by label, chronologically ordered — test set is always the most recent time window to prevent temporal leakage.

---

## 4. Shared Feature Extraction

All three models import from `features/engineering.py` exclusively. A single `FEATURE_COLUMNS` constant in `features/config.py` governs the feature list — changing it once changes it everywhere.

**Input window:** 6 hours of readings (t-1h through t-6h) used to predict condition at t+6h.

### Feature Groups

| Group | Features | Count |
|-------|----------|-------|
| Raw lags | temp, humidity, pressure × 6 timesteps | 18 |
| Pressure tendency | Δpressure over 1h, 3h, 6h | 3 |
| Pressure acceleration | 2nd derivative of pressure | 1 |
| Temp rate of change | Δtemp over 1h, 3h | 2 |
| Dew point | Derived from temp + humidity | 1 |
| Absolute humidity | Derived | 1 |
| Rolling stats | mean, std, min, max × 3h & 6h × 3 signals | 24 |
| Cyclical time | sin/cos of hour-of-day, day-of-year | 4 |
| BME688 extras (optional) | gas resistance, IAQ, eCO2, bVOC | 0–4 |

**Total feature vector:** 54 features (50 core + 4 optional BME688 extras)

### Key Engineering Notes
- **Cyclical encoding** (`sin`/`cos`) for time features prevents ordinal distance artifacts (23:00 and 00:00 are adjacent, not distant).
- **Pressure tendency** (1st derivative) and **pressure acceleration** (2nd derivative) are among the most predictive features for short-horizon weather — they distinguish a steady low from a developing storm.
- **Dew point** is derived as: `T_d = T - ((100 - RH) / 5)` (Magnus approximation).

---

## 5. Models

### 5.1 Logistic Regression (Linear Baseline)

> "Linear regression" for classification = Logistic Regression — a linear model with softmax output.

- **Type:** Multinomial Logistic Regression
- **Solver:** `lbfgs`
- **Regularization:** L2 (Ridge), `C=1.0` (tunable)
- **Class weights:** `balanced` to handle Snow class imbalance
- **Interpretability:** Coefficient matrix exported for feature importance analysis
- **Edge deployment:** Coefficients serialized as JSON → C float array header for LilyGo. Inference = single matrix multiply + softmax (~microseconds on ESP32-S3).

### 5.2 Random Forest Classifier

- **Trees:** 200 estimators
- **Max depth:** 12 (limits C code size from `m2cgen`)
- **Class weights:** `balanced_subsample`
- **Feature importance:** Gini-based importances exported for report
- **Edge deployment:** Converted to dependency-free C inference code via `m2cgen` library

### 5.3 Neural Network (TFLite)

- **Architecture:** `Input(54) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(5, Softmax)`
- **Optimizer:** Adam, lr=0.001
- **Loss:** Categorical crossentropy
- **Training:** Early stopping on val_loss (patience=10), ReduceLROnPlateau
- **Quantization:** INT8 post-training quantization → target model size ~8–16KB
- **Edge deployment:** `.tflite` model embedded as C array in Arduino sketch, run via TFLite Micro runtime on ESP32-S3

---

## 6. Evaluation

All metrics computed identically for all 3 models on the held-out test set.

| Metric | Details |
|--------|---------|
| Accuracy | Overall |
| Macro F1 | Fair across imbalanced classes |
| Per-class Precision / Recall / F1 | Per label |
| Confusion matrix | 5×5 heatmap |
| ROC-AUC (one-vs-rest) | Probabilistic discrimination |
| Model size | Serialized artifact size in KB |
| Inference time estimate | Estimated FLOPs / cycle count for ESP32-S3 |

---

## 7. Auto-Generated HTML Report

`report/generate.py` produces `docs/report_output.html` containing:

1. **Executive summary table** — all 3 models side-by-side on all metrics
2. **Dataset description** — class distribution chart, station locations
3. **Feature engineering** — feature list, importance rankings per model
4. **Confusion matrices** — 3 heatmaps (one per model)
5. **ROC curves** — all models overlaid on one chart
6. **Edge deployment section** — model sizes, quantization results, estimated LilyGo inference time
7. **Recommendation** — which model to deploy and why (balanced accuracy vs. size tradeoff)

---

## 8. Project Structure

```
ai-weather-prediction/
├── data/
│   ├── download.py          ← Open-Meteo fetch, cache to CSV
│   └── processed/           ← train.csv, val.csv, test.csv
├── features/
│   ├── __init__.py
│   ├── engineering.py       ← ALL feature engineering (shared)
│   └── config.py            ← FEATURE_COLUMNS, LABEL_MAP constants
├── models/
│   ├── logistic_regression.py
│   ├── random_forest.py
│   └── neural_network.py
├── evaluation/
│   ├── metrics.py           ← compute_all_metrics()
│   └── plots.py             ← confusion matrix, ROC, bar charts
├── report/
│   ├── generate.py          ← assembles HTML report
│   └── template.html        ← Jinja2 template
├── deployment/
│   └── export_tflite.py     ← TFLite conversion + INT8 quantization
├── main.py                  ← orchestrator (single entry point)
├── requirements.txt
└── docs/
    ├── report_output.html   ← generated report (gitignored)
    └── superpowers/specs/
        └── 2026-03-25-ai-weather-prediction-design.md
```

---

## 9. CLI Entry Point

```bash
# Full pipeline: download → features → train → evaluate → report
python main.py --location "London" --years 5 --report

# Individual stages
python main.py --only download
python main.py --only train
python main.py --only report
```

---

## 10. Dependencies

```
openmeteo-requests      # Open-Meteo API client
requests-cache          # Caching for API calls
pandas, numpy           # Data processing
scikit-learn            # LR + RF + metrics
tensorflow              # Neural network + TFLite export
m2cgen                  # RF → C code conversion
matplotlib, seaborn     # Plots
jinja2                  # Report templating
```

---

## 11. Deployment Artifacts

| Model | Artifact | Target size |
|-------|----------|-------------|
| Logistic Regression | `deployment/lr_coefficients.h` | < 5 KB |
| Random Forest | `deployment/rf_model.c` | < 50 KB |
| Neural Network | `deployment/model.tflite` → `deployment/model_data.h` | < 20 KB |

All artifacts are self-contained C/C++ files requiring no external libraries beyond TFLite Micro (for the NN).
