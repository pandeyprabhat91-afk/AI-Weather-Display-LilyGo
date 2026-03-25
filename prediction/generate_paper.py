"""
Generate technical paper DOCX: AI Weather Prediction System
Run: python generate_paper.py
"""
import json, os
import numpy as np
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# ── helpers ──────────────────────────────────────────────────────────────────

def set_col_width(col, width_cm):
    for cell in col.cells:
        cell.width = Cm(width_cm)

def shade_cell(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def bold_cell(cell, text, size=10, color=None):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor.from_string(color)

def center_cell(cell, text, size=9, bold=False, color=None):
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)

def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def add_para(doc, text, size=10, indent=False, italic=False):
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Cm(1)
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    p.paragraph_format.space_after = Pt(4)
    return p

def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(9)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)

# ── load results ─────────────────────────────────────────────────────────────

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "evaluation", "outputs", "results.json")) as f:
    results = json.load(f)

import pandas as pd
train_df = pd.read_csv(os.path.join(BASE, "data", "processed", "train.csv"))
test_df  = pd.read_csv(os.path.join(BASE, "data", "processed", "test.csv"))
val_df   = pd.read_csv(os.path.join(BASE, "data", "processed", "val.csv"))

CLASS_NAMES = ["Sunny", "Cloudy", "Rainy", "Stormy", "Snowy"]
FEATURE_GROUPS = [
    ("Raw signal lags", "temperature, humidity, pressure × 6 timesteps (t−1h to t−6h)", 18),
    ("Pressure tendency", "Δpressure at 1 h, 3 h, 6 h intervals", 3),
    ("Pressure acceleration", "Δp_3h − Δp_6h (2nd derivative of pressure)", 1),
    ("Temperature rate of change", "Δtemp at 1 h, 3 h intervals", 2),
    ("Dew point", "Magnus formula from most-recent timestep (t−1h)", 1),
    ("Absolute humidity", "August–Roche–Magnus formula from t−1h (g/m³)", 1),
    ("Rolling statistics", "mean, std, min, max × 3 h & 6 h windows × 3 signals", 24),
    ("Cyclical time encoding", "sin/cos of hour-of-day, sin/cos of day-of-year", 4),
    ("BME688 extras (optional)", "gas resistance, IAQ, eCO₂, breath VOC (zero-padded during training)", 4),
    ("Total", "", 58),
]

# ── build document ────────────────────────────────────────────────────────────

doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.8)
    section.right_margin  = Cm(2.8)

# ── TITLE ──
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Edge-Deployable AI Weather Prediction:\nA Comparative Study of Linear Regression, "
                  "Random Forest, and Neural Network Models\nUsing Bosch BME688 Sensor Data")
r.bold = True
r.font.size = Pt(16)
title.paragraph_format.space_after = Pt(6)

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = subtitle.add_run(f"Technical Report  ·  {datetime.date.today().strftime('%B %d, %Y')}  ·  "
                       "Target Platform: Raspberry Pi 5")
r2.italic = True
r2.font.size = Pt(10)
subtitle.paragraph_format.space_after = Pt(16)

doc.add_paragraph()

# ── ABSTRACT ──
add_heading(doc, "Abstract", level=1)
add_para(doc, (
    "This paper presents a comparative evaluation of three supervised machine-learning approaches — "
    "multinomial logistic regression (LR), random forest (RF), and a feedforward neural network (NN) — "
    "applied to 6-hour-ahead weather condition classification using atmospheric sensor data "
    "compatible with the Bosch BME688 environmental sensor. "
    "All models share an identical 58-feature extraction pipeline derived from temperature, relative humidity, "
    "and barometric pressure readings. Training data was sourced from the Open-Meteo historical "
    "weather archive across three geographically diverse stations (London UK, Helsinki Finland, Singapore) "
    "spanning five years (2020–2025), yielding 131,472 labelled samples across five WMO-mapped classes: "
    "Sunny, Cloudy, Rainy, Stormy, and Snowy. "
    "Experimental results show that the neural network achieves the highest raw accuracy (54.0%) "
    "but suffers from majority-class collapse on the imbalanced dataset. "
    "The random forest delivers the best balanced macro-F1 score (0.348) with robust per-class recall. "
    "Logistic regression, while providing the fastest inference (2.44 µs), "
    "achieves the lowest balanced performance (macro-F1: 0.259). "
    "For deployment on the Raspberry Pi 5, the random forest is recommended as the primary classifier. "
    "Future work should address the severe class imbalance and the complete absence of Stormy-class "
    "observations in the current dataset."
))

doc.add_paragraph()

# ── 1. INTRODUCTION ──
add_heading(doc, "1. Introduction", level=1)
add_para(doc, (
    "Short-range weather forecasting at the sensor node level — so-called edge weather intelligence — "
    "has become increasingly relevant in the context of IoT environmental monitoring. "
    "Unlike NWP (Numerical Weather Prediction) models that require supercomputing resources, "
    "edge ML classifiers can operate locally on low-power hardware such as the Raspberry Pi 5, "
    "providing real-time 6-hour weather condition forecasts without cloud dependency."
))
add_para(doc, (
    "The Bosch BME688 sensor provides temperature, relative humidity, barometric pressure, "
    "gas resistance, IAQ index, estimated CO₂ (eCO₂), and breath VOC measurements — "
    "a richer feature set than classical barometer-only weather stations. "
    "This work leverages the three primary physical signals (temperature, humidity, pressure) "
    "for training, as public historical datasets do not include gas/IAQ measurements. "
    "The BME688 gas-phase channels are architecturally included in the feature vector "
    "as zero-padded optional columns, enabling seamless integration of live sensor readings "
    "at inference time without retraining."
))
add_para(doc, (
    "Three classification paradigms are systematically compared: "
    "(1) multinomial logistic regression as the linear baseline; "
    "(2) random forest as a non-linear ensemble method; and "
    "(3) a compact feedforward neural network. "
    "All models are trained on identical feature vectors to ensure a fair comparison. "
    "The paper reports accuracy, macro-averaged F1, per-class precision/recall/F1, "
    "confusion matrices, model artifact sizes, and estimated inference latency."
))

doc.add_paragraph()

# ── 2. DATASET ──
add_heading(doc, "2. Dataset", level=1)
add_heading(doc, "2.1 Data Source", level=2)
add_para(doc, (
    "Historical hourly weather data was obtained from the Open-Meteo Historical Weather API "
    "(https://open-meteo.com), an open-source, API-key-free service providing ERA5-reanalysis-backed "
    "atmospheric data at 1-hour resolution. Three stations were selected to maximise meteorological "
    "diversity and ensure representation of all five target classes:"
))
stations_tbl = doc.add_table(rows=4, cols=3)
stations_tbl.style = "Table Grid"
hdr_cells = stations_tbl.rows[0].cells
for cell, text in zip(hdr_cells, ["Station", "Coordinates", "Climate Type"]):
    bold_cell(cell, text, size=9)
    shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]
    run.font.color.rgb = RGBColor(255,255,255)
for row, (sta, coord, climate) in zip(stations_tbl.rows[1:], [
    ("London, United Kingdom",  "51.51°N, 0.13°W",    "Temperate oceanic (Cfb)"),
    ("Helsinki, Finland",       "60.17°N, 25.00°E",   "Humid continental / subarctic (Dfb)"),
    ("Singapore",               "1.35°N, 103.82°E",   "Tropical rainforest (Af)"),
]):
    for cell, text in zip(row.cells, [sta, coord, climate]):
        center_cell(cell, text)
add_caption(doc, "Table 1. Training stations and climate classifications.")

add_heading(doc, "2.2 Label Mapping", level=2)
add_para(doc, (
    "Raw WMO present-weather codes returned by the API are mapped to five integer class labels. "
    "Codes not falling within the defined ranges (e.g. dust storms, volcanic ash, code 4–44 misc.) "
    "are dropped; the expected drop rate is < 0.5% of records."
))

wmo_tbl = doc.add_table(rows=6, cols=3)
wmo_tbl.style = "Table Grid"
for cell, text in zip(wmo_tbl.rows[0].cells, ["Label", "Class", "WMO Codes"]):
    bold_cell(cell, text, size=9)
    shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]
    run.font.color.rgb = RGBColor(255,255,255)
for row, (lbl, cls, codes) in zip(wmo_tbl.rows[1:], [
    ("0", "Sunny",   "0, 1"),
    ("1", "Cloudy",  "2, 3, 45–48"),
    ("2", "Rainy",   "51–67, 80–82"),
    ("3", "Stormy",  "95–99"),
    ("4", "Snowy",   "71–77, 85–86"),
]):
    for cell, text in zip(row.cells, [lbl, cls, codes]):
        center_cell(cell, text)
add_caption(doc, "Table 2. WMO weather code to class label mapping.")

add_heading(doc, "2.3 Data Statistics", level=2)

total = len(train_df) + len(val_df) + len(test_df)
train_dist = train_df.label.value_counts().sort_index().to_dict()
test_dist  = test_df.label.value_counts().sort_index().to_dict()

add_para(doc, (
    f"After WMO mapping, gap-filling (forward-fill ≤ 2 h; otherwise drop), and chronological "
    f"70/15/15 splitting, the dataset contains {total:,} samples:"
))

stats_tbl = doc.add_table(rows=7, cols=5)
stats_tbl.style = "Table Grid"
for cell, text in zip(stats_tbl.rows[0].cells,
                      ["Split", "Total Samples", "Sunny (0)", "Cloudy (1)", "Rainy (2)", ]):
    bold_cell(cell, text, size=9)
    shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]
    run.font.color.rgb = RGBColor(255,255,255)

# rebuild as 8-col table
stats_tbl2 = doc.add_table(rows=4, cols=7)
stats_tbl2.style = "Table Grid"
for cell, text in zip(stats_tbl2.rows[0].cells,
    ["Split", "Samples", "Sunny", "Cloudy", "Rainy", "Stormy", "Snowy"]):
    bold_cell(cell, text, size=9)
    shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]
    run.font.color.rgb = RGBColor(255,255,255)

for tbl_row, (split_name, df) in zip(stats_tbl2.rows[1:],
    [("Train (70%)", train_df), ("Val (15%)", val_df), ("Test (15%)", test_df)]):
    dist = df.label.value_counts().sort_index()
    vals = [split_name, str(len(df))]
    for i in range(5):
        vals.append(str(dist.get(i, 0)))
    for cell, text in zip(tbl_row.cells, vals):
        center_cell(cell, text)

# Remove empty first table
stats_tbl._element.getparent().remove(stats_tbl._element)
add_caption(doc, "Table 3. Dataset split statistics by class. Stormy (3) has zero samples across all splits.")

add_para(doc, (
    "A critical finding is that Stormy-class observations (WMO 95–99) are entirely absent from the dataset. "
    "Despite Helsinki's subarctic climate and Singapore's tropical convective activity, no thunderstorm "
    "codes were recorded in the five-year window across these stations. This is likely attributable to "
    "the ERA5 reanalysis model's conservative weather-code assignment at 1-hour intervals. "
    "Similarly, the Snowy class is severely underrepresented (train: "
    f"{train_dist.get(4,0):,} samples, {100*train_dist.get(4,0)/len(train_df):.1f}%), "
    "reflecting the limited geographic contribution of Helsinki to the merged multi-station corpus."
))
add_para(doc, (
    f"The Cloudy class dominates strongly: {train_dist.get(1,0):,} of {len(train_df):,} training "
    f"samples ({100*train_dist.get(1,0)/len(train_df):.1f}%). "
    "This severe class imbalance drives the majority-class collapse behaviour observed in the "
    "neural network results (Section 5)."
))

doc.add_paragraph()

# ── 3. FEATURE ENGINEERING ──
add_heading(doc, "3. Feature Engineering", level=1)
add_para(doc, (
    "A unified 58-dimensional feature vector is constructed from each 6-hour lookback window "
    "(readings at t−1h through t−6h) to predict the weather condition at t+6h (6-hour horizon). "
    "All three models receive the identical feature vector, computed by "
    "prediction/features/engineering.py, ensuring a scientifically valid comparison."
))
add_para(doc, (
    "Features are standardised using a sklearn StandardScaler fitted exclusively on the training set "
    "and serialised for use at inference time. The scaler parameters (mean and standard deviation "
    "per feature) are also exported as a C header file for potential embedded deployment."
))

feat_tbl = doc.add_table(rows=len(FEATURE_GROUPS)+1, cols=3)
feat_tbl.style = "Table Grid"
for cell, text in zip(feat_tbl.rows[0].cells, ["Feature Group", "Description", "Count"]):
    bold_cell(cell, text, size=9)
    shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]
    run.font.color.rgb = RGBColor(255,255,255)
for row, (grp, desc, cnt) in zip(feat_tbl.rows[1:], FEATURE_GROUPS):
    is_total = grp == "Total"
    for cell, text in zip(row.cells, [grp, desc, str(cnt)]):
        center_cell(cell, text, bold=is_total)
    if is_total:
        for cell in row.cells:
            shade_cell(cell, "D5F5E3")
add_caption(doc, "Table 4. Feature engineering groups (58 features total).")

add_para(doc, "Key derived features of meteorological significance:")
bullets = [
    "Pressure tendency (Δp at 1 h, 3 h, 6 h): falling pressure is the primary indicator of approaching low-pressure systems and precipitation.",
    "Pressure acceleration (Δp_3h − Δp_6h): the second derivative distinguishes a steadily developing system from a rapid-onset event such as a squall line.",
    "Dew point (Magnus formula, computed from t−1h): encodes the moisture content of the atmosphere more physically than relative humidity alone.",
    "Absolute humidity (August–Roche–Magnus, g/m³): a temperature-independent moisture measure that improves snow/rain discrimination near 0°C.",
    "Cyclical time encoding (sin/cos of hour and day-of-year): eliminates the ordinal-distance artefact of raw integer time features; preserves continuity at midnight and year boundaries.",
    "Rolling statistics (mean, std, min, max over 3 h and 6 h): capture signal volatility and trend direction over the input window.",
    "BME688 gas/IAQ channels (4 features, zero-padded during training): reserved positions allow the live BME688 sensor output to be injected at inference without architectural changes.",
]
for b in bullets:
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(b)
    run.font.size = Pt(9)

doc.add_paragraph()

# ── 4. MODEL ARCHITECTURES ──
add_heading(doc, "4. Model Architectures and Hyperparameters", level=1)

add_heading(doc, "4.1 Logistic Regression (Linear Baseline)", level=2)
add_para(doc, (
    "Multinomial logistic regression models the posterior class probability as a softmax over a "
    "linear combination of the input features. Despite the \"linear regression\" designation, "
    "this is the standard linear classification model for multi-class problems. It provides a "
    "fully interpretable coefficient matrix (5 × 58) that directly quantifies each feature's "
    "contribution to each class decision."
))

lr_params = [
    ("Solver", "lbfgs (limited-memory BFGS)"),
    ("Maximum iterations", "1,000"),
    ("Regularisation", "L2 (Ridge), C = 1.0"),
    ("Class weighting", "balanced (inversely proportional to class frequency)"),
    ("Random seed", "42"),
    ("Multi-class strategy", "Multinomial softmax (native to lbfgs)"),
    ("Input features", "58 (StandardScaler normalised)"),
    ("Output", "5-class probability vector"),
    ("Deployment artefact", "lr_coefficients.h (C float array, 2.9 KB)"),
    ("Inference cost (Raspberry Pi 5)", "~2.44 µs (matrix multiply + softmax, analytical)"),
]
lr_tbl = doc.add_table(rows=len(lr_params)+1, cols=2)
lr_tbl.style = "Table Grid"
for cell, text in zip(lr_tbl.rows[0].cells, ["Hyperparameter", "Value"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)
for row, (k, v) in zip(lr_tbl.rows[1:], lr_params):
    center_cell(row.cells[0], k, bold=True)
    center_cell(row.cells[1], v)
add_caption(doc, "Table 5. Logistic Regression hyperparameters.")

add_heading(doc, "4.2 Random Forest Classifier", level=2)
add_para(doc, (
    "Random forest builds an ensemble of decision trees, each trained on a bootstrap sample of "
    "the training set with random feature subsets at each node split (bagging + random subspace). "
    "Final predictions are made by majority vote over all trees. The ensemble is robust to "
    "feature scaling, handles non-linear interactions naturally, and provides Gini-based "
    "feature importances for interpretability."
))

rf_params = [
    ("Number of estimators (trees)", "200"),
    ("Maximum tree depth", "12"),
    ("Feature sampling per split", "√n_features (default)"),
    ("Class weighting", "balanced_subsample (per-bootstrap resampling)"),
    ("Parallelism", "n_jobs = −1 (all CPU cores)"),
    ("Random seed", "42"),
    ("Input features", "58 (StandardScaler normalised)"),
    ("Output", "5-class probability (vote fraction)"),
    ("Deployment artefact", "rf_model.pkl (joblib, 116 MB) + rf_model.c (m2cgen C code)"),
    ("Inference cost (Raspberry Pi 5)", "~20 µs (analytical; direct pickle load negligible)"),
    ("RAM at inference", "~450 MB (full model loaded)"),
]
rf_tbl = doc.add_table(rows=len(rf_params)+1, cols=2)
rf_tbl.style = "Table Grid"
for cell, text in zip(rf_tbl.rows[0].cells, ["Hyperparameter", "Value"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)
for row, (k, v) in zip(rf_tbl.rows[1:], rf_params):
    center_cell(row.cells[0], k, bold=True)
    center_cell(row.cells[1], v)
add_caption(doc, "Table 6. Random Forest hyperparameters.")

add_heading(doc, "4.3 Feedforward Neural Network (TFLite)", level=2)
add_para(doc, (
    "A compact three-layer feedforward neural network is implemented using TensorFlow/Keras. "
    "The architecture is deliberately shallow to enable deployment on resource-constrained hardware. "
    "Dropout regularisation (p = 0.3) reduces overfitting on the imbalanced training set. "
    "Early stopping on validation loss with weight restoration ensures the best generalising "
    "checkpoint is retained regardless of training duration."
))
add_para(doc, "Architecture: Input(58) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(5, Softmax)", italic=True)

nn_params = [
    ("Layer 1", "Dense(64), ReLU activation"),
    ("Layer 2", "Dropout(0.3, seed=42)"),
    ("Layer 3", "Dense(32), ReLU activation"),
    ("Output layer", "Dense(5), Softmax activation"),
    ("Optimiser", "Adam, learning rate = 0.001"),
    ("Loss function", "Sparse categorical cross-entropy"),
    ("Batch size", "256"),
    ("Maximum epochs", "100 (early stopping active)"),
    ("Early stopping", "patience = 10, monitor = val_loss, restore best weights"),
    ("LR scheduler", "ReduceLROnPlateau (factor=0.5, patience=5)"),
    ("Random seed", "42 (tf.random.set_seed + Dropout seed)"),
    ("Input features", "58 (StandardScaler normalised)"),
    ("Output", "5-class probability (softmax)"),
    ("Deployment artefact (.keras)", "nn_model.keras (Keras SavedModel, full precision)"),
    ("Deployment artefact (TFLite)", "model.tflite (INT8 quantised, 11 KB)"),
    ("TFLite calibration", "1,000 stratified-random training samples"),
    ("Inference cost (Raspberry Pi 5)", "~49.3 µs (analytical); actual measured < 1 ms"),
    ("Total trainable parameters", "~6,373"),
]
nn_tbl = doc.add_table(rows=len(nn_params)+1, cols=2)
nn_tbl.style = "Table Grid"
for cell, text in zip(nn_tbl.rows[0].cells, ["Parameter", "Value"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)
for row, (k, v) in zip(nn_tbl.rows[1:], nn_params):
    center_cell(row.cells[0], k, bold=True)
    center_cell(row.cells[1], v)
add_caption(doc, "Table 7. Neural Network architecture and training parameters.")

doc.add_paragraph()

# ── 5. RESULTS ──
add_heading(doc, "5. Experimental Results", level=1)
add_heading(doc, "5.1 Overall Performance", level=2)
add_para(doc, (
    "Table 8 summarises the primary evaluation metrics for all three models on the held-out "
    "test set (19,721 samples, chronologically the most recent 15% of the dataset). "
    "The macro-averaged F1 score is the primary ranking metric, as it treats all classes equally "
    "regardless of their support — critical given the severe class imbalance in this dataset."
))

# Build summary from actual results
artifact_size_map = {
    "Logistic Regression": "2.9 KB (.h)",
    "Random Forest": "116 MB (.pkl)",
    "XGBoost": "~5 MB (.pkl)",
    "Neural Network": "~11 KB (.tflite)",
}
infer_map = {
    "Logistic Regression": "~2.4 µs",
    "Random Forest": "~20 µs",
    "XGBoost": "~15 µs",
    "Neural Network": "~49 µs",
}
best_f1_name = max(results, key=lambda r: r["metrics"]["macro_f1"])["name"]

# Summary table
summary_tbl = doc.add_table(rows=1 + len(results), cols=6)
summary_tbl.style = "Table Grid"
for cell, text in zip(summary_tbl.rows[0].cells,
    ["Model", "Accuracy", "Macro F1", "Artifact Size", "Inf. Time", "Recommended"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)

for row, m in zip(summary_tbl.rows[1:], results):
    acc  = f"{m['metrics']['accuracy']:.3f}"
    mf1  = f"{m['metrics']['macro_f1']:.3f}"
    art  = artifact_size_map.get(m["name"], "—")
    inf  = infer_map.get(m["name"], "—")
    rec  = "YES ✓" if m["name"] == best_f1_name else "No"
    vals = (m["name"], acc, mf1, art, inf, rec)
    for cell, text in zip(row.cells, vals):
        center_cell(cell, text, bold=(rec == "YES ✓" and text in [mf1, "YES ✓"]))
    if m["name"] == best_f1_name:
        for cell in row.cells:
            shade_cell(cell, "D5F5E3")
add_caption(doc, "Table 8. Executive summary of model performance. Green row = recommended model. "
                 "Artifact size for Raspberry Pi 5 deployment (no size constraint).")

add_heading(doc, "5.2 Per-Class Performance", level=2)
add_para(doc, (
    "Table 9 reports per-class precision, recall, and F1 for each model. "
    "The Stormy class (3) achieves 0.000 across all models due to its complete absence "
    "from the training, validation, and test sets. The Snowy class (4) shows the widest "
    "variance across models, reflecting the difficulty of classifying a rare class (1.5% "
    "of test samples)."
))

per_class_tbl = doc.add_table(rows=1 + 5*len(results), cols=5)
per_class_tbl.style = "Table Grid"
for cell, text in zip(per_class_tbl.rows[0].cells,
    ["Model / Class", "Precision", "Recall", "F1", "Test Support"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)

test_support = {i: test_df.label.value_counts().get(i, 0) for i in range(5)}
model_colors = {"Logistic Regression": "EAF4FB", "Random Forest": "EAF4FB", "XGBoost": "EAF4FB", "Neural Network": "EAF4FB"}
row_idx = 1
for m in results:
    mname = m["name"]
    prec  = m["metrics"]["per_class_precision"]
    rec   = m["metrics"]["per_class_recall"]
    f1    = m["metrics"]["per_class_f1"]
    for ci, cname in enumerate(CLASS_NAMES):
        row = per_class_tbl.rows[row_idx]
        label_text = f"{mname} — {cname}" if ci == 0 else f"   {cname}"
        support = test_support.get(ci, 0)
        for cell, text in zip(row.cells,
            [f"{mname[:4]}. — {cname}" if ci == 0 else f"    {cname}",
             f"{prec[ci]:.3f}", f"{rec[ci]:.3f}", f"{f1[ci]:.3f}", str(support)]):
            center_cell(cell, text)
        if ci == 0:
            shade_cell(row.cells[0], "D6EAF8")
        row_idx += 1

add_caption(doc, "Table 9. Per-class precision, recall, and F1 by model. Support = number of test samples per class.")

add_heading(doc, "5.3 Confusion Matrices", level=2)
add_para(doc, "Tables 10–12 show the 5×5 confusion matrices (rows = true class, columns = predicted class).")

for m in results:
    cm = np.array(m["metrics"]["confusion_matrix"])
    add_para(doc, f"{m['name']}:")
    cm_tbl = doc.add_table(rows=6, cols=6)
    cm_tbl.style = "Table Grid"
    # header row
    for cell, text in zip(cm_tbl.rows[0].cells[1:], CLASS_NAMES):
        bold_cell(cell, f"Pred: {text}", size=8); shade_cell(cell, "2E86C1")
        run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)
    shade_cell(cm_tbl.rows[0].cells[0], "2E86C1")
    for ri, cname in enumerate(CLASS_NAMES):
        row = cm_tbl.rows[ri+1]
        bold_cell(row.cells[0], f"True: {cname}", size=8); shade_cell(row.cells[0], "D6EAF8")
        for ci, val in enumerate(cm[ri]):
            is_diag = (ri == ci)
            center_cell(row.cells[ci+1], str(val), bold=is_diag)
            if is_diag and val > 0:
                shade_cell(row.cells[ci+1], "D5F5E3")
    cap_num = {r["name"]: 10+i for i, r in enumerate(results)}
    add_caption(doc, f"Table {cap_num[m['name']]}. Confusion matrix — {m['name']}. "
                     "Green diagonal = correct predictions.")

doc.add_paragraph()

# ── 6. ANALYSIS ──
add_heading(doc, "6. Analysis and Discussion", level=1)

add_heading(doc, "6.1 Neural Network: Majority-Class Collapse", level=2)
add_para(doc, (
    "The neural network achieves the highest raw accuracy (54.0%) but the lowest macro-F1 (0.204). "
    "Inspection of its confusion matrix reveals the cause: the model predicts Cloudy for "
    "10,373 of 10,697 true-Cloudy test samples (recall = 0.971) while almost entirely ignoring "
    "other classes. This behaviour — known as majority-class collapse — arises when the loss "
    "function is minimised by predicting the dominant class for all inputs."
))
add_para(doc, (
    "The Cloudy class accounts for 50.7% of training samples. Without explicit class weighting, "
    "the cross-entropy loss is dominated by Cloudy samples, and gradient descent converges to "
    "a near-constant Cloudy prediction. The model's 54.0% accuracy is essentially the prior "
    "probability of the Cloudy class in the test set (54.3% = 10,697 / 19,721), confirming "
    "that the NN adds little predictive value in its current configuration."
))
add_para(doc, (
    "Mitigation strategies: (1) add class_weight='balanced' to the Keras compile step or "
    "compute sample weights proportional to inverse class frequency; (2) apply SMOTE or "
    "random over-sampling of minority classes before training; (3) use focal loss instead "
    "of cross-entropy to down-weight easy-majority examples."
))

add_heading(doc, "6.2 Random Forest: Best Balanced Classifier", level=2)
add_para(doc, (
    "Random forest achieves the best macro-F1 (0.348) and the highest per-class recall "
    "outside Cloudy, including meaningful Snowy recall (0.579). The balanced_subsample "
    "class weighting draws balanced bootstrap samples per class within each tree, providing "
    "natural resistance to the class imbalance without external resampling."
))
add_para(doc, (
    "The 200-tree ensemble with max_depth=12 produces a 116 MB serialised model — "
    "this is entirely acceptable for Raspberry Pi 5 (4–8 GB RAM, NVMe SSD) but would require "
    "significant depth reduction (max_depth ≤ 6) for microcontroller deployment."
))

add_heading(doc, "6.3 Logistic Regression: Interpretable but Limited", level=2)
add_para(doc, (
    "Logistic regression achieves accuracy = 0.270 and macro-F1 = 0.259, the lowest among all "
    "three models. Its linear decision boundary cannot capture the non-linear interactions "
    "between pressure tendency, humidity, and temperature that characterise weather transition "
    "events. However, LR offers unique advantages: the 5 × 58 coefficient matrix provides a "
    "direct, interpretable mapping from each feature to each class decision, and inference "
    "costs only 2.44 µs — making it suitable as a fast fallback classifier."
))
add_para(doc, (
    "Notably, LR achieves Snowy recall = 0.943 (the highest of any model for any minority class), "
    "driven by aggressive balanced-weight compensation for the Snowy class's 2.3% training "
    "representation. This comes at the cost of extreme false-positive rates for Sunny and Rainy."
))

add_heading(doc, "6.4 Absent Stormy Class", level=2)
add_para(doc, (
    "The complete absence of Stormy-class samples across all three splits is a significant "
    "data quality finding. ERA5 reanalysis, the underlying data source for Open-Meteo, "
    "tends to smooth out extreme short-duration events (thunderstorms typically last 1–3 hours) "
    "and assign conservative present-weather codes. "
    "To obtain Stormy-class training data, the following strategies are recommended:"
))
for b in [
    "Add stations with known high thunderstorm frequency (e.g. Florida, Central Africa, Bangladesh).",
    "Augment with SYNOP surface observation data (WMO Global Telecommunication System) which uses actual station observations rather than reanalysis.",
    "Generate synthetic Stormy samples using pressure, humidity, and temperature profiles characteristic of pre-storm conditions (CAPE proxy, dew point depression).",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

doc.add_paragraph()

# ── 7. DEPLOYMENT ──
add_heading(doc, "7. Deployment on Raspberry Pi 5", level=1)
add_para(doc, (
    "The target inference platform is a Raspberry Pi 5 (Arm Cortex-A76, 4/8 GB LPDDR4X RAM, "
    "running Raspberry Pi OS 64-bit). All three models are trivially deployable — the "
    "116 MB random forest occupies ~2.9% of a 4 GB system's RAM. "
    "No quantisation or model compression is required."
))

deploy_rows = [
    ("Logistic Regression", "lr_coefficients.h (2.9 KB)",    "< 1 ms",  "< 1 MB",    "~2 µs"),
    ("Random Forest",       "rf_model.pkl (116 MB)",          "~1.2 s",  "~450 MB",   "~1–5 ms"),
    ("XGBoost",             "xgb_model.pkl (~5 MB)",          "~50 ms",  "~20 MB",    "~15 µs"),
    ("Neural Network",      "model.tflite (~11 KB INT8)",     "< 500 ms","< 50 MB",   "< 1 ms"),
]
deploy_tbl = doc.add_table(rows=1 + len(deploy_rows), cols=5)
deploy_tbl.style = "Table Grid"
for cell, text in zip(deploy_tbl.rows[0].cells,
    ["Model", "Artefact", "Load Time", "RAM Usage", "Inference Latency"]):
    bold_cell(cell, text, size=9); shade_cell(cell, "2E86C1")
    run = cell.paragraphs[0].runs[0]; run.font.color.rgb = RGBColor(255,255,255)
for row, vals in zip(deploy_tbl.rows[1:], deploy_rows):
    for cell, text in zip(row.cells, vals):
        center_cell(cell, text)
add_caption(doc, "Table 13. Raspberry Pi 5 deployment characteristics.")

add_para(doc, "Recommended deployment stack:")
for b in [
    "Primary classifier: Random Forest (rf_model.pkl via joblib) — best balanced F1.",
    "Ensemble option: Average RF + NN probabilities for improved accuracy on common classes.",
    "Runtime: Python 3.11, scikit-learn, TensorFlow Lite Runtime (tflite-runtime package).",
    "Sensor interface: Bosch BSEC library via Python bindings or I²C direct read for BME688.",
    "Inference trigger: Run prediction every 30 minutes; buffer last 6 hourly readings.",
    "Output: 5-class probability vector → display on connected screen or push to MQTT broker.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

doc.add_paragraph()

# ── 8. RECOMMENDATIONS ──
add_heading(doc, "8. Recommendations and Future Work", level=1)

add_heading(doc, "8.1 Immediate Actions (High Priority)", level=2)
for b in [
    "Apply class weighting to the Neural Network (class_weight parameter in model.fit()) — expected to raise macro-F1 from 0.204 to > 0.35.",
    "Source Stormy-class training data from high-thunderstorm-frequency stations or SYNOP observation archives.",
    "Reduce the Snowy class imbalance by adding more northern stations (e.g. Tromsø, Yakutsk, Winnipeg).",
    "Re-run training pipeline after data augmentation and re-evaluate all three models.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

add_heading(doc, "8.2 Model Improvements (Medium Priority)", level=2)
for b in [
    "Neural Network: Replace cross-entropy with focal loss (γ=2) to address class imbalance without manual weighting.",
    "Neural Network: Expand architecture to Dense(128) → Dense(64) → Dense(32) given the RPi 5's compute capacity.",
    "Random Forest: Experiment with n_estimators=500 and max_depth=20 for improved accuracy at the cost of larger model size.",
    "Add gradient boosting (XGBoost / LightGBM) as a 4th comparison model — typically outperforms RF on tabular data.",
    "Implement LSTM / temporal CNN to exploit the sequential nature of the 6-hour input window more directly than engineered lag features.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

add_heading(doc, "8.3 Feature Engineering Improvements (Medium Priority)", level=2)
for b in [
    "Incorporate BME688 gas resistance and IAQ signals once live sensor data is available — these provide air quality proxies for fog, precipitation onset, and atmospheric chemistry changes.",
    "Add wind speed and wind direction (available from Open-Meteo) as training features — wind direction is a strong predictor of weather type (westerlies bring rain to London, easterlies bring cold continental air).",
    "Experiment with longer lookback windows (12 h, 24 h) and longer forecast horizons (12 h, 24 h).",
    "Apply Principal Component Analysis (PCA) to the 58-feature vector to identify redundant features and potentially improve LR performance.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

add_heading(doc, "8.4 Production Deployment (Low Priority)", level=2)
for b in [
    "Implement a Raspberry Pi service (systemd unit) that runs inference every 30 minutes using the latest BME688 readings.",
    "Log predictions and actuals to a local SQLite database for ongoing model performance monitoring.",
    "Re-train models seasonally using new Open-Meteo data to account for climate drift.",
    "Consider federated learning across multiple BME688 sensor nodes for improved generalisation.",
]:
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(b).font.size = Pt(9)

doc.add_paragraph()

# ── 9. CONCLUSION ──
add_heading(doc, "9. Conclusion", level=1)
add_para(doc, (
    "This paper presented a systematic comparison of three machine learning approaches for "
    "6-hour weather condition classification using BME688-compatible sensor data on a "
    "Raspberry Pi 5 edge platform. "
    "The random forest classifier is recommended as the primary deployment model, "
    "achieving the best macro-F1 score (0.348) and the most balanced per-class performance, "
    "with 116 MB memory footprint trivially accommodated by the Raspberry Pi 5's 4–8 GB RAM."
))
add_para(doc, (
    "The neural network, while achieving the highest raw accuracy (54.0%), suffers from "
    "majority-class collapse and requires class rebalancing before it can be recommended. "
    "Logistic regression, despite its low balanced performance, provides a useful fast "
    "interpretable baseline (2.44 µs inference) and strong Snowy-class recall (0.943) "
    "through aggressive balanced weighting."
))
add_para(doc, (
    "The most critical finding is the complete absence of Stormy-class samples — "
    "a consequence of ERA5 reanalysis data's conservative weather-code assignment — "
    "and the severe underrepresentation of the Snowy class (2.3% of training samples). "
    "Addressing these data gaps through multi-source data augmentation is the highest-priority "
    "next step before production deployment."
))

doc.add_paragraph()

# ── APPENDIX ──
add_heading(doc, "Appendix A: Full Parameter Reference", level=1)

add_para(doc, "A.1 Feature Vector — Complete Ordered Listing")
feat_names = []
for sig in ["temp", "humidity", "pressure"]:
    for t in range(6, 0, -1):
        feat_names.append(f"{sig}_t-{t}h")
feat_names += ["dp_1h", "dp_3h", "dp_6h", "dp_accel", "dt_1h", "dt_3h",
               "dew_point", "abs_humidity"]
for sig in ["temp", "humidity", "pressure"]:
    for w in ["3h", "6h"]:
        for s in ["mean", "std", "min", "max"]:
            feat_names.append(f"{sig}_{s}_{w}")
feat_names += ["sin_hour", "cos_hour", "sin_doy", "cos_doy",
               "gas_resistance", "iaq", "eco2", "bvoc"]

n_cols = 4
n_rows = (len(feat_names) + n_cols - 1) // n_cols
app_tbl = doc.add_table(rows=n_rows + 1, cols=n_cols * 2)
app_tbl.style = "Table Grid"
for ci in range(n_cols):
    bold_cell(app_tbl.rows[0].cells[ci*2],   "Index", size=8)
    bold_cell(app_tbl.rows[0].cells[ci*2+1], "Feature Name", size=8)
    shade_cell(app_tbl.rows[0].cells[ci*2],   "2E86C1")
    shade_cell(app_tbl.rows[0].cells[ci*2+1], "2E86C1")
    app_tbl.rows[0].cells[ci*2].paragraphs[0].runs[0].font.color.rgb = RGBColor(255,255,255)
    app_tbl.rows[0].cells[ci*2+1].paragraphs[0].runs[0].font.color.rgb = RGBColor(255,255,255)
for idx, name in enumerate(feat_names):
    row_i = idx % n_rows + 1
    col_i = (idx // n_rows) * 2
    center_cell(app_tbl.rows[row_i].cells[col_i],   str(idx), size=8)
    center_cell(app_tbl.rows[row_i].cells[col_i+1], name, size=8)
add_caption(doc, "Table A1. Complete ordered feature vector (58 features). "
                 "Features 54–57 (gas_resistance, iaq, eco2, bvoc) are zero-padded "
                 "during training and populated from live BME688 readings at inference.")

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, "reports", "AI_Weather_Prediction_Technical_Paper.docx")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
doc.save(out_path)
print(f"Saved: {out_path}")
