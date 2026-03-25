"""
AI Weather Prediction Pipeline — Main Orchestrator

Usage:
  python main.py                   # full pipeline (download+train+evaluate)
  python main.py --report          # full pipeline + generate HTML report
  python main.py --only download
  python main.py --only train
  python main.py --only evaluate
  python main.py --only report
  python main.py --locations "London, UK" "Helsinki, Finland" "Singapore" --years 5
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
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))

from data.download import prepare_data
from features.engineering import build_features
from features.config import RANDOM_SEED, TOTAL_FEATURE_COUNT, CLASS_NAMES, STATIONS
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.neural_network import NeuralNetworkModel
from evaluation.metrics import compute_all_metrics
from deployment.export import export_scaler_header, export_lr_header, export_rf_c, export_tflite
from report.generate import generate_report

BASE = os.path.dirname(__file__)
PROCESSED_DIR = os.path.join(BASE, "data", "processed")
DEPLOY_DIR    = os.path.join(BASE, "deployment")
REPORT_DIR    = os.path.join(BASE, "reports")
EVAL_DIR      = os.path.join(BASE, "evaluation", "outputs")


def _require(path: str, message: str) -> None:
    if not os.path.exists(path):
        print(f"ERROR: {message}")
        sys.exit(1)


def _file_kb(path: str) -> float:
    try:
        return round(os.path.getsize(path) / 1024, 1)
    except FileNotFoundError:
        return 0.0


def _stratified_sample_idx(y: np.ndarray, n: int) -> np.ndarray:
    n = min(n, len(y))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=RANDOM_SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return idx


def _lr_inference_us() -> float:
    flops = 2 * TOTAL_FEATURE_COUNT * 5 + 5
    return round(flops / 240, 2)


def _rf_inference_us(rf) -> float:
    flops = 2 * rf.model.max_depth * rf.model.n_estimators
    return round(flops / 240, 2)


def _nn_inference_us() -> float:
    flops = 2 * (TOTAL_FEATURE_COUNT * 64 + 64 * 32 + 32 * 5)
    return round(flops / 240, 2)


def _build_feature_names() -> list:
    names = []
    for sig in ["temp", "humidity", "pressure"]:
        for t in range(6, 0, -1):
            names.append(f"{sig}_t-{t}h")
    names += ["dp_1h", "dp_3h", "dp_6h", "dp_accel", "dt_1h", "dt_3h",
              "dew_point", "abs_humidity"]
    for sig in ["temp", "humidity", "pressure"]:
        for window in ["3h", "6h"]:
            for stat in ["mean", "std", "min", "max"]:
                names.append(f"{sig}_{stat}_{window}")
    names += ["sin_hour", "cos_hour", "sin_doy", "cos_doy",
              "gas_resistance", "iaq", "eco2", "bvoc"]
    return names


def stage_download(args):
    stations = None
    if args.locations:
        stations = []
        for loc in args.locations:
            matched = next((s for s in STATIONS if s[0].lower() == loc.lower()), None)
            if matched:
                stations.append(matched)
            else:
                print(f"WARNING: Station '{loc}' not found in defaults — using all defaults.")
                stations = None
                break
    prepare_data(stations=stations, years=args.years, force_download=args.force_download)


def stage_train(args):
    _require(os.path.join(PROCESSED_DIR, "train.csv"),
             "data/processed/train.csv not found. Run: python main.py --only download")
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"), parse_dates=["time"])
    val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"),   parse_dates=["time"])

    print("Building features...")
    X_train, y_train = build_features(train_df)
    X_val,   y_val   = build_features(val_df)
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    os.makedirs(DEPLOY_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(DEPLOY_DIR, "scaler.pkl"))
    export_scaler_header(scaler, os.path.join(DEPLOY_DIR, "scaler_params.h"))

    print("Training Logistic Regression...")
    lr = LogisticRegressionModel(random_seed=RANDOM_SEED)
    lr.fit(X_train_s, y_train)
    lr.save(os.path.join(DEPLOY_DIR, "lr_model.pkl"))
    export_lr_header(lr, os.path.join(DEPLOY_DIR, "lr_coefficients.h"))

    print("Training Random Forest...")
    rf = RandomForestModel(random_seed=RANDOM_SEED)
    rf.fit(X_train_s, y_train)
    rf.save(os.path.join(DEPLOY_DIR, "rf_model.pkl"))
    export_rf_c(rf, os.path.join(DEPLOY_DIR, "rf_model.c"))

    print("Training Neural Network...")
    nn = NeuralNetworkModel(random_seed=RANDOM_SEED)
    nn.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)
    nn.save(os.path.join(DEPLOY_DIR, "nn_model.keras"))

    cal_idx = _stratified_sample_idx(y_train, n=1000)
    export_tflite(nn.get_keras_model(), X_train_s[cal_idx],
                  os.path.join(DEPLOY_DIR, "model.tflite"),
                  header_path=os.path.join(DEPLOY_DIR, "model_data.h"))
    print("Training complete. Deployment artifacts in deployment/")


def stage_evaluate(args):
    for name in ["lr_model.pkl", "rf_model.pkl", "nn_model.keras", "scaler.pkl"]:
        _require(os.path.join(DEPLOY_DIR, name),
                 f"deployment/{name} not found. Run: python main.py --only train")
    _require(os.path.join(PROCESSED_DIR, "test.csv"),
             "data/processed/test.csv not found. Run: python main.py --only download")

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"), parse_dates=["time"])
    X_test, y_test = build_features(test_df)
    scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.pkl"))
    X_test_s = scaler.transform(X_test).astype(np.float32)

    lr = LogisticRegressionModel.load(os.path.join(DEPLOY_DIR, "lr_model.pkl"))
    rf = RandomForestModel.load(os.path.join(DEPLOY_DIR, "rf_model.pkl"))
    nn = NeuralNetworkModel.load(os.path.join(DEPLOY_DIR, "nn_model.keras"))

    results = []
    for name, model, importance, imp_type, infer_us in [
        ("Logistic Regression", lr, lr.get_feature_importances(),  "lr", _lr_inference_us()),
        ("Random Forest",       rf, rf.get_feature_importances(),  "rf", _rf_inference_us(rf)),
        ("Neural Network",      nn, lr.get_feature_importances(),  "lr", _nn_inference_us()),
    ]:
        y_pred  = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
        metrics = compute_all_metrics(y_test, y_pred, y_proba)
        artifact_paths = {
            "Logistic Regression": os.path.join(DEPLOY_DIR, "lr_coefficients.h"),
            "Random Forest":       os.path.join(DEPLOY_DIR, "rf_model.c"),
            "Neural Network":      os.path.join(DEPLOY_DIR, "model.tflite"),
        }
        results.append({
            "name": name,
            "metrics": metrics,
            "artifact_size_kb": _file_kb(artifact_paths[name]),
            "inference_us": infer_us,
            "importance": importance,
            "importance_type": imp_type,
            "y_proba": y_proba,
        })
        print(f"{name}: accuracy={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    serializable = []
    for r in results:
        m = dict(r["metrics"])
        m["confusion_matrix"] = m["confusion_matrix"].tolist()
        serializable.append({"name": r["name"], "metrics": m,
                              "artifact_size_kb": r["artifact_size_kb"],
                              "inference_us": r["inference_us"]})
    with open(os.path.join(EVAL_DIR, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)

    np.save(os.path.join(EVAL_DIR, "y_test.npy"), y_test)
    for r in results:
        key = r["name"].replace(" ", "_")
        np.save(os.path.join(EVAL_DIR, f"y_proba_{key}.npy"), r["y_proba"])
        np.save(os.path.join(EVAL_DIR, f"importance_{key}.npy"), r["importance"])

    print(f"Evaluation complete. Results in {EVAL_DIR}/results.json")


def stage_report(args):
    _require(os.path.join(EVAL_DIR, "results.json"),
             "evaluation/outputs/results.json not found. Run: python main.py --only evaluate")

    with open(os.path.join(EVAL_DIR, "results.json")) as f:
        serialized = json.load(f)

    y_true = np.load(os.path.join(EVAL_DIR, "y_test.npy"))
    models_for_report = []
    for s in serialized:
        key = s["name"].replace(" ", "_")
        s["metrics"]["confusion_matrix"] = np.array(s["metrics"]["confusion_matrix"])
        models_for_report.append({
            "name": s["name"],
            "metrics": s["metrics"],
            "artifact_size_kb": s["artifact_size_kb"],
            "inference_us": s["inference_us"],
            "importance": np.load(os.path.join(EVAL_DIR, f"importance_{key}.npy")),
            "importance_type": "lr" if "Logistic" in s["name"] else ("rf" if "Forest" in s["name"] else "lr"),
            "y_proba": np.load(os.path.join(EVAL_DIR, f"y_proba_{key}.npy")),
        })

    generate_report({
        "dataset_description": "Open-Meteo historical hourly data, 3 stations (London, Helsinki, Singapore), 5 years, 6-hour ahead forecast.",
        "models": models_for_report,
        "y_true": y_true,
        "feature_names": _build_feature_names(),
    }, os.path.join(REPORT_DIR, "report_output.html"))


def main():
    parser = argparse.ArgumentParser(description="AI Weather Prediction Pipeline")
    parser.add_argument("--only", choices=["download", "train", "evaluate", "report"])
    parser.add_argument("--locations", nargs="+")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--report", action="store_true")
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
