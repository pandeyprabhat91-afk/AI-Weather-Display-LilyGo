"""Standalone report regeneration script."""
import sys
import os

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from report.generate import generate_report
from main import _build_feature_names

EVAL = os.path.join(os.path.dirname(__file__), "evaluation", "outputs")
REPORTS = os.path.join(os.path.dirname(__file__), "reports")

with open(os.path.join(EVAL, "results.json")) as f:
    data = json.load(f)

y_true = np.load(os.path.join(EVAL, "y_test.npy"))

mods = []
for m in data:
    key = m["name"].replace(" ", "_")
    metrics = dict(m["metrics"])
    metrics["confusion_matrix"] = np.array(metrics["confusion_matrix"])
    mods.append({
        "name": m["name"],
        "metrics": metrics,
        "artifact_size_kb": m["artifact_size_kb"],
        "inference_us": m["inference_us"],
        "importance": np.load(os.path.join(EVAL, f"importance_{key}.npy")),
        "importance_type": "lr" if "Logistic" in m["name"] else "rf",
        "y_proba": np.load(os.path.join(EVAL, f"y_proba_{key}.npy")),
    })

generate_report(
    {
        "dataset_description": (
            "Open-Meteo historical hourly data, 6 stations "
            "(London, Helsinki, Singapore, Orlando, Dhaka, Manaus), "
            "5 years, 24-hour sliding window, 4-class forecast "
            "(Sunny / Cloudy / Rainy / Snowy)."
        ),
        "models": mods,
        "y_true": y_true,
        "feature_names": _build_feature_names(),
    },
    os.path.join(REPORTS, "report_output.html"),
)
print("REPORT_DONE")
