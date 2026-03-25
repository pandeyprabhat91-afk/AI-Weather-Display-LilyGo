import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from features.config import CLASS_NAMES


def _save_fig(fig, output_dir: str, name: str) -> str:
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return name


def generate_report(results: dict, output_path: str) -> None:
    """
    results = {
      "dataset_description": str,
      "models": [{"name", "metrics", "artifact_size_kb", "inference_us",
                  "importance", "importance_type", "y_proba"}, ...],
      "y_true": np.ndarray,
      "feature_names": list[str],
    }
    """
    from evaluation.plots import confusion_matrix_fig, roc_fig, feature_importance_fig

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    model_rows = []
    models_proba = {}

    for m in results["models"]:
        metrics = m["metrics"]
        model_rows.append({
            "name": m["name"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics.get("roc_auc_ovr", 0.0),
            "artifact_size_kb": m["artifact_size_kb"],
            "inference_us": m["inference_us"],
            "per_class": list(zip(CLASS_NAMES,
                                  metrics["per_class_precision"],
                                  metrics["per_class_recall"],
                                  metrics["per_class_f1"])),
        })
        models_proba[m["name"]] = m["y_proba"]

        fig = confusion_matrix_fig(metrics["confusion_matrix"],
                                   title=f"{m['name']} — Confusion Matrix")
        _save_fig(fig, output_dir, f"cm_{m['name'].lower().replace(' ', '_')}.png")

        fig = feature_importance_fig(m["importance"], results["feature_names"],
                                     model_type=m["importance_type"],
                                     title=f"{m['name']} — Feature Importance")
        _save_fig(fig, output_dir, f"fi_{m['name'].lower().replace(' ', '_')}.png")

    fig = roc_fig(results["y_true"], models_proba)
    _save_fig(fig, output_dir, "roc_curves.png")

    y = results["y_true"]
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([CLASS_NAMES[i] for i in unique], counts, color="steelblue")
    ax.set_xlabel("Class"); ax.set_ylabel("Count"); ax.set_title("Test Set Class Distribution")
    fig.tight_layout()
    _save_fig(fig, output_dir, "class_distribution.png")

    best = max(model_rows, key=lambda x: x["macro_f1"])
    best_name = best["name"]
    rec_text = (f"{best_name} achieved macro F1 of {best['macro_f1']:.3f}, "
                f"accuracy {best['accuracy']:.3f}. "
                f"Artifact: {best['artifact_size_kb']} KB. "
                f"Est. inference: {best['inference_us']} µs on ESP32-S3 at 240 MHz.")

    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    html = env.get_template("template.html").render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        dataset_description=results["dataset_description"],
        models=model_rows,
        best_model=best_name,
        recommendation_text=rec_text,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Report written to {output_path}")
