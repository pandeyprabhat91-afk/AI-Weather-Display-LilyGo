"""Quick per-class analysis of saved evaluation outputs."""
import json
import numpy as np
import sys
sys.path.insert(0, ".")
from sklearn.metrics import classification_report, confusion_matrix
from features.config import CLASS_NAMES

y_test = np.load("evaluation/outputs/y_test.npy")
unique, counts = np.unique(y_test, return_counts=True)
print("Test label distribution:")
for u, c in zip(unique, counts):
    print(f"  {CLASS_NAMES[u]} ({u}): {c}")
print()

for model in ["Logistic_Regression", "Random_Forest", "XGBoost", "Neural_Network"]:
    y_proba = np.load(f"evaluation/outputs/y_proba_{model}.npy")
    y_pred = y_proba.argmax(axis=-1)
    print(f"=== {model.replace('_', ' ')} ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    header = "        " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i]:>6}  " + "  ".join(f"{v:>8}" for v in row))
    print()
