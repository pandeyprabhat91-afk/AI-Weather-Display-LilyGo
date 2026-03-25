import os
import sys
import numpy as np
import tensorflow as tf
import m2cgen as m2c
from sklearn.preprocessing import StandardScaler
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel


def export_scaler_header(scaler: StandardScaler, output_path: str) -> None:
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
    coef = model.model.coef_
    intercept = model.model.intercept_
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
    # m2cgen recursively traverses every tree node; raise the limit so large
    # forests (400 trees × depth 12) don't hit Python's default 1000-frame cap.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 100_000))
    try:
        code = m2c.export_to_c(model.model)
    finally:
        sys.setrecursionlimit(old_limit)
    with open(output_path, "w") as f:
        f.write(code)


def export_tflite(keras_model, representative_data: np.ndarray,
                  tflite_path: str, header_path: str = None) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(representative_data)):
            yield [representative_data[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    if header_path is not None:
        _write_c_array(tflite_model, header_path, "model_data")


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
