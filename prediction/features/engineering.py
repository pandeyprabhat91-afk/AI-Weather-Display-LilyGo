import numpy as np
import pandas as pd
from features.config import LOOKBACK, LOOKAHEAD, BME688_EXTRAS, TOTAL_FEATURE_COUNT


def compute_dew_point(T: float, RH: float) -> float:
    """Magnus formula. T in °C, RH in %. Uses most recent (t-1h) values."""
    gamma = np.log(RH / 100.0) + (17.625 * T) / (243.04 + T)
    return 243.04 * gamma / (17.625 - gamma)


def compute_abs_humidity(T: float, RH: float) -> float:
    """August-Roche-Magnus absolute humidity g/m³. Uses most recent (t-1h) values."""
    return (6.112 * np.exp((17.67 * T) / (T + 243.5)) * RH * 2.1674) / (273.15 + T)


def build_features(df: pd.DataFrame):
    """
    Build feature matrix X and label vector y from hourly weather DataFrame.
    df must have: time, temperature, humidity, pressure, label
    Optionally: gas_resistance, iaq, eco2, bvoc (zero-padded if absent)
    Returns: X float32 (n, 58), y int32 (n,)
    """
    df = df.sort_values("time").reset_index(drop=True)
    has_extras = all(col in df.columns for col in BME688_EXTRAS)

    records = []
    labels = []

    for i in range(LOOKBACK, len(df) - LOOKAHEAD):
        past = df.iloc[i - LOOKBACK: i]
        target_label = df.iloc[i + LOOKAHEAD]["label"]
        if pd.isna(target_label):
            continue

        feat = []
        temp_vals = past["temperature"].values.astype(float)
        hum_vals  = past["humidity"].values.astype(float)
        pres_vals = past["pressure"].values.astype(float)

        # 1. Raw lags: temp, humidity, pressure × 6 (18)
        feat.extend(temp_vals)
        feat.extend(hum_vals)
        feat.extend(pres_vals)

        # 2. Pressure tendency at 1h, 3h, 6h (3)
        dp1h = pres_vals[-1] - pres_vals[-2]
        dp3h = pres_vals[-1] - pres_vals[-4]
        dp6h = pres_vals[-1] - pres_vals[0]
        feat.extend([dp1h, dp3h, dp6h])

        # 3. Pressure acceleration: Δp_3h − Δp_6h (1)
        feat.append(dp3h - dp6h)

        # 4. Temp rate of change at 1h, 3h (2)
        feat.append(temp_vals[-1] - temp_vals[-2])
        feat.append(temp_vals[-1] - temp_vals[-4])

        # 5. Dew point from t-1h (1)
        feat.append(compute_dew_point(T=temp_vals[-1], RH=hum_vals[-1]))

        # 6. Absolute humidity from t-1h (1)
        feat.append(compute_abs_humidity(T=temp_vals[-1], RH=hum_vals[-1]))

        # 7. Rolling stats: mean, std, min, max × 3h & 6h × 3 signals (24)
        for col_vals in [temp_vals, hum_vals, pres_vals]:
            for window in [col_vals[-3:], col_vals]:
                feat.extend([float(np.mean(window)), float(np.std(window)),
                              float(np.min(window)),  float(np.max(window))])

        # 8. Cyclical time (4)
        current_time = df.iloc[i]["time"]
        hour = current_time.hour
        doy  = current_time.day_of_year
        feat.extend([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * doy  / 365), np.cos(2 * np.pi * doy  / 365),
        ])

        # 9. BME688 extras (4, zero-padded)
        if has_extras:
            for col in BME688_EXTRAS:
                feat.append(float(past.iloc[-1][col]))
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        assert len(feat) == TOTAL_FEATURE_COUNT
        records.append(feat)
        labels.append(int(target_label))

    if not records:
        return np.empty((0, TOTAL_FEATURE_COUNT), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(records, dtype=np.float32), np.array(labels, dtype=np.int32)
