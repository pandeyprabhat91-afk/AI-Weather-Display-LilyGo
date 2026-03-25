import numpy as np
import pandas as pd
from features.config import LOOKBACK, LOOKAHEAD, TOTAL_FEATURE_COUNT


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
    When a 'station' column is present and multiple stations exist, windows are
    extracted per-station to avoid cross-station temporal contamination.
    Returns: X float32 (n, 54), y int32 (n,)
    """
    if "station" in df.columns and df["station"].nunique() > 1:
        parts_X, parts_y = [], []
        for _, group in df.groupby("station", sort=False):
            X, y = _build_features_single(group)
            if len(X) > 0:
                parts_X.append(X)
                parts_y.append(y)
        if not parts_X:
            return np.empty((0, TOTAL_FEATURE_COUNT), dtype=np.float32), np.empty(0, dtype=np.int32)
        return np.concatenate(parts_X, axis=0), np.concatenate(parts_y, axis=0)
    return _build_features_single(df)


def _build_features_single(df: pd.DataFrame):
    """
    Extract sliding-window features from a single-station hourly DataFrame.
    """
    df = df.sort_values("time").reset_index(drop=True)

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

        # 2. Pressure tendency at 1h, 12h, 24h (3)
        dp1h  = pres_vals[-1] - pres_vals[-2]
        dp12h = pres_vals[-1] - pres_vals[-13]
        dp24h = pres_vals[-1] - pres_vals[0]
        feat.extend([dp1h, dp12h, dp24h])

        # 3. Pressure acceleration: Δp_12h − Δp_24h (1)
        feat.append(dp12h - dp24h)

        # 4. Temp rate of change at 1h, 12h (2)
        feat.append(temp_vals[-1] - temp_vals[-2])
        feat.append(temp_vals[-1] - temp_vals[-13])

        # 5. Dew point from t-1h (1)
        feat.append(compute_dew_point(T=temp_vals[-1], RH=hum_vals[-1]))

        # 6. Absolute humidity from t-1h (1)
        feat.append(compute_abs_humidity(T=temp_vals[-1], RH=hum_vals[-1]))

        # 7. Rolling stats: mean, std, min, max × 6h, 12h, 24h × 3 signals (36)
        for col_vals in [temp_vals, hum_vals, pres_vals]:
            for window in [col_vals[-6:], col_vals[-12:], col_vals]:
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

        # 9. Discriminative threshold features (8)
        t_now   = float(temp_vals[-1])
        rh_now  = float(hum_vals[-1])
        dp_now  = compute_dew_point(T=t_now, RH=rh_now)
        dp_depression = t_now - dp_now          # near 0 = near-saturation (rain/snow)
        feat.append(dp_depression)              # 1: dew-point depression

        t_min24 = float(np.min(temp_vals))
        feat.append(float(t_min24 < 3.0))      # 2: any hour near-freezing in window
        feat.append(float(t_now < 0.0))         # 3: currently below freezing
        feat.append(float(t_now < 3.0))         # 4: currently near-freezing

        rh_mean24 = float(np.mean(hum_vals))
        pres_trend_sign = float(np.sign(dp24h)) # +1 rising / -1 falling / 0 steady
        feat.append(pres_trend_sign)            # 5: pressure trend direction

        # Wet-bulb approximation: Tw ≈ T * atan(0.151977*(RH+8.313659)^0.5) + ...
        # Simplified proxy: just dew-point depression normalised by temperature
        dp_dep_norm = dp_depression / (abs(t_now) + 1.0)
        feat.append(dp_dep_norm)                # 6: normalised dew-point depression

        # Snow composite: cold + humid (high value = likely snow)
        snow_score = float(t_min24 < 3.0) * rh_mean24 / 100.0
        feat.append(snow_score)                 # 7: cold-humid composite

        # Rain composite: high humidity + falling pressure
        rain_score = (rh_mean24 / 100.0) * max(0.0, -dp24h / 10.0)
        feat.append(rain_score)                 # 8: rain composite

        assert len(feat) == TOTAL_FEATURE_COUNT
        records.append(feat)
        labels.append(int(target_label))

    if not records:
        return np.empty((0, TOTAL_FEATURE_COUNT), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.array(records, dtype=np.float32), np.array(labels, dtype=np.int32)
