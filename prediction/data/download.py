# prediction/data/download.py
import os
import pandas as pd
import numpy as np
from features.config import WMO_MAP, STATIONS

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def _get_api_client():
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    session = requests_cache.CachedSession(
        os.path.join(CACHE_DIR, ".http_cache"), expire_after=-1
    )
    session = retry(session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=session)


def fetch_weather_data(stations=None, years: int = 5, force_download: bool = False) -> pd.DataFrame:
    if stations is None:
        stations = STATIONS
    frames = []
    client = _get_api_client()
    import datetime
    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today() - datetime.timedelta(days=365 * years)).isoformat()

    for name, lat, lon in stations:
        slug = name.lower().replace(" ", "_").replace(",", "")
        cache_path = os.path.join(CACHE_DIR, f"{slug}_{years}y.csv")
        if os.path.exists(cache_path) and not force_download:
            df = pd.read_csv(cache_path, parse_dates=["time"])
            frames.append(df)
            continue
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": ["temperature_2m", "relative_humidity_2m", "pressure_msl", "weather_code"],
            "start_date": start_date,
            "end_date": end_date,
        }
        responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        r = responses[0]
        hourly = r.Hourly()
        df = pd.DataFrame({
            "time": pd.date_range(
                start=pd.Timestamp(hourly.Time(), unit="s", tz="UTC"),
                end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="UTC"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            ),
            "temperature": hourly.Variables(0).ValuesAsNumpy(),
            "humidity": hourly.Variables(1).ValuesAsNumpy(),
            "pressure": hourly.Variables(2).ValuesAsNumpy(),
            "weather_code": hourly.Variables(3).ValuesAsNumpy().astype(int),
        })
        df["station"] = name
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_csv(cache_path, index=False)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values("time").reset_index(drop=True)


def apply_wmo_mapping(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label"] = df["weather_code"].map(WMO_MAP)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def forward_fill_gaps(df: pd.DataFrame, max_gap_hours: int = 2) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)
    numeric_cols = ["temperature", "humidity", "pressure"]
    for col in numeric_cols:
        is_nan = df[col].isna()
        run_id = (is_nan != is_nan.shift()).cumsum()
        run_lengths = is_nan.groupby(run_id).transform("sum")
        long_gap = is_nan & (run_lengths > max_gap_hours)
        # Only ffill short gaps (not long ones)
        df[col] = df[col].ffill()
        # Re-mark long gap positions as NaN so they get dropped
        df.loc[long_gap, col] = np.nan
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    """Split data per-station chronologically (70/15/15) then concatenate.
    This guarantees every climate zone and rare class (Stormy, Snowy) is
    represented in every split, regardless of when those events occurred."""
    if "station" not in df.columns or df["station"].nunique() <= 1:
        df = df.sort_values("time").reset_index(drop=True)
        n = len(df)
        i_train = int(n * 0.70)
        i_val = int(n * 0.85)
        return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()

    train_parts, val_parts, test_parts = [], [], []
    for _, group in df.groupby("station", sort=False):
        g = group.sort_values("time").reset_index(drop=True)
        n = len(g)
        i_train = int(n * 0.70)
        i_val   = int(n * 0.85)
        train_parts.append(g.iloc[:i_train])
        val_parts.append(g.iloc[i_train:i_val])
        test_parts.append(g.iloc[i_val:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts,   ignore_index=True),
        pd.concat(test_parts,  ignore_index=True),
    )


def prepare_data(stations=None, years: int = 5, force_download: bool = False):
    raw = fetch_weather_data(stations=stations, years=years, force_download=force_download)
    raw = apply_wmo_mapping(raw)
    raw = forward_fill_gaps(raw)
    train, val, test = split_data(raw)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print(f"Data prepared: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
