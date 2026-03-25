import pandas as pd
import numpy as np
import pytest
from data.download import apply_wmo_mapping, split_data, forward_fill_gaps


def make_raw_df(n=200):
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": times,
        "temperature": np.random.uniform(-10, 35, n),
        "humidity": np.random.uniform(20, 100, n),
        "pressure": np.random.uniform(980, 1030, n),
        "weather_code": np.random.choice([0, 1, 2, 61, 71, 95], n),
    })


def test_apply_wmo_mapping_maps_known_codes():
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=3, freq="h"),
        "temperature": [10.0, 10.0, 10.0],
        "humidity": [60.0, 60.0, 60.0],
        "pressure": [1013.0, 1013.0, 1013.0],
        "weather_code": [0, 61, 95],
    })
    result = apply_wmo_mapping(df)
    assert list(result["label"]) == [0, 2, 3]


def test_apply_wmo_mapping_drops_unmapped_codes():
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", periods=3, freq="h"),
        "temperature": [10.0, 10.0, 10.0],
        "humidity": [60.0, 60.0, 60.0],
        "pressure": [1013.0, 1013.0, 1013.0],
        "weather_code": [0, 4, 61],
    })
    result = apply_wmo_mapping(df)
    assert len(result) == 2
    assert 4 not in result["weather_code"].values


def test_forward_fill_gaps_fills_short_gaps():
    df = make_raw_df(10)
    df.loc[3, "temperature"] = np.nan
    df.loc[4, "temperature"] = np.nan
    result = forward_fill_gaps(df)
    assert result["temperature"].isna().sum() == 0


def test_forward_fill_gaps_drops_long_gaps():
    df = make_raw_df(10)
    df.loc[3:6, "temperature"] = np.nan
    result = forward_fill_gaps(df)
    assert result["temperature"].isna().sum() == 0
    assert len(result) < 10


def test_split_data_proportions():
    df = make_raw_df(1000)
    df["label"] = np.random.randint(0, 5, 1000)
    train, val, test = split_data(df)
    assert len(train) == pytest.approx(700, abs=5)
    assert len(val) == pytest.approx(150, abs=5)
    assert len(test) == pytest.approx(150, abs=5)


def test_split_data_is_chronological():
    df = make_raw_df(1000)
    df["label"] = 0
    train, val, test = split_data(df)
    assert train["time"].max() < val["time"].min()
    assert val["time"].max() < test["time"].min()
