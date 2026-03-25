from features.config import (
    RANDOM_SEED, LABEL_MAP, WMO_MAP, STATIONS,
    CLASS_NAMES, N_CLASSES, LOOKBACK, LOOKAHEAD,
    BME688_EXTRAS, CORE_FEATURE_COUNT, TOTAL_FEATURE_COUNT,
)

def test_random_seed_is_42():
    assert RANDOM_SEED == 42

def test_label_map_has_four_classes():
    assert set(LABEL_MAP.keys()) == {0, 1, 2, 3}
    assert LABEL_MAP[0] == "Sunny"
    assert LABEL_MAP[3] == "Snowy"

def test_wmo_map_covers_expected_codes():
    assert WMO_MAP[0] == 0
    assert WMO_MAP[1] == 0
    assert WMO_MAP[2] == 1
    assert WMO_MAP[45] == 1
    assert WMO_MAP[61] == 2
    assert WMO_MAP[80] == 2
    assert WMO_MAP[95] == 2  # Stormy merged into Rainy
    assert WMO_MAP[99] == 2  # Stormy merged into Rainy
    assert WMO_MAP[71] == 3
    assert WMO_MAP[85] == 3

def test_wmo_map_does_not_contain_unmapped_codes():
    assert 50 not in WMO_MAP

def test_feature_count():
    assert CORE_FEATURE_COUNT == 128
    assert len(BME688_EXTRAS) == 4  # list still defined for documentation
    assert TOTAL_FEATURE_COUNT == 128  # BME688 extras excluded; only temp/humidity/pressure used

def test_stations_has_six_entries():
    assert len(STATIONS) == 6
