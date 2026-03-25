from features.config import (
    RANDOM_SEED, LABEL_MAP, WMO_MAP, STATIONS,
    CLASS_NAMES, N_CLASSES, LOOKBACK, LOOKAHEAD,
    BME688_EXTRAS, CORE_FEATURE_COUNT, TOTAL_FEATURE_COUNT,
)

def test_random_seed_is_42():
    assert RANDOM_SEED == 42

def test_label_map_has_five_classes():
    assert set(LABEL_MAP.keys()) == {0, 1, 2, 3, 4}
    assert LABEL_MAP[0] == "Sunny"
    assert LABEL_MAP[4] == "Snowy"

def test_wmo_map_covers_expected_codes():
    assert WMO_MAP[0] == 0
    assert WMO_MAP[1] == 0
    assert WMO_MAP[2] == 1
    assert WMO_MAP[45] == 1
    assert WMO_MAP[61] == 2
    assert WMO_MAP[80] == 2
    assert WMO_MAP[95] == 3
    assert WMO_MAP[99] == 3
    assert WMO_MAP[71] == 4
    assert WMO_MAP[85] == 4

def test_wmo_map_does_not_contain_unmapped_codes():
    assert 4 not in WMO_MAP
    assert 50 not in WMO_MAP

def test_feature_count():
    assert CORE_FEATURE_COUNT == 54
    assert len(BME688_EXTRAS) == 4
    assert TOTAL_FEATURE_COUNT == 58

def test_stations_has_three_entries():
    assert len(STATIONS) == 3
