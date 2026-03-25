# prediction/features/config.py

RANDOM_SEED = 42

LOOKBACK = 6   # hours of history used as input
LOOKAHEAD = 6  # hours ahead to predict

# --- Labels ---
LABEL_MAP = {
    0: "Sunny",
    1: "Cloudy",
    2: "Rainy",
    3: "Stormy",
    4: "Snowy",
}
CLASS_NAMES = [LABEL_MAP[i] for i in range(5)]
N_CLASSES = 5

# --- WMO weather code → label int ---
WMO_MAP: dict = {}
for code in [0, 1]:
    WMO_MAP[code] = 0  # Sunny
for code in [2, 3, 45, 46, 47, 48]:
    WMO_MAP[code] = 1  # Cloudy
for code in [51, 52, 53, 54, 55, 56, 57,
             61, 62, 63, 64, 65, 66, 67,
             80, 81, 82]:
    WMO_MAP[code] = 2  # Rainy
for code in [95, 96, 97, 98, 99]:
    WMO_MAP[code] = 3  # Stormy
for code in [71, 72, 73, 74, 75, 76, 77, 85, 86]:
    WMO_MAP[code] = 4  # Snowy

# --- Default training stations: (name, latitude, longitude) ---
STATIONS = [
    ("London, UK",        51.5074, -0.1278),
    ("Helsinki, Finland", 60.1699, 25.0002),
    ("Singapore",          1.3521, 103.8198),
]

# --- BME688 optional columns (zero-padded if absent) ---
BME688_EXTRAS = ["gas_resistance", "iaq", "eco2", "bvoc"]

# --- Feature count bookkeeping ---
# 18 raw lags + 3 pressure tendency + 1 pressure accel +
# 2 temp rate + 1 dew point + 1 abs humidity +
# 24 rolling stats + 4 cyclical time = 54
CORE_FEATURE_COUNT = 54
TOTAL_FEATURE_COUNT = CORE_FEATURE_COUNT + len(BME688_EXTRAS)  # 58
