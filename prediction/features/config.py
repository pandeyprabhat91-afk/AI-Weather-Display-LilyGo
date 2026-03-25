# prediction/features/config.py

RANDOM_SEED = 42

LOOKBACK = 24  # hours of history used as input (captures full diurnal cycle)
LOOKAHEAD = 6  # hours ahead to predict

# --- Labels ---
# Stormy (WMO 95-99) is absent from Open-Meteo historical archive for all 6
# training stations → merged into Rainy (class 2).  Snowy moves to index 3
# so there is no dead neuron in the classifier.
LABEL_MAP = {
    0: "Sunny",
    1: "Cloudy",
    2: "Rainy",
    3: "Snowy",
}
CLASS_NAMES = [LABEL_MAP[i] for i in range(4)]
N_CLASSES = 4

# --- WMO weather code → label int ---
WMO_MAP: dict = {}
for code in [0, 1]:
    WMO_MAP[code] = 0  # Sunny
for code in [2, 3, 45, 46, 47, 48]:
    WMO_MAP[code] = 1  # Cloudy
for code in [51, 52, 53, 54, 55, 56, 57,
             61, 62, 63, 64, 65, 66, 67,
             80, 81, 82,
             95, 96, 97, 98, 99]:   # Stormy merged into Rainy
    WMO_MAP[code] = 2  # Rainy
for code in [71, 72, 73, 74, 75, 76, 77, 85, 86]:
    WMO_MAP[code] = 3  # Snowy

# --- Default training stations: (name, latitude, longitude) ---
# First 3: original coverage (Europe + tropical); last 3: high-thunderstorm
# stations that reliably produce WMO 95-99 (Stormy) labels
STATIONS = [
    ("London, UK",        51.5074,  -0.1278),
    ("Helsinki, Finland", 60.1699,  25.0002),
    ("Singapore",          1.3521, 103.8198),
    ("Orlando, US",       28.5383, -81.3792),
    ("Dhaka, Bangladesh", 23.7275,  90.4070),
    ("Manaus, Brazil",    -3.1190, -60.0217),
]

# --- BME688 optional columns (zero-padded if absent) ---
BME688_EXTRAS = ["gas_resistance", "iaq", "eco2", "bvoc"]

# --- Feature count bookkeeping ---
# 72 raw lags (3 signals × 24 steps)
#  + 3 pressure tendency (1h, 12h, 24h)
#  + 1 pressure acceleration
#  + 2 temp rate of change (1h, 12h)
#  + 1 dew point  + 1 abs humidity
#  + 36 rolling stats (3 signals × 3 windows × 4 stats)
#  + 4 cyclical time
#  + 8 discriminative threshold features (dew-point depression, freeze flags,
#       pressure-trend sign, normalised dp-dep, snow composite, rain composite)
# = 128
# BME688 extras (gas/IAQ/eCO2/bVOC) are EXCLUDED — inference uses only
# temperature, humidity and pressure from the BME688 sensor.
CORE_FEATURE_COUNT = 128
TOTAL_FEATURE_COUNT = CORE_FEATURE_COUNT  # 128
