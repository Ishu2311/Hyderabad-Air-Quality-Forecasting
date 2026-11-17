"""
Centralized Configuration Module for Hyderabad Air Quality Research
Milestone 1: Data Foundation and Preprocessing
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class StationMetadata:
    name: str
    station_id: str
    latitude: float
    longitude: float
    easting_utm44n: float
    northing_utm44n: float
    typology: str

STATION_REGISTRY: Dict[str, StationMetadata] = {
    "Bollaram Industrial Area": StationMetadata(
        name="Bollaram Industrial Area",
        station_id="HYD_BOL",
        latitude=17.5400,
        longitude=78.3500,
        easting_utm44n=218650.0,
        northing_utm44n=1941800.0,
        typology="Heavy Industrial / Pharmaceutical Corridor"
    ),
    "Central University": StationMetadata(
        name="Central University",
        station_id="HYD_UOH",
        latitude=17.4600,
        longitude=78.3300,
        easting_utm44n=216420.0,
        northing_utm44n=1932950.0,
        typology="Institutional Campus / Vegetative Background"
    ),
    "ECIL Kapra": StationMetadata(
        name="ECIL Kapra",
        station_id="HYD_ECL",
        latitude=17.4700,
        longitude=78.5700,
        easting_utm44n=241880.0,
        northing_utm44n=1933850.0,
        typology="Dense Commercial / Mixed Urban Residential"
    ),
    "ICRISAT Patancheru": StationMetadata(
        name="ICRISAT Patancheru",
        station_id="HYD_ICR",
        latitude=17.5100,
        longitude=78.2700,
        easting_utm44n=210180.0,
        northing_utm44n=1938550.0,
        typology="Peri-Urban / Agricultural-Industrial Boundary"
    )
}

STATION_NAME_ALIASES: Dict[str, str] = {
    "Hyderabad": "Central University",
    "Central University, Hyderabad": "Central University",
    "Bollaram Industrial Area, Hyderabad": "Bollaram Industrial Area",
    "ECIL, Hyderabad": "ECIL Kapra",
    "ICRISAT, Hyderabad": "ICRISAT Patancheru"
}

POLLUTANTS: List[str] = [
    "PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "Ozone"
]

WEATHER_VARIABLES: List[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure"
]

# Physical bounds for validation (min, max) in standard units
PHYSICAL_VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "PM2.5": (0.0, 1000.0),      # ug/m3
    "PM10": (0.0, 1500.0),       # ug/m3
    "NO": (0.0, 1000.0),         # ug/m3
    "NO2": (0.0, 1000.0),        # ug/m3
    "NH3": (0.0, 1500.0),        # ug/m3
    "SO2": (0.0, 1000.0),        # ug/m3
    "CO": (0.0, 100.0),          # mg/m3
    "Ozone": (0.0, 1000.0),      # ug/m3
    "temperature_2m": (-5.0, 60.0),           # deg C
    "relative_humidity_2m": (0.0, 100.0),     # %
    "wind_speed_10m": (0.0, 60.0),            # m/s
    "wind_direction_10m": (0.0, 360.0),       # deg
    "precipitation": (0.0, 500.0),            # mm
    "surface_pressure": (800.0, 1100.0)       # hPa
}

# Timeline specifications
DATASET_START = "2023-01-01 00:00:00"
DATASET_END = "2026-09-30 23:00:00"
EXPECTED_HOURS_PER_STATION = 32856
EXPECTED_TOTAL_STATION_HOURS = 131424

# Chronological partition dates
TRAIN_START = "2023-01-01 00:00:00"
TRAIN_END = "2024-12-31 23:00:00"
EXPECTED_TRAIN_HOURS = 70176  # 17,544 hours * 4 stations

VAL_START = "2025-01-01 00:00:00"
VAL_END = "2025-12-31 23:00:00"
EXPECTED_VAL_HOURS = 35040    # 8,760 hours * 4 stations

TEST_START = "2026-01-01 00:00:00"
TEST_END = "2026-09-30 23:00:00"
EXPECTED_TEST_HOURS = 26208   # 6,552 hours * 4 stations

# Target parameters
REGRESSION_HORIZONS: List[int] = [1, 6, 24, 48]
CLASSIFICATION_HORIZONS: List[int] = [1, 6, 24]

# Threshold values (ug/m3)
PM25_STATUTORY_24H_THRESHOLD = 60.0
PM25_ACUTE_EXCURSION_THRESHOLD = 60.0
PM25_SEVERE_SPIKE_THRESHOLD = 120.0       # Approved >= 120 definition
PM25_VERY_POOR_ENTRY_THRESHOLD = 120.0    # Strict > 120 definition
MIN_VALID_HOURS_24H = 16                  # Minimum valid hours for 24h average
MAX_PREDICTION_HORIZON_HOURS = 48         # Maximum forward window for boundary buffering

# Imputation gap thresholds
SHORT_GAP_MAX_HOURS = 3
MEDIUM_GAP_MAX_HOURS = 24

# Milestone 2: Feature Engineering Specifications
HISTORICAL_LAGS: List[int] = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS: List[int] = [3, 6, 12, 24]
TREND_DIFFERENCES: List[int] = [1, 3, 24]

# Diurnal congestion windows
MORNING_PEAK_HOURS: Tuple[int, ...] = (8, 9, 10, 11)
EVENING_PEAK_HOURS: Tuple[int, ...] = (17, 18, 19, 20, 21)
EXPLORATORY_FREIGHT_HOURS: Tuple[int, ...] = (23, 0, 1, 2, 3, 4, 5)

# OpenStreetMap (OSM) Roadway Spatial Proxy Specifications
# IMPORTANT SCIENTIFIC PROVENANCE AUDIT NOTICE:
# Overpass API queries failed in this environment due to server-side rate-limiting and timeouts.
# The roadway metrics below are ASSUMED / ESTIMATED benchmark proxies calibrated from literature
# and metropolitan road atlas references for Hyderabad. They were NOT computed directly from
# local downloaded raw OSM geometry vector files. They provide 4 spatial degrees of freedom
# and must NEVER be described as measured vehicle counts or empirical OSM geometric extractions.
@dataclass(frozen=True)
class OSMRoadMetrics:
    primary_length_1km: float      # km of motorway/trunk/primary in 1km buffer (assumed proxy)
    secondary_length_1km: float    # km of secondary/arterial in 1km buffer (assumed proxy)
    primary_length_3km: float      # km of motorway/trunk/primary in 3km buffer (assumed proxy)
    secondary_length_3km: float    # km of secondary/arterial in 3km buffer (assumed proxy)
    distance_to_highway_km: float  # Euclidean distance (km) to nearest national highway corridor
    intersection_density: float    # intersections per km2 (assumed proxy)
    data_provenance: str = "Assumed literature proxy (unverified by raw OSM geometry download)"

OSM_STATION_ROAD_PROXIES: Dict[str, OSMRoadMetrics] = {
    "Bollaram Industrial Area": OSMRoadMetrics(
        primary_length_1km=2.45,
        secondary_length_1km=4.10,
        primary_length_3km=14.80,
        secondary_length_3km=26.50,
        distance_to_highway_km=3.20,
        intersection_density=8.50
    ),
    "Central University": OSMRoadMetrics(
        primary_length_1km=1.20,
        secondary_length_1km=2.80,
        primary_length_3km=9.60,
        secondary_length_3km=18.40,
        distance_to_highway_km=3.80,
        intersection_density=3.80
    ),
    "ECIL Kapra": OSMRoadMetrics(
        primary_length_1km=4.80,
        secondary_length_1km=8.60,
        primary_length_3km=28.50,
        secondary_length_3km=46.20,
        distance_to_highway_km=4.10,
        intersection_density=16.20
    ),
    "ICRISAT Patancheru": OSMRoadMetrics(
        primary_length_1km=3.60,
        secondary_length_1km=3.10,
        primary_length_3km=18.20,
        secondary_length_3km=19.50,
        distance_to_highway_km=0.45,
        intersection_density=4.50
    )
}

# Buffer geometric areas in projected UTM Zone 44N (EPSG:32644)
BUFFER_AREA_1KM_KM2 = 3.141592653589793   # pi * 1^2
BUFFER_AREA_3KM_KM2 = 28.274333882308138  # pi * 3^2

# =============================================================================
# FEATURE SCHEMA DEFINITIONS: MUTUALLY EXCLUSIVE & EXHAUSTIVE COLUMN SUBSETS
# Reconciles exactly to the 173 physical columns in feature_store_primary.parquet:
# Identifiers (3) + Metadata/Quality (51) + Primary Predictors (84) +
# Exploratory Road Proxies (19) + Targets (16) = 173 Total Columns
# =============================================================================

# 1. Identifiers (3 columns) - Administrative keys not used as numerical predictors
IDENTIFIER_COLUMNS: List[str] = [
    "station",
    "station_id",
    "datetime"
]

# 2. Administrative Metadata & Telemetry Quality Flags (51 columns) - NOT model predictors
METADATA_AND_QUALITY_COLUMNS: List[str] = [
    # Geodetic / Spatial / Partition Administrative Metadata (7)
    "latitude",
    "longitude",
    "easting_utm44n",
    "northing_utm44n",
    "typology",
    "partition",
    "is_boundary_buffer",
    # Temporal Discrete Metadata (4)
    "hour",
    "dayofweek",
    "month",
    "season",
    # Sensor Gap Tracking Metadata (16)
    "PM2.5_gap_length", "PM2.5_gap_category",
    "PM10_gap_length", "PM10_gap_category",
    "NO_gap_length", "NO_gap_category",
    "NO2_gap_length", "NO2_gap_category",
    "NH3_gap_length", "NH3_gap_category",
    "SO2_gap_length", "SO2_gap_category",
    "CO_gap_length", "CO_gap_category",
    "Ozone_gap_length", "Ozone_gap_category",
    # Raw Uncleaned Sensor Telemetry (8)
    "PM2.5_raw", "PM10_raw", "NO_raw", "NO2_raw", "NH3_raw", "SO2_raw", "CO_raw", "Ozone_raw",
    # Missingness Mask Flags (8)
    "PM2.5_missing_mask", "PM10_missing_mask", "NO_missing_mask", "NO2_missing_mask",
    "NH3_missing_mask", "SO2_missing_mask", "CO_missing_mask", "Ozone_missing_mask",
    # Imputation Tracking Flags (8)
    "PM2.5_imputed_flag", "PM10_imputed_flag", "NO_imputed_flag", "NO2_imputed_flag",
    "NH3_imputed_flag", "SO2_imputed_flag", "CO_imputed_flag", "Ozone_imputed_flag"
]

# 3. Primary Model Predictors (84 columns) - VERIFIED PHYSICAL & METEOROLOGICAL PREDICTORS
# Strictly excludes all unverified road proxies and interaction features
PRIMARY_MODEL_PREDICTORS: List[str] = [
    # A. Prospective Ambient Pollutants at origin t (8)
    "PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "Ozone",
    # B. Historical PM2.5 Autoregressive Lags (6)
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    # C. Historical PM2.5 Rolling Statistics (8)
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_std_6h",
    "pm25_roll_mean_12h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
    "pm25_roll_min_24h", "pm25_roll_max_24h",
    # D. Historical Trends and Dynamic Acceleration (5)
    "pm25_trend_diff_1h", "pm25_trend_diff_3h", "pm25_trend_diff_24h",
    "pm25_acceleration_1h", "pm25_ratio_to_24h_mean",
    # E. Co-Pollutant Chemical Ratios and Lags (8)
    "ratio_pm25_pm10", "ratio_no_no2",
    "pm10_lag_1", "pm10_lag_24",
    "no2_lag_1", "no2_lag_24",
    "so2_lag_1", "so2_lag_24",
    # F. Cyclical Temporal and Seasonal Transformations (10)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_winter", "is_summer", "is_monsoon", "is_post_monsoon",
    # G. Meteorological Variables, Vectors, Moisture Deficit & Boundary Layer (32)
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "surface_pressure",
    "wind_dir_sin", "wind_dir_cos", "wind_u", "wind_v",
    "sat_vapor_pressure_es", "act_vapor_pressure_ea", "vapor_pressure_deficit_vpd",
    "dew_point_temp", "dew_point_depression",
    "is_atmospheric_stagnation", "is_wind_dispersion", "ventilation_index_proxy",
    "precip_roll_sum_6h", "precip_roll_sum_24h", "has_recent_rain_24h",
    "temp_lag_1", "temp_lag_24", "temp_diff_24h",
    "rh_lag_1", "rh_lag_24", "rh_diff_24h",
    "wind_speed_lag_1", "wind_speed_roll_mean_6h", "wind_speed_roll_mean_24h",
    "pressure_lag_1", "pressure_tendency_3h",
    # H. Diurnal Traffic Windows (Purely Temporal - NO road geometry multiplication) (7)
    "is_morning_peak", "is_evening_peak", "is_traffic_peak", "is_weekend",
    "is_weekday_peak", "is_freight_window_exploratory", "is_daytime_offpeak"
]

# 4. Excluded Exploratory Road Proxies (19 columns) - EXCLUDED FROM PRIMARY MODEL
# Preserved strictly for optional exploratory sensitivity experiments
EXPLORATORY_ROAD_PROXY_FEATURES: List[str] = [
    # Static Land-Use Spatial Roadway Proxies (14) - 4 spatial degrees of freedom
    "osm_primary_length_1km", "osm_secondary_length_1km", "osm_total_length_1km",
    "osm_primary_length_3km", "osm_secondary_length_3km", "osm_total_length_3km",
    "osm_road_density_1km", "osm_road_density_3km",
    "osm_primary_density_1km", "osm_primary_density_3km",
    "osm_primary_concentration_1km", "osm_primary_concentration_3km",
    "osm_distance_to_highway_km", "osm_intersection_density_km2",
    # Spatio-Temporal Traffic Pressure Interactions (5) - Derived from assumed road metrics
    "traffic_pressure_total_1km",
    "traffic_pressure_weekday_peak",
    "traffic_pressure_primary_1km",
    "freight_pressure_proxy",
    "highway_rush_pressure"
]

# 5. Multi-Horizon Prospective Targets (16 columns) - PROSPECTIVE TARGETS ONLY
TARGET_COLUMNS: List[str] = [
    # Continuous Regression Targets (4 single-horizon + 1 rolling mean + 1 count)
    "target_pm25_h1", "target_pm25_h6", "target_pm25_h24", "target_pm25_h48",
    "target_pm25_24h_mean", "target_pm25_24h_valid_count",
    # Binary Hotspot Targets (9 single-horizon + 1 statutory)
    "target_hotspot_acute_h1", "target_hotspot_spike_h1", "target_hotspot_very_poor_h1",
    "target_hotspot_acute_h6", "target_hotspot_spike_h6", "target_hotspot_very_poor_h6",
    "target_hotspot_acute_h24", "target_hotspot_spike_h24", "target_hotspot_very_poor_h24",
    "target_hotspot_24h_statutory"
]

# Combined candidate list (Primary + Exploratory)
ALL_CANDIDATE_PREDICTORS: List[str] = PRIMARY_MODEL_PREDICTORS + EXPLORATORY_ROAD_PROXY_FEATURES

# Exact schema identity verification assertions
assert len(IDENTIFIER_COLUMNS) == 3, f"Expected 3 identifiers, got {len(IDENTIFIER_COLUMNS)}"
assert len(METADATA_AND_QUALITY_COLUMNS) == 51, f"Expected 51 metadata/quality, got {len(METADATA_AND_QUALITY_COLUMNS)}"
assert len(PRIMARY_MODEL_PREDICTORS) == 84, f"Expected 84 primary predictors, got {len(PRIMARY_MODEL_PREDICTORS)}"
assert len(EXPLORATORY_ROAD_PROXY_FEATURES) == 19, f"Expected 19 road proxies, got {len(EXPLORATORY_ROAD_PROXY_FEATURES)}"
assert len(TARGET_COLUMNS) == 16, f"Expected 16 targets, got {len(TARGET_COLUMNS)}"
assert (
    len(IDENTIFIER_COLUMNS)
    + len(METADATA_AND_QUALITY_COLUMNS)
    + len(PRIMARY_MODEL_PREDICTORS)
    + len(EXPLORATORY_ROAD_PROXY_FEATURES)
    + len(TARGET_COLUMNS)
) == 173, "Column identity sum must equal exactly 173"


# Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMPLEMENTATION_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = IMPLEMENTATION_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = IMPLEMENTATION_ROOT / "configs"

MODELS_DIR = PROJECT_ROOT / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED: int = 42
PROJECT_RANDOM_SEED: int = 42

CONCENTRATION_THRESHOLDS: Dict[str, float] = {
    "acute_excursion": PM25_ACUTE_EXCURSION_THRESHOLD,
    "severe_spike": PM25_SEVERE_SPIKE_THRESHOLD
}

CPCB_RAW_FILE = RAW_DATA_DIR / "cpcb_hourly_raw.csv"
WEATHER_RAW_FILE = RAW_DATA_DIR / "open_meteo_weather_raw.csv"
MERGED_RAW_FILE = INTERIM_DATA_DIR / "merged_raw.csv"
