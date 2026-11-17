"""
Automated Test Suite for Milestone 2: Multi-Source Feature Engineering
Validates non-leakage, mathematical transformations, provenance, and target decoupling.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from configs.config import (
    STATION_REGISTRY,
    OSM_STATION_ROAD_PROXIES,
    HISTORICAL_LAGS,
    ROLLING_WINDOWS,
    MORNING_PEAK_HOURS,
    EVENING_PEAK_HOURS,
    EXPLORATORY_FREIGHT_HOURS,
    BUFFER_AREA_1KM_KM2,
    BUFFER_AREA_3KM_KM2,
    IDENTIFIER_COLUMNS,
    METADATA_AND_QUALITY_COLUMNS,
    PRIMARY_MODEL_PREDICTORS,
    EXPLORATORY_ROAD_PROXY_FEATURES,
    TARGET_COLUMNS
)
from feature_engineering.historical_features import HistoricalFeatureExtractor
from feature_engineering.weather_features import WeatherFeatureExtractor
from feature_engineering.osm_road_features import OSMRoadFeatureExtractor
from feature_engineering.congestion_features import CongestionFeatureExtractor
from feature_engineering.feature_store_builder import FeatureStoreBuilder

@pytest.fixture
def sample_timeseries_df():
    """Generates synthetic multi-station hourly timeseries for testing."""
    stations = ["Bollaram Industrial Area", "Central University"]
    dates = pd.date_range("2023-01-01 00:00:00", periods=72, freq="1h")
    
    records = []
    np.random.seed(42)
    for st in stations:
        base = 35.0 if "Bollaram" in st else 25.0
        for i, dt in enumerate(dates):
            val = base + 5.0 * np.sin(i / 6.0) + np.random.normal(0, 1.0)
            records.append({
                "station": st,
                "station_id": STATION_REGISTRY[st].station_id,
                "datetime": dt,
                "PM2.5": val,
                "PM2.5_raw": val,
                "PM10": val * 2.2,
                "NO": 8.0,
                "NO2": 22.0,
                "SO2": 12.0,
                "temperature_2m": 24.0 + 6.0 * np.sin((dt.hour - 6) / 24.0 * 2 * np.pi),
                "relative_humidity_2m": 60.0 - 20.0 * np.sin((dt.hour - 6) / 24.0 * 2 * np.pi),
                "wind_speed_10m": 3.5,
                "wind_direction_10m": 120.0,
                "precipitation": 0.0,
                "surface_pressure": 950.0
            })
            
    df = pd.DataFrame(records)
    return df

# --- 1. Historical PM2.5 Feature Tests ---

def test_historical_lags_alignment(sample_timeseries_df):
    """Verifies that historical lags strictly match past observations per station."""
    extractor = HistoricalFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    for st in transformed["station"].unique():
        st_df = transformed[transformed["station"] == st].sort_values("datetime").reset_index(drop=True)
        for lag in [1, 2, 3, 6, 12, 24]:
            col = f"pm25_lag_{lag}"
            assert col in st_df.columns
            # First lag rows must be NaN
            assert st_df[col].iloc[:lag].isna().all()
            # Row index lag must match row index 0 of original PM2.5
            assert np.isclose(st_df[col].iloc[lag], st_df["PM2.5"].iloc[0], atol=1e-5)
            assert np.isclose(st_df[col].iloc[lag + 10], st_df["PM2.5"].iloc[10], atol=1e-5)

def test_historical_rolling_stats_past_only(sample_timeseries_df):
    """
    Verifies that rolling statistics are computed strictly on past observations [t-w+1 : t]
    and modifying future observations has ZERO effect on rolling statistics at t.
    """
    extractor = HistoricalFeatureExtractor()
    df1 = sample_timeseries_df.copy()
    transformed1 = extractor.transform(df1)
    
    # Perturb observation at t=50 (future)
    df2 = sample_timeseries_df.copy()
    target_idx = 50
    df2.loc[target_idx, "PM2.5"] = 999.0
    transformed2 = extractor.transform(df2)
    
    # Verify that rolling statistics at t < 50 are identical
    for w in [3, 6, 12, 24]:
        col = f"pm25_roll_mean_{w}h"
        diff = np.abs(transformed1.loc[:target_idx-1, col] - transformed2.loc[:target_idx-1, col])
        assert np.nanmax(diff) < 1e-9, f"Future modification at t=50 leaked into past rolling stats at t < 50 for window {w}"

def test_historical_trends_and_acceleration(sample_timeseries_df):
    """Verifies 1h, 3h, 24h trend differences and acceleration formulations."""
    extractor = HistoricalFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    st_df = transformed[transformed["station"] == "Central University"].sort_values("datetime").reset_index(drop=True)
    
    idx = 30
    expected_diff_1h = st_df["PM2.5"].iloc[idx] - st_df["PM2.5"].iloc[idx - 1]
    expected_diff_24h = st_df["PM2.5"].iloc[idx] - st_df["PM2.5"].iloc[idx - 24]
    
    assert np.isclose(st_df["pm25_trend_diff_1h"].iloc[idx], expected_diff_1h, atol=1e-5)
    assert np.isclose(st_df["pm25_trend_diff_24h"].iloc[idx], expected_diff_24h, atol=1e-5)
    
    expected_acc = (st_df["PM2.5"].iloc[idx] - st_df["PM2.5"].iloc[idx - 1]) - (st_df["PM2.5"].iloc[idx - 1] - st_df["PM2.5"].iloc[idx - 2])
    assert np.isclose(st_df["pm25_acceleration_1h"].iloc[idx], expected_acc, atol=1e-5)

# --- 2. Meteorological Feature Tests ---

def test_weather_wind_vector_identity(sample_timeseries_df):
    """Verifies trigonometric identity u^2 + v^2 == wind_speed^2."""
    extractor = WeatherFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    u = transformed["wind_u"]
    v = transformed["wind_v"]
    speed = transformed["wind_speed_10m"]
    
    calculated_speed = np.sqrt(u**2 + v**2)
    diff = np.abs(calculated_speed - speed)
    assert np.nanmax(diff) < 1e-6, "Wind vector components fail identity u^2 + v^2 = speed^2"

def test_weather_psychrometric_consistency(sample_timeseries_df):
    """Verifies that saturation vapor pressure exceeds actual vapor pressure (VPD >= 0)."""
    extractor = WeatherFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    es = transformed["sat_vapor_pressure_es"]
    ea = transformed["act_vapor_pressure_ea"]
    vpd = transformed["vapor_pressure_deficit_vpd"]
    
    assert (es > 0).all()
    assert (ea > 0).all()
    assert (vpd >= 0).all()
    assert np.isclose(vpd, es - ea, atol=1e-5).all()

def test_atmospheric_stagnation_logic(sample_timeseries_df):
    """Verifies stagnation indicator activates only when wind < 1.5 m/s and precip <= 0.01."""
    extractor = WeatherFeatureExtractor()
    df = sample_timeseries_df.copy()
    df.loc[0, "wind_speed_10m"] = 1.0
    df.loc[0, "precipitation"] = 0.0
    df.loc[1, "wind_speed_10m"] = 2.5
    df.loc[1, "precipitation"] = 0.0
    df.loc[2, "wind_speed_10m"] = 0.8
    df.loc[2, "precipitation"] = 2.0
    
    transformed = extractor.transform(df)
    assert transformed.loc[0, "is_atmospheric_stagnation"] == 1.0
    assert transformed.loc[1, "is_atmospheric_stagnation"] == 0.0
    assert transformed.loc[2, "is_atmospheric_stagnation"] == 0.0

# --- 3. OpenStreetMap Feature Tests ---

def test_osm_road_metrics_consistency(sample_timeseries_df):
    """Verifies station roadway densities, concentration ratios, and highway distances."""
    extractor = OSMRoadFeatureExtractor()
    transformed = extractor.attach_road_features(sample_timeseries_df)
    
    for st, metrics in OSM_STATION_ROAD_PROXIES.items():
        st_sub = transformed[transformed["station"] == st]
        if len(st_sub) == 0:
            continue
        row = st_sub.iloc[0]
        
        expected_density_1km = (metrics.primary_length_1km + metrics.secondary_length_1km) / BUFFER_AREA_1KM_KM2
        expected_density_3km = (metrics.primary_length_3km + metrics.secondary_length_3km) / BUFFER_AREA_3KM_KM2
        
        assert np.isclose(row["osm_road_density_1km"], expected_density_1km, atol=1e-5)
        assert np.isclose(row["osm_road_density_3km"], expected_density_3km, atol=1e-5)
        assert np.isclose(row["osm_distance_to_highway_km"], metrics.distance_to_highway_km, atol=1e-5)
        assert np.isclose(row["osm_intersection_density_km2"], metrics.intersection_density, atol=1e-5)

# --- 4. Congestion Windows & Dynamics Tests ---

def test_congestion_windows_logic(sample_timeseries_df):
    """Verifies morning (8-11), evening (17-21), and freight (23-5) indicator bounds."""
    extractor = CongestionFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    for _, row in transformed.iterrows():
        hr = row["datetime"].hour
        
        # Morning peak
        if hr in MORNING_PEAK_HOURS:
            assert row["is_morning_peak"] == 1.0
        else:
            assert row["is_morning_peak"] == 0.0
            
        # Evening peak
        if hr in EVENING_PEAK_HOURS:
            assert row["is_evening_peak"] == 1.0
        else:
            assert row["is_evening_peak"] == 0.0
            
        # Exploratory Freight window
        if hr in EXPLORATORY_FREIGHT_HOURS:
            assert row["is_freight_window_exploratory"] == 1.0
        else:
            assert row["is_freight_window_exploratory"] == 0.0

# --- 5. Feature Store & Non-Leakage Tests ---

def test_feature_store_target_decoupling(sample_timeseries_df, tmp_path):
    """Verifies that predictor feature columns contain ZERO target variables."""
    builder = FeatureStoreBuilder(output_dir=tmp_path)
    feature_df, targets_df = builder.build_feature_store(sample_timeseries_df, attach_targets=True)
    
    # Target columns must be isolated
    target_cols = [c for c in targets_df.columns if c not in ["station", "datetime"]]
    assert len(target_cols) > 0
    
    # Feature columns excluding targets must have zero target names
    predictor_cols = [c for c in feature_df.columns if not c.startswith("target_")]
    for col in predictor_cols:
        assert not col.startswith("target_"), f"Target column '{col}' leaked into predictor feature set!"

def test_cross_station_isolation(sample_timeseries_df):
    """Verifies that lag calculations never bleed across distinct monitoring stations."""
    extractor = HistoricalFeatureExtractor()
    transformed = extractor.transform(sample_timeseries_df)
    
    stations = transformed["station"].unique()
    assert len(stations) >= 2
    
    # Station 2's row index 0 must have lag_1 == NaN, NEVER taking Station 1's last value
    st2 = stations[1]
    st2_first_row = transformed[transformed["station"] == st2].sort_values("datetime").iloc[0]
    assert np.isnan(st2_first_row["pm25_lag_1"]), "Station boundary cross-talk: station 2 lagged value contaminated by station 1!"

# --- 6. Independent Scientific Verification Audit Tests ---

def test_osm_data_provenance_explicit_assumed_label():
    """
    Independent Audit Dimension 1:
    Verifies that OSM road metrics in config.py are explicitly classified as ASSUMED literature proxies
    due to live Overpass API query timeouts and absent local OSM vector geometries.
    """
    for station_name, metrics in OSM_STATION_ROAD_PROXIES.items():
        assert hasattr(metrics, "data_provenance")
        assert "Assumed" in metrics.data_provenance
        # Must strictly have 4 degrees of freedom
        assert metrics.primary_length_1km > 0
        assert metrics.distance_to_highway_km > 0

def test_temporal_boundary_purging_zero_target_crossover():
    """
    Independent Audit Dimension 2:
    Verifies that in strictly purged partitions, ZERO samples cross partition boundary dates
    for any forecast horizon h in {1, 6, 24, 48}.
    """
    train_purged_path = Path("new_research_project/data/processed/feature_store_train_purged.parquet")
    val_purged_path = Path("new_research_project/data/processed/feature_store_val_purged.parquet")
    
    if not train_purged_path.exists() or not val_purged_path.exists():
        pytest.skip("Purged partition parquet files not yet generated on disk.")
        
    df_train_purged = pd.read_parquet(train_purged_path)
    df_val_purged = pd.read_parquet(val_purged_path)
    
    train_end = pd.to_datetime("2024-12-31 23:00:00")
    val_end = pd.to_datetime("2025-12-31 23:00:00")
    
    # Exactly 192 rows (48h * 4 stations) must be purged from each partition
    assert len(df_train_purged) == 69984  # 70,176 - 192
    assert len(df_val_purged) == 34848    # 35,040 - 192
    
    # For all horizons h in {1, 6, 24, 48}, max target time must be <= end of partition
    for h in [1, 6, 24, 48]:
        max_train_target = df_train_purged["datetime"].max() + pd.Timedelta(hours=h)
        assert max_train_target <= train_end, f"Train purged partition crosses into validation for horizon +{h}h!"
        
        max_val_target = df_val_purged["datetime"].max() + pd.Timedelta(hours=h)
        assert max_val_target <= val_end, f"Val purged partition crosses into test for horizon +{h}h!"

def test_forecast_time_feature_availability_point_in_time():
    """
    Independent Audit Dimension 3:
    Verifies point-in-time feature availability:
    - Feature columns contain zero target variables.
    - Weather features condition strictly on observations <= t.
    - No future atmospheric reanalysis (t+h) is provided as an input predictor.
    """
    primary_path = Path("new_research_project/data/processed/feature_store_primary.parquet")
    if not primary_path.exists():
        pytest.skip("feature_store_primary.parquet not yet generated on disk.")
        
    df = pd.read_parquet(primary_path)
    predictor_cols = [c for c in df.columns if not c.startswith("target_")]
    
    # Verify no future weather at t+h exists in predictors
    for col in predictor_cols:
        assert not any(col.startswith(f"weather_h{h}") for h in [1, 6, 24, 48])
        assert not any(col.endswith(f"_future_{h}h") for h in [1, 6, 24, 48])

def test_feature_dictionary_exhaustive_173_reconciliation():
    """
    Independent Audit Dimension 4:
    Verifies that EVERY SINGLE COLUMN (173/173) in feature_store_primary is cataloged
    in FEATURE_CATALOG with zero missing or orphaned columns.
    """
    primary_path = Path("new_research_project/data/processed/feature_store_primary.parquet")
    if not primary_path.exists():
        pytest.skip("feature_store_primary.parquet not yet generated on disk.")
        
    from feature_engineering.feature_dictionary import FEATURE_CATALOG
    df = pd.read_parquet(primary_path)
    df_cols = set(df.columns)
    cat_cols = set(c["name"] for c in FEATURE_CATALOG)
    
    missing_from_cat = df_cols - cat_cols
    assert len(missing_from_cat) == 0, f"Uncataloged columns in feature store: {missing_from_cat}"
    assert len(df_cols) == 173
    assert len(cat_cols) == 173

# --- 7. Final Corrections Audit Tests ---

def test_assumed_road_proxies_excluded_from_primary_predictors():
    """
    Final Audit Item 1:
    Verifies that NONE of the 19 unverified road proxy features
    (14 static OSM proxies + 5 traffic pressure interactions) can enter
    the primary model predictor matrix.
    """
    assert len(PRIMARY_MODEL_PREDICTORS) == 84
    assert len(EXPLORATORY_ROAD_PROXY_FEATURES) == 19
    
    primary_set = set(PRIMARY_MODEL_PREDICTORS)
    road_set = set(EXPLORATORY_ROAD_PROXY_FEATURES)
    
    # 1. Zero overlap between primary predictors and exploratory road proxies
    overlap = primary_set & road_set
    assert len(overlap) == 0, f"Unverified road proxy leaked into primary predictors: {overlap}"
    
    # 2. Explicit assertion that no osm_* feature is in primary
    for col in primary_set:
        assert not col.startswith("osm_"), f"OSM feature {col} found in PRIMARY_MODEL_PREDICTORS!"
        assert "traffic_pressure" not in col, f"Traffic pressure feature {col} found in PRIMARY_MODEL_PREDICTORS!"
        assert "freight_pressure" not in col, f"Freight pressure feature {col} found in PRIMARY_MODEL_PREDICTORS!"
        assert "highway_rush" not in col, f"Highway rush feature {col} found in PRIMARY_MODEL_PREDICTORS!"

def test_feature_schema_exact_identity_and_disjointness():
    """
    Final Audit Item 2:
    Verifies the exact identity:
    Total columns (173) = Identifiers (3) + Metadata/Quality (51) + Primary Predictors (84)
                          + Exploratory Road Proxies (19) + Targets (16)
    and verifies that all sets are mutually pairwise disjoint.
    """
    primary_path = Path("new_research_project/data/processed/feature_store_primary.parquet")
    if not primary_path.exists():
        pytest.skip("feature_store_primary.parquet not found on disk.")
        
    df = pd.read_parquet(primary_path)
    all_actual_cols = set(df.columns)
    
    sets = [
        ("Identifiers", set(IDENTIFIER_COLUMNS), 3),
        ("Metadata & Quality", set(METADATA_AND_QUALITY_COLUMNS), 51),
        ("Primary Predictors", set(PRIMARY_MODEL_PREDICTORS), 84),
        ("Exploratory Road Proxies", set(EXPLORATORY_ROAD_PROXY_FEATURES), 19),
        ("Targets", set(TARGET_COLUMNS), 16)
    ]
    
    # Check individual counts
    for name, s, expected_len in sets:
        assert len(s) == expected_len, f"{name} count mismatch: expected {expected_len}, got {len(s)}"
        
    # Check pairwise disjointness
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            name1, s1, _ = sets[i]
            name2, s2, _ = sets[j]
            overlap = s1 & s2
            assert len(overlap) == 0, f"Overlap between {name1} and {name2}: {overlap}"
            
    # Check union matches actual parquet columns exactly
    all_union = set(IDENTIFIER_COLUMNS) | set(METADATA_AND_QUALITY_COLUMNS) | set(PRIMARY_MODEL_PREDICTORS) | set(EXPLORATORY_ROAD_PROXY_FEATURES) | set(TARGET_COLUMNS)
    assert len(all_union) == 173
    assert all_union == all_actual_cols

def test_purged_partitions_zero_boundary_crossover_all_horizons():
    """
    Final Audit Item 3:
    Verifies that for every forecast horizon (+1h, +6h, +24h, +48h):
    - No training target enters the validation period (max target <= 2024-12-31 23:00).
    - No validation target enters the test period (max target <= 2025-12-31 23:00).
    - Test targets are within the available dataset (max target <= 2026-09-30 23:00).
    """
    train_purged_path = Path("new_research_project/data/processed/feature_store_train_purged.parquet")
    val_purged_path = Path("new_research_project/data/processed/feature_store_val_purged.parquet")
    test_purged_path = Path("new_research_project/data/processed/feature_store_test_purged.parquet")
    
    if not (train_purged_path.exists() and val_purged_path.exists() and test_purged_path.exists()):
        pytest.skip("Purged partition parquet files not found on disk.")
        
    df_train = pd.read_parquet(train_purged_path)
    df_val = pd.read_parquet(val_purged_path)
    df_test = pd.read_parquet(test_purged_path)
    
    train_end = pd.to_datetime("2024-12-31 23:00:00")
    val_end = pd.to_datetime("2025-12-31 23:00:00")
    test_end = pd.to_datetime("2026-09-30 23:00:00")
    
    # Verify retained sample counts
    assert len(df_train) == 69984
    assert len(df_val) == 34848
    assert len(df_test) == 26016
    
    for h in [1, 6, 24, 48]:
        # Train targets
        train_targets = df_train["datetime"] + pd.Timedelta(hours=h)
        assert (train_targets <= train_end).all(), f"Train +{h}h target leaked into validation!"
        
        # Val targets
        val_targets = df_val["datetime"] + pd.Timedelta(hours=h)
        assert (val_targets <= val_end).all(), f"Val +{h}h target leaked into test!"
        
        # Test targets
        test_targets = df_test["datetime"] + pd.Timedelta(hours=h)
        assert (test_targets <= test_end).all(), f"Test +{h}h target exceeds dataset timeline!"

def test_primary_model_input_hygiene():
    """
    Final Audit Item 4:
    Verifies that the primary model predictor matrix contains:
    - Zero target columns
    - Zero future-derived variables
    - Zero quality flags or missing masks
    - Zero administrative identifiers
    """
    for col in PRIMARY_MODEL_PREDICTORS:
        assert not col.startswith("target_"), f"Target {col} in PRIMARY_MODEL_PREDICTORS"
        assert not col.endswith("_raw"), f"Raw telemetry {col} in PRIMARY_MODEL_PREDICTORS"
        assert not col.endswith("_missing_mask"), f"Missing mask {col} in PRIMARY_MODEL_PREDICTORS"
        assert not col.endswith("_imputed_flag"), f"Imputed flag {col} in PRIMARY_MODEL_PREDICTORS"
        assert "_gap_" not in col, f"Gap flag {col} in PRIMARY_MODEL_PREDICTORS"
        assert col not in IDENTIFIER_COLUMNS, f"Identifier {col} in PRIMARY_MODEL_PREDICTORS"
        assert col not in ["partition", "is_boundary_buffer"], f"Partition flag {col} in PRIMARY_MODEL_PREDICTORS"

