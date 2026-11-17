"""
Model Registry and Experiment Configuration Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Defines:
- Model specifications and hyperparameter configurations
- Approved target columns and forecast horizons
- Feature tier definitions for ablation protocols
- Serialization directory paths and artifact naming conventions
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))
from typing import Dict, List, Any, Optional

try:
    from configs.config import (
        PRIMARY_MODEL_PREDICTORS,
        EXPLORATORY_ROAD_PROXY_FEATURES,
        MODELS_DIR,
        PREDICTIONS_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR
    )
except (ImportError, ValueError):
    from ..configs.config import (
        PRIMARY_MODEL_PREDICTORS,
        EXPLORATORY_ROAD_PROXY_FEATURES,
        MODELS_DIR,
        PREDICTIONS_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR
    )

# 1. Approved Target Definitions
REGRESSION_TARGETS: Dict[str, str] = {
    "h1": "target_pm25_h1",
    "h6": "target_pm25_h6",
    "h24": "target_pm25_h24",
    "h48": "target_pm25_h48"
}

CLASSIFICATION_TARGETS_ACUTE: Dict[str, str] = {
    "h1": "target_hotspot_acute_h1",
    "h6": "target_hotspot_acute_h6",
    "h24": "target_hotspot_acute_h24"
}

CLASSIFICATION_TARGETS_SPIKE: Dict[str, str] = {
    "h1": "target_hotspot_spike_h1",
    "h6": "target_hotspot_spike_h6",
    "h24": "target_hotspot_spike_h24"
}

# 2. Approved Feature Groups for Ablation Protocols
# Tier A: Historical Pollution & Calendar Only (45 features)
TIER_A_HISTORICAL_FEATURES: List[str] = [
    # Ambient pollutants at origin t
    "PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "Ozone",
    # Autoregressive lags
    "pm25_lag_1", "pm25_lag_2", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    # Rolling stats
    "pm25_roll_mean_3h", "pm25_roll_mean_6h", "pm25_roll_std_6h",
    "pm25_roll_mean_12h", "pm25_roll_mean_24h", "pm25_roll_std_24h",
    "pm25_roll_min_24h", "pm25_roll_max_24h",
    # Trends and acceleration
    "pm25_trend_diff_1h", "pm25_trend_diff_3h", "pm25_trend_diff_24h",
    "pm25_acceleration_1h", "pm25_ratio_to_24h_mean",
    # Co-pollutant ratios and lags
    "ratio_pm25_pm10", "ratio_no_no2",
    "pm10_lag_1", "pm10_lag_24",
    "no2_lag_1", "no2_lag_24",
    "so2_lag_1", "so2_lag_24",
    # Cyclical calendar features
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_winter", "is_summer", "is_monsoon", "is_post_monsoon"
]

# Tier B: Primary Model Predictors = Tier A + Weather + Purely Temporal Diurnal Traffic Windows (84 features)
TIER_B_PRIMARY_FEATURES: List[str] = PRIMARY_MODEL_PREDICTORS

# Tier C: Exploratory Sensitivity = Tier B + 19 Unverified Road Proxies (103 features)
# Explicitly quarantined for exploratory sensitivity testing only
TIER_C_EXPLORATORY_FEATURES: List[str] = PRIMARY_MODEL_PREDICTORS + EXPLORATORY_ROAD_PROXY_FEATURES

# 3. Model Default Configurations
RANDOM_SEED = 42

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": RANDOM_SEED
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "tree_method": "hist"
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1
    }
}
