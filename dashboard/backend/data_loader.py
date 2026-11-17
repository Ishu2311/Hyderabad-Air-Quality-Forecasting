"""
Dashboard Backend: Data Loader Service
Provides cached, thread-safe access to Parquet datasets, prediction tables,
evaluation metrics, and spatial summaries.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import numpy as np
import pandas as pd

# Determine Project Roots safely
CURRENT_DIR = Path(__file__).resolve().parent
DASHBOARD_DIR = CURRENT_DIR.parent
PROJECT_ROOT = DASHBOARD_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Optional Streamlit caching decorator fallback for standalone tests
try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    def cache_data(func):
        return func


@cache_data
def load_cpcb_stations_metadata() -> Dict[str, Dict[str, Any]]:
    """Returns canonical metadata for the 4 Hyderabad CPCB CAAQMS monitoring stations."""
    return {
        "Bollaram Industrial Area": {
            "name": "Bollaram Industrial Area",
            "code": "AP001",
            "station_type": "Industrial Corridor",
            "latitude": 17.5405,
            "longitude": 78.3512,
            "utm_x": 218942.3,
            "utm_y": 1941830.1,
            "elevation_m": 574.0,
            "ward": "Jinnaram / Bollaram",
            "surrounding_typology": "Pharmaceutical and chemical manufacturing plants, heavy freight corridors."
        },
        "Central University": {
            "name": "Central University",
            "code": "AP003",
            "station_type": "Academic Enclave / Urban Forest",
            "latitude": 17.4565,
            "longitude": 78.3262,
            "utm_x": 216110.8,
            "utm_y": 1932560.4,
            "elevation_m": 542.0,
            "ward": "Gachibowli / Serilingampally",
            "surrounding_typology": "Forested academic campus, semi-urban institutional buffer, low direct industrial emissions."
        },
        "ECIL Kapra": {
            "name": "ECIL Kapra",
            "code": "AP004",
            "station_type": "Dense Residential / Commercial",
            "latitude": 17.4728,
            "longitude": 78.5583,
            "utm_x": 240751.2,
            "utm_y": 1934091.8,
            "elevation_m": 530.0,
            "ward": "Kapra / Secunderabad",
            "surrounding_typology": "Dense residential settlement, mixed commercial street retail, isolated eastern metropolitan sector."
        },
        "ICRISAT Patancheru": {
            "name": "ICRISAT Patancheru",
            "code": "AP005",
            "station_type": "Semi-Arid Agricultural / Semi-Rural",
            "latitude": 17.5111,
            "longitude": 78.2752,
            "utm_x": 210850.5,
            "utm_y": 1938640.2,
            "elevation_m": 545.0,
            "ward": "Patancheru",
            "surrounding_typology": "Agricultural research fields, highway access periphery, regional background boundary."
        }
    }


@cache_data
def load_historical_observations(
    station: Optional[str] = None,
    pollutants: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Loads historical physical observations from the processed feature store.
    
    Parameters
    ----------
    station : Optional[str]
        Filter by specific station name.
    pollutants : Optional[List[str]]
        Subset of pollutants to return.
    """
    parquet_path = PROCESSED_DIR / "feature_store_primary.parquet"
    if not parquet_path.exists():
        # Fallback to test purged
        parquet_path = PROCESSED_DIR / "feature_store_test_purged.parquet"
    
    if not parquet_path.exists():
        return pd.DataFrame()

    default_cols = ["station", "datetime", "PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
    target_cols = default_cols if pollutants is None else ["station", "datetime"] + pollutants
    
    # Read available columns safely
    schema_df = pd.read_parquet(parquet_path, engine="pyarrow")
    available_cols = [c for c in target_cols if c in schema_df.columns]
    df = schema_df[available_cols].copy()
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    if station is not None and station in df["station"].values:
        df = df[df["station"] == station]
        
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


@cache_data
def load_regression_predictions() -> pd.DataFrame:
    """Loads out-of-time test predictions across all models and horizons."""
    parquet_path = PREDICTIONS_DIR / "regression_predictions_test.parquet"
    if not parquet_path.exists():
        return pd.DataFrame()
        
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


@cache_data
def load_regression_metrics() -> Dict[str, Any]:
    """Loads comprehensive model performance metrics."""
    metrics_path = PREDICTIONS_DIR / "regression_metrics.json"
    stacking_path = PREDICTIONS_DIR / "stacking_regression_metrics.json"
    
    data: Dict[str, Any] = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            data["base_models"] = json.load(f)
            
    if stacking_path.exists():
        with open(stacking_path, "r", encoding="utf-8") as f:
            data["stacking"] = json.load(f)
            
    return data


@cache_data
def load_spatial_validation_results() -> Dict[str, Any]:
    """Loads Leave-One-Station-Out Cross-Validation spatial results."""
    json_path = PREDICTIONS_DIR / "spatial_validation_results.json"
    if not json_path.exists():
        return {}
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_hotspot_summary() -> Dict[str, Any]:
    """Loads spatial hotspot delineation summary and grid cell metrics."""
    json_path = PREDICTIONS_DIR / "hotspot_detection_summary.json"
    if not json_path.exists():
        return {}
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_hotspot_narratives() -> Dict[str, str]:
    """Loads human-readable hotspot narratives generated via TreeSHAP."""
    json_path = PREDICTIONS_DIR / "human_readable_hotspot_narratives.json"
    if not json_path.exists():
        return {}
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_global_feature_importance() -> Dict[str, Any]:
    """Loads global SHAP importance rankings and category shares."""
    json_path = PREDICTIONS_DIR / "global_feature_importance.json"
    if not json_path.exists():
        return {}
        
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_shap_values(horizon: str = "h1") -> Optional[Dict[str, Any]]:
    """Loads serialized SHAP matrices, feature names, and metadata."""
    npz_path = PREDICTIONS_DIR / f"shap_values_{horizon}.npz"
    if not npz_path.exists():
        return None
        
    data = np.load(npz_path, allow_pickle=True)
    return {
        "shap_values": data["shap_values"],
        "X_exp": data["X_exp"],
        "features": list(data["features"]),
        "expected_value": float(data["expected_value"]),
        "primary_model": str(data["primary_model"]),
        "meta_weights": data["meta_weights"],
        "intercept": float(data["intercept"])
    }


def get_figure_path(figure_name: str) -> Optional[Path]:
    """Resolves absolute path of pre-rendered high-resolution figures."""
    path = FIGURES_DIR / figure_name
    if path.exists():
        return path
    
    # Also check implementation/reports/figures
    impl_path = PROJECT_ROOT / "implementation" / "reports" / "figures" / figure_name
    if impl_path.exists():
        return impl_path
        
    return None
