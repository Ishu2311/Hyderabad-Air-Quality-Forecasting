"""
Pytest Fixtures for Milestone 1 Test Suite
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure implementation directory, project root, and repo root are in sys.path
TEST_DIR = Path(__file__).resolve().parent
IMPL_DIR = TEST_DIR.parent
PROJECT_ROOT = IMPL_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent

for _p in [str(REPO_ROOT), str(PROJECT_ROOT), str(IMPL_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.config import (
    STATION_REGISTRY,
    CPCB_RAW_FILE,
    WEATHER_RAW_FILE
)
from data_ingestion.dataset_loader import DatasetLoader
from preprocessing.prospective_imputer import ProspectiveImputer
from target_generation.target_builder import TargetBuilder

@pytest.fixture(scope="session")
def sample_synthetic_timeseries():
    """
    Creates a controlled synthetic hourly dataset with known gaps:
    - Station A and Station B.
    - 100 hours each.
    - Station A has:
      - Short gap: 2 consecutive NaNs (hours 10-11).
      - Medium gap: 6 consecutive NaNs (hours 30-35).
      - Extended outage: 28 consecutive NaNs (hours 50-77).
    """
    dates = pd.date_range("2023-01-01 00:00:00", periods=100, freq="h")
    
    # Station A values
    vals_a = np.linspace(20, 80, 100)
    # Insert known gaps
    vals_a[10:12] = np.nan  # 2h gap (short)
    vals_a[30:36] = np.nan  # 6h gap (medium)
    vals_a[50:78] = np.nan  # 28h gap (extended)
    
    # Station B values
    vals_b = np.linspace(30, 90, 100)
    vals_b[5:8] = np.nan    # 3h gap (short)
    vals_b[20:30] = np.nan  # 10h gap (medium)
    
    df_a = pd.DataFrame({
        "datetime": dates,
        "station": "Bollaram Industrial Area",
        "PM2.5": vals_a,
        "temperature_2m": np.linspace(20, 35, 100),
        "relative_humidity_2m": np.linspace(40, 80, 100),
        "wind_speed_10m": np.linspace(1, 5, 100),
        "wind_direction_10m": np.linspace(90, 270, 100),
        "precipitation": np.zeros(100),
        "surface_pressure": np.ones(100) * 1013.25
    })
    
    df_b = pd.DataFrame({
        "datetime": dates,
        "station": "Central University",
        "PM2.5": vals_b,
        "temperature_2m": np.linspace(19, 34, 100),
        "relative_humidity_2m": np.linspace(45, 85, 100),
        "wind_speed_10m": np.linspace(1.2, 4.8, 100),
        "wind_direction_10m": np.linspace(80, 260, 100),
        "precipitation": np.zeros(100),
        "surface_pressure": np.ones(100) * 1013.25
    })
    
    df = pd.concat([df_a, df_b], ignore_index=True)
    return df

@pytest.fixture(scope="session")
def real_merged_dataset():
    """Loads and merges real CPCB and weather datasets."""
    loader = DatasetLoader()
    merged, audit = loader.load_and_merge(save_interim=False)
    return merged
