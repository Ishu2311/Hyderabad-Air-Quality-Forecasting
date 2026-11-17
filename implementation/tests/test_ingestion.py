"""
Automated Ingestion Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from configs.config import (
    STATION_REGISTRY,
    EXPECTED_HOURS_PER_STATION,
    EXPECTED_TOTAL_STATION_HOURS,
    WEATHER_VARIABLES
)
from data_ingestion.cpcb_ingestion import CPCBIngestion
from data_ingestion.weather_ingestion import WeatherIngestion

def test_cpcb_duplicate_timestamps():
    """Verify zero duplicate timestamps in CPCB observations."""
    ingest = CPCBIngestion()
    df, summary = ingest.process()
    dups = df.duplicated(subset=["station", "datetime"]).sum()
    assert dups == 0, f"Found {dups} duplicate records in CPCB data."

def test_weather_duplicate_timestamps():
    """Verify zero duplicate timestamps in weather reanalysis."""
    ingest = WeatherIngestion()
    df, summary = ingest.process()
    dups = df.duplicated(subset=["station", "datetime"]).sum()
    assert dups == 0, f"Found {dups} duplicate records in weather data."

def test_negative_readings_cleaning():
    """Verify negative PM2.5 readings are cleaned to NaN and NEVER replaced with zero."""
    ingest = CPCBIngestion()
    # Create test df with synthetic negative reading
    test_df = pd.DataFrame({
        "datetime": ["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
        "station": ["Bollaram Industrial Area", "Bollaram Industrial Area"],
        "PM2.5": [50.0, -15.5]
    })
    cleaned, counts = ingest.detect_invalid_physical_readings(test_df)
    assert counts["PM2.5"] == 1
    assert np.isnan(cleaned.loc[1, "PM2.5"]), "Negative reading must evaluate to NaN"
    assert cleaned.loc[1, "PM2.5"] != 0.0, "Negative reading must NEVER be coerced to zero"

def test_station_identifier_standardization():
    """Verify alias mapping correctly resolves 'Hyderabad' to 'Central University'."""
    ingest = CPCBIngestion()
    test_df = pd.DataFrame({
        "station": ["Hyderabad", "Bollaram Industrial Area", "ECIL Kapra", "ICRISAT Patancheru"],
        "datetime": ["2023-01-01 00:00:00"] * 4
    })
    standardized = ingest.standardize_station_names(test_df)
    assert "Hyderabad" not in standardized["station"].values
    assert "Central University" in standardized["station"].values
    assert len(set(standardized["station"].unique()) - set(STATION_REGISTRY.keys())) == 0

def test_hourly_continuity():
    """Verify strictly monotonic 1-hour continuity per station."""
    ingest = CPCBIngestion()
    df, _ = ingest.process()
    for st in sorted(STATION_REGISTRY.keys()):
        st_df = df[df["station"] == st].sort_values(by="datetime")
        diffs = st_df["datetime"].diff().dropna()
        assert (diffs == pd.Timedelta(hours=1)).all(), f"Hourly continuity broken in station {st}"

def test_exact_dimensions_and_valid_counts():
    """
    Verify exact match against audited research specifications:
    - Exactly 131,424 total station-hours (32,856 per station).
    - Exactly 111,400 valid PM2.5 observations.
    - Station valid counts:
      - Bollaram: 23,662
      - Central University: 29,127
      - ECIL Kapra: 28,519
      - ICRISAT Patancheru: 30,092
    """
    ingest = CPCBIngestion()
    df, summary = ingest.process()
    
    assert len(df) == EXPECTED_TOTAL_STATION_HOURS, f"Total rows {len(df)} != {EXPECTED_TOTAL_STATION_HOURS}"
    
    expected_station_valid = {
        "Bollaram Industrial Area": 23662,
        "Central University": 29127,
        "ECIL Kapra": 28519,
        "ICRISAT Patancheru": 30092
    }
    
    for _, row in summary.iterrows():
        st = row["station_name"]
        exp = expected_station_valid[st]
        act = row["pm25_valid_observations"]
        assert act == exp, f"Station {st}: actual {act} != expected {exp}"
        assert row["total_station_hours"] == EXPECTED_HOURS_PER_STATION
        
    total_valid = summary["pm25_valid_observations"].sum()
    assert total_valid == 111400, f"Total valid {total_valid} != 111,400"

def test_weather_completeness(real_merged_dataset):
    """Verify weather reanalysis has 100% completeness (zero NaNs) across all 131,424 hours."""
    for var in WEATHER_VARIABLES:
        assert var in real_merged_dataset.columns
        assert real_merged_dataset[var].isna().sum() == 0, f"Weather variable {var} contains NaN"
