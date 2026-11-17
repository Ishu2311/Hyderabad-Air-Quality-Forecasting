"""
Weather Ingestion and Validation Module
Milestone 1: Data Foundation and Preprocessing

Ingests hourly atmospheric model reanalysis variables from Open-Meteo.
Validates station identifiers, timestamps, physical value bounds, and temporal continuity.
Strictly decoupled from physical CPCB CAAQMS sensor observations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        STATION_REGISTRY,
        STATION_NAME_ALIASES,
        WEATHER_VARIABLES,
        PHYSICAL_VALID_RANGES,
        DATASET_START,
        DATASET_END,
        WEATHER_RAW_FILE
    )
except (ImportError, ValueError):
    from configs.config import (
        STATION_REGISTRY,
        STATION_NAME_ALIASES,
        WEATHER_VARIABLES,
        PHYSICAL_VALID_RANGES,
        DATASET_START,
        DATASET_END,
        WEATHER_RAW_FILE
    )

class WeatherIngestion:
    """Ingests and validates atmospheric reanalysis variables."""

    def __init__(self, raw_file_path: Optional[Path] = None):
        self.raw_file_path = Path(raw_file_path) if raw_file_path else WEATHER_RAW_FILE
        self.data: Optional[pd.DataFrame] = None
        self.validation_summary: Dict = {}

    def load_raw(self) -> pd.DataFrame:
        """Loads raw weather CSV."""
        if not self.raw_file_path.exists():
            raise FileNotFoundError(f"Weather raw file not found at: {self.raw_file_path}")
        df = pd.read_csv(self.raw_file_path)
        self.data = df
        return df

    def standardize_station_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes station names."""
        df = df.copy()
        if "station" not in df.columns:
            raise KeyError("Required column 'station' not found in weather data.")
        df["station"] = df["station"].replace(STATION_NAME_ALIASES)
        
        unknown = set(df["station"].unique()) - set(STATION_REGISTRY.keys())
        if unknown:
            raise ValueError(f"Unknown station identifiers in weather: {unknown}")
        return df

    def standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes datetime column to datetime64[ns]."""
        df = df.copy()
        if "datetime" not in df.columns:
            raise KeyError("Required column 'datetime' not found in weather data.")
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def detect_and_handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detects and reports duplicate (station, datetime) records."""
        dup_mask = df.duplicated(subset=["station", "datetime"], keep="first")
        num_duplicates = int(dup_mask.sum())
        if num_duplicates > 0:
            df = df.drop_duplicates(subset=["station", "datetime"], keep="first").copy()
        return df, num_duplicates

    def validate_physical_bounds(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Validates that meteorological variables fall within physical ranges."""
        df = df.copy()
        violations: Dict[str, int] = {}
        for var in WEATHER_VARIABLES:
            if var in df.columns:
                min_val, max_val = PHYSICAL_VALID_RANGES.get(var, (-100.0, 1000.0))
                invalid_mask = (df[var] < min_val) | (df[var] > max_val)
                count = int(invalid_mask.sum())
                violations[var] = count
                if count > 0:
                    df.loc[invalid_mask, var] = np.nan
        return df, violations

    def enforce_continuous_hourly_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures complete 1-hour grid per station."""
        df = df.copy()
        full_timeline = pd.date_range(
            start=DATASET_START,
            end=DATASET_END,
            freq="h"
        )
        
        station_dfs = []
        for station_name in sorted(STATION_REGISTRY.keys()):
            st_df = df[df["station"] == station_name].copy()
            st_df = st_df.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
            st_reindexed = st_df.reindex(full_timeline)
            st_reindexed["station"] = station_name
            station_dfs.append(st_reindexed)
            
        aligned_df = pd.concat(station_dfs).reset_index().rename(columns={"index": "datetime"})
        return aligned_df

    def generate_weather_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates statistical summary of weather variables."""
        records = []
        for var in WEATHER_VARIABLES:
            if var in df.columns:
                series = df[var]
                records.append({
                    "variable": var,
                    "total_hours": len(series),
                    "valid_count": int(series.notna().sum()),
                    "missing_count": int(series.isna().sum()),
                    "completeness_pct": round(float(series.notna().sum() / len(series) * 100.0), 2),
                    "min": round(float(series.min()), 2),
                    "median": round(float(series.median()), 2),
                    "mean": round(float(series.mean()), 2),
                    "max": round(float(series.max()), 2),
                    "std": round(float(series.std()), 2)
                })
        return pd.DataFrame(records)

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Executes full weather ingestion pipeline."""
        df = self.load_raw()
        df = self.standardize_station_names(df)
        df = self.standardize_timestamps(df)
        df, num_dups = self.detect_and_handle_duplicates(df)
        df, bounds_violations = self.validate_physical_bounds(df)
        df = self.enforce_continuous_hourly_grid(df)
        summary_df = self.generate_weather_summary(df)
        
        self.validation_summary = {
            "total_rows": len(df),
            "duplicates_removed": num_dups,
            "bounds_violations": bounds_violations,
            "completeness_all_vars": bool(df[WEATHER_VARIABLES].notna().all().all())
        }
        return df, summary_df
