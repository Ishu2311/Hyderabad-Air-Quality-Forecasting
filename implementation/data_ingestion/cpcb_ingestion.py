"""
CPCB Ingestion and Validation Module
Milestone 1: Data Foundation and Preprocessing

Ingests physical Continuous Ambient Air Quality Monitoring Station (CAAQMS)
observations from the Central Pollution Control Board (CPCB) network.
Validates station identifiers, standardizes timestamps and units, identifies duplicates,
and detects invalid physical sensor readings without lookahead or data fabrication.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        STATION_REGISTRY,
        STATION_NAME_ALIASES,
        POLLUTANTS,
        PHYSICAL_VALID_RANGES,
        DATASET_START,
        DATASET_END,
        EXPECTED_HOURS_PER_STATION,
        EXPECTED_TOTAL_STATION_HOURS,
        CPCB_RAW_FILE
    )
except (ImportError, ValueError):
    from configs.config import (
        STATION_REGISTRY,
        STATION_NAME_ALIASES,
        POLLUTANTS,
        PHYSICAL_VALID_RANGES,
        DATASET_START,
        DATASET_END,
        EXPECTED_HOURS_PER_STATION,
        EXPECTED_TOTAL_STATION_HOURS,
        CPCB_RAW_FILE
    )

class CPCBIngestion:
    """Ingests and validates raw physical CPCB ambient air quality observations."""

    def __init__(self, raw_file_path: Optional[Path] = None):
        self.raw_file_path = Path(raw_file_path) if raw_file_path else CPCB_RAW_FILE
        self.data: Optional[pd.DataFrame] = None
        self.validation_summary: Dict = {}

    def load_raw(self) -> pd.DataFrame:
        """Loads the raw CPCB CSV file."""
        if not self.raw_file_path.exists():
            raise FileNotFoundError(f"CPCB raw file not found at: {self.raw_file_path}")
        df = pd.read_csv(self.raw_file_path)
        self.data = df
        return df

    def standardize_station_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes station names using official registry and aliases."""
        df = df.copy()
        if "station" not in df.columns:
            raise KeyError("Required column 'station' not found in CPCB data.")
        
        # Apply alias mapping
        df["station"] = df["station"].replace(STATION_NAME_ALIASES)
        
        # Validate that all stations are known
        unknown_stations = set(df["station"].unique()) - set(STATION_REGISTRY.keys())
        if unknown_stations:
            raise ValueError(f"Unknown station identifiers encountered: {unknown_stations}")
            
        return df

    def standardize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes datetime column to hourly datetime64[ns] IST."""
        df = df.copy()
        if "datetime" not in df.columns:
            raise KeyError("Required column 'datetime' not found in CPCB data.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def detect_and_handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Detects and reports duplicate (station, datetime) records."""
        dup_mask = df.duplicated(subset=["station", "datetime"], keep="first")
        num_duplicates = int(dup_mask.sum())
        if num_duplicates > 0:
            df = df.drop_duplicates(subset=["station", "datetime"], keep="first").copy()
        return df, num_duplicates

    def detect_invalid_physical_readings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Detects invalid physical readings:
        - Negative concentrations (< 0) are physically impossible and replaced with NaN.
        - Preserves missingness as NaN; NEVER replaces with 0.
        - Extreme unphysical readings beyond sensor saturation bounds are recorded.
        """
        df = df.copy()
        invalid_counts: Dict[str, int] = {}
        
        for col in POLLUTANTS:
            if col in df.columns:
                min_valid, max_valid = PHYSICAL_VALID_RANGES.get(col, (0.0, 1000.0))
                
                # Check negative values
                neg_mask = df[col] < min_valid
                neg_count = int(neg_mask.sum())
                if neg_count > 0:
                    df.loc[neg_mask, col] = np.nan
                    
                # Check extreme sensor saturation anomalies
                extreme_mask = df[col] > max_valid
                extreme_count = int(extreme_mask.sum())
                if extreme_count > 0:
                    # In research, we flag or cap; here we record and retain or replace with NaN
                    df.loc[extreme_mask, col] = np.nan
                    
                invalid_counts[col] = neg_count + extreme_count
                
        return df, invalid_counts

    def enforce_continuous_hourly_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures an uninterrupted, chronological 1-hour grid per station.
        Any missing timestamps in the calendar range are explicitly created with NaN.
        """
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
            
            # Reindex to full chronological grid
            st_reindexed = st_df.reindex(full_timeline)
            st_reindexed["station"] = station_name
            st_reindexed["station_id"] = STATION_REGISTRY[station_name].station_id
            
            station_dfs.append(st_reindexed)
            
        aligned_df = pd.concat(station_dfs).reset_index().rename(columns={"index": "datetime"})
        return aligned_df

    def generate_station_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates comprehensive station-wise observation summary."""
        records = []
        for station_name, meta in sorted(STATION_REGISTRY.items()):
            st_df = df[df["station"] == station_name]
            total_hours = len(st_df)
            
            pm25 = st_df["PM2.5"]
            valid_obs = int(pm25.notna().sum())
            missing_obs = int(pm25.isna().sum())
            missing_pct = (missing_obs / total_hours * 100.0) if total_hours > 0 else 0.0
            
            records.append({
                "station_name": station_name,
                "station_id": meta.station_id,
                "typology": meta.typology,
                "total_station_hours": total_hours,
                "pm25_valid_observations": valid_obs,
                "pm25_missing_hours": missing_obs,
                "pm25_missing_percentage": round(missing_pct, 2),
                "pm25_min": round(float(pm25.min()), 2) if valid_obs > 0 else np.nan,
                "pm25_median": round(float(pm25.median()), 2) if valid_obs > 0 else np.nan,
                "pm25_mean": round(float(pm25.mean()), 2) if valid_obs > 0 else np.nan,
                "pm25_max": round(float(pm25.max()), 2) if valid_obs > 0 else np.nan,
                "pm25_std": round(float(pm25.std()), 2) if valid_obs > 0 else np.nan
            })
            
        summary_df = pd.DataFrame(records)
        return summary_df

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Executes full CPCB ingestion pipeline."""
        df = self.load_raw()
        df = self.standardize_station_names(df)
        df = self.standardize_timestamps(df)
        df, num_dups = self.detect_and_handle_duplicates(df)
        df, invalid_counts = self.detect_invalid_physical_readings(df)
        df = self.enforce_continuous_hourly_grid(df)
        summary_df = self.generate_station_summary(df)
        
        self.validation_summary = {
            "total_rows": len(df),
            "duplicates_removed": num_dups,
            "invalid_readings_cleaned": invalid_counts,
            "stations_present": sorted(df["station"].unique().tolist()),
            "total_valid_pm25": int(df["PM2.5"].notna().sum()),
            "total_missing_pm25": int(df["PM2.5"].isna().sum())
        }
        
        return df, summary_df
