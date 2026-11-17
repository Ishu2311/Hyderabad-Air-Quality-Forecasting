"""
Dataset Loader and Multimodal Alignment Module
Milestone 1: Data Foundation and Preprocessing

Combines ground-truth CPCB CAAQMS sensor telemetry with Open-Meteo meteorological reanalysis.
Preserves explicit separation of modalities and attaches geodetic station metadata.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        STATION_REGISTRY,
        EXPECTED_TOTAL_STATION_HOURS,
        MERGED_RAW_FILE
    )
    from .cpcb_ingestion import CPCBIngestion
    from .weather_ingestion import WeatherIngestion
except (ImportError, ValueError):
    from configs.config import (
        STATION_REGISTRY,
        EXPECTED_TOTAL_STATION_HOURS,
        MERGED_RAW_FILE
    )
    from data_ingestion.cpcb_ingestion import CPCBIngestion
    from data_ingestion.weather_ingestion import WeatherIngestion

class DatasetLoader:
    """Loads, integrates, and validates multimodal environmental datasets."""

    def __init__(self, cpcb_path: Optional[Path] = None, weather_path: Optional[Path] = None):
        self.cpcb_ingestion = CPCBIngestion(cpcb_path)
        self.weather_ingestion = WeatherIngestion(weather_path)
        self.merged_data: Optional[pd.DataFrame] = None

    def load_and_merge(self, save_interim: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Loads both modalities, merges on (station, datetime), and attaches station metadata."""
        df_cpcb, cpcb_summary = self.cpcb_ingestion.process()
        df_weather, weather_summary = self.weather_ingestion.process()
        
        # Merge on station and datetime (1:1 alignment)
        merged = pd.merge(df_cpcb, df_weather, on=["station", "datetime"], how="inner")
        
        if len(merged) != EXPECTED_TOTAL_STATION_HOURS:
            raise ValueError(
                f"Merged row count mismatch: expected {EXPECTED_TOTAL_STATION_HOURS}, got {len(merged)}"
            )
            
        # Attach geodetic station metadata
        for station_name, meta in STATION_REGISTRY.items():
            mask = merged["station"] == station_name
            merged.loc[mask, "station_id"] = meta.station_id
            merged.loc[mask, "latitude"] = meta.latitude
            merged.loc[mask, "longitude"] = meta.longitude
            merged.loc[mask, "easting_utm44n"] = meta.easting_utm44n
            merged.loc[mask, "northing_utm44n"] = meta.northing_utm44n
            merged.loc[mask, "typology"] = meta.typology
            
        # Sort chronologically by station
        merged = merged.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        self.merged_data = merged
        
        if save_interim:
            MERGED_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(MERGED_RAW_FILE, index=False)
            
        audit_info = {
            "cpcb_summary": cpcb_summary.to_dict(orient="records"),
            "weather_summary": weather_summary.to_dict(orient="records"),
            "total_station_hours": len(merged),
            "stations": sorted(merged["station"].unique().tolist()),
            "date_range": [str(merged["datetime"].min()), str(merged["datetime"].max())],
            "pm25_valid_count": int(merged["PM2.5"].notna().sum()),
            "pm25_missing_count": int(merged["PM2.5"].isna().sum())
        }
        
        return merged, audit_info
