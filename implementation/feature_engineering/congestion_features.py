"""
Congestion Windows and Traffic Dynamics Feature Engineering Module
Milestone 2: Multi-Source Feature Engineering

Implements temporal congestion windows and spatio-temporal traffic pressure interactions:
- Morning peak rush window: 08:00 to 11:00 inclusive
- Evening peak rush window: 17:00 to 21:00 inclusive
- Composite traffic peak: Morning or Evening peak
- Day-type indicator: Weekend vs. Weekday
- Weekday rush indicator: Peak rush during business weekdays
- Exploratory freight restriction window: 23:00 to 05:00 inclusive
- Spatio-temporal interaction proxies: Interaction of localized road metrics with temporal congestion

EXPLORATORY FREIGHT ASSUMPTION NOTICE:
The 23:00–05:00 freight entry window is implemented strictly as an exploratory proxy assumption
reflecting typical metropolitan commercial vehicle regulations in urban India. It must NOT be
presented as an established statutory certainty until verified against official gazette notifications
issued by the Hyderabad City Police Commissionerate.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        MORNING_PEAK_HOURS,
        EVENING_PEAK_HOURS,
        EXPLORATORY_FREIGHT_HOURS
    )
except (ImportError, ValueError):
    from configs.config import (
        MORNING_PEAK_HOURS,
        EVENING_PEAK_HOURS,
        EXPLORATORY_FREIGHT_HOURS
    )

class CongestionFeatureExtractor:
    """Extracts diurnal congestion indicators and spatio-temporal traffic interaction features."""

    def __init__(
        self,
        morning_hours: Tuple[int, ...] = MORNING_PEAK_HOURS,
        evening_hours: Tuple[int, ...] = EVENING_PEAK_HOURS,
        freight_hours: Tuple[int, ...] = EXPLORATORY_FREIGHT_HOURS
    ):
        self.morning_hours = morning_hours
        self.evening_hours = evening_hours
        self.freight_hours = freight_hours

    def extract_temporal_congestion_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts binary calendar and diurnal congestion indicator windows."""
        df = df.copy()
        if "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'datetime' column.")
            
        dt = pd.to_datetime(df["datetime"])
        hour = dt.dt.hour
        dow = dt.dt.dayofweek
        
        # 1. Morning Peak Rush Window (08:00 - 11:00)
        df["is_morning_peak"] = hour.isin(self.morning_hours).astype(float)
        
        # 2. Evening Peak Rush Window (17:00 - 21:00)
        df["is_evening_peak"] = hour.isin(self.evening_hours).astype(float)
        
        # 3. Composite Traffic Peak Window
        df["is_traffic_peak"] = ((df["is_morning_peak"] == 1.0) | (df["is_evening_peak"] == 1.0)).astype(float)
        
        # 4. Weekend Indicator (Saturday=5, Sunday=6)
        df["is_weekend"] = dow.isin([5, 6]).astype(float)
        
        # 5. Weekday Traffic Peak (Active commuter flow)
        df["is_weekday_peak"] = ((df["is_traffic_peak"] == 1.0) & (df["is_weekend"] == 0.0)).astype(float)
        
        # 6. Exploratory Freight Restriction Window (23:00 - 05:00)
        # Unverified exploratory assumption documented per methodology guidelines
        df["is_freight_window_exploratory"] = hour.isin(self.freight_hours).astype(float)
        
        # 7. Off-Peak Daytime Window (11:00 - 17:00 on weekdays)
        df["is_daytime_offpeak"] = (
            (hour >= 11) & (hour < 17) & (df["is_weekend"] == 0.0)
        ).astype(float)
        
        return df

    def extract_traffic_pressure_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates spatio-temporal interaction features combining localized road metrics
        with active temporal congestion windows.
        
        Rationale:
        A high road density station at 02:00 has low active emission generation, while the same
        station at 09:00 experiences concentrated tailpipe emissions and stop-and-go idling.
        Multiplying static road density by dynamic congestion flags captures this non-linear interaction.
        """
        df = df.copy()
        
        # Verify presence of OSM road metrics
        has_osm = "osm_road_density_1km" in df.columns
        has_primary = "osm_primary_length_1km" in df.columns
        has_highway_dist = "osm_distance_to_highway_km" in df.columns
        
        if not has_osm:
            return df
            
        peak = df.get("is_traffic_peak", 0.0)
        weekday_peak = df.get("is_weekday_peak", 0.0)
        freight = df.get("is_freight_window_exploratory", 0.0)
        
        # 1. Total Roadway Traffic Pressure within 1km buffer
        df["traffic_pressure_total_1km"] = df["osm_road_density_1km"] * peak
        
        # 2. Weekday Commuter Traffic Pressure
        df["traffic_pressure_weekday_peak"] = df["osm_road_density_1km"] * weekday_peak
        
        # 3. Primary Arterial Peak Pressure
        if has_primary:
            df["traffic_pressure_primary_1km"] = df["osm_primary_length_1km"] * peak
            
        # 4. Exploratory Night-time Freight Heavy Vehicle Pressure (within 3km regional buffer)
        if "osm_road_density_3km" in df.columns:
            df["freight_pressure_proxy"] = df["osm_road_density_3km"] * freight
            
        # 5. Highway Corridor Congestion Proximity
        if has_highway_dist:
            # Inverse distance weighted by active peak flag
            df["highway_rush_pressure"] = peak / (df["osm_distance_to_highway_km"] + 0.1)
            
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full congestion and traffic dynamics feature extraction pipeline."""
        df = self.extract_temporal_congestion_indicators(df)
        df = self.extract_traffic_pressure_interactions(df)
        return df
