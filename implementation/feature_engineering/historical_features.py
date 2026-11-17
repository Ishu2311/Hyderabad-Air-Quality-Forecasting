"""
Historical PM2.5 and Atmospheric Feature Engineering Module
Milestone 2: Multi-Source Feature Engineering

Implements station-wise historical features:
- Autoregressive lags: 1h, 2h, 3h, 6h, 12h, 24h
- Rolling statistical features (past-only): 3h, 6h, 12h, 24h means; 6h, 24h std; 24h min/max
- Historical trend dynamics: 1h, 3h, 24h differences and rate-of-change acceleration
- Diurnal and seasonal cyclical features: hour, day-of-week, month sine/cosine, and IMD seasons
- Multi-pollutant co-exposure: PM10, NO2, SO2 historical lags and chemical source ratios

Enforces strict zero-lookahead: all rolling calculations use past observations only (<= t).
Prevents information leakage across stations and chronological partitions.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        HISTORICAL_LAGS,
        ROLLING_WINDOWS,
        TREND_DIFFERENCES
    )
except (ImportError, ValueError):
    from configs.config import (
        HISTORICAL_LAGS,
        ROLLING_WINDOWS,
        TREND_DIFFERENCES
    )

def get_imd_season(month: int) -> str:
    """
    Standard Indian Meteorological Department (IMD) seasonal classification:
    - Winter: Dec, Jan, Feb (12, 1, 2)
    - Summer / Pre-Monsoon: Mar, Apr, May (3, 4, 5)
    - Monsoon (Southwest): Jun, Jul, Aug, Sep (6, 7, 8, 9)
    - Post-Monsoon / Retreating: Oct, Nov (10, 11)
    """
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Monsoon"
    else:
        return "Post_Monsoon"

class HistoricalFeatureExtractor:
    """Extracts station-wise autoregressive, rolling, trend, and cyclical features."""

    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        trend_diffs: Optional[List[int]] = None
    ):
        self.lags = lags if lags is not None else HISTORICAL_LAGS
        self.rolling_windows = rolling_windows if rolling_windows is not None else ROLLING_WINDOWS
        self.trend_diffs = trend_diffs if trend_diffs is not None else TREND_DIFFERENCES

    def extract_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts mathematically continuous cyclical time transformations.
        Hour, day-of-week, and month are transformed via sine and cosine.
        """
        df = df.copy()
        if "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'datetime' column.")
            
        dt = pd.to_datetime(df["datetime"])
        
        # Hour of day (0-23)
        hour = dt.dt.hour
        df["hour"] = hour
        df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        
        # Day of week (0=Monday, 6=Sunday)
        dow = dt.dt.dayofweek
        df["dayofweek"] = dow
        df["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        
        # Month of year (1-12)
        month = dt.dt.month
        df["month"] = month
        df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        
        # IMD Season
        df["season"] = month.apply(get_imd_season)
        df["is_winter"] = (df["season"] == "Winter").astype(float)
        df["is_summer"] = (df["season"] == "Summer").astype(float)
        df["is_monsoon"] = (df["season"] == "Monsoon").astype(float)
        df["is_post_monsoon"] = (df["season"] == "Post_Monsoon").astype(float)
        
        return df

    def extract_historical_pm25_features(
        self,
        df: pd.DataFrame,
        target_col: str = "PM2.5"
    ) -> pd.DataFrame:
        """
        Extracts station-wise autoregressive lags, rolling statistics, and trends.
        
        CRITICAL NON-LEAKAGE ENFORCEMENT:
        - Grouped strictly by 'station'.
        - Datetime sorted monotonically.
        - Lags use strictly positive shifts: shift(k), k in {1, 2, 3, 6, 12, 24}.
        - Rolling windows are computed over [t-w+1 : t], strictly past observations only.
        - Future values (> t) are never accessed.
        """
        df = df.copy()
        if "station" not in df.columns or "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'station' and 'datetime' columns.")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        
        grouped = df.groupby("station")[target_col]
        
        # 1. Autoregressive Historical Lags
        for lag in self.lags:
            df[f"pm25_lag_{lag}"] = grouped.shift(lag)
            
        # 2. Past-Only Rolling Statistics
        # Note: rolling over series at index t aggregates {y_{t-w+1}, ..., y_t}, which is strictly past-only.
        for w in self.rolling_windows:
            df[f"pm25_roll_mean_{w}h"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).mean())
            if w in (6, 24):
                df[f"pm25_roll_std_{w}h"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=2).std())
            if w == 24:
                df[f"pm25_roll_min_24h"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).min())
                df[f"pm25_roll_max_24h"] = grouped.transform(lambda s: s.rolling(window=w, min_periods=1).max())
                
        # 3. Historical Short & Diurnal Trends
        # Trend = y_t - y_{t-k} (rate of change over historical window)
        for d in self.trend_diffs:
            df[f"pm25_trend_diff_{d}h"] = df[target_col] - df[f"pm25_lag_{d}"]
            
        # 4. Acceleration / Curvature: (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})
        df["pm25_acceleration_1h"] = df["pm25_trend_diff_1h"] - (df["pm25_lag_1"] - df["pm25_lag_2"])
        
        # 5. Volatility Ratio: current concentration relative to 24-hour baseline
        # Ratio y_t / (roll_mean_24h + eps) indicates acute surge relative to diurnal background
        df["pm25_ratio_to_24h_mean"] = df[target_col] / (df["pm25_roll_mean_24h"] + 1e-5)
        
        return df

    def extract_co_pollutant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts multi-pollutant lags and photochemical/combustion ratios.
        - Coarse vs fine aerosol ratio: PM2.5 / (PM10 + eps)
        - Combustion oxidation ratio: NO / (NO2 + eps)
        - Historical 1h and 24h lags for key co-pollutants (PM10, NO2, SO2)
        """
        df = df.copy()
        grouped = df.groupby("station")
        
        # Chemical source ratios
        if "PM2.5" in df.columns and "PM10" in df.columns:
            df["ratio_pm25_pm10"] = df["PM2.5"] / (df["PM10"] + 1e-5)
            # Clip physical ratio to [0, 1.5] to suppress noise from division by near-zero PM10
            df["ratio_pm25_pm10"] = df["ratio_pm25_pm10"].clip(lower=0.0, upper=1.5)
            
        if "NO" in df.columns and "NO2" in df.columns:
            df["ratio_no_no2"] = df["NO"] / (df["NO2"] + 1e-5)
            df["ratio_no_no2"] = df["ratio_no_no2"].clip(lower=0.0, upper=10.0)
            
        # Co-pollutant lags (strictly positive past shifts)
        co_pollutants = [p for p in ["PM10", "NO2", "SO2"] if p in df.columns]
        for cp in co_pollutants:
            p_lower = cp.lower()
            df[f"{p_lower}_lag_1"] = grouped[cp].shift(1)
            df[f"{p_lower}_lag_24"] = grouped[cp].shift(24)
            
        return df

    def transform(self, df: pd.DataFrame, target_col: str = "PM2.5") -> pd.DataFrame:
        """Executes full historical feature extraction pipeline."""
        df = self.extract_cyclical_features(df)
        df = self.extract_historical_pm25_features(df, target_col=target_col)
        df = self.extract_co_pollutant_features(df)
        return df
