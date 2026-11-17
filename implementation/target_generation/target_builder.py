"""
Target Generation Module
Milestone 1: Data Foundation and Preprocessing

Constructs multi-horizon regression and hotspot classification targets:
- Regression targets: +1h, +6h, +24h, +48h instantaneous concentrations.
- 24-hour forward average: Computed over t+1 to t+24 requiring >= 16 valid observations.
  Denominator is strictly N_valid, never a fixed 24. Evaluates to NaN if N_valid < 16.
- Hotspot classification targets:
  - Acute excursion (60 ug/m3): evaluated at +1h, +6h, +24h.
  - Exploratory severe spike (>= 120 ug/m3): evaluated at +1h, +6h.
  - Exploratory Very Poor entry (> 120 ug/m3): evaluated at +1h, +6h.
  - Statutory 24-hour hotspot (>= 60 ug/m3 over forward 24h mean).
- Preserves raw sensor missingness: missing targets evaluate strictly to NaN, NEVER zero.
- Zero-lookahead guarantee: targets are derived strictly from raw ground-truth sensor observations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        REGRESSION_HORIZONS,
        CLASSIFICATION_HORIZONS,
        PM25_STATUTORY_24H_THRESHOLD,
        PM25_ACUTE_EXCURSION_THRESHOLD,
        PM25_SEVERE_SPIKE_THRESHOLD,
        PM25_VERY_POOR_ENTRY_THRESHOLD,
        MIN_VALID_HOURS_24H
    )
except (ImportError, ValueError):
    from configs.config import (
        REGRESSION_HORIZONS,
        CLASSIFICATION_HORIZONS,
        PM25_STATUTORY_24H_THRESHOLD,
        PM25_ACUTE_EXCURSION_THRESHOLD,
        PM25_SEVERE_SPIKE_THRESHOLD,
        PM25_VERY_POOR_ENTRY_THRESHOLD,
        MIN_VALID_HOURS_24H
    )

class TargetBuilder:
    """Constructs multi-horizon targets with missing-aware statutory and acute rules."""

    def __init__(
        self,
        regression_horizons: Optional[List[int]] = None,
        classification_horizons: Optional[List[int]] = None,
        statutory_threshold: float = PM25_STATUTORY_24H_THRESHOLD,
        acute_threshold: float = PM25_ACUTE_EXCURSION_THRESHOLD,
        spike_threshold: float = PM25_SEVERE_SPIKE_THRESHOLD,
        very_poor_threshold: float = PM25_VERY_POOR_ENTRY_THRESHOLD,
        min_valid_hours_24h: int = MIN_VALID_HOURS_24H
    ):
        self.regression_horizons = regression_horizons if regression_horizons else REGRESSION_HORIZONS
        self.classification_horizons = classification_horizons if classification_horizons else CLASSIFICATION_HORIZONS
        self.statutory_threshold = statutory_threshold
        self.acute_threshold = acute_threshold
        self.spike_threshold = spike_threshold
        self.very_poor_threshold = very_poor_threshold
        self.min_valid_hours_24h = min_valid_hours_24h

    def _compute_forward_24h_mean(self, series: pd.Series) -> pd.Series:
        """
        Computes the forward 24-hour mean concentration for window t+1 to t+24.
        
        Mathematical Formulation:
            y_bar_{t+1:t+24} = (1 / N_valid) * sum_{k=1}^{24} y_{t+k} * m_{t+k}
            where N_valid = sum_{k=1}^{24} m_{t+k} >= 16.
            If N_valid < 16, evaluates to NaN.
            Denominator is N_valid, NEVER 24.
        """
        # Reversing the series allows using standard rolling mean
        # When reversed, looking backwards in reversed series corresponds to looking forwards in time
        rev = series.iloc[::-1]
        
        # rolling(24, min_periods=16) computes arithmetic mean of valid observations
        # in the 24-step window, naturally dividing by the count of non-NaNs (N_valid)
        rolled_rev = rev.rolling(window=24, min_periods=self.min_valid_hours_24h).mean()
        
        # Reverse back and shift by -1 so that index t contains mean of t+1 to t+24
        forward_24h_mean = rolled_rev.iloc[::-1].shift(-1)
        return forward_24h_mean

    def _compute_forward_24h_valid_count(self, series: pd.Series) -> pd.Series:
        """Computes count of valid hourly observations in forward window t+1 to t+24."""
        is_valid = series.notna().astype(float)
        rev = is_valid.iloc[::-1]
        rolled_rev = rev.rolling(window=24, min_periods=1).sum()
        forward_valid_count = rolled_rev.iloc[::-1].shift(-1)
        return forward_valid_count

    def build_targets(self, df: pd.DataFrame, pm25_col: str = "PM2.5_raw") -> pd.DataFrame:
        """
        Builds all regression and classification targets per station.
        
        Args:
            df: DataFrame containing at least 'station', 'datetime', and raw PM2.5.
            pm25_col: Column name containing raw un-imputed PM2.5 observations.
            
        Returns:
            DataFrame with original columns and appended target columns.
        """
        df = df.copy()
        if "station" not in df.columns or "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'station' and 'datetime' columns.")
            
        # Fallback to 'PM2.5' if raw col not present
        if pm25_col not in df.columns:
            if "PM2.5" in df.columns:
                pm25_col = "PM2.5"
            else:
                raise KeyError(f"Neither '{pm25_col}' nor 'PM2.5' found in columns.")
                
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        stations = sorted(df["station"].unique())
        
        station_dfs = []
        
        for st in stations:
            st_df = df[df["station"] == st].copy()
            raw_pm25 = st_df[pm25_col]
            
            # --- 1. Multi-Horizon Instantaneous Regression Targets ---
            for h in self.regression_horizons:
                # Shift future value into current timestamp t: shift(-h)
                # If future value is missing, target evaluates to NaN (never zero)
                target_reg = raw_pm25.shift(-h)
                st_df[f"target_pm25_h{h}"] = target_reg
                
            # --- 2. Forward 24-Hour Average Target ---
            forward_24h_mean = self._compute_forward_24h_mean(raw_pm25)
            forward_valid_count = self._compute_forward_24h_valid_count(raw_pm25)
            
            st_df["target_pm25_24h_mean"] = forward_24h_mean
            st_df["target_pm25_24h_valid_count"] = forward_valid_count
            
            # --- 3. Hotspot Classification Targets ---
            for h in self.classification_horizons:
                target_reg = st_df[f"target_pm25_h{h}"]
                
                # A. Acute Excursion Target (60 ug/m3)
                # 1 if y(t+h) >= 60, 0 if y(t+h) < 60, NaN if unobserved
                st_df[f"target_hotspot_acute_h{h}"] = np.where(
                    target_reg.isna(),
                    np.nan,
                    (target_reg >= self.acute_threshold).astype(float)
                )
                
                # B. Exploratory Severe-Spike Alert Target (>= 120 ug/m3 as approved)
                st_df[f"target_hotspot_spike_h{h}"] = np.where(
                    target_reg.isna(),
                    np.nan,
                    (target_reg >= self.spike_threshold).astype(float)
                )
                
                # C. Exploratory Very Poor Entry Alert Target (> 120 ug/m3 CPCB NAQI boundary)
                st_df[f"target_hotspot_very_poor_h{h}"] = np.where(
                    target_reg.isna(),
                    np.nan,
                    (target_reg > self.very_poor_threshold).astype(float)
                )
                
            # D. Statutory 24-Hour Hotspot Target (>= 60 ug/m3 over forward 24h mean)
            # Evaluates to NaN if forward_24h_mean is NaN (i.e., N_valid < 16)
            st_df["target_hotspot_24h_statutory"] = np.where(
                forward_24h_mean.isna(),
                np.nan,
                (forward_24h_mean >= self.statutory_threshold).astype(float)
            )
            
            station_dfs.append(st_df)
            
        result_df = pd.concat(station_dfs).sort_values(by=["station", "datetime"]).reset_index(drop=True)
        return result_df

    def summarize_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates comprehensive distribution summary for all generated targets."""
        target_cols = [c for c in df.columns if c.startswith("target_")]
        records = []
        
        for col in target_cols:
            series = df[col]
            valid_count = int(series.notna().sum())
            missing_count = int(series.isna().sum())
            total = len(series)
            valid_pct = round(valid_count / total * 100.0, 2)
            
            is_binary = set(series.dropna().unique()).issubset({0.0, 1.0})
            
            rec = {
                "target_name": col,
                "target_type": "binary_classification" if is_binary else "continuous_regression",
                "total_rows": total,
                "valid_count": valid_count,
                "missing_count": missing_count,
                "completeness_pct": valid_pct,
                "min": round(float(series.min()), 2) if valid_count > 0 else np.nan,
                "mean": round(float(series.mean()), 4) if valid_count > 0 else np.nan,
                "max": round(float(series.max()), 2) if valid_count > 0 else np.nan,
                "std": round(float(series.std()), 4) if valid_count > 0 else np.nan
            }
            
            if is_binary and valid_count > 0:
                pos_count = int((series == 1.0).sum())
                neg_count = int((series == 0.0).sum())
                rec["positive_class_count"] = pos_count
                rec["negative_class_count"] = neg_count
                rec["positive_class_prevalence_pct"] = round(pos_count / valid_count * 100.0, 2)
            else:
                rec["positive_class_count"] = np.nan
                rec["negative_class_count"] = np.nan
                rec["positive_class_prevalence_pct"] = np.nan
                
            records.append(rec)
            
        summary_df = pd.DataFrame(records)
        return summary_df
