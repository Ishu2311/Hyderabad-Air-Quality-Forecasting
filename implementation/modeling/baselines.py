"""
Forecasting Baselines Implementation Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Implements:
1. Zero-Order Persistence Baseline (Markovian):
   y_hat(i, t+h) = y(i, t) for all h in {1, 6, 24, 48}
2. Corrected Seasonal-Naive Baseline (Diurnal Cycle S = 24h, Single Exact Source Timestamps):
   h = +1h:  source = y(i, t - 23h)
   h = +6h:  source = y(i, t - 18h)
   h = +24h: source = y(i, t - 24h)
   h = +48h: source = y(i, t - 24h)
   
Strict non-leakage: all source timestamps are strictly <= t.
Missing source values evaluate strictly to NaN and are excluded from metric loss.
Zero substitution is strictly prohibited.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class BaselineForecaster:
    """Computes zero-order persistence and seasonal-naive benchmark forecasts."""

    def __init__(self, pm25_col: str = "PM2.5"):
        self.pm25_col = pm25_col

    def generate_persistence_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Zero-Order Persistence forecast across all horizons:
        y_hat(i, t+h) = y(i, t)
        """
        preds = pd.DataFrame(index=df.index)
        preds["station"] = df["station"]
        preds["datetime"] = df["datetime"]
        
        # At prediction origin t, persistence uses current available PM2.5 observation
        current_obs = df[self.pm25_col].values
        
        for h in [1, 6, 24, 48]:
            preds[f"pred_persistence_h{h}"] = current_obs
            
        return preds

    def generate_seasonal_naive_forecasts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Corrected Seasonal-Naive forecasts using approved single exact historical source lags:
        - +1h:  y(i, t - 23h)
        - +6h:  y(i, t - 18h)
        - +24h: y(i, t - 24h)
        - +48h: y(i, t - 24h)
        """
        df_sorted = df.sort_values(by=["station", "datetime"]).copy()
        
        preds_list = []
        for station, group in df_sorted.groupby("station"):
            st_preds = pd.DataFrame(index=group.index)
            st_preds["station"] = group["station"]
            st_preds["datetime"] = group["datetime"]
            
            s = group[self.pm25_col]
            
            # Horizon +1h: Diurnal match in preceding cycle observed before t -> Lag 23
            st_preds["pred_seasonal_naive_h1"] = s.shift(23).values
            
            # Horizon +6h: Diurnal match in preceding cycle observed before t -> Lag 18
            st_preds["pred_seasonal_naive_h6"] = s.shift(18).values
            
            # Horizon +24h: Corresponding hour in preceding diurnal cycle -> Lag 24
            st_preds["pred_seasonal_naive_h24"] = s.shift(24).values
            
            # Horizon +48h: Single exact source timestamp rule -> Lag 24
            st_preds["pred_seasonal_naive_h48"] = s.shift(24).values
            
            preds_list.append(st_preds)
            
        preds = pd.concat(preds_list).loc[df.index]
        return preds

    def generate_hotspot_persistence_forecasts(
        self,
        df: pd.DataFrame,
        threshold: float = 60.0
    ) -> pd.DataFrame:
        """
        Computes persistence hotspot forecast:
        z_hat(i, t+h) = 1 if y(i, t) >= threshold else 0 (NaN if y(i, t) is NaN)
        """
        preds = pd.DataFrame(index=df.index)
        preds["station"] = df["station"]
        preds["datetime"] = df["datetime"]
        
        current_obs = df[self.pm25_col]
        hotspot_flag = (current_obs >= threshold).astype(float)
        hotspot_flag[current_obs.isna()] = np.nan
        
        for h in [1, 6, 24]:
            preds[f"pred_hotspot_persistence_h{h}"] = hotspot_flag.values
            
        return preds

    def generate_hotspot_seasonal_naive_forecasts(
        self,
        df: pd.DataFrame,
        threshold: float = 60.0
    ) -> pd.DataFrame:
        """
        Computes seasonal-naive hotspot forecast:
        z_hat(i, t+h) = 1 if y_hat_seasonal_naive(i, t+h) >= threshold else 0 (NaN if source is NaN)
        """
        sn_preds = self.generate_seasonal_naive_forecasts(df)
        preds = pd.DataFrame(index=df.index)
        preds["station"] = df["station"]
        preds["datetime"] = df["datetime"]
        
        for h in [1, 6, 24]:
            sn_val = sn_preds[f"pred_seasonal_naive_h{h}"]
            flag = (sn_val >= threshold).astype(float)
            flag[sn_val.isna()] = np.nan
            preds[f"pred_hotspot_seasonal_naive_h{h}"] = flag.values
            
        return preds
