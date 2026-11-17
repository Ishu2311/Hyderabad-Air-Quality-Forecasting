"""
Prospective Leakage-Safe Imputer Module
Milestone 1: Data Foundation and Preprocessing

Implements strict zero-lookahead feature preprocessing:
- Preserves raw physical observations and missingness masks.
- Short gaps (<= 3h): Forward-filled strictly from past valid observations (ffill past-only, limit=3).
- Medium gaps (4 to 24h): Imputed using diurnal monthly/hourly lookup tables fitted on training data only.
- Extended outages (> 24h): Retained as NaN for native gradient boosting tree handling (primary feature set).
- Dense option provided for linear/neural models.
- Tracking indicator flags added for full auditability.
- Strict isolation: Target labels are NEVER imputed or modified.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        POLLUTANTS,
        SHORT_GAP_MAX_HOURS,
        MEDIUM_GAP_MAX_HOURS
    )
    from .gap_analyzer import GapAnalyzer
except (ImportError, ValueError):
    from configs.config import (
        POLLUTANTS,
        SHORT_GAP_MAX_HOURS,
        MEDIUM_GAP_MAX_HOURS
    )
    from preprocessing.gap_analyzer import GapAnalyzer

class ProspectiveImputer:
    """Prospective feature imputer enforcing zero lookahead across all partitions."""

    def __init__(
        self,
        pollutants: Optional[List[str]] = None,
        short_max: int = SHORT_GAP_MAX_HOURS,
        medium_max: int = MEDIUM_GAP_MAX_HOURS,
        train_years: Tuple[int, ...] = (2023, 2024)
    ):
        self.pollutants = pollutants if pollutants else POLLUTANTS
        self.short_max = short_max
        self.medium_max = medium_max
        self.train_years = train_years
        self.gap_analyzer = GapAnalyzer(short_max=short_max, medium_max=medium_max)
        self.diurnal_tables: Dict[str, Dict[str, Dict]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "ProspectiveImputer":
        """
        Fits diurnal lookup tables strictly on the specified training split.
        Parameters are calculated exclusively from training-partition observations.
        """
        df = df.copy()
        if "datetime" not in df.columns or "station" not in df.columns:
            raise KeyError("DataFrame must contain 'datetime' and 'station' columns.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        train_mask = df["datetime"].dt.year.isin(self.train_years)
        train_df = df[train_mask]
        
        if len(train_df) == 0:
            raise ValueError(f"No training data found for years {self.train_years}")
            
        stations = sorted(train_df["station"].unique())
        self.diurnal_tables = {}
        
        for st in stations:
            st_train = train_df[train_df["station"] == st]
            self.diurnal_tables[st] = {}
            
            for p in self.pollutants:
                if p not in st_train.columns:
                    continue
                    
                series = st_train[p]
                # Diurnal monthly and hourly group means
                mh_groups = st_train.groupby([st_train["datetime"].dt.month, st_train["datetime"].dt.hour])[p].mean()
                overall_mean = float(series.mean()) if series.notna().any() else 0.0
                
                # Convert to serializable dictionary format {"month_hour": mean_value}
                lookup_dict = {
                    f"{m}_{h}": float(val)
                    for (m, h), val in mh_groups.items()
                    if not np.isnan(val)
                }
                
                self.diurnal_tables[st][p] = {
                    "monthly_hourly": lookup_dict,
                    "overall_train_mean": round(overall_mean, 4)
                }
                
        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Applies prospective imputation to features:
        
        Returns:
            primary_df: Features with short gaps ffilled, medium gaps diurnal-imputed,
                        and extended outages (>24h) preserved as NaN.
            dense_df: Fully dense features where extended outages are also diurnal-imputed.
            imputation_stats_df: Detailed statistics on imputed counts per station and pollutant.
        """
        if not self.is_fitted:
            raise RuntimeError("ProspectiveImputer must be fitted before transform().")
            
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        
        # Analyze gaps across full dataset
        df_gap_annotated, _ = self.gap_analyzer.analyze_dataframe(df, self.pollutants)
        
        primary_df = df_gap_annotated.copy()
        dense_df = df_gap_annotated.copy()
        
        imputation_stats = []
        stations = sorted(df["station"].unique())
        
        for st in stations:
            st_mask = df["station"] == st
            st_diurnal = self.diurnal_tables.get(st, {})
            
            for p in self.pollutants:
                if p not in df.columns:
                    continue
                    
                # 1. Preserve raw observations and missing-value mask
                raw_series = df.loc[st_mask, p].copy()
                is_missing = raw_series.isna()
                orig_missing_count = int(is_missing.sum())
                total_hours = len(raw_series)
                
                primary_df.loc[st_mask, f"{p}_raw"] = raw_series
                primary_df.loc[st_mask, f"{p}_missing_mask"] = is_missing.astype(int)
                dense_df.loc[st_mask, f"{p}_raw"] = raw_series
                dense_df.loc[st_mask, f"{p}_missing_mask"] = is_missing.astype(int)
                
                # Gap category masks
                gap_cat = df_gap_annotated.loc[st_mask, f"{p}_gap_category"]
                short_mask = gap_cat == "short"
                medium_mask = gap_cat == "medium"
                extended_mask = gap_cat == "extended"
                
                # 2. Short gaps (<= 3h): Forward-fill strictly from past
                # ffill with limit=3 propagates only backwards-to-forwards in time
                ffilled_series = raw_series.ffill(limit=self.short_max)
                
                # Apply ffill strictly to short gaps
                primary_series = raw_series.copy()
                dense_series = raw_series.copy()
                
                primary_series[short_mask] = ffilled_series[short_mask]
                dense_series[short_mask] = ffilled_series[short_mask]
                
                # 3. Medium gaps (4 to 24h): Training-fitted diurnal lookup
                st_p_diurnal = st_diurnal.get(p, {})
                lookup = st_p_diurnal.get("monthly_hourly", {})
                overall_fallback = st_p_diurnal.get("overall_train_mean", 0.0)
                
                st_dates = df.loc[st_mask, "datetime"]
                diurnal_values = np.array([
                    lookup.get(f"{dt.month}_{dt.hour}", overall_fallback)
                    for dt in st_dates
                ])
                
                primary_series[medium_mask] = diurnal_values[medium_mask]
                dense_series[medium_mask] = diurnal_values[medium_mask]
                
                # 4. Extended outages (> 24h):
                # Primary dataset retains NaN
                # Dense dataset uses diurnal lookup
                dense_series[extended_mask] = diurnal_values[extended_mask]
                
                # 5. Imputation indicator flag:
                # 0 = Raw valid observation
                # 1 = Short gap forward-filled (<= 3h)
                # 2 = Medium gap diurnal-imputed (4-24h)
                # 3 = Extended outage (> 24h)
                imputed_flag = pd.Series(0, index=raw_series.index, dtype=int)
                imputed_flag[short_mask] = 1
                imputed_flag[medium_mask] = 2
                imputed_flag[extended_mask] = 3
                
                primary_df.loc[st_mask, p] = primary_series
                primary_df.loc[st_mask, f"{p}_imputed_flag"] = imputed_flag
                
                dense_df.loc[st_mask, p] = dense_series
                dense_df.loc[st_mask, f"{p}_imputed_flag"] = imputed_flag
                
                imputation_stats.append({
                    "station": st,
                    "pollutant": p,
                    "total_hours": total_hours,
                    "original_missing": orig_missing_count,
                    "short_gaps_ffill": int(short_mask.sum()),
                    "medium_gaps_diurnal": int(medium_mask.sum()),
                    "extended_outages_gt24h": int(extended_mask.sum()),
                    "primary_final_nan": int(primary_series.isna().sum()),
                    "dense_final_nan": int(dense_series.isna().sum()),
                    "train_overall_mean": round(overall_fallback, 2)
                })
                
        stats_df = pd.DataFrame(imputation_stats)
        return primary_df, dense_df, stats_df

    def save_lookup_tables(self, file_path: Union[str, Path]) -> None:
        """Saves fitted diurnal tables to JSON."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.diurnal_tables, f, indent=2)

    def load_lookup_tables(self, file_path: Union[str, Path]) -> None:
        """Loads fitted diurnal tables from JSON."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Lookup file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            self.diurnal_tables = json.load(f)
        self.is_fitted = True
