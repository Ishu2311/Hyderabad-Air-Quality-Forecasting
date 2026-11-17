"""
Gap Analyzer Module
Milestone 1: Data Foundation and Preprocessing

Analyzes continuous missing blocks (gaps) for each station and pollutant.
Categorizes gap durations into short (<=3h), medium (4-24h), and extended outages (>24h).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        POLLUTANTS,
        SHORT_GAP_MAX_HOURS,
        MEDIUM_GAP_MAX_HOURS
    )
except (ImportError, ValueError):
    from configs.config import (
        POLLUTANTS,
        SHORT_GAP_MAX_HOURS,
        MEDIUM_GAP_MAX_HOURS
    )

class GapAnalyzer:
    """Analyzes telemetry outages and contiguous missing-value block distributions."""

    def __init__(
        self,
        short_max: int = SHORT_GAP_MAX_HOURS,
        medium_max: int = MEDIUM_GAP_MAX_HOURS
    ):
        self.short_max = short_max
        self.medium_max = medium_max

    def analyze_station_pollutant(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, pd.Series, Dict[str, int]]:
        """
        Analyzes contiguous missing blocks for a single station-pollutant series.
        
        Returns:
            gap_lengths: pd.Series with the contiguous gap length for each row (0 if not missing).
            gap_category: pd.Series categorizing the missingness:
                          'valid' (not missing), 'short' (<=3h), 'medium' (4-24h), 'extended' (>24h).
            counts: Dict with counts of hours in each category.
        """
        is_missing = series.isna()
        
        # Group contiguous missing blocks
        # Non-missing values increment the group id
        group_ids = (~is_missing).cumsum()
        
        # Calculate size of each contiguous missing group
        gap_lengths = pd.Series(0, index=series.index, dtype=int)
        
        if is_missing.any():
            missing_groups = group_ids[is_missing]
            group_sizes = missing_groups.groupby(missing_groups).transform("count")
            gap_lengths[is_missing] = group_sizes
            
        gap_category = pd.Series("valid", index=series.index, dtype=object)
        
        short_mask = is_missing & (gap_lengths <= self.short_max)
        medium_mask = is_missing & (gap_lengths > self.short_max) & (gap_lengths <= self.medium_max)
        extended_mask = is_missing & (gap_lengths > self.medium_max)
        
        gap_category[short_mask] = "short"
        gap_category[medium_mask] = "medium"
        gap_category[extended_mask] = "extended"
        
        counts = {
            "total_hours": len(series),
            "valid_hours": int((~is_missing).sum()),
            "total_missing": int(is_missing.sum()),
            "short_gap_hours": int(short_mask.sum()),
            "medium_gap_hours": int(medium_mask.sum()),
            "extended_outage_hours": int(extended_mask.sum())
        }
        
        return gap_lengths, gap_category, counts

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        pollutants: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyzes gap distributions across all stations and pollutants in the DataFrame.
        
        Returns:
            gap_flagged_df: DataFrame augmented with gap length and category columns.
            summary_df: Summary statistics per station and pollutant.
        """
        if pollutants is None:
            pollutants = [p for p in POLLUTANTS if p in df.columns]
            
        df = df.copy()
        if "station" not in df.columns or "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'station' and 'datetime' columns.")
            
        df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        stations = sorted(df["station"].unique())
        
        summary_records = []
        
        for st in stations:
            st_mask = df["station"] == st
            for p in pollutants:
                series = df.loc[st_mask, p]
                gap_len, gap_cat, counts = self.analyze_station_pollutant(series)
                
                df.loc[st_mask, f"{p}_gap_length"] = gap_len
                df.loc[st_mask, f"{p}_gap_category"] = gap_cat
                
                summary_records.append({
                    "station": st,
                    "pollutant": p,
                    **counts,
                    "missing_pct": round(counts["total_missing"] / counts["total_hours"] * 100.0, 2),
                    "short_gap_pct": round(counts["short_gap_hours"] / counts["total_hours"] * 100.0, 2),
                    "medium_gap_pct": round(counts["medium_gap_hours"] / counts["total_hours"] * 100.0, 2),
                    "extended_outage_pct": round(counts["extended_outage_hours"] / counts["total_hours"] * 100.0, 2),
                })
                
        summary_df = pd.DataFrame(summary_records)
        return df, summary_df
