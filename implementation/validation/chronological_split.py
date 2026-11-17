"""
Chronological Partitioning and Partition Leakage Guard Module
Milestone 1: Data Foundation and Preprocessing

Implements strict chronological data partitioning:
- Train: 2023-01-01 00:00:00 to 2024-12-31 23:00:00 (70,176 station-hours)
- Validation: 2025-01-01 00:00:00 to 2025-12-31 23:00:00 (35,040 station-hours)
- Test: 2026-01-01 00:00:00 to 2026-09-30 23:00:00 (26,208 station-hours)
- Enforces zero random shuffling.
- Purges/buffers boundary transition windows (max horizon H=48h) to prevent future
  target windows from introducing cross-partition data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        TRAIN_START,
        TRAIN_END,
        VAL_START,
        VAL_END,
        TEST_START,
        TEST_END,
        EXPECTED_TRAIN_HOURS,
        EXPECTED_VAL_HOURS,
        EXPECTED_TEST_HOURS,
        EXPECTED_TOTAL_STATION_HOURS,
        MAX_PREDICTION_HORIZON_HOURS,
        PROCESSED_DATA_DIR
    )
except (ImportError, ValueError):
    from configs.config import (
        TRAIN_START,
        TRAIN_END,
        VAL_START,
        VAL_END,
        TEST_START,
        TEST_END,
        EXPECTED_TRAIN_HOURS,
        EXPECTED_VAL_HOURS,
        EXPECTED_TEST_HOURS,
        EXPECTED_TOTAL_STATION_HOURS,
        MAX_PREDICTION_HORIZON_HOURS,
        PROCESSED_DATA_DIR
    )

class ChronologicalSplitter:
    """Partitions time series chronologically and prevents target window boundary leakage."""

    def __init__(
        self,
        train_start: str = TRAIN_START,
        train_end: str = TRAIN_END,
        val_start: str = VAL_START,
        val_end: str = VAL_END,
        test_start: str = TEST_START,
        test_end: str = TEST_END,
        buffer_hours: int = MAX_PREDICTION_HORIZON_HOURS
    ):
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.val_start = pd.to_datetime(val_start)
        self.val_end = pd.to_datetime(val_end)
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)
        self.buffer_hours = buffer_hours

    def add_partition_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns 'partition' label ('train', 'val', 'test') and 'is_boundary_buffer' flag.
        
        Boundary buffer logic:
        For horizon H_max = 48h, the last 48 hours of Train have target timestamps in Val.
        Flagging 'is_boundary_buffer = True' allows researchers to exclude these overlapping
        transition hours during evaluation.
        """
        df = df.copy()
        if "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'datetime' column.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["partition"] = "unassigned"
        df["is_boundary_buffer"] = False
        
        # Partition masks
        train_mask = (df["datetime"] >= self.train_start) & (df["datetime"] <= self.train_end)
        val_mask = (df["datetime"] >= self.val_start) & (df["datetime"] <= self.val_end)
        test_mask = (df["datetime"] >= self.test_start) & (df["datetime"] <= self.test_end)
        
        df.loc[train_mask, "partition"] = "train"
        df.loc[val_mask, "partition"] = "val"
        df.loc[test_mask, "partition"] = "test"
        
        # Boundary buffer masks: last buffer_hours of each partition
        train_buffer_start = self.train_end - pd.Timedelta(hours=self.buffer_hours - 1)
        val_buffer_start = self.val_end - pd.Timedelta(hours=self.buffer_hours - 1)
        test_buffer_start = self.test_end - pd.Timedelta(hours=self.buffer_hours - 1)
        
        train_buffer_mask = (df["datetime"] >= train_buffer_start) & (df["datetime"] <= self.train_end)
        val_buffer_mask = (df["datetime"] >= val_buffer_start) & (df["datetime"] <= self.val_end)
        test_buffer_mask = (df["datetime"] >= test_buffer_start) & (df["datetime"] <= self.test_end)
        
        df.loc[train_buffer_mask | val_buffer_mask | test_buffer_mask, "is_boundary_buffer"] = True
        return df

    def split(
        self,
        df: pd.DataFrame,
        purge_boundary_buffer: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits DataFrame into Train, Validation, and Test sets.
        
        Args:
            df: DataFrame to split.
            purge_boundary_buffer: If True, excludes boundary buffer rows so future targets
                                  never cross partition thresholds.
                                  
        Returns:
            df_train, df_val, df_test
        """
        df_annotated = self.add_partition_metadata(df)
        
        df_train = df_annotated[df_annotated["partition"] == "train"].copy()
        df_val = df_annotated[df_annotated["partition"] == "val"].copy()
        df_test = df_annotated[df_annotated["partition"] == "test"].copy()
        
        if purge_boundary_buffer:
            df_train = df_train[~df_train["is_boundary_buffer"]].copy()
            df_val = df_val[~df_val["is_boundary_buffer"]].copy()
            df_test = df_test[~df_test["is_boundary_buffer"]].copy()
            
        return df_train, df_val, df_test

    def save_splits(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        output_dir: Optional[Union[str, Path]] = None,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """Saves partitioned CSV files."""
        out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        
        p = f"{prefix}_" if prefix else ""
        train_path = out_dir / f"{p}train_data.csv"
        val_path = out_dir / f"{p}val_data.csv"
        test_path = out_dir / f"{p}test_data.csv"
        
        df_train.to_csv(train_path, index=False)
        df_val.to_csv(val_path, index=False)
        df_test.to_csv(test_path, index=False)
        
        summary = {
            "train": {
                "rows": len(df_train),
                "start": str(df_train["datetime"].min()),
                "end": str(df_train["datetime"].max()),
                "path": str(train_path)
            },
            "validation": {
                "rows": len(df_val),
                "start": str(df_val["datetime"].min()),
                "end": str(df_val["datetime"].max()),
                "path": str(val_path)
            },
            "test": {
                "rows": len(df_test),
                "start": str(df_test["datetime"].min()),
                "end": str(df_test["datetime"].max()),
                "path": str(test_path)
            }
        }
        
        summary_path = out_dir / f"{p}partition_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        return {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "summary": summary_path
        }
