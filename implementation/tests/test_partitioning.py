"""
Automated Chronological Partitioning and Boundary Leakage Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from configs.config import (
    EXPECTED_TRAIN_HOURS,
    EXPECTED_VAL_HOURS,
    EXPECTED_TEST_HOURS,
    EXPECTED_TOTAL_STATION_HOURS,
    MAX_PREDICTION_HORIZON_HOURS
)
from validation.chronological_split import ChronologicalSplitter

def test_partition_dimensions_and_chronology(real_merged_dataset):
    """Verify chronological split counts match research specs with zero temporal overlap."""
    splitter = ChronologicalSplitter()
    df_train, df_val, df_test = splitter.split(real_merged_dataset, purge_boundary_buffer=False)
    
    assert len(df_train) == EXPECTED_TRAIN_HOURS, f"Train hours {len(df_train)} != {EXPECTED_TRAIN_HOURS}"
    assert len(df_val) == EXPECTED_VAL_HOURS, f"Val hours {len(df_val)} != {EXPECTED_VAL_HOURS}"
    assert len(df_test) == EXPECTED_TEST_HOURS, f"Test hours {len(df_test)} != {EXPECTED_TEST_HOURS}"
    assert len(df_train) + len(df_val) + len(df_test) == EXPECTED_TOTAL_STATION_HOURS
    
    # Verify strict non-overlapping temporal order
    assert df_train["datetime"].max() < df_val["datetime"].min(), "Train overlaps into Validation"
    assert df_val["datetime"].max() < df_test["datetime"].min(), "Validation overlaps into Test"
    
    # Verify exact date bounds
    assert str(df_train["datetime"].min()) == "2023-01-01 00:00:00"
    assert str(df_train["datetime"].max()) == "2024-12-31 23:00:00"
    assert str(df_val["datetime"].min()) == "2025-01-01 00:00:00"
    assert str(df_val["datetime"].max()) == "2025-12-31 23:00:00"
    assert str(df_test["datetime"].min()) == "2026-01-01 00:00:00"
    assert str(df_test["datetime"].max()) == "2026-09-30 23:00:00"

def test_boundary_buffer_leakage_guard(real_merged_dataset):
    """
    Verify boundary buffer identifies the last 48 hours of partitions
    to guard against target window leakage.
    """
    splitter = ChronologicalSplitter(buffer_hours=48)
    df_annotated = splitter.add_partition_metadata(real_merged_dataset)
    
    # Check that is_boundary_buffer flags exactly 48 hours per station in train (48 * 4 = 192)
    train_buffer = df_annotated[(df_annotated["partition"] == "train") & df_annotated["is_boundary_buffer"]]
    assert len(train_buffer) == 48 * 4, f"Expected 192 train buffer rows, got {len(train_buffer)}"
    
    val_buffer = df_annotated[(df_annotated["partition"] == "val") & df_annotated["is_boundary_buffer"]]
    assert len(val_buffer) == 48 * 4, f"Expected 192 val buffer rows, got {len(val_buffer)}"
    
    # Test purged splits
    df_train_clean, df_val_clean, df_test_clean = splitter.split(real_merged_dataset, purge_boundary_buffer=True)
    assert len(df_train_clean) == EXPECTED_TRAIN_HOURS - (48 * 4)
    assert len(df_val_clean) == EXPECTED_VAL_HOURS - (48 * 4)
