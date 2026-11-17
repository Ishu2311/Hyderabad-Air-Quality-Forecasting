"""
Automated Reproducibility Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from data_ingestion.dataset_loader import DatasetLoader
from target_generation.target_builder import TargetBuilder

def test_pipeline_reproducibility(real_merged_dataset):
    """Verify target generation and preprocessing produce deterministic outputs on re-runs."""
    builder = TargetBuilder()
    
    targets1 = builder.build_targets(real_merged_dataset, pm25_col="PM2.5")
    targets2 = builder.build_targets(real_merged_dataset, pm25_col="PM2.5")
    
    target_cols = [c for c in targets1.columns if c.startswith("target_")]
    for col in target_cols:
        # Check equality including NaNs
        s1 = targets1[col]
        s2 = targets2[col]
        both_nan = s1.isna() & s2.isna()
        both_equal = (s1 == s2) | both_nan
        assert both_equal.all(), f"Target column {col} is non-deterministic!"
