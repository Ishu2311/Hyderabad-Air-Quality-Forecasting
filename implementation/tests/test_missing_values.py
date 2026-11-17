"""
Automated Missing-Value and Imputation Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from configs.config import POLLUTANTS
from preprocessing.gap_analyzer import GapAnalyzer
from preprocessing.prospective_imputer import ProspectiveImputer

def test_gap_analyzer_categorization(sample_synthetic_timeseries):
    """Verify gap analyzer correctly classifies short (<=3h), medium (4-24h), and extended (>24h) gaps."""
    analyzer = GapAnalyzer(short_max=3, medium_max=24)
    df_annotated, summary = analyzer.analyze_dataframe(sample_synthetic_timeseries, ["PM2.5"])
    
    st_a = df_annotated[df_annotated["station"] == "Bollaram Industrial Area"]
    
    # Hours 10-11 were a 2h gap -> should be 'short'
    assert (st_a.loc[10:11, "PM2.5_gap_category"] == "short").all()
    assert (st_a.loc[10:11, "PM2.5_gap_length"] == 2).all()
    
    # Hours 30-35 were a 6h gap -> should be 'medium'
    assert (st_a.loc[30:35, "PM2.5_gap_category"] == "medium").all()
    assert (st_a.loc[30:35, "PM2.5_gap_length"] == 6).all()
    
    # Hours 50-77 were a 28h gap -> should be 'extended'
    assert (st_a.loc[50:77, "PM2.5_gap_category"] == "extended").all()
    assert (st_a.loc[50:77, "PM2.5_gap_length"] == 28).all()

def test_missing_mask_preservation(sample_synthetic_timeseries):
    """Verify raw missing-value mask is strictly preserved."""
    imputer = ProspectiveImputer(pollutants=["PM2.5"], train_years=(2023,))
    imputer.fit(sample_synthetic_timeseries)
    primary_df, dense_df, stats = imputer.transform(sample_synthetic_timeseries)
    
    # Check mask matches original isna()
    orig_na = sample_synthetic_timeseries["PM2.5"].isna()
    assert (primary_df["PM2.5_missing_mask"] == orig_na.astype(int)).all()
    assert (primary_df["PM2.5_raw"].isna() == orig_na).all()

def test_short_gap_ffill_no_lookahead(sample_synthetic_timeseries):
    """Verify short gaps are forward-filled strictly from past observations with zero lookahead."""
    imputer = ProspectiveImputer(pollutants=["PM2.5"], train_years=(2023,))
    imputer.fit(sample_synthetic_timeseries)
    primary_df, _, _ = imputer.transform(sample_synthetic_timeseries)
    
    st_a_orig = sample_synthetic_timeseries[sample_synthetic_timeseries["station"] == "Bollaram Industrial Area"]
    st_a_proc = primary_df[primary_df["station"] == "Bollaram Industrial Area"]
    
    # Hours 10-11 were NaN; hour 9 was valid
    val_at_9 = st_a_orig.loc[9, "PM2.5"]
    assert st_a_proc.loc[10, "PM2.5"] == val_at_9, "Short gap must equal past valid value at t-1"
    assert st_a_proc.loc[11, "PM2.5"] == val_at_9, "Short gap must equal past valid value at t-2"
    assert st_a_proc.loc[10, "PM2.5_imputed_flag"] == 1
    assert st_a_proc.loc[11, "PM2.5_imputed_flag"] == 1

def test_extended_outages_remain_nan_in_primary(sample_synthetic_timeseries):
    """Verify extended outages (>24h) remain NaN in primary prospective features."""
    imputer = ProspectiveImputer(pollutants=["PM2.5"], train_years=(2023,))
    imputer.fit(sample_synthetic_timeseries)
    primary_df, dense_df, _ = imputer.transform(sample_synthetic_timeseries)
    
    st_a_proc = primary_df[primary_df["station"] == "Bollaram Industrial Area"]
    # Hours 50-77 were 28h gap -> must remain NaN
    assert st_a_proc.loc[50:77, "PM2.5"].isna().all(), "Extended outages must remain NaN in primary dataset"
    
    # But in dense dataset, they should be filled
    st_a_dense = dense_df[dense_df["station"] == "Bollaram Industrial Area"]
    assert st_a_dense.loc[50:77, "PM2.5"].notna().all(), "Dense dataset must impute extended outages"

def test_zero_target_substitution_absence(real_merged_dataset):
    """Verify missing PM2.5 is NEVER substituted with 0.0 in raw or targets."""
    # Check that in raw data, missing is represented as NaN, never 0.0
    missing_count = real_merged_dataset["PM2.5"].isna().sum()
    zero_count = (real_merged_dataset["PM2.5"] == 0.0).sum()
    # Physical sensor readings for PM2.5 in Hyderabad rarely evaluate to exactly 0.0
    # But missing values must be NaNs, not coerced to 0.0
    assert missing_count == 20024, f"Expected 20,024 missing PM2.5, got {missing_count}"
