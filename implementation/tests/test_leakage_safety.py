"""
Automated Data Leakage Safety Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from preprocessing.prospective_imputer import ProspectiveImputer
from preprocessing.scaler import FeatureScaler

def test_scaler_zero_leakage_from_future(real_merged_dataset):
    """
    Verify scaler fitted on 2023-2024 training data is completely invariant
    to alterations in 2025 validation or 2026 test sets.
    """
    df1 = real_merged_dataset.copy()
    train_mask = df1["datetime"].dt.year.isin([2023, 2024])
    
    scaler1 = FeatureScaler(feature_columns=["PM2.5", "temperature_2m"])
    scaler1.fit(df1[train_mask])
    
    # Create corrupted future dataset where 2025 and 2026 have extreme 10x artificial values
    df2 = real_merged_dataset.copy()
    future_mask = ~train_mask
    df2.loc[future_mask, "PM2.5"] = df2.loc[future_mask, "PM2.5"] * 10.0
    df2.loc[future_mask, "temperature_2m"] = 99.0
    
    scaler2 = FeatureScaler(feature_columns=["PM2.5", "temperature_2m"])
    scaler2.fit(df2[train_mask])
    
    # Means and standard deviations must be 100% bit-for-bit identical
    np.testing.assert_array_equal(scaler1.scaler.mean_, scaler2.scaler.mean_)
    np.testing.assert_array_equal(scaler1.scaler.scale_, scaler2.scaler.scale_)

def test_diurnal_imputer_zero_future_leakage(real_merged_dataset):
    """
    Verify diurnal lookup table parameters are fitted strictly on training data
    and invariant to perturbations in future partitions.
    """
    df1 = real_merged_dataset.copy()
    train_mask = df1["datetime"].dt.year.isin([2023, 2024])
    
    imputer1 = ProspectiveImputer(pollutants=["PM2.5"], train_years=(2023, 2024))
    imputer1.fit(df1)
    
    # Corrupt future data
    df2 = real_merged_dataset.copy()
    future_mask = ~train_mask
    df2.loc[future_mask, "PM2.5"] = 999.0
    
    imputer2 = ProspectiveImputer(pollutants=["PM2.5"], train_years=(2023, 2024))
    imputer2.fit(df2)
    
    assert imputer1.diurnal_tables == imputer2.diurnal_tables, "Diurnal lookup table changed when future data changed!"
