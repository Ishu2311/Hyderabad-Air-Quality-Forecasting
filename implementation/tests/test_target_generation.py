"""
Automated Target Generation Tests
Milestone 1: Data Foundation and Preprocessing
"""

import pytest
import pandas as pd
import numpy as np

from target_generation.target_builder import TargetBuilder

def test_multi_horizon_shift_alignment():
    """Verify regression targets strictly align with forward shifts t+h per station."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=60, freq="h")
    df = pd.DataFrame({
        "datetime": dates,
        "station": "Bollaram Industrial Area",
        "PM2.5": range(60)
    })
    
    builder = TargetBuilder(regression_horizons=[1, 6, 24, 48])
    df_targets = builder.build_targets(df, pm25_col="PM2.5")
    
    # Check t+1
    assert df_targets.loc[0, "target_pm25_h1"] == 1
    assert df_targets.loc[10, "target_pm25_h1"] == 11
    assert np.isnan(df_targets.loc[59, "target_pm25_h1"])
    
    # Check t+6
    assert df_targets.loc[0, "target_pm25_h6"] == 6
    assert df_targets.loc[10, "target_pm25_h6"] == 16
    assert np.isnan(df_targets.loc[54, "target_pm25_h6"])
    
    # Check t+24
    assert df_targets.loc[0, "target_pm25_h24"] == 24
    assert df_targets.loc[10, "target_pm25_h24"] == 34
    
    # Check t+48
    assert df_targets.loc[0, "target_pm25_h48"] == 48
    assert np.isnan(df_targets.loc[12, "target_pm25_h48"])

def test_missing_target_preservation_nan():
    """Verify that if future reading is NaN, regression and classification targets are NaN, never 0."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=10, freq="h")
    vals = [50.0] * 10
    vals[5] = np.nan  # Future observation at index 5 is unobserved
    
    df = pd.DataFrame({
        "datetime": dates,
        "station": "Central University",
        "PM2.5": vals
    })
    
    builder = TargetBuilder(regression_horizons=[1], classification_horizons=[1])
    df_targets = builder.build_targets(df, pm25_col="PM2.5")
    
    # At index 4, future target t+1 is index 5 (which is NaN)
    assert np.isnan(df_targets.loc[4, "target_pm25_h1"]), "Missing future observation must yield NaN target"
    assert np.isnan(df_targets.loc[4, "target_hotspot_acute_h1"]), "Missing future observation must yield NaN acute target"
    assert np.isnan(df_targets.loc[4, "target_hotspot_spike_h1"]), "Missing future observation must yield NaN spike target"
    
    # Verify non-zero
    assert df_targets.loc[4, "target_pm25_h1"] != 0.0
    assert df_targets.loc[4, "target_hotspot_acute_h1"] != 0.0

def test_forward_24h_mean_completeness_and_denominator():
    """
    Verify forward 24-hour mean:
    1. Requires >= 16 valid observations in forward window.
    2. Denominator is N_valid, NOT 24.
    3. If N_valid < 16, evaluates to NaN.
    """
    dates = pd.date_range("2023-01-01 00:00:00", periods=30, freq="h")
    # Construct window from index 1 to 24 with exactly 16 valid values of 30.0, and 8 NaNs
    vals = [np.nan] * 30
    # At index 0, forward window is 1..24
    for i in range(1, 17):
        vals[i] = 30.0  # 16 values of 30.0
    # indices 17..24 are NaN (8 NaNs)
    
    df = pd.DataFrame({
        "datetime": dates,
        "station": "ECIL Kapra",
        "PM2.5": vals
    })
    
    builder = TargetBuilder()
    df_targets = builder.build_targets(df, pm25_col="PM2.5")
    
    # For index 0: exactly 16 valid observations with sum = 16 * 30 = 480
    # Mean must be 480 / 16 = 30.0! (If denominator were 24, it would be 480/24 = 20.0)
    mean_val = df_targets.loc[0, "target_pm25_24h_mean"]
    valid_count = df_targets.loc[0, "target_pm25_24h_valid_count"]
    
    assert valid_count == 16, f"Expected valid count 16, got {valid_count}"
    assert np.isclose(mean_val, 30.0), f"Expected mean 30.0 (divided by 16), got {mean_val}"
    
    # Now test with 15 valid observations (one fewer)
    vals[16] = np.nan  # Now only 15 valid
    df2 = pd.DataFrame({
        "datetime": dates,
        "station": "ECIL Kapra",
        "PM2.5": vals
    })
    df_targets2 = builder.build_targets(df2, pm25_col="PM2.5")
    assert np.isnan(df_targets2.loc[0, "target_pm25_24h_mean"]), "Target must evaluate to NaN when N_valid < 16"
    assert np.isnan(df_targets2.loc[0, "target_hotspot_24h_statutory"]), "Hotspot target must evaluate to NaN when N_valid < 16"

def test_threshold_boundaries():
    """
    Verify exact threshold boundary behaviors:
    - 60.0 ug/m3 acute excursion: < 60 is 0, >= 60 is 1
    - 120.0 ug/m3 exploratory severe spike (approved >= 120): < 120 is 0, >= 120 is 1
    - 120.0 ug/m3 Very Poor entry (> 120): <= 120 is 0, > 120 is 1
    """
    dates = pd.date_range("2023-01-01 00:00:00", periods=10, freq="h")
    # Test values placed at index 1..6 so that index 0..5 targets hit each test case
    test_values = [0.0, 59.9, 60.0, 119.9, 120.0, 120.1, 200.0, 0.0, 0.0, 0.0]
    df = pd.DataFrame({
        "datetime": dates,
        "station": "ICRISAT Patancheru",
        "PM2.5": test_values
    })
    
    builder = TargetBuilder(regression_horizons=[1], classification_horizons=[1])
    df_targets = builder.build_targets(df, pm25_col="PM2.5")
    
    # At index 0, target is value at index 1 (59.9)
    assert df_targets.loc[0, "target_hotspot_acute_h1"] == 0.0
    assert df_targets.loc[0, "target_hotspot_spike_h1"] == 0.0
    
    # At index 1, target is value at index 2 (60.0)
    assert df_targets.loc[1, "target_hotspot_acute_h1"] == 1.0  # 60.0 is acute
    assert df_targets.loc[1, "target_hotspot_spike_h1"] == 0.0  # 60.0 is not spike
    
    # At index 2, target is value at index 3 (119.9)
    assert df_targets.loc[2, "target_hotspot_acute_h1"] == 1.0
    assert df_targets.loc[2, "target_hotspot_spike_h1"] == 0.0
    
    # At index 3, target is value at index 4 (120.0)
    assert df_targets.loc[3, "target_hotspot_spike_h1"] == 1.0  # approved >= 120 definition
    assert df_targets.loc[3, "target_hotspot_very_poor_h1"] == 0.0  # strict > 120 definition
    
    # At index 4, target is value at index 5 (120.1)
    assert df_targets.loc[4, "target_hotspot_spike_h1"] == 1.0
    assert df_targets.loc[4, "target_hotspot_very_poor_h1"] == 1.0  # > 120 is 1.0
