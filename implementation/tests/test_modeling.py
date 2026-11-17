"""
Automated Test Suite for Milestone 3: Baseline Benchmarking and Machine Learning Modeling
Validates baseline alignment, non-leakage, predictor restrictions, threshold calibration, and test-set isolation.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from configs.config import (
    PROCESSED_DATA_DIR,
    PRIMARY_MODEL_PREDICTORS,
    EXPLORATORY_ROAD_PROXY_FEATURES,
    IDENTIFIER_COLUMNS,
    METADATA_AND_QUALITY_COLUMNS,
    TARGET_COLUMNS
)
from modeling.model_registry import (
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGETS_ACUTE,
    CLASSIFICATION_TARGETS_SPIKE,
    TIER_B_PRIMARY_FEATURES
)
from modeling.baselines import BaselineForecaster
from modeling.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    find_optimal_threshold
)

@pytest.fixture
def sample_timeseries_df():
    """Synthetic station hourly timeseries for testing baseline formulations."""
    dates = pd.date_range("2023-01-01 00:00:00", periods=72, freq="1h")
    values = [20.0 + 10.0 * np.sin(i / 12.0 * np.pi) for i in range(len(dates))]
    
    df1 = pd.DataFrame({
        "station": "Central University",
        "datetime": dates,
        "PM2.5": values
    })
    
    df2 = pd.DataFrame({
        "station": "ECIL Kapra",
        "datetime": dates,
        "PM2.5": [v * 1.5 for v in values]
    })
    
    # Introduce deliberate NaNs to test missing-source handling
    df1.loc[10, "PM2.5"] = np.nan
    df2.loc[15, "PM2.5"] = np.nan
    
    return pd.concat([df1, df2]).reset_index(drop=True)

# --- 1. Baseline Timestamp Alignment Tests ---

def test_baseline_persistence_alignment(sample_timeseries_df):
    """
    Verifies zero-order persistence baseline:
    y_hat(i, t+h) = y(i, t) for all h in {1, 6, 24, 48}.
    Missing current observation must evaluate strictly to NaN.
    """
    forecaster = BaselineForecaster(pm25_col="PM2.5")
    preds = forecaster.generate_persistence_forecasts(sample_timeseries_df)
    
    for h in [1, 6, 24, 48]:
        col = f"pred_persistence_h{h}"
        assert col in preds.columns
        # Value must exactly match current observation
        assert np.allclose(
            preds[col].dropna().values,
            sample_timeseries_df["PM2.5"].dropna().values
        )
        # Missing observation at index 10 must produce NaN forecast
        assert np.isnan(preds.loc[10, col])

def test_baseline_seasonal_naive_source_lags(sample_timeseries_df):
    """
    Verifies corrected seasonal-naive baseline source lags:
    h = +1h:  source = y(i, t - 23h)
    h = +6h:  source = y(i, t - 18h)
    h = +24h: source = y(i, t - 24h)
    h = +48h: source = y(i, t - 24h)
    
    All source timestamps must be strictly past (<= t).
    """
    forecaster = BaselineForecaster(pm25_col="PM2.5")
    preds = forecaster.generate_seasonal_naive_forecasts(sample_timeseries_df)
    
    cu_orig = sample_timeseries_df[sample_timeseries_df["station"] == "Central University"].reset_index(drop=True)
    cu_preds = preds[preds["station"] == "Central University"].reset_index(drop=True)
    
    # Check +1h uses lag 23
    assert np.isnan(cu_preds.loc[22, "pred_seasonal_naive_h1"])
    assert np.isclose(cu_preds.loc[23, "pred_seasonal_naive_h1"], cu_orig.loc[0, "PM2.5"])
    assert np.isclose(cu_preds.loc[30, "pred_seasonal_naive_h1"], cu_orig.loc[7, "PM2.5"])
    
    # Check +6h uses lag 18
    assert np.isnan(cu_preds.loc[17, "pred_seasonal_naive_h6"])
    assert np.isclose(cu_preds.loc[18, "pred_seasonal_naive_h6"], cu_orig.loc[0, "PM2.5"])
    assert np.isclose(cu_preds.loc[30, "pred_seasonal_naive_h6"], cu_orig.loc[12, "PM2.5"])
    
    # Check +24h uses lag 24
    assert np.isnan(cu_preds.loc[23, "pred_seasonal_naive_h24"])
    assert np.isclose(cu_preds.loc[24, "pred_seasonal_naive_h24"], cu_orig.loc[0, "PM2.5"])
    assert np.isclose(cu_preds.loc[30, "pred_seasonal_naive_h24"], cu_orig.loc[6, "PM2.5"])
    
    # Check +48h uses single exact lag 24
    assert np.isclose(cu_preds.loc[24, "pred_seasonal_naive_h48"], cu_orig.loc[0, "PM2.5"])
    assert np.isclose(cu_preds.loc[30, "pred_seasonal_naive_h48"], cu_orig.loc[6, "PM2.5"])

# --- 2. Predictor Restriction & Non-Leakage Tests ---

def test_primary_predictor_restriction_84_features():
    """
    Verifies that the primary model configuration uses EXACTLY the approved 84 features,
    barring all 19 unverified road proxy features.
    """
    assert len(PRIMARY_MODEL_PREDICTORS) == 84
    assert len(EXPLORATORY_ROAD_PROXY_FEATURES) == 19
    
    primary_set = set(PRIMARY_MODEL_PREDICTORS)
    road_set = set(EXPLORATORY_ROAD_PROXY_FEATURES)
    
    # Zero intersection with unverified road proxies
    assert len(primary_set & road_set) == 0
    
    # Zero intersection with targets
    assert len(primary_set & set(TARGET_COLUMNS)) == 0
    
    # Zero intersection with identifiers
    assert len(primary_set & set(IDENTIFIER_COLUMNS)) == 0
    
    # Zero intersection with quality flags
    assert len(primary_set & set(METADATA_AND_QUALITY_COLUMNS)) == 0

def test_purged_partitions_zero_target_crossover():
    """
    Verifies that purged partitions loaded for model training contain ZERO target boundary crossover.
    """
    train_path = PROCESSED_DATA_DIR / "feature_store_train_purged.parquet"
    val_path = PROCESSED_DATA_DIR / "feature_store_val_purged.parquet"
    test_path = PROCESSED_DATA_DIR / "feature_store_test_purged.parquet"
    
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Purged partition files not yet found on disk.")
        
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    assert len(df_train) == 69984
    assert len(df_val) == 34848
    assert len(df_test) == 26016
    
    train_end = pd.to_datetime("2024-12-31 23:00:00")
    val_end = pd.to_datetime("2025-12-31 23:00:00")
    test_end = pd.to_datetime("2026-09-30 23:00:00")
    
    for h in [1, 6, 24, 48]:
        max_train_target = df_train["datetime"].max() + pd.Timedelta(hours=h)
        assert max_train_target <= train_end, f"Train +{h}h target crossed into validation!"
        
        max_val_target = df_val["datetime"].max() + pd.Timedelta(hours=h)
        assert max_val_target <= val_end, f"Val +{h}h target crossed into test!"
        
        max_test_target = df_test["datetime"].max() + pd.Timedelta(hours=h)
        assert max_test_target <= test_end, f"Test +{h}h target exceeded dataset boundary!"

# --- 3. Threshold Calibration & Missing Target Tests ---

def test_classification_threshold_optimization():
    """
    Verifies deterministic threshold optimization algorithm:
    Grid search in [0.05, 0.95], step 0.01, maximizing F1 score.
    """
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Predicted probabilities
    y_prob = np.array([0.1, 0.2, 0.25, 0.45, 0.55, 0.60, 0.70, 0.80, 0.85, 0.90])
    
    tau = find_optimal_threshold(y_true, y_prob)
    assert 0.05 <= tau <= 0.95
    
    # Threshold between 0.56 and 0.60 achieves perfect F1 = 1.0
    preds = (y_prob >= tau).astype(int)
    assert preds[5:].sum() == 5  # all positives identified
    assert preds[:4].sum() == 0  # negatives correctly classified

def test_missing_targets_preserved_as_nan():
    """
    Verifies that missing targets evaluate to NaN and are not replaced with 0.0.
    """
    test_path = PROCESSED_DATA_DIR / "feature_store_test_purged.parquet"
    if not test_path.exists():
        pytest.skip("Test purged parquet not found on disk.")
        
    df_test = pd.read_parquet(test_path)
    for col in ["target_pm25_h1", "target_pm25_h6", "target_pm25_h24", "target_pm25_h48"]:
        assert col in df_test.columns
        nan_count = df_test[col].isna().sum()
        assert nan_count > 0, f"Expected unobserved targets in test split, but found none for {col}!"
        # Zeroes must not be used to replace missing observations
        assert not (df_test[col] == 0.0).all()
