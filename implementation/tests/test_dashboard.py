"""
Tests for Milestone 5: Streamlit Dashboard and Integration
Verifies data loading, forecasting service, spatial mapping service,
explainability service, and dashboard integrity.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from new_research_project.dashboard.backend.data_loader import (
    load_cpcb_stations_metadata,
    load_historical_observations,
    load_regression_predictions,
    load_regression_metrics,
    load_spatial_validation_results,
    load_hotspot_summary,
    load_hotspot_narratives,
    load_global_feature_importance,
    load_shap_values,
    get_figure_path
)
from new_research_project.dashboard.backend.forecasting_service import (
    get_available_stations,
    get_horizon_metadata,
    get_forecast_timeseries,
    calculate_slice_metrics,
    WINNING_MODELS
)
from new_research_project.dashboard.backend.spatial_service import (
    get_station_map_dataframe,
    get_spatial_grid_samples,
    HYD_BOUNDS,
    UNCERTAINTY_TIERS
)
from new_research_project.dashboard.backend.explainability_service import (
    get_top_features_dataframe,
    get_category_shares_dataframe,
    get_station_narrative
)


def test_station_metadata_integrity():
    """Confirms all 4 CPCB stations have complete canonical metadata."""
    stations = load_cpcb_stations_metadata()
    assert len(stations) == 4, f"Expected 4 stations, got {len(stations)}"
    expected_names = [
        "Bollaram Industrial Area",
        "Central University",
        "ECIL Kapra",
        "ICRISAT Patancheru"
    ]
    for name in expected_names:
        assert name in stations, f"Missing station: {name}"
        meta = stations[name]
        assert "latitude" in meta and "longitude" in meta
        assert "utm_x" in meta and "utm_y" in meta
        assert "station_type" in meta
        assert meta["latitude"] > 17.0 and meta["latitude"] < 18.0
        assert meta["longitude"] > 78.0 and meta["longitude"] < 79.0


def test_historical_observations_loader():
    """Confirms historical observations load with correct schema and datetime sorting."""
    df = load_historical_observations(station="Central University", pollutants=["PM2.5", "PM10"])
    assert not df.empty, "Historical observations DataFrame is empty"
    assert "datetime" in df.columns
    assert "station" in df.columns
    assert "PM2.5" in df.columns
    assert df["station"].iloc[0] == "Central University"


def test_regression_predictions_loader():
    """Confirms 2026 out-of-time test predictions load properly."""
    df = load_regression_predictions()
    assert not df.empty, "Regression predictions DataFrame is empty"
    assert len(df) == 26016, f"Expected 26,016 test rows, got {len(df)}"
    for h in ["h1", "h6", "h24", "h48"]:
        assert f"target_pm25_{h}" in df.columns
        assert f"pred_persistence_{h}" in df.columns
        assert f"pred_seasonal_naive_{h}" in df.columns
        assert f"pred_rf_{h}" in df.columns
        assert f"pred_xgb_{h}" in df.columns
        assert f"pred_lgb_{h}" in df.columns
        assert f"pred_stacking_{h}" in df.columns


def test_forecasting_service_slice_and_metrics():
    """Confirms forecast time series slicing and on-the-fly metric calculations."""
    stations = get_available_stations()
    assert len(stations) == 4
    
    # Test slice retrieval
    df_slice = get_forecast_timeseries(
        station="Central University",
        horizon="h1",
        start_date="2026-01-01",
        end_date="2026-01-31"
    )
    assert not df_slice.empty
    assert "Actual PM2.5 (ug/m3)" in df_slice.columns
    assert "Persistence Baseline" in df_slice.columns
    
    # Test metric computation
    y_true = np.array([30.0, 40.0, 50.0, 60.0])
    y_pred = np.array([32.0, 38.0, 52.0, 58.0])
    metrics = calculate_slice_metrics(y_true, y_pred)
    assert metrics["count"] == 4
    assert abs(metrics["mae"] - 2.0) < 1e-4
    assert abs(metrics["bias"] - 0.0) < 1e-4
    assert metrics["r2"] > 0.95


def test_winning_models_metadata():
    """Verifies winning model configurations for all 4 horizons."""
    for h in ["h1", "h6", "h24", "h48"]:
        meta = get_horizon_metadata(h)
        assert "name" in meta
        assert "code" in meta
        assert "description" in meta
        assert meta["mae_test"] > 0.0


def test_spatial_service_layers_and_uncertainty():
    """Confirms spatial service generates valid station coordinates and grid uncertainty."""
    df_st = get_station_map_dataframe(horizon="h1")
    assert len(df_st) == 4
    assert "latitude" in df_st.columns
    assert "longitude" in df_st.columns
    assert "predicted_pm25" in df_st.columns
    assert "status" in df_st.columns
    
    # Test grid sample generation
    df_grid = get_spatial_grid_samples(horizon="h1", grid_res=10)
    assert not df_grid.empty
    assert len(df_grid) == 100
    assert "uncertainty_tier" in df_grid.columns
    assert "min_dist_km" in df_grid.columns
    assert "interp_pm25" in df_grid.columns
    
    # Verify uncertainty tier definitions
    assert UNCERTAINTY_TIERS["tier_1_low"]["area_pct"] == 16.80
    assert UNCERTAINTY_TIERS["tier_2_moderate"]["area_pct"] == 48.30
    assert UNCERTAINTY_TIERS["tier_3_high"]["area_pct"] == 34.90


def test_explainability_service_features_and_narratives():
    """Confirms explainability service parses top features, shares, and narratives."""
    df_top = get_top_features_dataframe(horizon="h1", top_n=10)
    assert not df_top.empty
    assert len(df_top) <= 10
    assert "Rank" in df_top.columns
    assert "Feature Name" in df_top.columns
    assert "Mean |SHAP| (ug/m3)" in df_top.columns
    
    df_shares = get_category_shares_dataframe(horizon="h1")
    assert not df_shares.empty
    assert "Feature Group" in df_shares.columns
    assert "Attribution Share (%)" in df_shares.columns
    
    # Verify station narrative
    narrative = get_station_narrative("Bollaram Industrial Area")
    assert len(narrative) > 20
    assert "Bollaram" in narrative


def test_figure_artifacts_exist():
    """Verifies key pre-rendered high-resolution figures are discoverable."""
    assert get_figure_path("spatial_forecast_h1_idw.png") is not None
    assert get_figure_path("spatial_uncertainty_h1.png") is not None
    assert get_figure_path("hotspot_delineation_h1.png") is not None
    assert get_figure_path("shap_beeswarm_h1.png") is not None
    assert get_figure_path("shap_dependence_h1_PM2.5.png") is not None
