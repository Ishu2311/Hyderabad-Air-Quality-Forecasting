"""
Test Suite for SHAP Explainability and Narrative Modules
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability
"""

import numpy as np
import pytest

from new_research_project.implementation.configs.config import PRIMARY_MODEL_PREDICTORS
from new_research_project.implementation.explainability.feature_importance import classify_feature_category
from new_research_project.implementation.explainability.hotspot_narrative import generate_human_readable_hotspot_narrative


def test_primary_predictor_set_cleanliness():
    """Confirms explainability module exclusively references the 84 approved primary predictors."""
    assert len(PRIMARY_MODEL_PREDICTORS) == 84
    forbidden = ["target", "station", "datetime", "road_length", "road_density", "highway_dist"]
    for f in PRIMARY_MODEL_PREDICTORS:
        for fb in forbidden:
            assert fb not in f.lower(), f"Forbidden substring '{fb}' found in predictor '{f}'"


def test_feature_taxonomy_classification():
    """Confirms feature categories are appropriately assigned."""
    assert classify_feature_category("pm25_lag_1h") == "Historical PM2.5 Dynamics"
    assert classify_feature_category("pm25_rolling_mean_24h") == "Historical PM2.5 Dynamics"
    assert classify_feature_category("temperature_2m") == "Thermodynamics & Boundary Layer"
    assert classify_feature_category("boundary_layer_height") == "Thermodynamics & Boundary Layer"
    assert classify_feature_category("wind_speed_10m") == "Wind & Dispersion Dynamics"
    assert classify_feature_category("is_atmospheric_stagnation") == "Wind & Dispersion Dynamics"
    assert classify_feature_category("is_evening_rush_hour") == "Temporal Congestion Windows"
    assert classify_feature_category("hour_sin") == "Cyclical Calendar & Solar Timing"


def test_hotspot_narrative_generation():
    """Confirms natural language narrative generator dynamically includes all components."""
    elev = [
        {"feature": "pm25_lag_1h", "shap_impact": 12.5, "observed_value": 75.0},
        {"feature": "is_atmospheric_stagnation", "shap_impact": 4.2, "observed_value": 1.0}
    ]
    mit = [
        {"feature": "wind_speed_10m", "shap_impact": -3.1, "observed_value": 4.5}
    ]

    narrative = generate_human_readable_hotspot_narrative(
        station_name="ECIL Kapra",
        horizon_label="6 hours",
        predicted_pm25=68.5,
        observed_baseline=58.0,
        top_elevating_factors=elev,
        top_mitigating_factors=mit,
        uncertainty_tier="Low Uncertainty (Proximate <= 5km)",
        acute_threshold=60.0
    )

    assert "ECIL Kapra" in narrative
    assert "exceeding the statutory 24-hour NAAQS benchmark" in narrative
    assert "68.5 ug/m3" in narrative
    assert "high recent persistence" in narrative
    assert "Methodological Note" in narrative
    assert "association" in narrative.lower()
    assert "Low Uncertainty" in narrative
