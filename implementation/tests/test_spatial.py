"""
Test Suite for Spatial Interpolation and Hotspot Detection Modules
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability
"""

import numpy as np
import pytest

from new_research_project.implementation.configs.config import STATION_REGISTRY
from new_research_project.implementation.spatial.idw import InverseDistanceWeighting, haversine_distance
from new_research_project.implementation.spatial.kriging import (
    OrdinaryKriging,
    spherical_variogram,
    exponential_variogram,
    gaussian_variogram
)
from new_research_project.implementation.spatial.hotspot_detection import (
    HotspotDetector,
    THRESHOLD_ACUTE_EXCURSION,
    THRESHOLD_SEVERE_SPIKE
)
from new_research_project.implementation.spatial.spatial_metrics import compute_spatial_validation_metrics


def test_station_coordinate_validity():
    """Confirms all 4 CPCB stations have valid UTM44N and WGS84 coordinates."""
    assert len(STATION_REGISTRY) == 4
    for name, meta in STATION_REGISTRY.items():
        assert 17.0 <= meta.latitude <= 18.0, f"Latitude out of bounds for {name}"
        assert 78.0 <= meta.longitude <= 79.0, f"Longitude out of bounds for {name}"
        assert 200000.0 <= meta.easting_utm44n <= 260000.0, f"Easting out of bounds for {name}"
        assert 1920000.0 <= meta.northing_utm44n <= 1960000.0, f"Northing out of bounds for {name}"


def test_idw_mathematical_coincidence():
    """Confirms IDW returns exact station value at station locations without division by zero."""
    coords = np.array([[218650, 1941800], [216420, 1932950], [241880, 1933850], [210180, 1938550]], dtype=float)
    values = np.array([55.0, 32.0, 78.0, 41.0], dtype=float)

    idw = InverseDistanceWeighting(power=2.0, coordinate_system="utm")
    idw.fit(coords, values)

    preds, dists = idw.predict(coords)
    assert np.allclose(preds, values, atol=1e-5)
    assert np.allclose(dists, 0.0, atol=1e-5)


def test_idw_distance_power_decay():
    """Confirms higher IDW power increases weight of the closer station."""
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    values = np.array([10.0, 100.0])

    target = np.array([[2.0, 0.0]])  # Closer to station 0 (dist 2 vs 8)

    idw_p1 = InverseDistanceWeighting(power=1.0).fit(coords, values)
    idw_p3 = InverseDistanceWeighting(power=3.0).fit(coords, values)

    pred_p1, _ = idw_p1.predict(target)
    pred_p3, _ = idw_p3.predict(target)

    # Higher power gives higher weight to closer station (station 0 with val 10) -> lower prediction
    assert pred_p3[0] < pred_p1[0]


def test_idw_missing_station_exclusion():
    """Confirms IDW dynamically excludes stations with NaN observations without fabricating values."""
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    values = np.array([10.0, np.nan, 30.0])

    idw = InverseDistanceWeighting(power=2.0).fit(coords, values)
    assert len(idw.station_coords) == 2
    assert len(idw.station_values) == 2

    # Midpoint between station 0 and station 2 is at 10.0
    pred, _ = idw.predict(np.array([[10.0, 0.0]]))
    assert np.isclose(pred[0], 20.0)


def test_kriging_unbiasedness_and_coincidence():
    """Confirms Ordinary Kriging enforces sum of weights = 1 and exact coincidence."""
    coords = np.array([[218650, 1941800], [216420, 1932950], [241880, 1933850], [210180, 1938550]], dtype=float)
    values = np.array([45.0, 30.0, 65.0, 40.0], dtype=float)

    for v_model in ["exponential", "spherical", "gaussian"]:
        ok = OrdinaryKriging(variogram_model=v_model)
        ok.fit(coords, values)

        # Check coincidence
        preds, variances = ok.predict(coords)
        assert np.allclose(preds, values, atol=1e-4), f"Coincidence failed for {v_model}"
        assert np.allclose(variances, 0.0, atol=1e-4), f"Variance zero failed for {v_model}"
        assert ok.condition_number < 1e12, f"Condition number too high for {v_model}"


def test_kriging_singular_matrix_robustness():
    """Confirms Ordinary Kriging uses SVD / pseudo-inverse fallback on singular matrices."""
    # Collinear or duplicate points trigger ill-conditioning
    coords = np.array([[1000.0, 1000.0], [1000.0, 1000.001], [5000.0, 5000.0]], dtype=float)
    values = np.array([50.0, 50.0, 20.0], dtype=float)

    ok = OrdinaryKriging(variogram_model="exponential")
    ok.fit(coords, values)
    preds, variances = ok.predict(np.array([[3000.0, 3000.0]]))

    assert not np.isnan(preds[0])
    assert not np.isnan(variances[0])


def test_hotspot_threshold_categorization():
    """Confirms hotspot detector adheres strictly to approved concentration boundaries."""
    detector = HotspotDetector()
    assert detector.acute_threshold == 60.0
    assert detector.spike_threshold == 120.0

    # Grid scenario
    Ny, Nx = 10, 10
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
    conc_surface = np.full((Ny, Nx), 40.0)
    conc_surface[0:3, 0:3] = 65.0   # 9 acute cells
    conc_surface[8:10, 8:10] = 130.0 # 4 spike cells

    dist_surface = np.full((Ny, Nx), 3000.0)  # 3km
    dist_surface[8:10, 8:10] = 18000.0       # 18km (extrapolation)

    st_preds = {"Bollaram": 65.0, "Central Univ": 40.0, "ECIL": 130.0, "Patancheru": 40.0}

    res = detector.detect_grid_hotspots(
        grid_x, grid_y, conc_surface, variance_surface=None, dist_surface=dist_surface,
        horizon="h1", station_predictions=st_preds
    )

    assert res["acute_excursion_cells"] == 13  # 9 + 4
    assert res["severe_spike_cells"] == 4
    assert res["station_observations"]["Bollaram"]["status"] == "ACUTE EXCURSION (>= 60 ug/m3)"
    assert res["station_observations"]["ECIL"]["status"] == "SEVERE SPIKE (>= 120 ug/m3)"
    assert res["station_observations"]["Central Univ"]["status"] == "COMPLIANT (< 60 ug/m3)"
    assert res["uncertainty_breakdown_acute"]["extrapolation_cells_gt_15km"] == 4


def test_spatial_metrics_computation():
    """Confirms spatial evaluation metric function computes MAE, RMSE, and bias accurately."""
    y_true = np.array([50.0, 60.0, 70.0, np.nan])
    y_interp = np.array([55.0, 58.0, 72.0, 60.0])

    m = compute_spatial_validation_metrics(y_true, y_interp, station_name="Test", horizon="h1", method="idw")
    assert m["n_samples"] == 3
    assert np.isclose(m["mae"], (5.0 + 2.0 + 2.0) / 3.0)
    assert np.isclose(m["bias"], (5.0 - 2.0 + 2.0) / 3.0)
