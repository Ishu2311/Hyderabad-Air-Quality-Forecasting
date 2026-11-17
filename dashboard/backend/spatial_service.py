"""
Dashboard Backend: Spatial Mapping and Uncertainty Service
Handles station coordinates, IDW interpolation surfaces, 3-tier uncertainty
buffer geometries, and hotspot contouring across Greater Hyderabad.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.data_loader import load_cpcb_stations_metadata, load_hotspot_summary, load_spatial_validation_results


# Bounding box for Greater Hyderabad domain
HYD_BOUNDS = {
    "min_lat": 17.30,
    "max_lat": 17.65,
    "min_lon": 78.20,
    "max_lon": 78.65,
    "center_lat": 17.47,
    "center_lon": 78.42
}

UNCERTAINTY_TIERS = {
    "tier_1_low": {
        "label": "Low Uncertainty (Proximate <= 5 km)",
        "threshold_km": 5.0,
        "area_pct": 16.80,
        "cell_count": 1075,
        "color": [46, 204, 113, 140],  # Soft Emerald Green
        "description": "Reliable local sensor envelope immediately adjacent to CAAQMS monitoring sites."
    },
    "tier_2_moderate": {
        "label": "Moderate Uncertainty (Intermediate 5 - 15 km)",
        "threshold_km": 15.0,
        "area_pct": 48.30,
        "cell_count": 3091,
        "color": [241, 196, 15, 120],  # Amber Gold
        "description": "Intermediate spatial decay zone; reflects regional dispersion between active donor stations."
    },
    "tier_3_high": {
        "label": "High Uncertainty (Severe Extrapolation > 15 km)",
        "threshold_km": 999.0,
        "area_pct": 34.90,
        "cell_count": 2234,
        "color": [231, 76, 60, 110],  # Alert Crimson
        "description": "High-variance extrapolation regime unmonitored by sensors (e.g. Old City, Airport, Uppal)."
    }
}


def get_station_map_dataframe(horizon: str = "h1") -> pd.DataFrame:
    """
    Constructs station dataframe with geographical coordinates,
    station typology, and latest predicted concentrations.
    """
    stations = load_cpcb_stations_metadata()
    summary = load_hotspot_summary()
    
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    h_data = summary.get(h_key, {})
    obs_map = h_data.get("station_observations", {})
    
    records = []
    for s_name, meta in stations.items():
        st_obs = obs_map.get(s_name, {})
        pred_pm25 = st_obs.get("predicted_pm25", 35.0)
        status = st_obs.get("status", "COMPLIANT (< 60 ug/m3)")
        
        # Color coding: Green for compliant (<60), Orange for acute excursion (>=60), Red for spike (>=120)
        if pred_pm25 >= 120.0:
            color = [217, 30, 24, 220]  # Deep Red
            status_tag = "SEVERE SPIKE (>= 120 ug/m3)"
        elif pred_pm25 >= 60.0:
            color = [230, 126, 34, 220]  # Orange
            status_tag = "ACUTE EXCURSION (>= 60 ug/m3)"
        else:
            color = [39, 174, 96, 220]  # Emerald Green
            status_tag = "COMPLIANT (< 60 ug/m3)"
            
        records.append({
            "station": s_name,
            "station_code": meta["code"],
            "station_type": meta["station_type"],
            "latitude": meta["latitude"],
            "longitude": meta["longitude"],
            "predicted_pm25": round(float(pred_pm25), 2),
            "status": status_tag,
            "color": color,
            "ward": meta["ward"],
            "typology": meta["surrounding_typology"]
        })
        
    return pd.DataFrame(records)


def get_spatial_grid_samples(horizon: str = "h1", grid_res: int = 30) -> pd.DataFrame:
    """
    Generates a regular 2D spatial grid over Hyderabad and evaluates IDW
    interpolated concentrations and distance-to-nearest-sensor uncertainty tiers.
    """
    stations = load_cpcb_stations_metadata()
    summary = load_hotspot_summary()
    
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    obs_map = summary.get(h_key, {}).get("station_observations", {})
    
    # Station locations in lat/lon
    st_lats = np.array([meta["latitude"] for meta in stations.values()])
    st_lons = np.array([meta["longitude"] for meta in stations.values()])
    st_vals = np.array([obs_map.get(name, {}).get("predicted_pm25", 35.0) for name in stations.keys()])
    
    lats = np.linspace(HYD_BOUNDS["min_lat"], HYD_BOUNDS["max_lat"], grid_res)
    lons = np.linspace(HYD_BOUNDS["min_lon"], HYD_BOUNDS["max_lon"], grid_res)
    
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()
    
    # Approximate metric conversion: 1 deg lat ~ 111 km, 1 deg lon ~ 106 km at 17.5 deg
    dx_km = (flat_lon[:, None] - st_lons[None, :]) * 106.0
    dy_km = (flat_lat[:, None] - st_lats[None, :]) * 111.0
    dists_km = np.sqrt(dx_km**2 + dy_km**2)
    
    min_dist_km = np.min(dists_km, axis=1)
    
    # IDW weighting with p=2.0
    eps = 1e-4
    weights = 1.0 / (dists_km + eps)**2
    weights /= np.sum(weights, axis=1, keepdims=True)
    interp_pm25 = np.sum(weights * st_vals[None, :], axis=1)
    
    # Assign uncertainty tiers
    tiers = []
    tier_colors = []
    for d in min_dist_km:
        if d <= 5.0:
            tiers.append("Low Uncertainty (<= 5 km)")
            tier_colors.append([46, 204, 113, 130])
        elif d <= 15.0:
            tiers.append("Moderate Uncertainty (5-15 km)")
            tier_colors.append([241, 196, 15, 110])
        else:
            tiers.append("High Extrapolation (> 15 km)")
            tier_colors.append([231, 76, 60, 90])
            
    return pd.DataFrame({
        "latitude": flat_lat,
        "longitude": flat_lon,
        "min_dist_km": np.round(min_dist_km, 2),
        "interp_pm25": np.round(interp_pm25, 2),
        "uncertainty_tier": tiers,
        "color": tier_colors
    })
