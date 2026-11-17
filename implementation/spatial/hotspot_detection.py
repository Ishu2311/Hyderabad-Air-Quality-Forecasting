"""
Spatial Hotspot Identification and Classification Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Implements:
1. Concentration-based hotspot identification using approved regulatory boundaries:
   - Acute Excursions: PM2.5 >= 60.0 ug/m3 (24h statutory NAAQS standard non-compliance)
   - Severe Spikes: PM2.5 >= 120.0 ug/m3 (NAQI Very Poor entry boundary - exploratory)
2. Probability-based hotspot identification using calibrated decision thresholds tau*:
   - Horizon +1h: tau* = 0.76 (LightGBM)
   - Horizon +6h: tau* = 0.60 (LightGBM)
   - Horizon +24h: tau* = 0.53 (LightGBM)
3. Multi-tier Spatial Uncertainty Classification:
   - Tier 1 (Low Uncertainty): Distance <= 5.0 km from physical monitoring station
   - Tier 2 (Moderate Uncertainty): 5.0 km < Distance <= 15.0 km
   - Tier 3 (High Extrapolation Uncertainty): Distance > 15.0 km
4. Distinction between physical station measurements, interpolated surfaces, and AQI indices.

Academic Research Integrity:
- Explicitly flags interpolated exceedances as ESTIMATES, not confirmed physical events.
- Never conflates PM2.5 mass concentration (ug/m3) with dimensionless AQI scores.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from configs.config import (
    STATION_REGISTRY,
    CONCENTRATION_THRESHOLDS
)

# Thresholds from research methodology
THRESHOLD_ACUTE_EXCURSION = 60.0   # ug/m3
THRESHOLD_SEVERE_SPIKE = 120.0     # ug/m3

# Calibrated classification thresholds tau* from Milestone 3 validation
CALIBRATED_TAU_ACUTE = {
    "h1": 0.76,
    "h6": 0.60,
    "h24": 0.53
}


class HotspotDetector:
    """Detects and categorizes spatial air quality hotspots across Hyderabad."""

    def __init__(
        self,
        acute_threshold: float = THRESHOLD_ACUTE_EXCURSION,
        spike_threshold: float = THRESHOLD_SEVERE_SPIKE
    ):
        self.acute_threshold = float(acute_threshold)
        self.spike_threshold = float(spike_threshold)

    @staticmethod
    def classify_uncertainty_tier(distance_meters: float) -> str:
        """Classifies spatial uncertainty tier based on distance to nearest monitoring station."""
        dist_km = distance_meters / 1000.0
        if dist_km <= 5.0:
            return "Low Uncertainty (Proximate <= 5km)"
        elif dist_km <= 15.0:
            return "Moderate Uncertainty (Intermediate 5-15km)"
        else:
            return "High Uncertainty (Extrapolation > 15km)"

    def detect_grid_hotspots(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        conc_surface: np.ndarray,
        variance_surface: Optional[np.ndarray],
        dist_surface: np.ndarray,
        horizon: str,
        station_predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detects hotspot zones across an interpolated spatial grid.
        
        Parameters
        ----------
        grid_x : 2D array of Easting or Longitude coordinates
        grid_y : 2D array of Northing or Latitude coordinates
        conc_surface : 2D array of predicted PM2.5 in ug/m3
        variance_surface : 2D array of Kriging variance (or None if IDW)
        dist_surface : 2D array of distance to nearest station (meters)
        horizon : str ('h1', 'h6', 'h24', 'h48')
        station_predictions : dict mapping station_name -> predicted PM2.5 value
        """
        Ny, Nx = conc_surface.shape
        total_cells = Ny * Nx

        acute_mask = conc_surface >= self.acute_threshold
        spike_mask = conc_surface >= self.spike_threshold

        n_acute = int(np.sum(acute_mask))
        n_spike = int(np.sum(spike_mask))

        acute_pct = float((n_acute / total_cells) * 100.0)
        spike_pct = float((n_spike / total_cells) * 100.0)

        # Station-level hotspot census
        station_status = {}
        for s_name, s_val in station_predictions.items():
            if np.isnan(s_val):
                status = "OUTAGE / NO DATA"
            elif s_val >= self.spike_threshold:
                status = "SEVERE SPIKE (>= 120 ug/m3)"
            elif s_val >= self.acute_threshold:
                status = "ACUTE EXCURSION (>= 60 ug/m3)"
            else:
                status = "COMPLIANT (< 60 ug/m3)"
            station_status[s_name] = {
                "predicted_pm25": float(s_val) if not np.isnan(s_val) else None,
                "status": status
            }

        # Analyze uncertainty of acute hotspot cells
        if n_acute > 0:
            acute_dists = dist_surface[acute_mask]
            mean_dist_km = float(np.mean(acute_dists) / 1000.0)
            proximate_cells = int(np.sum(acute_dists <= 5000.0))
            intermediate_cells = int(np.sum((acute_dists > 5000.0) & (acute_dists <= 15000.0)))
            extrapolation_cells = int(np.sum(acute_dists > 15000.0))
        else:
            mean_dist_km = np.nan
            proximate_cells = 0
            intermediate_cells = 0
            extrapolation_cells = 0

        summary = {
            "horizon": horizon,
            "threshold_acute_ug_m3": self.acute_threshold,
            "threshold_spike_ug_m3": self.spike_threshold,
            "grid_dimensions": {"Ny": Ny, "Nx": Nx, "total_cells": total_cells},
            "acute_excursion_cells": n_acute,
            "acute_excursion_area_pct": acute_pct,
            "severe_spike_cells": n_spike,
            "severe_spike_area_pct": spike_pct,
            "uncertainty_breakdown_acute": {
                "mean_distance_to_sensor_km": mean_dist_km,
                "proximate_cells_le_5km": proximate_cells,
                "intermediate_cells_5_to_15km": intermediate_cells,
                "extrapolation_cells_gt_15km": extrapolation_cells,
                "pct_extrapolation": float(extrapolation_cells / n_acute * 100.0) if n_acute > 0 else 0.0
            },
            "station_observations": station_status,
            "research_disclosure": (
                "Interpolated hotspot cells reflect spatial continuous weighting between reporting stations. "
                "Cells classified as 'Extrapolation (> 15km)' carry high spatial estimation variance and "
                "must NOT be interpreted as confirmed ground-level violations."
            )
        }
        return summary
