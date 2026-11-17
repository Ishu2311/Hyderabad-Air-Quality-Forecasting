"""
Inverse Distance Weighting (IDW) Spatial Interpolation Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Implements:
1. Exact mathematical formulation of Inverse Distance Weighting:
   Z_hat(s_0) = sum_k w_k Z(s_k) / sum_k w_k  where w_k = 1 / d(s_0, s_k)^p
2. Robust station coincidence handling (exact return within epsilon threshold).
3. Dynamic missing-value handling (excludes NaN sensor values from interpolation set).
4. Dual coordinate support: UTM Zone 44N planar Euclidean (meters) or WGS84 Haversine (km).
5. Nearest-sensor distance surface computation as a baseline spatial uncertainty metric.

Academic Research Integrity:
- Preserves the strict distinction between physical sensor observations and spatial estimates.
- Interpolated values are exploratory spatial visualizations, NOT verified ground truth.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


def haversine_distance(
    lat1: Union[float, np.ndarray],
    lon1: Union[float, np.ndarray],
    lat2: Union[float, np.ndarray],
    lon2: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Computes Great Circle distance between coordinates (degrees) in kilometers using Haversine formula.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


class InverseDistanceWeighting:
    """
    Inverse Distance Weighting (IDW) Spatial Interpolator.
    
    Parameters
    ----------
    power : float, default=2.0
        Distance decay exponent p in w = 1 / d^p.
    coordinate_system : str, default="utm"
        Coordinate system to use: "utm" (planar Euclidean meters) or "wgs84" (Haversine km).
    coincidence_epsilon : float, default=1e-5
        Distance threshold below which target point is treated as identical to the station.
    """

    def __init__(
        self,
        power: float = 2.0,
        coordinate_system: str = "utm",
        coincidence_epsilon: float = 1e-5
    ):
        if power <= 0:
            raise ValueError(f"Distance power must be positive, got {power}")
        if coordinate_system not in ["utm", "wgs84"]:
            raise ValueError(f"Unsupported coordinate_system '{coordinate_system}'. Must be 'utm' or 'wgs84'.")

        self.power = float(power)
        self.coordinate_system = coordinate_system
        self.coincidence_epsilon = float(coincidence_epsilon)
        self.station_coords: Optional[np.ndarray] = None
        self.station_values: Optional[np.ndarray] = None
        self.station_names: Optional[List[str]] = None

    def fit(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        station_names: Optional[List[str]] = None
    ) -> "InverseDistanceWeighting":
        """
        Fits IDW with spatial observation/prediction points.
        
        Parameters
        ----------
        coords : np.ndarray of shape (K, 2)
            Station spatial coordinates: (Easting, Northing) if utm, or (Latitude, Longitude) if wgs84.
        values : np.ndarray of shape (K,)
            Continuous PM2.5 values at the stations.
        station_names : list of str, optional
            Names of the stations.
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)

        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"coords must be of shape (K, 2), got {coords.shape}")
        if values.ndim != 1 or len(values) != len(coords):
            raise ValueError("values must be 1D array of same length as coords")

        # Exclude missing observations without imputation
        valid_mask = ~np.isnan(values) & ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
        self.station_coords = coords[valid_mask]
        self.station_values = values[valid_mask]
        if station_names is not None:
            self.station_names = [name for name, v in zip(station_names, valid_mask) if v]
        else:
            self.station_names = [f"Station_{i}" for i in range(len(self.station_coords))]

        return self

    def _compute_distances(
        self,
        target_coords: np.ndarray
    ) -> np.ndarray:
        """Computes distance matrix between target coordinates (M, 2) and station coordinates (K, 2)."""
        if self.station_coords is None or len(self.station_coords) == 0:
            raise RuntimeError("Interpolator must be fitted with valid stations before computing distances.")

        M = len(target_coords)
        K = len(self.station_coords)

        if self.coordinate_system == "utm":
            # Euclidean distance in meters
            dx = target_coords[:, 0:1] - self.station_coords[:, 0:1].T  # (M, K)
            dy = target_coords[:, 1:2] - self.station_coords[:, 1:2].T  # (M, K)
            distances = np.sqrt(dx**2 + dy**2)
        else:
            # Haversine distance in km
            lat_t = target_coords[:, 0:1]  # (M, 1)
            lon_t = target_coords[:, 1:2]
            lat_s = self.station_coords[:, 0:1].T  # (1, K)
            lon_s = self.station_coords[:, 1:2].T
            distances = haversine_distance(lat_t, lon_t, lat_s, lon_s)

        return distances

    def predict(
        self,
        target_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates PM2.5 concentrations at target coordinates.
        
        Returns
        -------
        interpolated_values : np.ndarray of shape (M,)
            Estimated PM2.5 concentration in ug/m3.
        nearest_distance : np.ndarray of shape (M,)
            Distance to the nearest reporting physical station (meters if utm, km if wgs84).
        """
        target_coords = np.asarray(target_coords, dtype=float)
        if target_coords.ndim != 2 or target_coords.shape[1] != 2:
            raise ValueError(f"target_coords must be of shape (M, 2), got {target_coords.shape}")

        if self.station_coords is None or len(self.station_coords) == 0:
            # Zero reporting stations -> return NaN
            nan_arr = np.full(len(target_coords), np.nan)
            return nan_arr, nan_arr

        if len(self.station_coords) == 1:
            # Single reporting station -> uniform field with distance
            dists = self._compute_distances(target_coords)
            val = float(self.station_values[0])
            return np.full(len(target_coords), val), dists[:, 0]

        dists = self._compute_distances(target_coords)  # Shape (M, K)
        nearest_dist = np.min(dists, axis=1)

        M, K = dists.shape
        predictions = np.zeros(M, dtype=float)

        # Handle station coincidence
        coincident_mask = dists < self.coincidence_epsilon  # Shape (M, K)
        has_coincidence = np.any(coincident_mask, axis=1)

        # For points that coincide with a station, assign that station's exact value
        if np.any(has_coincidence):
            coincident_indices = np.where(has_coincidence)[0]
            for idx in coincident_indices:
                station_idx = np.where(coincident_mask[idx])[0][0]
                predictions[idx] = self.station_values[station_idx]

        # For remaining points, apply standard IDW
        non_coincident_indices = np.where(~has_coincidence)[0]
        if len(non_coincident_indices) > 0:
            d_sub = dists[non_coincident_indices, :]  # Shape (N_nc, K)
            weights = 1.0 / (d_sub ** self.power)     # Shape (N_nc, K)
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            norm_weights = weights / weights_sum
            predictions[non_coincident_indices] = np.sum(norm_weights * self.station_values, axis=1)

        return predictions, nearest_dist

    def interpolate_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates over a 2D meshgrid.
        
        Parameters
        ----------
        grid_x : 2D np.ndarray of shape (Ny, Nx)
            Easting or Longitude coordinates of meshgrid.
        grid_y : 2D np.ndarray of shape (Ny, Nx)
            Northing or Latitude coordinates of meshgrid.
            
        Returns
        -------
        z_grid : 2D np.ndarray of shape (Ny, Nx)
            Interpolated continuous PM2.5 concentration surface.
        dist_grid : 2D np.ndarray of shape (Ny, Nx)
            Distance surface to nearest physical monitoring station.
        """
        Ny, Nx = grid_x.shape
        target_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        preds, dists = self.predict(target_pts)
        return preds.reshape(Ny, Nx), dists.reshape(Ny, Nx)
