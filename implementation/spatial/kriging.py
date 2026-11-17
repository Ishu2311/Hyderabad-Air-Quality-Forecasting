"""
Ordinary Kriging Spatial Interpolation Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Implements from first principles:
1. Ordinary Kriging (Best Linear Unbiased Estimator - BLUE) under intrinsic spatial stationarity:
   Z_hat(s_0) = sum_k w_k Z(s_k)  subject to sum_k w_k = 1
2. Kriging linear system solver:
   [ Gamma   1 ] [ w ]   [ gamma_0 ]
   [  1^T    0 ] [ u ] = [    1    ]
3. Kriging estimation variance computation:
   sigma_k^2(s_0) = w^T gamma_0 + u
4. Theoretical variogram models:
   - Spherical
   - Exponential
   - Gaussian
5. Numerical condition-number diagnostics (kappa(K)) and singular value decomposition (SVD) fallback.
6. Statistical Identifiability Analysis for sparse monitoring networks (N=4 stations -> 6 distance pairs).

Academic Research Integrity:
- Documents mathematical limitations of variogram fitting on small N.
- Distinguishes physical observations from interpolated estimates.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import scipy.linalg as la
from scipy.optimize import curve_fit


def spherical_variogram(h: np.ndarray, nugget: float, sill: float, range_a: float) -> np.ndarray:
    """Spherical semivariogram model."""
    h = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h)
    
    # Avoid zero division
    range_a = max(range_a, 1e-3)
    sill = max(sill, 1e-6)
    nugget = max(nugget, 0.0)
    
    mask_within = (h > 0) & (h <= range_a)
    mask_beyond = (h > range_a)
    
    gamma[mask_within] = nugget + sill * (1.5 * (h[mask_within] / range_a) - 0.5 * (h[mask_within] / range_a)**3)
    gamma[mask_beyond] = nugget + sill
    return gamma


def exponential_variogram(h: np.ndarray, nugget: float, sill: float, range_a: float) -> np.ndarray:
    """Exponential semivariogram model."""
    h = np.asarray(h, dtype=float)
    range_a = max(range_a, 1e-3)
    sill = max(sill, 1e-6)
    nugget = max(nugget, 0.0)
    
    gamma = np.zeros_like(h)
    mask = h > 0
    gamma[mask] = nugget + sill * (1.0 - np.exp(-3.0 * h[mask] / range_a))
    return gamma


def gaussian_variogram(h: np.ndarray, nugget: float, sill: float, range_a: float) -> np.ndarray:
    """Gaussian semivariogram model."""
    h = np.asarray(h, dtype=float)
    range_a = max(range_a, 1e-3)
    sill = max(sill, 1e-6)
    nugget = max(nugget, 0.0)
    
    gamma = np.zeros_like(h)
    mask = h > 0
    gamma[mask] = nugget + sill * (1.0 - np.exp(-3.0 * (h[mask] / range_a)**2))
    return gamma


VARIOGRAM_MODELS = {
    "spherical": spherical_variogram,
    "exponential": exponential_variogram,
    "gaussian": gaussian_variogram
}


class OrdinaryKriging:
    """
    Ordinary Kriging Spatial Interpolator with Condition-Number Diagnostics.
    
    Parameters
    ----------
    variogram_model : str, default="exponential"
        Theoretical variogram model ('spherical', 'exponential', 'gaussian').
    coordinate_system : str, default="utm"
        Coordinate system ('utm' for Euclidean meters, 'wgs84' for degrees).
    nugget : float, optional
        Nugget effect c0. If None, estimated from data.
    sill : float, optional
        Partial sill c. If None, estimated from sample variance.
    range_a : float, optional
        Effective range a. If None, estimated from maximum inter-station distance.
    coincidence_epsilon : float, default=1e-5
        Distance threshold below which target point is treated as identical to a station.
    """

    def __init__(
        self,
        variogram_model: str = "exponential",
        coordinate_system: str = "utm",
        nugget: Optional[float] = None,
        sill: Optional[float] = None,
        range_a: Optional[float] = None,
        coincidence_epsilon: float = 1e-5
    ):
        if variogram_model not in VARIOGRAM_MODELS:
            raise ValueError(f"Unknown variogram_model '{variogram_model}'. Choose from {list(VARIOGRAM_MODELS.keys())}")
        if coordinate_system not in ["utm", "wgs84"]:
            raise ValueError(f"Unsupported coordinate_system '{coordinate_system}'.")

        self.variogram_model_name = variogram_model
        self.variogram_func = VARIOGRAM_MODELS[variogram_model]
        self.coordinate_system = coordinate_system
        self.coincidence_epsilon = float(coincidence_epsilon)

        self.user_nugget = nugget
        self.user_sill = sill
        self.user_range_a = range_a

        # Fitted parameters
        self.nugget: float = 0.0
        self.sill: float = 1.0
        self.range_a: float = 10000.0  # meters
        self.condition_number: float = 1.0
        self.is_singular: bool = False
        self.fit_diagnostics: Dict[str, Any] = {}

        self.station_coords: Optional[np.ndarray] = None
        self.station_values: Optional[np.ndarray] = None
        self.station_names: Optional[List[str]] = None
        self.K_inv: Optional[np.ndarray] = None

    def _compute_inter_station_distances(self) -> np.ndarray:
        """Computes K x K pairwise Euclidean distance matrix."""
        coords = self.station_coords
        K = len(coords)
        dx = coords[:, 0:1] - coords[:, 0:1].T
        dy = coords[:, 1:2] - coords[:, 1:2].T
        return np.sqrt(dx**2 + dy**2)

    def fit_variogram(self, coords: np.ndarray, values: np.ndarray) -> Dict[str, Any]:
        """
        Estimates variogram parameters from experimental semivariance points.
        Documents identifiability limitations for N=4 stations (6 pairs).
        """
        K = len(coords)
        n_pairs = K * (K - 1) // 2

        # Extract experimental semivariance pairs
        h_pairs = []
        gamma_pairs = []
        for i in range(K):
            for j in range(i + 1, K):
                dist = np.sqrt(np.sum((coords[i] - coords[j])**2))
                semivar = 0.5 * (values[i] - values[j])**2
                h_pairs.append(dist)
                gamma_pairs.append(semivar)

        h_arr = np.array(h_pairs)
        gamma_arr = np.array(gamma_pairs)

        sample_var = float(np.var(values)) if np.var(values) > 1e-4 else 1.0
        max_dist = float(np.max(h_arr)) if len(h_arr) > 0 else 10000.0

        # Identifiability check: with 6 or fewer pairs, non-linear 3-parameter fitting is ill-posed
        can_fit = (K >= 4 and len(h_arr) >= 6 and sample_var > 1e-3)
        fitted_by_optim = False

        if can_fit and self.user_nugget is None and self.user_sill is None and self.user_range_a is None:
            try:
                # Try bounded non-linear least squares
                p0 = [0.05 * sample_var, 0.95 * sample_var, 0.5 * max_dist]
                bounds = ([0.0, 1e-4, 0.1 * max_dist], [sample_var, 5.0 * sample_var, 3.0 * max_dist])
                popt, _ = curve_fit(self.variogram_func, h_arr, gamma_arr, p0=p0, bounds=bounds, maxfev=2000)
                self.nugget, self.sill, self.range_a = float(popt[0]), float(popt[1]), float(popt[2])
                fitted_by_optim = True
            except Exception:
                fitted_by_optim = False

        if not fitted_by_optim:
            # Robust analytical fallback based on empirical dispersion
            self.nugget = float(self.user_nugget if self.user_nugget is not None else 0.05 * sample_var)
            self.sill = float(self.user_sill if self.user_sill is not None else 0.95 * sample_var)
            self.range_a = float(self.user_range_a if self.user_range_a is not None else 0.6 * max_dist)

        diagnostics = {
            "n_stations": K,
            "n_pairs": n_pairs,
            "sample_variance": sample_var,
            "max_interstation_distance_m": max_dist,
            "fitted_by_optimizer": fitted_by_optim,
            "variogram_model": self.variogram_model_name,
            "nugget": self.nugget,
            "sill": self.sill,
            "total_sill": self.nugget + self.sill,
            "range_a": self.range_a,
            "identifiability_limitation": (
                f"Network contains only {K} stations ({n_pairs} distance pairs). "
                "Empirical variogram is statistically under-determined. Variogram parameters "
                "represent regularized heuristic estimates rather than fully identified field properties."
            )
        }
        self.fit_diagnostics = diagnostics
        return diagnostics

    def fit(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        station_names: Optional[List[str]] = None
    ) -> "OrdinaryKriging":
        """
        Fits Ordinary Kriging model and constructs the (K+1)x(K+1) Kriging matrix.
        """
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float)

        valid_mask = ~np.isnan(values) & ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1])
        self.station_coords = coords[valid_mask]
        self.station_values = values[valid_mask]
        if station_names is not None:
            self.station_names = [name for name, v in zip(station_names, valid_mask) if v]
        else:
            self.station_names = [f"Station_{i}" for i in range(len(self.station_coords))]

        K = len(self.station_coords)
        if K < 2:
            self.K_inv = None
            return self

        # Fit variogram parameters
        self.fit_variogram(self.station_coords, self.station_values)

        # Build Kriging matrix K_mat of shape (K+1, K+1)
        D_stations = self._compute_inter_station_distances()
        Gamma = self.variogram_func(D_stations, self.nugget, self.sill, self.range_a)

        # Ensure diagonal is exactly 0.0
        np.fill_diagonal(Gamma, 0.0)

        K_mat = np.zeros((K + 1, K + 1), dtype=float)
        K_mat[:K, :K] = Gamma
        K_mat[:K, K] = 1.0
        K_mat[K, :K] = 1.0
        K_mat[K, K] = 0.0

        # Assess condition number and invert
        try:
            cond = float(np.linalg.cond(K_mat))
            self.condition_number = cond
            if cond > 1e12 or np.isnan(cond):
                # Ill-conditioned -> use Moore-Penrose pseudo-inverse with SVD truncation
                self.is_singular = True
                self.K_inv = la.pinv(K_mat, rcond=1e-10)
            else:
                self.is_singular = False
                self.K_inv = la.inv(K_mat)
        except la.LinAlgError:
            self.is_singular = True
            self.K_inv = la.pinv(K_mat, rcond=1e-8)

        self.fit_diagnostics["kriging_matrix_condition_number"] = self.condition_number
        self.fit_diagnostics["is_matrix_singular"] = self.is_singular

        return self

    def predict(
        self,
        target_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolates PM2.5 concentrations and computes Kriging estimation variance.
        
        Returns
        -------
        z_pred : np.ndarray of shape (M,)
            Kriging PM2.5 concentration estimates in ug/m3.
        kriging_variance : np.ndarray of shape (M,)
            Theoretical estimation variance sigma_k^2 (ug/m3)^2.
        """
        target_coords = np.asarray(target_coords, dtype=float)
        M = len(target_coords)

        if self.station_coords is None or len(self.station_coords) == 0:
            nan_arr = np.full(M, np.nan)
            return nan_arr, nan_arr

        K = len(self.station_coords)
        if K == 1 or self.K_inv is None:
            # Single station -> uniform field with maximum sill variance
            val = float(self.station_values[0])
            var = float(self.nugget + self.sill)
            return np.full(M, val), np.full(M, var)

        # Distance matrix from target coordinates to stations (M, K)
        dx = target_coords[:, 0:1] - self.station_coords[:, 0:1].T  # (M, K)
        dy = target_coords[:, 1:2] - self.station_coords[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)

        predictions = np.zeros(M, dtype=float)
        variances = np.zeros(M, dtype=float)

        # Semivariance vector gamma_0 of shape (M, K)
        gamma_0 = self.variogram_func(dists, self.nugget, self.sill, self.range_a)

        # Augment with unbiasedness constraint 1.0 -> shape (M, K+1)
        k_0 = np.ones((M, K + 1), dtype=float)
        k_0[:, :K] = gamma_0

        # Weights W = k_0 @ K_inv.T of shape (M, K+1)
        weights_and_mu = k_0 @ self.K_inv.T
        weights = weights_and_mu[:, :K]
        mu = weights_and_mu[:, K]

        # Station coincidence check
        coincident_mask = dists < self.coincidence_epsilon
        has_coincidence = np.any(coincident_mask, axis=1)

        # Calculate standard kriging estimates
        preds_all = np.sum(weights * self.station_values, axis=1)
        vars_all = np.sum(weights * gamma_0, axis=1) + mu
        # Variances must be strictly non-negative
        vars_all = np.maximum(vars_all, 0.0)

        predictions[:] = preds_all
        variances[:] = vars_all

        # Coincidence override
        if np.any(has_coincidence):
            coincident_indices = np.where(has_coincidence)[0]
            for idx in coincident_indices:
                station_idx = np.where(coincident_mask[idx])[0][0]
                predictions[idx] = self.station_values[station_idx]
                variances[idx] = 0.0

        return predictions, variances

    def interpolate_grid(
        self,
        grid_x: np.ndarray,
        grid_y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolates over a 2D meshgrid."""
        Ny, Nx = grid_x.shape
        target_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        preds, variances = self.predict(target_pts)
        return preds.reshape(Ny, Nx), variances.reshape(Ny, Nx)
