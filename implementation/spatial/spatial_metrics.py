"""
Spatial Evaluation Metrics Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Computes:
1. Spatial cross-validation continuous error metrics (MAE, RMSE, Bias, R2).
2. Spatial Relative Error (SRE) normalized by observed station mean.
3. Spatial Extrapolation Distance and Uncertainty Index.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_spatial_validation_metrics(
    y_true: np.ndarray,
    y_interp: np.ndarray,
    station_name: str,
    horizon: str,
    method: str
) -> Dict[str, Any]:
    """
    Computes spatial validation error metrics on held-out station observations.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_interp = np.asarray(y_interp, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_interp)
    n_valid = int(np.sum(mask))

    if n_valid == 0:
        return {
            "station": station_name,
            "horizon": horizon,
            "method": method,
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
            "mean_true": np.nan,
            "mean_interp": np.nan,
            "normalized_rmse_pct": np.nan
        }

    yt = y_true[mask]
    yp = y_interp[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    bias = float(np.mean(yp - yt))
    r2 = float(r2_score(yt, yp)) if np.var(yt) > 1e-9 else 0.0

    mean_yt = float(np.mean(yt))
    mean_yp = float(np.mean(yp))
    nrmse_pct = float((rmse / mean_yt) * 100.0) if abs(mean_yt) > 1e-6 else np.nan

    return {
        "station": station_name,
        "horizon": horizon,
        "method": method,
        "n_samples": n_valid,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "mean_true": mean_yt,
        "mean_interp": mean_yp,
        "normalized_rmse_pct": nrmse_pct
    }
