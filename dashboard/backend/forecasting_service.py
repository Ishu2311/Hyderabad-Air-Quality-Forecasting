"""
Dashboard Backend: Forecasting Service
Provides prediction slicing, actual vs predicted time series alignment,
winning model identification, and live slice metric calculations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.data_loader import load_regression_predictions, load_regression_metrics


WINNING_MODELS = {
    "h1": {
        "name": "Stacking Ensemble (Bayesian Ridge)",
        "code": "pred_stacking_h1",
        "description": "Out-of-fold linear combination of Random Forest (12.5%), XGBoost (44.8%), and LightGBM (43.9%).",
        "mae_test": 3.011,
        "rmse_test": 4.148,
        "r2_test": 0.942
    },
    "h6": {
        "name": "Stacking Ensemble (Bayesian Ridge)",
        "code": "pred_stacking_h6",
        "description": "Out-of-fold linear combination of Random Forest (6.5%), XGBoost (19.1%), and LightGBM (78.2%).",
        "mae_test": 5.284,
        "rmse_test": 7.391,
        "r2_test": 0.817
    },
    "h24": {
        "name": "LightGBM Gradient Boosted Regressor",
        "code": "pred_lgb_h24",
        "description": "Optimized LightGBM regressor with 84 primary predictors and early stopping.",
        "mae_test": 7.164,
        "rmse_test": 9.878,
        "r2_test": 0.673
    },
    "h48": {
        "name": "LightGBM Gradient Boosted Regressor",
        "code": "pred_lgb_h48",
        "description": "Optimized LightGBM regressor utilizing synoptic thermodynamics and boundary layer dynamics.",
        "mae_test": 8.542,
        "rmse_test": 11.637,
        "r2_test": 0.546
    }
}


def get_available_stations() -> List[str]:
    """Returns list of four CPCB monitoring stations."""
    return [
        "Bollaram Industrial Area",
        "Central University",
        "ECIL Kapra",
        "ICRISAT Patancheru"
    ]


def get_horizon_metadata(horizon: str) -> Dict[str, Any]:
    """Returns metadata for the chosen forecast horizon."""
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    return WINNING_MODELS.get(h_key, WINNING_MODELS["h1"])


def get_forecast_timeseries(
    station: str,
    horizon: str = "h1",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Extracts time series comparing Ground Truth PM2.5, Selected Model,
    Persistence Baseline, and Seasonal Naive Baseline.
    """
    df = load_regression_predictions()
    if df.empty:
        return pd.DataFrame()
        
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    target_col = f"target_pm25_{h_key}"
    
    # Filter by station
    sub_df = df[df["station"] == station].copy()
    
    # Apply date filters
    if start_date:
        sub_df = sub_df[sub_df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        sub_df = sub_df[sub_df["datetime"] <= pd.to_datetime(end_date)]
        
    model_meta = WINNING_MODELS.get(h_key, WINNING_MODELS["h1"])
    model_col = model_meta["code"]
    pers_col = f"pred_persistence_{h_key}"
    seas_col = f"pred_seasonal_naive_{h_key}"
    
    selected_cols = ["station", "datetime", target_col, model_col, pers_col, seas_col]
    available_cols = [c for c in selected_cols if c in sub_df.columns]
    
    res_df = sub_df[available_cols].copy()
    res_df = res_df.rename(columns={
        target_col: "Actual PM2.5 (ug/m3)",
        model_col: f"Forecast ({model_meta['name']})",
        pers_col: "Persistence Baseline",
        seas_col: "Seasonal-Naive (24h) Baseline"
    })
    
    return res_df.sort_values("datetime").reset_index(drop=True)


def calculate_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes standard evaluation metrics on valid observation pairs."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(mask) == 0:
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "r2": np.nan, "count": 0}
        
    yt = y_true[mask]
    yp = y_pred[mask]
    
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    bias = float(np.mean(yp - yt))
    r2 = float(r2_score(yt, yp)) if len(yt) > 1 and np.var(yt) > 1e-6 else 0.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "count": int(np.sum(mask))
    }
