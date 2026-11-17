"""
Model Evaluation and Metrics Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Implements:
1. Continuous Regression Metrics:
   - MAE, RMSE, R2, Mean Error (Bias), MDA, and valid sample count (N).
2. Binary Hotspot Classification Metrics:
   - ROC-AUC, PR-AUC, Brier Score, Precision, Recall, F1-score, Confusion Matrix, Prevalence.
   - Deterministic threshold calibration grid-search (0.05 - 0.95, step 0.01) on validation data.
3. Diebold-Mariano Statistical Significance Test:
   - Incorporates Harvey-Leybourne-Newbold (HLN) finite-sample correction for horizon h.
4. Station-by-station and network-level reporting with exact valid sample counts.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)
from scipy import stats

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes continuous regression evaluation metrics on valid (non-NaN) sample pairs.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if baseline_pred is not None:
        mask = mask & ~np.isnan(baseline_pred)
        
    n_valid = int(np.sum(mask))
    if n_valid == 0:
        return {
            "n_samples": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "mda": np.nan
        }
        
    yt = y_true[mask]
    yp = y_pred[mask]
    
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp)) if np.var(yt) > 1e-9 else 0.0
    bias = float(np.mean(yp - yt))
    
    # Mean Directional Accuracy (MDA) if length >= 2
    if len(yt) >= 2:
        diff_true = np.diff(yt)
        diff_pred = np.diff(yp)
        mda = float(np.mean((diff_true * diff_pred) >= 0))
    else:
        mda = np.nan
        
    return {
        "n_samples": n_valid,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias,
        "mda": mda
    }

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid_step: float = 0.01
) -> float:
    """
    Finds optimal decision threshold tau* in [0.05, 0.95] maximizing F1-score on validation data.
    Ties broken deterministically by selecting threshold closest to class prevalence.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]
    
    if len(yt) == 0 or np.sum(yt) == 0:
        return 0.50
        
    prevalence = float(np.mean(yt))
    thresholds = np.arange(0.05, 0.96, grid_step)
    
    best_f1 = -1.0
    best_tau = 0.50
    best_diff = 999.0
    
    for tau in thresholds:
        pred_binary = (yp >= tau).astype(int)
        f1 = f1_score(yt, pred_binary, zero_division=0)
        diff_prev = abs(tau - prevalence)
        
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
            best_diff = diff_prev
        elif np.isclose(f1, best_f1, atol=1e-5):
            # Deterministic tie-breaking: threshold closest to empirical prevalence
            if diff_prev < best_diff:
                best_tau = tau
                best_diff = diff_prev
                
    return float(best_tau)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes binary hotspot classification metrics on valid samples.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    n_valid = int(np.sum(mask))
    
    if n_valid == 0:
        return {
            "n_samples": 0,
            "prevalence": np.nan,
            "threshold": np.nan,
            "roc_auc": np.nan,
            "pr_auc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0
        }
        
    yt = y_true[mask].astype(int)
    yp = y_prob[mask]
    prevalence = float(np.mean(yt))
    
    # Select threshold if not specified
    if threshold is None:
        threshold = find_optimal_threshold(yt, yp)
        
    y_pred_binary = (yp >= threshold).astype(int)
    
    # ROC-AUC and PR-AUC require at least 1 positive and 1 negative sample
    if len(np.unique(yt)) > 1:
        roc_auc = float(roc_auc_score(yt, yp))
        pr_auc = float(average_precision_score(yt, yp))
    else:
        roc_auc = np.nan
        pr_auc = np.nan
        
    prec = float(precision_score(yt, y_pred_binary, zero_division=0))
    rec = float(recall_score(yt, y_pred_binary, zero_division=0))
    f1 = float(f1_score(yt, y_pred_binary, zero_division=0))
    brier = float(brier_score_loss(yt, yp))
    
    cm = confusion_matrix(yt, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    
    return {
        "n_samples": n_valid,
        "prevalence": prevalence,
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier": brier,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

def compute_diebold_mariano_test(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    h: int = 1
) -> Dict[str, Any]:
    """
    Computes Diebold-Mariano test statistic with Harvey-Leybourne-Newbold (HLN)
    finite-sample correction for multi-step ahead forecasts.
    Loss differential d_t = e_baseline^2 - e_model^2 (positive d_t implies model superiority).
    """
    mask = ~np.isnan(y_true) & ~np.isnan(pred_model) & ~np.isnan(pred_baseline)
    T = int(np.sum(mask))
    if T < 30:
        return {"dm_stat": np.nan, "p_value": np.nan, "significant_01": False, "T": T}
        
    yt = y_true[mask]
    e_base = (yt - pred_baseline[mask]) ** 2
    e_mod = (yt - pred_model[mask]) ** 2
    d = e_base - e_mod
    
    d_mean = np.mean(d)
    
    # Auto-covariance estimation for horizon h
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        if len(d) > k:
            gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            gamma_sum += 2.0 * gamma_k
            
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 1e-12:
        return {"dm_stat": 0.0, "p_value": 1.0, "significant_01": False, "T": T}
        
    dm = d_mean / np.sqrt(var_d)
    
    # Harvey-Leybourne-Newbold (HLN) finite-sample correction factor
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_corrected = float(dm * hln_factor)
    
    # Two-sided p-value using Student's t distribution with (T-1) degrees of freedom
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_corrected), df=T - 1)))
    significant_01 = bool(p_val < 0.01 and dm_corrected > 0)
    
    return {
        "dm_stat": dm_corrected,
        "p_value": p_val,
        "significant_01": significant_01,
        "T": T
    }
