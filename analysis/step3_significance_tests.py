"""
Step 3: Statistical Significance Testing
Computes:
1. Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) finite-sample correction,
   HAC autocovariance truncation at lag h-1, squared-error loss.
2. 24-hour moving-block bootstrap (2,000 resamples, seed 42) 95% CI for MAE differences.
3. Holm-Bonferroni family-wise error adjusted p-values.

Evaluated on:
  - Pooled test rows (all station-hour rows combined)
  - Network-averaged hourly series (hourly mean across reporting stations)
  - Station-specific series

Model pairs:
  - stacking vs persistence
  - stacking vs xgb
  - stacking vs lgb
  - lgb vs persistence
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error


def find_project_root() -> Path:
    """Find the root directory of the project."""
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "new_research_project",
        cwd.parent,
        cwd.parent / "new_research_project",
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent / "new_research_project",
    ]
    for c in candidates:
        if (c / "data" / "predictions" / "regression_predictions_test.parquet").exists():
            return c
    raise FileNotFoundError("Could not locate project root containing test predictions parquet.")


def holm_adjust_pvalues(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down adjustment to p-values."""
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m <= 1:
        return [float(p) for p in p_arr]
        
    order = np.argsort(p_arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(m)
    
    sorted_p = p_arr[order]
    adj = np.zeros(m)
    cum_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        cum_max = max(cum_max, val)
        adj[i] = min(1.0, cum_max)
    return [float(x) for x in adj[ranks]]


def compute_hln_diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int = 1
) -> Dict[str, Any]:
    """
    Computes Diebold-Mariano test statistic with Harvey-Leybourne-Newbold (HLN)
    finite-sample correction and HAC autocovariance truncation at lag h-1.
    
    Loss differential:
      d_t = e_b(t)^2 - e_a(t)^2
    Positive d_t indicates that model A has lower squared-error loss than model B.
    """
    mask = (~np.isnan(y_true)) & (~np.isnan(pred_a)) & (~np.isnan(pred_b))
    T = int(np.sum(mask))
    if T < 30:
        return {"dm_stat": np.nan, "p_value": np.nan, "T": T, "d_mean": np.nan}
        
    yt = y_true[mask]
    pa = pred_a[mask]
    pb = pred_b[mask]
    
    e_b_sq = (yt - pb) ** 2
    e_a_sq = (yt - pa) ** 2
    d = e_b_sq - e_a_sq
    d_mean = float(np.mean(d))
    
    # Autocovariances up to lag h-1
    gamma_0 = float(np.var(d, ddof=0))
    gamma_sum = 0.0
    for k in range(1, h):
        if len(d) > k:
            gamma_k = float(np.mean((d[k:] - d_mean) * (d[:-k] - d_mean)))
            gamma_sum += 2.0 * gamma_k
            
    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 1e-12:
        return {"dm_stat": 0.0, "p_value": 1.0, "T": T, "d_mean": d_mean}
        
    dm_raw = d_mean / np.sqrt(var_d)
    
    # Harvey-Leybourne-Newbold correction factor
    hln_factor = np.sqrt(max(1e-12, (T + 1 - 2 * h + h * (h - 1) / T) / T))
    dm_corrected = float(dm_raw * hln_factor)
    
    # Two-sided p-value from Student's t distribution with (T-1) degrees of freedom
    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(dm_corrected), df=T - 1)))
    
    return {
        "dm_stat": dm_corrected,
        "p_value": p_val,
        "T": T,
        "d_mean": d_mean
    }


def compute_series_moving_block_bootstrap_ci(
    y: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    block_length: int = 24,
    n_resamples: int = 2000,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    24-hour moving-block bootstrap on univariate series (e.g. station series or network mean).
    Delta_MAE = MAE(A) - MAE(B)
    """
    N = len(y)
    if N < block_length:
        return np.nan, np.nan, np.nan
        
    valid = (~np.isnan(y)) & (~np.isnan(pa)) & (~np.isnan(pb))
    if np.sum(valid) == 0:
        return np.nan, np.nan, np.nan
        
    obs_diff = float(mean_absolute_error(y[valid], pa[valid]) - mean_absolute_error(y[valid], pb[valid]))
    
    rng = np.random.RandomState(seed)
    n_blocks = int(np.ceil(N / block_length))
    max_start = N - block_length + 1
    
    start_indices = rng.randint(0, max_start, size=(n_resamples, n_blocks))
    block_offsets = np.arange(block_length)
    full_indices = (start_indices[:, :, None] + block_offsets[None, None, :]).reshape(n_resamples, -1)[:, :N]
    
    err_a = np.abs(y - pa)
    err_b = np.abs(y - pb)
    
    sample_err_a = err_a[full_indices]
    sample_err_b = err_b[full_indices]
    
    mae_a = np.nanmean(sample_err_a, axis=1)
    mae_b = np.nanmean(sample_err_b, axis=1)
    boot_diffs = mae_a - mae_b
    
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return obs_diff, ci_lower, ci_upper


def run_significance_tests():
    project_root = find_project_root()
    print(f"[Step 3] Using project root: {project_root}")
    
    out_dir = project_root / "analysis" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_parquet = project_root / "data" / "predictions" / "regression_predictions_test.parquet"
    print(f"[Step 3] Loading predictions from {test_parquet}")
    df = pd.read_parquet(test_parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    
    horizons = ["h1", "h6", "h24", "h48"]
    horizon_steps = {"h1": 1, "h6": 6, "h24": 24, "h48": 48}
    
    pairs = [
        ("stacking", "persistence", "pred_stacking_{h}", "pred_persistence_{h}"),
        ("stacking", "xgb", "pred_stacking_{h}", "pred_xgb_{h}"),
        ("stacking", "lgb", "pred_stacking_{h}", "pred_lgb_{h}"),
        ("lgb", "persistence", "pred_lgb_{h}", "pred_persistence_{h}")
    ]
    
    stations = sorted(df["station"].unique())
    all_records = []

    # Prepare common calendar timeline for 24h moving block bootstrap
    dts = np.sort(df["datetime"].unique())
    T_hours = len(dts)
    block_len = 24
    n_blocks = int(np.ceil(T_hours / block_len))
    max_start = T_hours - block_len + 1
    rng_pooled = np.random.RandomState(42)
    start_indices_pooled = rng_pooled.randint(0, max_start, size=(2000, n_blocks))
    block_offsets_pooled = np.arange(block_len)
    full_indices_pooled = (start_indices_pooled[:, :, None] + block_offsets_pooled[None, None, :]).reshape(2000, -1)[:, :T_hours]

    # -------------------------------------------------------------
    # 1. POOLED TEST ROWS SCOPE
    # -------------------------------------------------------------
    print("\n--- Evaluating Scope: Pooled Test Rows ---")
    pooled_records = []
    for h in horizons:
        h_int = horizon_steps[h]
        target_col = f"target_pm25_{h}"
        pers_col = f"pred_persistence_{h}"
        
        # Primary persistence mask
        pers_mask = (~df[target_col].isna()) & (~df[pers_col].isna())
        df_prim = df[pers_mask]
        
        piv_y = df.pivot(index="datetime", columns="station", values=target_col).values
        piv_pers = df.pivot(index="datetime", columns="station", values=pers_col).values
        mask_wide = (~np.isnan(piv_y)) & (~np.isnan(piv_pers))
        
        for mod_a, mod_b, pat_a, pat_b in pairs:
            col_a = pat_a.format(h=h)
            col_b = pat_b.format(h=h)
            
            # Squared-error Diebold-Mariano on pooled rows
            dm_res = compute_hln_diebold_mariano(
                df_prim[target_col].values,
                df_prim[col_a].values,
                df_prim[col_b].values,
                h=h_int
            )
            
            # 24-hour moving block bootstrap on pooled multi-station panel
            piv_a = df.pivot(index="datetime", columns="station", values=col_a).values
            piv_b = df.pivot(index="datetime", columns="station", values=col_b).values
            
            err_a = np.where(mask_wide, np.abs(piv_y - piv_a), np.nan)
            err_b = np.where(mask_wide, np.abs(piv_y - piv_b), np.nan)
            
            obs_diff = float(np.nanmean(err_a) - np.nanmean(err_b))
            
            sample_err_a = err_a[full_indices_pooled]
            sample_err_b = err_b[full_indices_pooled]
            
            mae_a = np.nanmean(sample_err_a, axis=(1, 2))
            mae_b = np.nanmean(sample_err_b, axis=(1, 2))
            boot_diffs = mae_a - mae_b
            
            ci_low = float(np.percentile(boot_diffs, 2.5))
            ci_high = float(np.percentile(boot_diffs, 97.5))
            
            record = {
                "scope": "Pooled Test Rows",
                "horizon": h,
                "model_pair": f"{mod_a} vs {mod_b}",
                "model_a": mod_a,
                "model_b": mod_b,
                "n_samples": dm_res["T"],
                "dm_stat": dm_res["dm_stat"],
                "p_value_raw": dm_res["p_value"],
                "mae_diff_obs": obs_diff,
                "bootstrap_ci_95_lower": ci_low,
                "bootstrap_ci_95_upper": ci_high,
                "ci_str": f"[{ci_low:+.4f}, {ci_high:+.4f}]"
            }
            pooled_records.append(record)
            
    # Apply Holm adjustment per horizon (across 4 pairs) and across all tests in scope
    for h in horizons:
        h_rows = [r for r in pooled_records if r["horizon"] == h]
        h_adj = holm_adjust_pvalues([r["p_value_raw"] for r in h_rows])
        for r, padj in zip(h_rows, h_adj):
            r["p_value_holm"] = padj
            print(f"  {h} | {r['model_pair']:<24} | DM={r['dm_stat']:+6.3f} (p_raw={r['p_value_raw']:.4f}, p_holm={r['p_value_holm']:.4f}) | dMAE={r['mae_diff_obs']:+6.4f} 95%CI={r['ci_str']}")
            
    all_records.extend(pooled_records)

    # -------------------------------------------------------------
    # 2. NETWORK-AVERAGED HOURLY SERIES SCOPE
    # -------------------------------------------------------------
    print("\n--- Evaluating Scope: Network-Averaged Hourly Series ---")
    net_records = []
    for h in horizons:
        h_int = horizon_steps[h]
        target_col = f"target_pm25_{h}"
        pers_col = f"pred_persistence_{h}"
        
        pers_mask = (~df[target_col].isna()) & (~df[pers_col].isna())
        df_valid = df[pers_mask]
        
        agg_cols = [target_col] + [p[2].format(h=h) for p in pairs] + [p[3].format(h=h) for p in pairs]
        agg_cols = list(dict.fromkeys(agg_cols))
        
        df_net = df_valid.groupby("datetime")[agg_cols].mean().reset_index()
        df_net = df_net.sort_values("datetime").reset_index(drop=True)
        
        for mod_a, mod_b, pat_a, pat_b in pairs:
            col_a = pat_a.format(h=h)
            col_b = pat_b.format(h=h)
            
            dm_res = compute_hln_diebold_mariano(
                df_net[target_col].values,
                df_net[col_a].values,
                df_net[col_b].values,
                h=h_int
            )
            
            diff_obs, ci_low, ci_high = compute_series_moving_block_bootstrap_ci(
                df_net[target_col].values,
                df_net[col_a].values,
                df_net[col_b].values,
                block_length=24,
                n_resamples=2000,
                seed=42
            )
            
            record = {
                "scope": "Network-Averaged Series",
                "horizon": h,
                "model_pair": f"{mod_a} vs {mod_b}",
                "model_a": mod_a,
                "model_b": mod_b,
                "n_samples": dm_res["T"],
                "dm_stat": dm_res["dm_stat"],
                "p_value_raw": dm_res["p_value"],
                "mae_diff_obs": diff_obs,
                "bootstrap_ci_95_lower": ci_low,
                "bootstrap_ci_95_upper": ci_high,
                "ci_str": f"[{ci_low:+.4f}, {ci_high:+.4f}]"
            }
            net_records.append(record)
            
    for h in horizons:
        h_rows = [r for r in net_records if r["horizon"] == h]
        h_adj = holm_adjust_pvalues([r["p_value_raw"] for r in h_rows])
        for r, padj in zip(h_rows, h_adj):
            r["p_value_holm"] = padj
            print(f"  {h} | {r['model_pair']:<24} | DM={r['dm_stat']:+6.3f} (p_raw={r['p_value_raw']:.4f}, p_holm={r['p_value_holm']:.4f}) | dMAE={r['mae_diff_obs']:+6.4f} 95%CI={r['ci_str']}")
            
    all_records.extend(net_records)

    # -------------------------------------------------------------
    # 3. STATION-SPECIFIC SCOPES
    # -------------------------------------------------------------
    for st in stations:
        df_st = df[df["station"] == st].sort_values("datetime").reset_index(drop=True)
        st_records = []
        for h in horizons:
            h_int = horizon_steps[h]
            target_col = f"target_pm25_{h}"
            pers_col = f"pred_persistence_{h}"
            
            pers_mask = (~df_st[target_col].isna()) & (~df_st[pers_col].isna())
            df_pair = df_st[pers_mask].copy()
            
            for mod_a, mod_b, pat_a, pat_b in pairs:
                col_a = pat_a.format(h=h)
                col_b = pat_b.format(h=h)
                
                dm_res = compute_hln_diebold_mariano(
                    df_pair[target_col].values,
                    df_pair[col_a].values,
                    df_pair[col_b].values,
                    h=h_int
                )
                
                diff_obs, ci_low, ci_high = compute_series_moving_block_bootstrap_ci(
                    df_pair[target_col].values,
                    df_pair[col_a].values,
                    df_pair[col_b].values,
                    block_length=24,
                    n_resamples=2000,
                    seed=42
                )
                
                record = {
                    "scope": f"Station: {st}",
                    "horizon": h,
                    "model_pair": f"{mod_a} vs {mod_b}",
                    "model_a": mod_a,
                    "model_b": mod_b,
                    "n_samples": dm_res["T"],
                    "dm_stat": dm_res["dm_stat"],
                    "p_value_raw": dm_res["p_value"],
                    "mae_diff_obs": diff_obs,
                    "bootstrap_ci_95_lower": ci_low,
                    "bootstrap_ci_95_upper": ci_high,
                    "ci_str": f"[{ci_low:+.4f}, {ci_high:+.4f}]"
                }
                st_records.append(record)
                
        for h in horizons:
            h_rows = [r for r in st_records if r["horizon"] == h]
            h_adj = holm_adjust_pvalues([r["p_value_raw"] for r in h_rows])
            for r, padj in zip(h_rows, h_adj):
                r["p_value_holm"] = padj
        all_records.extend(st_records)

    df_out = pd.DataFrame(all_records)
    csv_path = out_dir / "significance_tests.csv"
    json_path = out_dir / "significance_tests.json"
    
    df_out.to_csv(csv_path, index=False)
    print(f"\n[Step 3] Saved updated significance tests CSV ({len(df_out)} rows) to {csv_path}")
    
    structured_json = {
        "metadata": {
            "methodology": "Diebold-Mariano test with Harvey-Leybourne-Newbold (HLN) finite-sample correction and HAC autocovariance truncation at lag h-1 (squared-error loss); 24-hour moving-block bootstrap (2000 resamples, seed 42) for MAE difference 95% CI; Holm-Bonferroni step-down adjusted p-values.",
            "scopes": ["Pooled Test Rows", "Network-Averaged Series", "Station Specific"],
            "pairs": [p[0] + " vs " + p[1] for p in pairs],
            "horizons": horizons
        },
        "tests": all_records
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(structured_json, f, indent=2)
    print(f"[Step 3] Saved updated significance tests JSON to {json_path}")

    # Mirror
    alt_out_dirs = [
        Path("C:/Users/harsha madala/Documents/Hyderabad_Air_Research/new_research_project/analysis/outputs"),
        Path("E:/new_research_project_BACKUP/new_research_project/analysis/outputs")
    ]
    for alt_dir in alt_out_dirs:
        try:
            alt_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(alt_dir / "significance_tests.csv", index=False)
            with open(alt_dir / "significance_tests.json", "w", encoding="utf-8") as f:
                json.dump(structured_json, f, indent=2)
            print(f"[Step 3] Mirrored outputs to {alt_dir}")
        except Exception as e:
            print(f"[Step 3] Note: Mirror failed for {alt_dir}: {e}")

    return df_out


if __name__ == "__main__":
    run_significance_tests()
