"""
Step 4: Spatial Baselines Comparison (LOSOCV)
Re-evaluates Leave-One-Station-Out cross-validation on operational forecast columns:
  - h1:  Stacking (pred_stacking_h1)
  - h6:  Stacking (pred_stacking_h6)
  - h24: LightGBM (pred_lgb_h24)
  - h48: LightGBM (pred_lgb_h48)

Methods compared:
  1. idw_p2: Inverse Distance Weighting (p=2.0, UTM planar coordinates) using donor forecasts
  2. donor_mean: Simple arithmetic unweighted mean of 3 donor stations' forecasts
  3. station_train_mean: Held-out station's historical training-period mean (2023-2024)
  4. direct_model: Direct local ML model forecast at held-out station (benchmark)

Evaluated on:
  (a) All test rows
  (b) Common-timestamp subset (where all 4 stations have valid observations)

Reports:
  - Pooled MAE (all rows combined)
  - Macro Average MAE (mean across stations)
  - Per-station MAE and sample size n
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
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


# Station spatial coordinates in UTM Zone 44N (meters)
STATION_COORDS_UTM = {
    "Bollaram Industrial Area": np.array([218650.0, 1941800.0]),
    "Central University": np.array([216420.0, 1932950.0]),
    "ECIL Kapra": np.array([241880.0, 1933850.0]),
    "ICRISAT Patancheru": np.array([210180.0, 1938550.0])
}


def compute_idw_prediction(donor_coords: np.ndarray, donor_values: np.ndarray, target_coord: np.ndarray, power: float = 2.0) -> float:
    """Computes IDW interpolation from donor points to target coordinate."""
    diffs = donor_coords - target_coord.reshape(1, 2)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    
    # Check for coincidence
    if np.any(dists < 1e-5):
        return float(donor_values[np.argmin(dists)])
        
    weights = 1.0 / (dists ** power)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        return float(np.mean(donor_values))
    return float(np.sum(weights * donor_values) / w_sum)


def run_spatial_baselines():
    project_root = find_project_root()
    print(f"[Step 4] Using project root: {project_root}")
    
    out_dir = project_root / "analysis" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load training data to compute held-out stations' historical training mean
    train_parquet = project_root / "data" / "processed" / "feature_store_train_purged.parquet"
    print(f"[Step 4] Loading training partition from {train_parquet}")
    df_train = pd.read_parquet(train_parquet, columns=["station", "PM2.5"])
    station_train_means = df_train.groupby("station")["PM2.5"].mean().to_dict()
    print("  Historical Training Means (2023-2024):", {s: round(m, 2) for s, m in station_train_means.items()})
    
    # 2. Load test predictions
    test_parquet = project_root / "data" / "predictions" / "regression_predictions_test.parquet"
    print(f"[Step 4] Loading test predictions from {test_parquet}")
    df_test = pd.read_parquet(test_parquet)
    df_test["datetime"] = pd.to_datetime(df_test["datetime"])
    
    horizon_configs = {
        "h1": {"model": "stacking", "pred_col": "pred_stacking_h1", "target_col": "target_pm25_h1", "pers_col": "pred_persistence_h1"},
        "h6": {"model": "stacking", "pred_col": "pred_stacking_h6", "target_col": "target_pm25_h6", "pers_col": "pred_persistence_h6"},
        "h24": {"model": "lgb", "pred_col": "pred_lgb_h24", "target_col": "target_pm25_h24", "pers_col": "pred_persistence_h24"},
        "h48": {"model": "lgb", "pred_col": "pred_lgb_h48", "target_col": "target_pm25_h48", "pers_col": "pred_persistence_h48"},
    }
    
    stations = sorted(list(STATION_COORDS_UTM.keys()))
    
    # Pre-compute common timestamps from Step 1
    common_dts = {}
    for h, cfg in horizon_configs.items():
        t_col = cfg["target_col"]
        p_col = cfg["pers_col"]
        piv = df_test[(~df_test[t_col].isna()) & (~df_test[p_col].isna())].pivot(index="datetime", columns="station", values=t_col)
        common_dts[h] = set(piv.dropna().index)
        
    records = []
    summary_results = {"metadata": {"stations": stations, "horizon_configs": {h: cfg["model"] for h, cfg in horizon_configs.items()}}, "results": {}}
    
    methods = ["direct_model", "idw_p2", "donor_mean", "station_train_mean"]

    for h, cfg in horizon_configs.items():
        summary_results["results"][h] = {}
        target_col = cfg["target_col"]
        pred_col = cfg["pred_col"]
        pers_col = cfg["pers_col"]
        
        # Wide pivot tables of targets and forecasts
        pivot_target = df_test.pivot(index="datetime", columns="station", values=target_col)
        pivot_pred = df_test.pivot(index="datetime", columns="station", values=pred_col)
        pivot_pers = df_test.pivot(index="datetime", columns="station", values=pers_col)
        
        datetimes_all = sorted(pivot_target.index)
        
        for subset_name, dts_allowed in [("all_rows", None), ("common_timestamps", common_dts[h])]:
            summary_results["results"][h][subset_name] = {}
            print(f"\n--- Horizon {h} ({cfg['model']}) | Subset: {subset_name} ---")
            
            # Store predictions across all stations for pooling
            pooled_y_true = {m: [] for m in methods}
            pooled_y_pred = {m: [] for m in methods}
            
            station_maes = {m: {} for m in methods}
            station_ns = {m: {} for m in methods}
            
            for held_out_st in stations:
                donor_sts = [s for s in stations if s != held_out_st]
                target_coord = STATION_COORDS_UTM[held_out_st]
                donor_coords_arr = np.array([STATION_COORDS_UTM[s] for s in donor_sts])
                hist_mean = station_train_means[held_out_st]
                
                y_true_list = []
                p_direct_list = []
                p_idw_list = []
                p_donor_mean_list = []
                p_train_mean_list = []
                
                for dt in datetimes_all:
                    if dts_allowed is not None and dt not in dts_allowed:
                        continue
                        
                    y_val = pivot_target.loc[dt, held_out_st]
                    pers_val = pivot_pers.loc[dt, held_out_st]
                    
                    # Persistence-valid primary convention
                    if np.isnan(y_val) or np.isnan(pers_val):
                        continue
                        
                    donor_forecasts = np.array([pivot_pred.loc[dt, s] for s in donor_sts])
                    valid_donors = ~np.isnan(donor_forecasts)
                    if np.sum(valid_donors) == 0:
                        continue
                        
                    v_coords = donor_coords_arr[valid_donors]
                    v_vals = donor_forecasts[valid_donors]
                    
                    # Compute spatial predictions
                    pred_idw = compute_idw_prediction(v_coords, v_vals, target_coord, power=2.0)
                    pred_mean = float(np.mean(v_vals))
                    pred_direct = pivot_pred.loc[dt, held_out_st]
                    
                    y_true_list.append(float(y_val))
                    p_direct_list.append(float(pred_direct))
                    p_idw_list.append(pred_idw)
                    p_donor_mean_list.append(pred_mean)
                    p_train_mean_list.append(hist_mean)
                    
                n_st = len(y_true_list)
                yt = np.array(y_true_list)
                
                st_preds = {
                    "direct_model": np.array(p_direct_list),
                    "idw_p2": np.array(p_idw_list),
                    "donor_mean": np.array(p_donor_mean_list),
                    "station_train_mean": np.array(p_train_mean_list)
                }
                
                for m in methods:
                    mae_m = float(mean_absolute_error(yt, st_preds[m])) if n_st > 0 else np.nan
                    station_maes[m][held_out_st] = mae_m
                    station_ns[m][held_out_st] = n_st
                    
                    pooled_y_true[m].extend(y_true_list)
                    pooled_y_pred[m].extend(st_preds[m].tolist())
                    
                    records.append({
                        "horizon": h,
                        "model_used": cfg["model"],
                        "subset": subset_name,
                        "method": m,
                        "station": held_out_st,
                        "n_samples": n_st,
                        "mae": mae_m,
                        "aggregation": "station_level"
                    })
                    
                print(f"  {held_out_st:<25} (n={n_st}): Direct={station_maes['direct_model'][held_out_st]:5.2f} | IDW(p=2)={station_maes['idw_p2'][held_out_st]:5.2f} | DonorMean={station_maes['donor_mean'][held_out_st]:5.2f} | TrainMean={station_maes['station_train_mean'][held_out_st]:5.2f}")

            # Compute Pooled MAE and Macro Average across stations
            summary_results["results"][h][subset_name]["by_station"] = station_maes
            summary_results["results"][h][subset_name]["macro_average"] = {}
            summary_results["results"][h][subset_name]["pooled_mae"] = {}
            summary_results["results"][h][subset_name]["n_stations"] = len(stations)
            
            print(f"  --- {h} {subset_name} SUMMARY ---")
            for m in methods:
                macro_mae = float(np.mean([station_maes[m][st] for st in stations]))
                pooled_mae = float(mean_absolute_error(np.array(pooled_y_true[m]), np.array(pooled_y_pred[m])))
                total_n = len(pooled_y_true[m])
                
                summary_results["results"][h][subset_name]["macro_average"][m] = macro_mae
                summary_results["results"][h][subset_name]["pooled_mae"][m] = pooled_mae
                
                records.append({
                    "horizon": h,
                    "model_used": cfg["model"],
                    "subset": subset_name,
                    "method": m,
                    "station": "MACRO_AVERAGE",
                    "n_samples": total_n,
                    "mae": macro_mae,
                    "aggregation": "macro_average"
                })
                records.append({
                    "horizon": h,
                    "model_used": cfg["model"],
                    "subset": subset_name,
                    "method": m,
                    "station": "POOLED_ALL_STATIONS",
                    "n_samples": total_n,
                    "mae": pooled_mae,
                    "aggregation": "pooled_all_rows"
                })
                
                print(f"    {m:<20}: Macro MAE = {macro_mae:5.2f} ug/m3 | Pooled MAE = {pooled_mae:5.2f} ug/m3 (total n={total_n})")

    df_out = pd.DataFrame(records)
    csv_path = out_dir / "spatial_baselines.csv"
    json_path = out_dir / "spatial_baselines.json"
    
    df_out.to_csv(csv_path, index=False)
    print(f"\n[Step 4] Saved spatial baselines CSV ({len(df_out)} rows) to {csv_path}")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"[Step 4] Saved spatial baselines JSON to {json_path}")

    # Mirror
    alt_out_dirs = [
        Path("C:/Users/harsha madala/Documents/Hyderabad_Air_Research/new_research_project/analysis/outputs"),
        Path("E:/new_research_project_BACKUP/new_research_project/analysis/outputs")
    ]
    for alt_dir in alt_out_dirs:
        try:
            alt_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(alt_dir / "spatial_baselines.csv", index=False)
            with open(alt_dir / "spatial_baselines.json", "w", encoding="utf-8") as f:
                json.dump(summary_results, f, indent=2)
            print(f"[Step 4] Mirrored outputs to {alt_dir}")
        except Exception as e:
            print(f"[Step 4] Note: Mirror failed for {alt_dir}: {e}")

    return df_out, summary_results


if __name__ == "__main__":
    run_spatial_baselines()
