"""
Step 1: Bollaram-Safe Evaluation
Evaluates multi-horizon PM2.5 forecasts under:
(a) Full test set (asserting consistency with existing JSON metrics)
(b) Common timestamps where all 4 stations have valid data
(c) Station-by-station breakdown
(d) Month-by-month breakdown

CONVENTION:
Primary evaluation is the 'persistence-valid mask' (rows where both target and persistence prediction are non-NaN).
Also reports the 'all valid rows' version for complete comparison.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute MAE, RMSE, R2, bias, and n for valid pairs."""
    valid = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    n = int(np.sum(valid))
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan}
    
    yt = y_true[valid]
    yp = y_pred[valid]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    if np.var(yt) > 1e-12:
        r2 = float(r2_score(yt, yp))
    else:
        r2 = np.nan
    bias = float(np.mean(yp - yt))
    return {
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": bias
    }


def run_bollaram_safe_evaluation():
    project_root = find_project_root()
    print(f"[Step 1] Using project root: {project_root}")
    
    # Define outputs directory
    out_dir = project_root / "analysis" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test parquet
    test_parquet = project_root / "data" / "predictions" / "regression_predictions_test.parquet"
    print(f"[Step 1] Loading predictions from {test_parquet}")
    df = pd.read_parquet(test_parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    
    # Load existing reference metrics
    reg_metrics_path = project_root / "data" / "predictions" / "regression_metrics.json"
    stk_metrics_path = project_root / "data" / "predictions" / "stacking_regression_metrics.json"
    
    with open(reg_metrics_path, "r", encoding="utf-8") as f:
        existing_reg = json.load(f)["test"]
    with open(stk_metrics_path, "r", encoding="utf-8") as f:
        existing_stk = json.load(f)["test"]
        
    horizons = ["h1", "h6", "h24", "h48"]
    model_col_map = {
        "persistence": "pred_persistence_{h}",
        "seasonal_naive": "pred_seasonal_naive_{h}",
        "rf": "pred_rf_{h}",
        "xgb": "pred_xgb_{h}",
        "lgb": "pred_lgb_{h}",
        "stacking": "pred_stacking_{h}"
    }
    
    existing_model_key_map = {
        "persistence": "persistence",
        "seasonal_naive": "seasonal_naive",
        "rf": "random_forest",
        "xgb": "xgboost",
        "lgb": "lightgbm"
    }

    results: Dict[str, Any] = {
        "metadata": {
            "evaluation_convention_primary": "persistence-valid mask: rows where both target and persistence prediction are non-NaN",
            "evaluation_convention_secondary": "all valid rows: rows where target and model prediction are both non-NaN",
            "total_test_rows": len(df),
            "stations": sorted(df["station"].unique().tolist()),
            "horizons": horizons,
            "models": list(model_col_map.keys())
        },
        "full_test_set": {"primary_mask": {}, "all_valid_rows": {}},
        "all_4_stations_valid": {"primary_mask": {}, "all_valid_rows": {}},
        "by_station": {"primary_mask": {}, "all_valid_rows": {}},
        "by_month": {"primary_mask": {}, "all_valid_rows": {}}
    }
    
    csv_records = []
    
    print("\n--- Evaluating (a) Full Test Set and Asserting Matches ---")
    for h in horizons:
        results["full_test_set"]["primary_mask"][h] = {}
        results["full_test_set"]["all_valid_rows"][h] = {}
        
        target_col = f"target_pm25_{h}"
        pers_col = f"pred_persistence_{h}"
        
        # Primary persistence-valid mask
        pers_mask = (~df[target_col].isna()) & (~df[pers_col].isna())
        
        for model, col_pattern in model_col_map.items():
            pred_col = col_pattern.format(h=h)
            
            # 1. Primary mask evaluation
            y_true_prim = df.loc[pers_mask, target_col].values
            y_pred_prim = df.loc[pers_mask, pred_col].values
            m_prim = compute_metrics(y_true_prim, y_pred_prim)
            results["full_test_set"]["primary_mask"][h][model] = m_prim
            
            # Reference metrics from existing JSON files
            if model == "stacking":
                ref = existing_stk[h]
            else:
                ref = existing_reg[h][existing_model_key_map[model]]
            
            # Assert match with existing JSON metrics:
            # - persistence, rf, xgb, lgb, stacking match exactly on the primary persistence-valid mask.
            # - seasonal_naive in existing JSON was evaluated on its own all_valid_rows.
            if model != "seasonal_naive":
                np.testing.assert_allclose(
                    m_prim["mae"], ref["mae"], rtol=1e-3, atol=1e-3,
                    err_msg=f"MAE mismatch for {model} at {h}: computed {m_prim['mae']}, ref {ref['mae']}"
                )
                np.testing.assert_allclose(
                    m_prim["rmse"], ref["rmse"], rtol=1e-3, atol=1e-3,
                    err_msg=f"RMSE mismatch for {model} at {h}: computed {m_prim['rmse']}, ref {ref['rmse']}"
                )
                np.testing.assert_allclose(
                    m_prim["r2"], ref["r2"], rtol=1e-3, atol=1e-3,
                    err_msg=f"R2 mismatch for {model} at {h}: computed {m_prim['r2']}, ref {ref['r2']}"
                )
                np.testing.assert_allclose(
                    m_prim["bias"], ref["bias"], rtol=1e-3, atol=1e-3,
                    err_msg=f"Bias mismatch for {model} at {h}: computed {m_prim['bias']}, ref {ref['bias']}"
                )
                assert m_prim["n"] == ref["n_samples"], f"n mismatch for {model} at {h}: {m_prim['n']} vs {ref['n_samples']}"
                print(f"  [PASS primary mask] {h} {model:<15} MAE={m_prim['mae']:.4f} RMSE={m_prim['rmse']:.4f} R2={m_prim['r2']:.4f} Bias={m_prim['bias']:.4f} n={m_prim['n']}")
            else:
                print(f"  [INFO primary mask] {h} {model:<15} MAE={m_prim['mae']:.4f} RMSE={m_prim['rmse']:.4f} R2={m_prim['r2']:.4f} Bias={m_prim['bias']:.4f} n={m_prim['n']}")
            
            csv_records.append({
                "subset": "full_test_set",
                "subset_detail": "all_stations_all_months",
                "mask_type": "primary_persistence_valid",
                "horizon": h,
                "model": model,
                **m_prim
            })
            
            # 2. All valid rows evaluation
            all_valid_mask = (~df[target_col].isna()) & (~df[pred_col].isna())
            y_true_all = df.loc[all_valid_mask, target_col].values
            y_pred_all = df.loc[all_valid_mask, pred_col].values
            m_all = compute_metrics(y_true_all, y_pred_all)
            results["full_test_set"]["all_valid_rows"][h][model] = m_all
            
            if model == "seasonal_naive":
                np.testing.assert_allclose(
                    m_all["mae"], ref["mae"], rtol=1e-3, atol=1e-3,
                    err_msg=f"MAE mismatch for seasonal_naive at {h} (all_valid): computed {m_all['mae']}, ref {ref['mae']}"
                )
                assert m_all["n"] == ref["n_samples"], f"n mismatch for seasonal_naive at {h}: {m_all['n']} vs {ref['n_samples']}"
                print(f"  [PASS all-valid mask] {h} {model:<15} MAE={m_all['mae']:.4f} RMSE={m_all['rmse']:.4f} R2={m_all['r2']:.4f} Bias={m_all['bias']:.4f} n={m_all['n']}")
            
            csv_records.append({
                "subset": "full_test_set",
                "subset_detail": "all_stations_all_months",
                "mask_type": "all_valid_rows",
                "horizon": h,
                "model": model,
                **m_all
            })

    print("\n--- Evaluating (b) Only Timestamps Where All 4 Stations Have Valid Data ---")
    for h in horizons:
        results["all_4_stations_valid"]["primary_mask"][h] = {}
        results["all_4_stations_valid"]["all_valid_rows"][h] = {}
        
        target_col = f"target_pm25_{h}"
        pers_col = f"pred_persistence_{h}"
        
        # Primary persistence-valid mask timestamps
        pers_mask = (~df[target_col].isna()) & (~df[pers_col].isna())
        piv_prim = df[pers_mask].pivot(index="datetime", columns="station", values=target_col)
        valid_dts_prim = set(piv_prim.dropna().index)
        
        # All valid target timestamps
        piv_all = df[~df[target_col].isna()].pivot(index="datetime", columns="station", values=target_col)
        valid_dts_all = set(piv_all.dropna().index)
        
        print(f"  Horizon {h}: {len(valid_dts_prim)} common timestamps (primary mask), {len(valid_dts_all)} (all valid)")
        
        for model, col_pattern in model_col_map.items():
            pred_col = col_pattern.format(h=h)
            
            # Primary mask subset
            sub_prim = df[pers_mask & df["datetime"].isin(valid_dts_prim)]
            m_prim = compute_metrics(sub_prim[target_col].values, sub_prim[pred_col].values)
            results["all_4_stations_valid"]["primary_mask"][h][model] = m_prim
            csv_records.append({
                "subset": "all_4_stations_valid",
                "subset_detail": f"common_timestamps_n_{len(valid_dts_prim)}",
                "mask_type": "primary_persistence_valid",
                "horizon": h,
                "model": model,
                **m_prim
            })
            
            # All valid rows subset
            sub_all = df[(~df[target_col].isna()) & (~df[pred_col].isna()) & df["datetime"].isin(valid_dts_all)]
            m_all = compute_metrics(sub_all[target_col].values, sub_all[pred_col].values)
            results["all_4_stations_valid"]["all_valid_rows"][h][model] = m_all
            csv_records.append({
                "subset": "all_4_stations_valid",
                "subset_detail": f"common_timestamps_n_{len(valid_dts_all)}",
                "mask_type": "all_valid_rows",
                "horizon": h,
                "model": model,
                **m_all
            })

    print("\n--- Evaluating (c) Each Station Separately ---")
    stations = sorted(df["station"].unique())
    for st in stations:
        results["by_station"]["primary_mask"][st] = {}
        results["by_station"]["all_valid_rows"][st] = {}
        df_st = df[df["station"] == st]
        
        for h in horizons:
            results["by_station"]["primary_mask"][st][h] = {}
            results["by_station"]["all_valid_rows"][st][h] = {}
            
            target_col = f"target_pm25_{h}"
            pers_col = f"pred_persistence_{h}"
            pers_mask = (~df_st[target_col].isna()) & (~df_st[pers_col].isna())
            
            for model, col_pattern in model_col_map.items():
                pred_col = col_pattern.format(h=h)
                
                # Primary
                m_prim = compute_metrics(df_st.loc[pers_mask, target_col].values, df_st.loc[pers_mask, pred_col].values)
                results["by_station"]["primary_mask"][st][h][model] = m_prim
                csv_records.append({
                    "subset": "by_station",
                    "subset_detail": st,
                    "mask_type": "primary_persistence_valid",
                    "horizon": h,
                    "model": model,
                    **m_prim
                })
                
                # All valid
                all_mask = (~df_st[target_col].isna()) & (~df_st[pred_col].isna())
                m_all = compute_metrics(df_st.loc[all_mask, target_col].values, df_st.loc[all_mask, pred_col].values)
                results["by_station"]["all_valid_rows"][st][h][model] = m_all
                csv_records.append({
                    "subset": "by_station",
                    "subset_detail": st,
                    "mask_type": "all_valid_rows",
                    "horizon": h,
                    "model": model,
                    **m_all
                })

    print("\n--- Evaluating (d) Each Calendar Month Separately ---")
    months = sorted(df["month"].unique())
    for mo in months:
        mo_name = pd.to_datetime(f"2026-{mo:02d}-01").strftime("%b_2026")
        results["by_month"]["primary_mask"][mo_name] = {}
        results["by_month"]["all_valid_rows"][mo_name] = {}
        df_mo = df[df["month"] == mo]
        
        for h in horizons:
            results["by_month"]["primary_mask"][mo_name][h] = {}
            results["by_month"]["all_valid_rows"][mo_name][h] = {}
            
            target_col = f"target_pm25_{h}"
            pers_col = f"pred_persistence_{h}"
            pers_mask = (~df_mo[target_col].isna()) & (~df_mo[pers_col].isna())
            
            for model, col_pattern in model_col_map.items():
                pred_col = col_pattern.format(h=h)
                
                # Primary
                m_prim = compute_metrics(df_mo.loc[pers_mask, target_col].values, df_mo.loc[pers_mask, pred_col].values)
                results["by_month"]["primary_mask"][mo_name][h][model] = m_prim
                csv_records.append({
                    "subset": "by_month",
                    "subset_detail": mo_name,
                    "mask_type": "primary_persistence_valid",
                    "horizon": h,
                    "model": model,
                    **m_prim
                })
                
                # All valid
                all_mask = (~df_mo[target_col].isna()) & (~df_mo[pred_col].isna())
                m_all = compute_metrics(df_mo.loc[all_mask, target_col].values, df_mo.loc[all_mask, pred_col].values)
                results["by_month"]["all_valid_rows"][mo_name][h][model] = m_all
                csv_records.append({
                    "subset": "by_month",
                    "subset_detail": mo_name,
                    "mask_type": "all_valid_rows",
                    "horizon": h,
                    "model": model,
                    **m_all
                })

    # Save outputs
    json_path = out_dir / "bollaram_safe_metrics.json"
    csv_path = out_dir / "bollaram_safe_metrics.csv"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Step 1] Saved JSON metrics to {json_path}")
    
    df_csv = pd.DataFrame(csv_records)
    df_csv.to_csv(csv_path, index=False)
    print(f"[Step 1] Saved CSV metrics table ({len(df_csv)} rows) to {csv_path}")

    # Also mirror to both working trees if applicable
    alt_out_dirs = [
        Path("C:/Users/harsha madala/Documents/Hyderabad_Air_Research/new_research_project/analysis/outputs"),
        Path("E:/new_research_project_BACKUP/new_research_project/analysis/outputs")
    ]
    for alt_dir in alt_out_dirs:
        try:
            alt_dir.mkdir(parents=True, exist_ok=True)
            with open(alt_dir / "bollaram_safe_metrics.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            df_csv.to_csv(alt_dir / "bollaram_safe_metrics.csv", index=False)
            print(f"[Step 1] Mirrored output to {alt_dir}")
        except Exception as e:
            print(f"[Step 1] Note: Could not mirror to {alt_dir}: {e}")

    return results, df_csv


if __name__ == "__main__":
    run_bollaram_safe_evaluation()
