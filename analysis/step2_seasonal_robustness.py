"""
Step 2: Seasonal Robustness Analysis
Computes MAE and Skill Score versus Persistence:
  Skill = 1 - (MAE_model / MAE_persistence)
by calendar quarter in the 2026 test partition:
  - Q1: Jan - Mar 2026 (Winter / Pre-Monsoon transition)
  - Q2: Apr - Jun 2026 (Summer / Monsoon onset)
  - Q3: Jul - Sep 2026 (Monsoon / Post-Monsoon)

Saves:
  - outputs/seasonal_robustness.csv
  - outputs/seasonal_robustness.png
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def run_seasonal_robustness():
    project_root = find_project_root()
    print(f"[Step 2] Using project root: {project_root}")
    
    out_dir = project_root / "analysis" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_parquet = project_root / "data" / "predictions" / "regression_predictions_test.parquet"
    print(f"[Step 2] Loading predictions from {test_parquet}")
    df = pd.read_parquet(test_parquet)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["month"] = df["datetime"].dt.month
    
    # Assign quarters
    def get_quarter(m):
        if m in [1, 2, 3]:
            return "Q1 (Jan-Mar)"
        elif m in [4, 5, 6]:
            return "Q2 (Apr-Jun)"
        elif m in [7, 8, 9]:
            return "Q3 (Jul-Sep)"
        else:
            return "Other"
            
    df["quarter"] = df["month"].apply(get_quarter)
    quarters = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)"]
    horizons = ["h1", "h6", "h24", "h48"]
    models = ["persistence", "seasonal_naive", "rf", "xgb", "lgb", "stacking"]
    
    model_col_map = {
        "persistence": "pred_persistence_{h}",
        "seasonal_naive": "pred_seasonal_naive_{h}",
        "rf": "pred_rf_{h}",
        "xgb": "pred_xgb_{h}",
        "lgb": "pred_lgb_{h}",
        "stacking": "pred_stacking_{h}"
    }

    records = []

    print("\n--- Computing Seasonal Robustness (MAE & Skill vs Persistence) ---")
    for q in quarters:
        df_q = df[df["quarter"] == q]
        print(f"\nQuarter: {q} (n_rows={len(df_q)})")
        
        for h in horizons:
            target_col = f"target_pm25_{h}"
            pers_col = f"pred_persistence_{h}"
            
            # 1. PRIMARY EVALUATION: Persistence-valid mask
            pers_mask = (~df_q[target_col].isna()) & (~df_q[pers_col].isna())
            y_prim = df_q.loc[pers_mask, target_col].values
            p_pers_prim = df_q.loc[pers_mask, pers_col].values
            
            mae_pers_prim = float(mean_absolute_error(y_prim, p_pers_prim))
            
            for m in models:
                p_col = model_col_map[m].format(h=h)
                p_m_prim = df_q.loc[pers_mask, p_col].values
                valid_m = (~np.isnan(y_prim)) & (~np.isnan(p_m_prim))
                
                if np.sum(valid_m) > 0:
                    mae_m_prim = float(mean_absolute_error(y_prim[valid_m], p_m_prim[valid_m]))
                    rmse_m_prim = float(np.sqrt(mean_squared_error(y_prim[valid_m], p_m_prim[valid_m])))
                    skill_prim = float(1.0 - (mae_m_prim / mae_pers_prim)) if mae_pers_prim > 0 else 0.0
                    n_prim = int(np.sum(valid_m))
                else:
                    mae_m_prim, rmse_m_prim, skill_prim, n_prim = np.nan, np.nan, np.nan, 0
                
                records.append({
                    "quarter": q,
                    "horizon": h,
                    "model": m,
                    "mask_type": "primary_persistence_valid",
                    "n_samples": n_prim,
                    "mae": mae_m_prim,
                    "rmse": rmse_m_prim,
                    "mae_persistence": mae_pers_prim,
                    "skill_vs_persistence": skill_prim,
                    "skill_pct": skill_prim * 100.0
                })
                
                # 2. SECONDARY EVALUATION: All valid rows
                all_valid_mask = (~df_q[target_col].isna()) & (~df_q[p_col].isna())
                if np.sum(all_valid_mask) > 0:
                    y_all = df_q.loc[all_valid_mask, target_col].values
                    p_all = df_q.loc[all_valid_mask, p_col].values
                    mae_m_all = float(mean_absolute_error(y_all, p_all))
                    rmse_m_all = float(np.sqrt(mean_squared_error(y_all, p_all)))
                    n_all = int(len(y_all))
                else:
                    mae_m_all, rmse_m_all, n_all = np.nan, np.nan, 0
                    
                records.append({
                    "quarter": q,
                    "horizon": h,
                    "model": m,
                    "mask_type": "all_valid_rows",
                    "n_samples": n_all,
                    "mae": mae_m_all,
                    "rmse": rmse_m_all,
                    "mae_persistence": mae_pers_prim,  # benchmark reference
                    "skill_vs_persistence": float(1.0 - (mae_m_all / mae_pers_prim)) if mae_pers_prim > 0 else 0.0,
                    "skill_pct": (1.0 - (mae_m_all / mae_pers_prim)) * 100.0 if mae_pers_prim > 0 else 0.0
                })

            # Print quick summary for primary mask
            row_lgb = [r for r in records if r["quarter"] == q and r["horizon"] == h and r["model"] == "lgb" and r["mask_type"] == "primary_persistence_valid"][0]
            row_stk = [r for r in records if r["quarter"] == q and r["horizon"] == h and r["model"] == "stacking" and r["mask_type"] == "primary_persistence_valid"][0]
            print(f"  {h}: Pers MAE={mae_pers_prim:5.2f} | Stacking MAE={row_stk['mae']:5.2f} (Skill={row_stk['skill_pct']:+5.1f}%) | LGB MAE={row_lgb['mae']:5.2f} (Skill={row_lgb['skill_pct']:+5.1f}%)")

    df_results = pd.DataFrame(records)
    csv_path = out_dir / "seasonal_robustness.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[Step 2] Saved CSV table to {csv_path}")

    # Generate Publication-Quality Visualization
    print("[Step 2] Generating visualization chart...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.flatten()
    
    # Palette and styling
    colors = {
        "persistence": "#7f7f7f",
        "seasonal_naive": "#bcbd22",
        "rf": "#2ca02c",
        "xgb": "#ff7f0e",
        "lgb": "#1f77b4",
        "stacking": "#d62728"
    }
    
    model_labels = {
        "persistence": "Persistence",
        "seasonal_naive": "Seasonal Naive",
        "rf": "Random Forest",
        "xgb": "XGBoost",
        "lgb": "LightGBM",
        "stacking": "Stacking"
    }

    df_prim = df_results[df_results["mask_type"] == "primary_persistence_valid"]
    
    x_indices = np.arange(len(quarters))
    bar_width = 0.13
    
    for ax_idx, h in enumerate(horizons):
        ax = axes[ax_idx]
        df_h = df_prim[df_prim["horizon"] == h]
        
        for m_idx, m in enumerate(models):
            df_m = df_h[df_h["model"] == m]
            skills = [df_m[df_m["quarter"] == q]["skill_pct"].values[0] for q in quarters]
            pos = x_indices + (m_idx - (len(models) - 1) / 2.0) * bar_width
            ax.bar(pos, skills, width=bar_width, color=colors[m], label=model_labels[m] if ax_idx == 0 else "", alpha=0.9, edgecolor="black", linewidth=0.5)
            
        ax.set_title(f"Forecast Horizon +{h[1:]}h", fontsize=12, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels(["Q1 (Winter/Trans)", "Q2 (Summer/Onset)", "Q3 (Monsoon)"], fontsize=10)
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        if ax_idx % 2 == 0:
            ax.set_ylabel("Skill vs. Persistence (%)", fontsize=11)
            
    fig.suptitle("Seasonal Forecast Robustness by Quarter (2026 Test Partition)\nSkill = 100 × (1 - MAE_model / MAE_persistence) [Primary Persistence-Valid Mask]", fontsize=14, fontweight="bold", y=0.98)
    fig.legend(loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.07, 1, 0.94])
    
    chart_path = out_dir / "seasonal_robustness.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[Step 2] Saved chart to {chart_path}")

    # Mirror to other directories
    alt_out_dirs = [
        Path("C:/Users/harsha madala/Documents/Hyderabad_Air_Research/new_research_project/analysis/outputs"),
        Path("E:/new_research_project_BACKUP/new_research_project/analysis/outputs")
    ]
    for alt_dir in alt_out_dirs:
        try:
            alt_dir.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(alt_dir / "seasonal_robustness.csv", index=False)
            import shutil
            shutil.copyfile(chart_path, alt_dir / "seasonal_robustness.png")
            print(f"[Step 2] Mirrored outputs to {alt_dir}")
        except Exception as e:
            print(f"[Step 2] Note: Mirror failed for {alt_dir}: {e}")

    return df_results


if __name__ == "__main__":
    run_seasonal_robustness()
