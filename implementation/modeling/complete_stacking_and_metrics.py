"""
Finalizes stacking evaluation and exports stacking_regression_metrics.json
using the trained meta-models on disk.
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

import json
import joblib
import numpy as np
import pandas as pd

from configs.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR
)
from modeling.model_registry import REGRESSION_TARGETS
from modeling.evaluation import compute_regression_metrics, compute_diebold_mariano_test

def finalize_stacking():
    df_val = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_val_purged.parquet")
    df_test = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_test_purged.parquet")
    
    val_preds_df = pd.read_parquet(PREDICTIONS_DIR / "regression_predictions_val.parquet")
    test_preds_df = pd.read_parquet(PREDICTIONS_DIR / "regression_predictions_test.parquet")
    
    stacking_results = {"validation": {}, "test": {}, "station_level": {}}
    
    stations = sorted(df_test["station"].unique().tolist())
    
    for h_key, target_col in REGRESSION_TARGETS.items():
        h_int = int(h_key.replace("h", ""))
        y_val = df_val[target_col].values
        y_test = df_test[target_col].values
        
        meta_learner = joblib.load(MODELS_DIR / f"stacking_meta_regressor_{h_key}.joblib")
        
        # Validation meta predictions
        val_rf = val_preds_df[f"pred_rf_{h_key}"].values
        val_xgb = val_preds_df[f"pred_xgb_{h_key}"].values
        val_lgb = val_preds_df[f"pred_lgb_{h_key}"].values
        X_meta_val = np.column_stack([val_rf, val_xgb, val_lgb])
        pred_stack_val = meta_learner.predict(X_meta_val)
        val_preds_df[f"pred_stacking_{h_key}"] = pred_stack_val
        
        # Test meta predictions
        test_rf = test_preds_df[f"pred_rf_{h_key}"].values
        test_xgb = test_preds_df[f"pred_xgb_{h_key}"].values
        test_lgb = test_preds_df[f"pred_lgb_{h_key}"].values
        X_meta_test = np.column_stack([test_rf, test_xgb, test_lgb])
        pred_stack_test = meta_learner.predict(X_meta_test)
        test_preds_df[f"pred_stacking_{h_key}"] = pred_stack_test
        
        p_val_pers = val_preds_df[f"pred_persistence_{h_key}"].values
        p_test_pers = test_preds_df[f"pred_persistence_{h_key}"].values
        
        stacking_results["validation"][h_key] = compute_regression_metrics(
            y_val, pred_stack_val, baseline_pred=p_val_pers
        )
        stacking_results["validation"][h_key]["meta_weights"] = {
            "rf": float(meta_learner.coef_[0]),
            "xgb": float(meta_learner.coef_[1]),
            "lgb": float(meta_learner.coef_[2]),
            "intercept": float(meta_learner.intercept_)
        }
        
        stacking_results["test"][h_key] = compute_regression_metrics(
            y_test, pred_stack_test, baseline_pred=p_test_pers
        )
        stacking_results["test"][h_key]["dm_test_vs_persistence"] = compute_diebold_mariano_test(
            y_test, pred_stack_test, p_test_pers, h=h_int
        )
        
        stacking_results["station_level"][h_key] = {}
        for st in stations:
            st_mask = (df_test["station"] == st).values
            stacking_results["station_level"][h_key][st] = compute_regression_metrics(
                y_test[st_mask], pred_stack_test[st_mask]
            )
            
    val_preds_df.to_parquet(PREDICTIONS_DIR / "regression_predictions_val.parquet", index=False)
    val_preds_df.to_csv(PREDICTIONS_DIR / "regression_predictions_val.csv", index=False)
    test_preds_df.to_parquet(PREDICTIONS_DIR / "regression_predictions_test.parquet", index=False)
    test_preds_df.to_csv(PREDICTIONS_DIR / "regression_predictions_test.csv", index=False)
    
    with open(PREDICTIONS_DIR / "stacking_regression_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stacking_results, f, indent=2)
        
    print("Stacking finalization complete! Saved stacking_regression_metrics.json and updated test predictions.")
    return stacking_results

if __name__ == "__main__":
    finalize_stacking()
