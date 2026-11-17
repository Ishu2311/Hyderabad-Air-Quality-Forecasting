"""
Hyperparameter Tuning Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Implements controlled, reproducible hyperparameter search for gradient boosted ensembles
(LightGBM and XGBoost) evaluated on chronological validation data (2025).
Zero random row splitting; zero test set exposure.
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

from typing import Dict, List, Optional, Tuple, Any
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb

from modeling.model_registry import (
    REGRESSION_TARGETS,
    RANDOM_SEED
)
from configs.config import (
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR,
    PRIMARY_MODEL_PREDICTORS
)

class HyperparameterTuner:
    """Conducts time-aware hyperparameter exploration on training split, validated on 2025 split."""

    def __init__(self, predictions_dir: Optional[Path] = None):
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.features = PRIMARY_MODEL_PREDICTORS

    def run_tuning(self) -> Dict[str, Any]:
        """Runs search across representative horizons h1 and h24."""
        df_train = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_train_purged.parquet")
        df_val = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_val_purged.parquet")
        
        tuning_results: Dict[str, Any] = {
            "random_seed": RANDOM_SEED,
            "search_space": {
                "lightgbm": {
                    "learning_rate": [0.01, 0.03, 0.05],
                    "num_leaves": [15, 31, 63],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },
                "xgboost": {
                    "learning_rate": [0.01, 0.03, 0.05],
                    "max_depth": [4, 6, 8],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                }
            },
            "horizons": {}
        }
        
        for h_key in ["h1", "h24"]:
            target_col = REGRESSION_TARGETS[h_key]
            
            y_tr_full = df_train[target_col].values
            y_val_full = df_val[target_col].values
            
            tr_mask = ~np.isnan(y_tr_full)
            val_mask = ~np.isnan(y_val_full)
            
            X_tr = df_train.loc[tr_mask, self.features].values
            y_tr = y_tr_full[tr_mask]
            
            X_val = df_val.loc[val_mask, self.features].values
            y_val = y_val_full[val_mask]
            
            # 1. LightGBM Grid Search
            t0 = time.time()
            best_lgb_rmse = 999.0
            best_lgb_params = {}
            lgb_trials = []
            
            for lr in [0.01, 0.03, 0.05]:
                for leaves in [15, 31, 63]:
                    for colsample in [0.7, 0.8, 1.0]:
                        model = lgb.LGBMRegressor(
                            n_estimators=300,
                            learning_rate=lr,
                            num_leaves=leaves,
                            colsample_bytree=colsample,
                            subsample=0.8,
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                            verbose=-1
                        )
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_val)
                        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                        
                        trial = {
                            "learning_rate": lr,
                            "num_leaves": leaves,
                            "colsample_bytree": colsample,
                            "val_rmse": rmse
                        }
                        lgb_trials.append(trial)
                        if rmse < best_lgb_rmse:
                            best_lgb_rmse = rmse
                            best_lgb_params = trial
                            
            lgb_elapsed = round(time.time() - t0, 2)
            
            # 2. XGBoost Grid Search (Focused candidate set for computational efficiency)
            t0 = time.time()
            best_xgb_rmse = 999.0
            best_xgb_params = {}
            xgb_trials = []
            
            for lr in [0.01, 0.03, 0.05]:
                for depth in [4, 6, 8]:
                    for colsample in [0.7, 0.8]:
                        model = xgb.XGBRegressor(
                            n_estimators=300,
                            learning_rate=lr,
                            max_depth=depth,
                            colsample_bytree=colsample,
                            subsample=0.8,
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                            tree_method="hist"
                        )
                        model.fit(X_tr, y_tr)
                        preds = model.predict(X_val)
                        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
                        
                        trial = {
                            "learning_rate": lr,
                            "max_depth": depth,
                            "colsample_bytree": colsample,
                            "val_rmse": rmse
                        }
                        xgb_trials.append(trial)
                        if rmse < best_xgb_rmse:
                            best_xgb_rmse = rmse
                            best_xgb_params = trial
                            
            xgb_elapsed = round(time.time() - t0, 2)
            
            tuning_results["horizons"][h_key] = {
                "target_col": target_col,
                "lightgbm": {
                    "best_params": best_lgb_params,
                    "best_val_rmse": best_lgb_rmse,
                    "elapsed_seconds": lgb_elapsed,
                    "total_trials": len(lgb_trials)
                },
                "xgboost": {
                    "best_params": best_xgb_params,
                    "best_val_rmse": best_xgb_rmse,
                    "elapsed_seconds": xgb_elapsed,
                    "total_trials": len(xgb_trials)
                }
            }
            
        with open(self.predictions_dir / "hyperparameter_tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(tuning_results, f, indent=2)
            
        print("Hyperparameter tuning completed and logged to hyperparameter_tuning_results.json")
        return tuning_results

if __name__ == "__main__":
    tuner = HyperparameterTuner()
    tuner.run_tuning()
