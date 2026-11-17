"""
Feature-Group Ablation Experiment Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Implements Cumulative Feature-Tier Ablation:
- Tier A: Historical Pollution & Calendar Only (45 features)
- Tier B: Primary Model Predictors = Tier A + Weather + Temporal Traffic Windows (84 features)
- Tier C: Exploratory Sensitivity = Tier B + 19 Unverified Road Proxies (103 features)
  (Explicitly quarantined to test whether assumed proxies provide genuine signal or noise)

Evaluated across horizons (+1h, +6h, +24h, +48h) on Validation (2025) and Test (2026) splits.
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from modeling.model_registry import (
    REGRESSION_TARGETS,
    TIER_A_HISTORICAL_FEATURES,
    TIER_B_PRIMARY_FEATURES,
    TIER_C_EXPLORATORY_FEATURES,
    MODEL_CONFIGS,
    RANDOM_SEED
)
from modeling.evaluation import compute_regression_metrics
from configs.config import (
    PROCESSED_DATA_DIR,
    PREDICTIONS_DIR
)

class AblationRunner:
    """Executes multi-tier feature ablation and sensitivity evaluation."""

    def __init__(self, predictions_dir: Optional[Path] = None):
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.imputer = SimpleImputer(strategy="median")

    def run_ablation(self) -> Dict[str, Any]:
        """Runs Tier A, Tier B, and Tier C ablation using LightGBM across all 4 horizons."""
        df_train = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_train_purged.parquet")
        df_val = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_val_purged.parquet")
        df_test = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_test_purged.parquet")
        
        tiers = {
            "Tier_A_Historical_Pollution_Only": {
                "features": TIER_A_HISTORICAL_FEATURES,
                "count": len(TIER_A_HISTORICAL_FEATURES),
                "description": "Historical PM2.5 lags, rolling stats, dynamic trends, co-pollutant chemical ratios, cyclical calendar."
            },
            "Tier_B_Primary_Physical_and_Weather": {
                "features": TIER_B_PRIMARY_FEATURES,
                "count": len(TIER_B_PRIMARY_FEATURES),
                "description": "Tier A + Open-Meteo ERA5 atmospheric thermodynamics, wind vectors, stability + temporal traffic windows (Approved Primary Predictors)."
            },
            "Tier_C_Exploratory_Sensitivity_Road_Proxies": {
                "features": TIER_C_EXPLORATORY_FEATURES,
                "count": len(TIER_C_EXPLORATORY_FEATURES),
                "description": "Tier B + 19 unverified OpenStreetMap road proxies and traffic pressure interaction terms (Exploratory Sensitivity Only)."
            }
        }
        
        ablation_results: Dict[str, Any] = {
            "tiers_metadata": {
                k: {"count": v["count"], "description": v["description"]}
                for k, v in tiers.items()
            },
            "horizons": {}
        }
        
        for h_key, target_col in REGRESSION_TARGETS.items():
            ablation_results["horizons"][h_key] = {"validation": {}, "test": {}}
            
            y_tr_full = df_train[target_col].values
            y_val_full = df_val[target_col].values
            y_test_full = df_test[target_col].values
            
            tr_mask = ~np.isnan(y_tr_full)
            y_tr = y_tr_full[tr_mask]
            
            for tier_name, tier_info in tiers.items():
                feats = tier_info["features"]
                
                # Prepare matrices with median imputation strictly fitted on train
                X_tr = self.imputer.fit_transform(df_train.loc[tr_mask, feats].values)
                X_val = self.imputer.transform(df_val[feats].values)
                X_test = self.imputer.transform(df_test[feats].values)
                
                model = lgb.LGBMRegressor(**MODEL_CONFIGS["lightgbm"])
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val[~np.isnan(y_val_full)], y_val_full[~np.isnan(y_val_full)])],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
                
                pred_val = model.predict(X_val)
                pred_test = model.predict(X_test)
                
                ablation_results["horizons"][h_key]["validation"][tier_name] = compute_regression_metrics(
                    y_val_full, pred_val
                )
                ablation_results["horizons"][h_key]["test"][tier_name] = compute_regression_metrics(
                    y_test_full, pred_test
                )
                
        with open(self.predictions_dir / "ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(ablation_results, f, indent=2)
            
        print("Feature-group ablation experiment completed and logged to ablation_results.json")
        return ablation_results

if __name__ == "__main__":
    runner = AblationRunner()
    runner.run_ablation()
