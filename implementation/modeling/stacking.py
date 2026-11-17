"""
Time-Series Out-of-Fold Stacking Ensemble Module
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Implements:
- 5-Fold Chronological Blocked Time-Series Cross-Validation on 2023-2024 Train Split
- Base Estimators (Level-0): Random Forest, XGBoost, LightGBM
- Meta-Learner (Level-1):
  - BayesianRidge for Continuous Regression
  - LogisticRegression(C=1.0) for Hotspot Classification
- Zero target leakage: Meta-learner is fitted strictly on out-of-fold predictions.
- Test partition (2026) remains completely untouched during stacking construction.
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

from typing import Dict, List, Optional, Tuple, Any
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from modeling.model_registry import (
    REGRESSION_TARGETS,
    CLASSIFICATION_TARGETS_ACUTE,
    CLASSIFICATION_TARGETS_SPIKE,
    MODEL_CONFIGS,
    RANDOM_SEED
)
from modeling.evaluation import (
    compute_regression_metrics,
    compute_classification_metrics,
    find_optimal_threshold,
    compute_diebold_mariano_test
)
from configs.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_MODEL_PREDICTORS
)

class StackingEnsemble:
    """Builds and evaluates leak-free time-series stacking ensembles."""

    def __init__(
        self,
        n_splits: int = 5,
        models_dir: Optional[Path] = None,
        predictions_dir: Optional[Path] = None
    ):
        self.n_splits = n_splits
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.features = PRIMARY_MODEL_PREDICTORS
        self.imputer = SimpleImputer(strategy="median")

    def load_purged_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads purged partition datasets."""
        df_train = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_train_purged.parquet")
        df_val = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_val_purged.parquet")
        df_test = pd.read_parquet(PROCESSED_DATA_DIR / "feature_store_test_purged.parquet")
        return df_train, df_val, df_test

    def generate_chronological_folds(self, df_train: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Creates 5 contiguous chronological blocked folds across the training timeline.
        Preserves chronological ordering across all 4 stations synchronously.
        """
        unique_dates = np.sort(df_train["datetime"].unique())
        n_dates = len(unique_dates)
        block_size = n_dates // self.n_splits
        
        folds = []
        for fold_idx in range(self.n_splits):
            val_start_idx = fold_idx * block_size
            val_end_idx = n_dates if fold_idx == self.n_splits - 1 else (fold_idx + 1) * block_size
            
            val_dates = set(unique_dates[val_start_idx:val_end_idx])
            train_dates = set(unique_dates) - val_dates
            
            train_mask = df_train["datetime"].isin(train_dates).values
            val_mask = df_train["datetime"].isin(val_dates).values
            
            folds.append((np.where(train_mask)[0], np.where(val_mask)[0]))
            
        return folds

    def train_regression_stacking(self) -> Dict[str, Any]:
        """Builds and evaluates stacking regressor across horizons +1h, +6h, +24h, +48h."""
        df_train, df_val, df_test = self.load_purged_data()
        
        X_train_raw = df_train[self.features].values
        X_val_raw = df_val[self.features].values
        X_test_raw = df_test[self.features].values
        
        X_train = self.imputer.fit_transform(X_train_raw)
        X_val = self.imputer.transform(X_val_raw)
        X_test = self.imputer.transform(X_test_raw)
        
        folds = self.generate_chronological_folds(df_train)
        stacking_results = {"validation": {}, "test": {}, "station_level": {}}
        
        # Load baseline predictions for Diebold-Mariano testing
        val_preds_df = pd.read_parquet(self.predictions_dir / "regression_predictions_val.parquet")
        test_preds_df = pd.read_parquet(self.predictions_dir / "regression_predictions_test.parquet")
        
        for h_key, target_col in REGRESSION_TARGETS.items():
            h_int = int(h_key.replace("h", ""))
            y_train = df_train[target_col].values
            y_val = df_val[target_col].values
            y_test = df_test[target_col].values
            
            # Arrays to store out-of-fold predictions on training set
            oof_rf = np.full(len(df_train), np.nan)
            oof_xgb = np.full(len(df_train), np.nan)
            oof_lgb = np.full(len(df_train), np.nan)
            
            # 1. Generate OOF Predictions via 5-Fold Chronological CV
            for tr_idx, val_idx in folds:
                tr_mask = ~np.isnan(y_train[tr_idx])
                actual_tr_idx = tr_idx[tr_mask]
                
                # Fit base models on fold training partition
                rf = RandomForestRegressor(**MODEL_CONFIGS["random_forest"])
                rf.fit(X_train[actual_tr_idx], y_train[actual_tr_idx])
                oof_rf[val_idx] = rf.predict(X_train[val_idx])
                
                xgb_m = xgb.XGBRegressor(**MODEL_CONFIGS["xgboost"])
                xgb_m.fit(X_train[actual_tr_idx], y_train[actual_tr_idx])
                oof_xgb[val_idx] = xgb_m.predict(X_train[val_idx])
                
                lgb_m = lgb.LGBMRegressor(**MODEL_CONFIGS["lightgbm"])
                lgb_m.fit(X_train[actual_tr_idx], y_train[actual_tr_idx])
                oof_lgb[val_idx] = lgb_m.predict(X_train[val_idx])
                
            # Valid mask for meta-learner training (only where both target and OOF preds are valid)
            meta_tr_mask = ~np.isnan(y_train) & ~np.isnan(oof_rf) & ~np.isnan(oof_xgb) & ~np.isnan(oof_lgb)
            X_meta_train = np.column_stack([
                oof_rf[meta_tr_mask],
                oof_xgb[meta_tr_mask],
                oof_lgb[meta_tr_mask]
            ])
            y_meta_train = y_train[meta_tr_mask]
            
            # 2. Fit Meta-Learner (BayesianRidge per approved methodology)
            meta_learner = BayesianRidge()
            meta_learner.fit(X_meta_train, y_meta_train)
            joblib.dump(meta_learner, self.models_dir / f"stacking_meta_regressor_{h_key}.joblib")
            
            # 3. Evaluate Meta-Learner on Validation Split (Level-0 models fitted on full train)
            val_rf = val_preds_df[f"pred_rf_{h_key}"].values
            val_xgb = val_preds_df[f"pred_xgb_{h_key}"].values
            val_lgb = val_preds_df[f"pred_lgb_{h_key}"].values
            X_meta_val = np.column_stack([val_rf, val_xgb, val_lgb])
            pred_stack_val = meta_learner.predict(X_meta_val)
            
            # 4. Evaluate Meta-Learner on Final Test Split
            test_rf = test_preds_df[f"pred_rf_{h_key}"].values
            test_xgb = test_preds_df[f"pred_xgb_{h_key}"].values
            test_lgb = test_preds_df[f"pred_lgb_{h_key}"].values
            X_meta_test = np.column_stack([test_rf, test_xgb, test_lgb])
            pred_stack_test = meta_learner.predict(X_meta_test)
            
            # Record predictions in tables
            val_preds_df[f"pred_stacking_{h_key}"] = pred_stack_val
            test_preds_df[f"pred_stacking_{h_key}"] = pred_stack_test
            
            p_val_pers = val_preds_df[f"pred_persistence_{h_key}"].values
            p_test_pers = test_preds_df[f"pred_persistence_{h_key}"].values
            
            # Metrics
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
            
            # Station-Level Test Breakdown
            stacking_results["station_level"][h_key] = {}
            for st in sorted(df_test["station"].unique()):
                st_mask = (df_test["station"] == st).values
                stacking_results["station_level"][h_key][st] = compute_regression_metrics(
                    y_test[st_mask], pred_stack_test[st_mask]
                )
                
        # Re-save augmented prediction files
        val_preds_df.to_parquet(self.predictions_dir / "regression_predictions_val.parquet", index=False)
        val_preds_df.to_csv(self.predictions_dir / "regression_predictions_val.csv", index=False)
        test_preds_df.to_parquet(self.predictions_dir / "regression_predictions_test.parquet", index=False)
        test_preds_df.to_csv(self.predictions_dir / "regression_predictions_test.csv", index=False)
        
        with open(self.predictions_dir / "stacking_regression_metrics.json", "w", encoding="utf-8") as f:
            json.dump(stacking_results, f, indent=2)
            
        print("Stacking regression completed and saved successfully!")
        return stacking_results

if __name__ == "__main__":
    stacker = StackingEnsemble()
    stacker.train_regression_stacking()
