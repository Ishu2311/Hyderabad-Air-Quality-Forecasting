"""
Regression Model Training and Evaluation Pipeline
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Trains:
1. Zero-Order Persistence Baseline
2. Seasonal-Naive Baseline
3. Random Forest Regressor
4. XGBoost Regressor
5. LightGBM Regressor

Horizons: +1h, +6h, +24h, +48h.
Predictors: Strictly PRIMARY_MODEL_PREDICTORS (84 features).
Data: Purged partitions only (train_purged, val_purged, test_purged).
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

from modeling.model_registry import (
    REGRESSION_TARGETS,
    TIER_B_PRIMARY_FEATURES,
    MODEL_CONFIGS,
    RANDOM_SEED
)
from modeling.baselines import BaselineForecaster
from modeling.evaluation import compute_regression_metrics, compute_diebold_mariano_test
from configs.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_MODEL_PREDICTORS
)

class RegressionTrainer:
    """Orchestrates multi-horizon continuous PM2.5 forecasting model training and evaluation."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        models_dir: Optional[Path] = None,
        predictions_dir: Optional[Path] = None
    ):
        self.features = features if features is not None else PRIMARY_MODEL_PREDICTORS
        self.models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_forecaster = BaselineForecaster(pm25_col="PM2.5")
        
        # Learned preprocessing: Imputer fitted strictly on training data
        self.imputer = SimpleImputer(strategy="median")
        self.is_imputer_fitted = False

    def load_purged_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads verified purged partitions guaranteeing zero target boundary crossover."""
        train_path = PROCESSED_DATA_DIR / "feature_store_train_purged.parquet"
        val_path = PROCESSED_DATA_DIR / "feature_store_val_purged.parquet"
        test_path = PROCESSED_DATA_DIR / "feature_store_test_purged.parquet"
        
        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)
        
        return df_train, df_val, df_test

    def prepare_feature_matrices(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepares imputed feature matrices fitting exclusively on training data."""
        X_train_raw = df_train[self.features].values
        X_val_raw = df_val[self.features].values
        X_test_raw = df_test[self.features].values
        
        # Fit on training data only
        X_train = self.imputer.fit_transform(X_train_raw)
        X_val = self.imputer.transform(X_val_raw)
        X_test = self.imputer.transform(X_test_raw)
        self.is_imputer_fitted = True
        
        # Save imputer
        joblib.dump(self.imputer, self.models_dir / "regression_feature_imputer.joblib")
        return X_train, X_val, X_test

    def train_horizon_models(
        self,
        h_key: str,
        target_col: str,
        X_train: np.ndarray,
        y_train_full: np.ndarray,
        X_val: np.ndarray,
        y_val_full: np.ndarray,
        X_test: np.ndarray,
        y_test_full: np.ndarray
    ) -> Dict[str, Any]:
        """Trains RF, XGBoost, and LightGBM for a specific forecast horizon."""
        # Mask out NaN targets strictly during training
        train_mask = ~np.isnan(y_train_full)
        X_tr = X_train[train_mask]
        y_tr = y_train_full[train_mask]
        
        horizon_results = {
            "horizon": h_key,
            "target_col": target_col,
            "train_valid_samples": int(len(y_tr)),
            "models": {}
        }
        
        # 1. Random Forest Regressor
        rf_config = MODEL_CONFIGS["random_forest"].copy()
        rf_model = RandomForestRegressor(**rf_config)
        rf_model.fit(X_tr, y_tr)
        joblib.dump(rf_model, self.models_dir / f"rf_regressor_{h_key}.joblib")
        
        pred_val_rf = rf_model.predict(X_val)
        pred_test_rf = rf_model.predict(X_test)
        
        # 2. XGBoost Regressor
        xgb_config = MODEL_CONFIGS["xgboost"].copy()
        xgb_model = xgb.XGBRegressor(**xgb_config)
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val[~np.isnan(y_val_full)], y_val_full[~np.isnan(y_val_full)])],
            verbose=False
        )
        joblib.dump(xgb_model, self.models_dir / f"xgb_regressor_{h_key}.joblib")
        
        pred_val_xgb = xgb_model.predict(X_val)
        pred_test_xgb = xgb_model.predict(X_test)
        
        # 3. LightGBM Regressor
        lgb_config = MODEL_CONFIGS["lightgbm"].copy()
        lgb_model = lgb.LGBMRegressor(**lgb_config)
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val[~np.isnan(y_val_full)], y_val_full[~np.isnan(y_val_full)])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        joblib.dump(lgb_model, self.models_dir / f"lgb_regressor_{h_key}.joblib")
        
        pred_val_lgb = lgb_model.predict(X_val)
        pred_test_lgb = lgb_model.predict(X_test)
        
        return {
            "rf": {"model": rf_model, "val_pred": pred_val_rf, "test_pred": pred_test_rf},
            "xgb": {"model": xgb_model, "val_pred": pred_val_xgb, "test_pred": pred_test_xgb},
            "lgb": {"model": lgb_model, "val_pred": pred_val_lgb, "test_pred": pred_test_lgb}
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Executes full training, prediction, and multi-station evaluation pipeline."""
        df_train, df_val, df_test = self.load_purged_data()
        X_train, X_val, X_test = self.prepare_feature_matrices(df_train, df_val, df_test)
        
        # Baselines
        base_val_pers = self.baseline_forecaster.generate_persistence_forecasts(df_val)
        base_val_sn = self.baseline_forecaster.generate_seasonal_naive_forecasts(df_val)
        
        base_test_pers = self.baseline_forecaster.generate_persistence_forecasts(df_test)
        base_test_sn = self.baseline_forecaster.generate_seasonal_naive_forecasts(df_test)
        
        # Result containers
        predictions_val = df_val[["station", "datetime"]].copy()
        predictions_test = df_test[["station", "datetime"]].copy()
        
        all_metrics: Dict[str, Any] = {
            "validation": {},
            "test": {},
            "station_level": {}
        }
        
        stations = sorted(df_val["station"].unique().tolist())
        
        for h_key, target_col in REGRESSION_TARGETS.items():
            h_int = int(h_key.replace("h", ""))
            y_train = df_train[target_col].values
            y_val = df_val[target_col].values
            y_test = df_test[target_col].values
            
            # Ground truth targets
            predictions_val[target_col] = y_val
            predictions_test[target_col] = y_test
            
            # Baseline predictions
            p_val_pers = base_val_pers[f"pred_persistence_{h_key}"].values
            p_val_sn = base_val_sn[f"pred_seasonal_naive_{h_key}"].values
            p_test_pers = base_test_pers[f"pred_persistence_{h_key}"].values
            p_test_sn = base_test_sn[f"pred_seasonal_naive_{h_key}"].values
            
            predictions_val[f"pred_persistence_{h_key}"] = p_val_pers
            predictions_val[f"pred_seasonal_naive_{h_key}"] = p_val_sn
            predictions_test[f"pred_persistence_{h_key}"] = p_test_pers
            predictions_test[f"pred_seasonal_naive_{h_key}"] = p_test_sn
            
            # Train ML models
            trained = self.train_horizon_models(
                h_key, target_col,
                X_train, y_train,
                X_val, y_val,
                X_test, y_test
            )
            
            for m_name in ["rf", "xgb", "lgb"]:
                predictions_val[f"pred_{m_name}_{h_key}"] = trained[m_name]["val_pred"]
                predictions_test[f"pred_{m_name}_{h_key}"] = trained[m_name]["test_pred"]
                
            # Compute Network Metrics for Validation Split
            all_metrics["validation"][h_key] = {
                "persistence": compute_regression_metrics(y_val, p_val_pers),
                "seasonal_naive": compute_regression_metrics(y_val, p_val_sn),
                "random_forest": compute_regression_metrics(y_val, trained["rf"]["val_pred"], baseline_pred=p_val_pers),
                "xgboost": compute_regression_metrics(y_val, trained["xgb"]["val_pred"], baseline_pred=p_val_pers),
                "lightgbm": compute_regression_metrics(y_val, trained["lgb"]["val_pred"], baseline_pred=p_val_pers)
            }
            
            # Compute Network Metrics for Final Test Split
            all_metrics["test"][h_key] = {
                "persistence": compute_regression_metrics(y_test, p_test_pers),
                "seasonal_naive": compute_regression_metrics(y_test, p_test_sn),
                "random_forest": compute_regression_metrics(y_test, trained["rf"]["test_pred"], baseline_pred=p_test_pers),
                "xgboost": compute_regression_metrics(y_test, trained["xgb"]["test_pred"], baseline_pred=p_test_pers),
                "lightgbm": compute_regression_metrics(y_test, trained["lgb"]["test_pred"], baseline_pred=p_test_pers),
                "dm_test_vs_persistence": {
                    "random_forest": compute_diebold_mariano_test(y_test, trained["rf"]["test_pred"], p_test_pers, h=h_int),
                    "xgboost": compute_diebold_mariano_test(y_test, trained["xgb"]["test_pred"], p_test_pers, h=h_int),
                    "lightgbm": compute_diebold_mariano_test(y_test, trained["lgb"]["test_pred"], p_test_pers, h=h_int)
                }
            }
            
            # Compute Station-Level Breakdown on Test Split
            all_metrics["station_level"][h_key] = {}
            for st in stations:
                st_mask_test = (df_test["station"] == st).values
                y_st_test = y_test[st_mask_test]
                p_st_pers = p_test_pers[st_mask_test]
                p_st_sn = p_test_sn[st_mask_test]
                
                all_metrics["station_level"][h_key][st] = {
                    "n_total_rows": int(np.sum(st_mask_test)),
                    "n_valid_targets": int(np.sum(~np.isnan(y_st_test))),
                    "persistence": compute_regression_metrics(y_st_test, p_st_pers),
                    "seasonal_naive": compute_regression_metrics(y_st_test, p_st_sn),
                    "random_forest": compute_regression_metrics(y_st_test, trained["rf"]["test_pred"][st_mask_test]),
                    "xgboost": compute_regression_metrics(y_st_test, trained["xgb"]["test_pred"][st_mask_test]),
                    "lightgbm": compute_regression_metrics(y_st_test, trained["lgb"]["test_pred"][st_mask_test])
                }
                
        # Save Predictions to Parquet and CSV
        predictions_val.to_parquet(self.predictions_dir / "regression_predictions_val.parquet", index=False)
        predictions_val.to_csv(self.predictions_dir / "regression_predictions_val.csv", index=False)
        
        predictions_test.to_parquet(self.predictions_dir / "regression_predictions_test.parquet", index=False)
        predictions_test.to_csv(self.predictions_dir / "regression_predictions_test.csv", index=False)
        
        # Save Metrics JSON
        with open(self.predictions_dir / "regression_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
            
        print("Continuous regression training and evaluation completed successfully!")
        return all_metrics

if __name__ == "__main__":
    trainer = RegressionTrainer()
    trainer.run_full_pipeline()
