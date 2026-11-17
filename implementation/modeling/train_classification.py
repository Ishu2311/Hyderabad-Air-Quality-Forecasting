"""
Hotspot Classification Training and Evaluation Pipeline
Milestone 3: Baseline Benchmarking and Machine Learning Model Development

Trains binary classifiers for:
1. Operational Acute Excursions (PM2.5 >= 60 ug/m3): Horizons +1h, +6h, +24h
2. Exploratory Severe Spikes (PM2.5 >= 120 ug/m3): Horizons +1h, +6h, +24h

Models:
- Hotspot Persistence Baseline
- Hotspot Seasonal-Naive Baseline
- Random Forest Classifier (balanced weights)
- XGBoost Classifier (scale_pos_weight from train)
- LightGBM Classifier (scale_pos_weight from train)

Threshold Calibration:
- Optimal tau* selected deterministically on 2025 validation split.
- Test split (2026) evaluated strictly at calibrated tau*.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import xgboost as xgb

from modeling.model_registry import (
    CLASSIFICATION_TARGETS_ACUTE,
    CLASSIFICATION_TARGETS_SPIKE,
    TIER_B_PRIMARY_FEATURES,
    MODEL_CONFIGS,
    RANDOM_SEED
)
from modeling.baselines import BaselineForecaster
from modeling.evaluation import (
    compute_classification_metrics,
    find_optimal_threshold
)
from configs.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PRIMARY_MODEL_PREDICTORS
)

class ClassificationTrainer:
    """Orchestrates multi-horizon hotspot classification training and evaluation."""

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
        self.imputer = SimpleImputer(strategy="median")

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
        
        X_train = self.imputer.fit_transform(X_train_raw)
        X_val = self.imputer.transform(X_val_raw)
        X_test = self.imputer.transform(X_test_raw)
        return X_train, X_val, X_test

    def train_horizon_classifier(
        self,
        target_name: str,
        X_train: np.ndarray,
        y_train_full: np.ndarray,
        X_val: np.ndarray,
        y_val_full: np.ndarray,
        X_test: np.ndarray,
        y_test_full: np.ndarray
    ) -> Dict[str, Any]:
        """Trains RF, XGBoost, and LightGBM classifiers for a specific binary hotspot target."""
        # Filter out NaN target observations strictly during training
        train_mask = ~np.isnan(y_train_full)
        X_tr = X_train[train_mask]
        y_tr = y_train_full[train_mask].astype(int)
        
        # Calculate class imbalance strictly from training partition
        n_pos = int(np.sum(y_tr == 1))
        n_neg = int(np.sum(y_tr == 0))
        scale_pos = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        
        # 1. Random Forest Classifier
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_SEED
        )
        rf_model.fit(X_tr, y_tr)
        joblib.dump(rf_model, self.models_dir / f"rf_classifier_{target_name}.joblib")
        
        prob_val_rf = rf_model.predict_proba(X_val)[:, 1]
        prob_test_rf = rf_model.predict_proba(X_test)[:, 1]
        
        # 2. XGBoost Classifier
        val_mask = ~np.isnan(y_val_full)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            tree_method="hist",
            eval_metric="logloss"
        )
        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val[val_mask], y_val_full[val_mask].astype(int))],
            verbose=False
        )
        joblib.dump(xgb_model, self.models_dir / f"xgb_classifier_{target_name}.joblib")
        
        prob_val_xgb = xgb_model.predict_proba(X_val)[:, 1]
        prob_test_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        # 3. LightGBM Classifier
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=-1
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val[val_mask], y_val_full[val_mask].astype(int))],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        joblib.dump(lgb_model, self.models_dir / f"lgb_classifier_{target_name}.joblib")
        
        prob_val_lgb = lgb_model.predict_proba(X_val)[:, 1]
        prob_test_lgb = lgb_model.predict_proba(X_test)[:, 1]
        
        return {
            "scale_pos_weight": scale_pos,
            "rf": {"model": rf_model, "val_prob": prob_val_rf, "test_prob": prob_test_rf},
            "xgb": {"model": xgb_model, "val_prob": prob_val_xgb, "test_prob": prob_test_xgb},
            "lgb": {"model": lgb_model, "val_prob": prob_val_lgb, "test_prob": prob_test_lgb}
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Executes full hotspot classification pipeline across Acute and Spike targets."""
        df_train, df_val, df_test = self.load_purged_data()
        X_train, X_val, X_test = self.prepare_feature_matrices(df_train, df_val, df_test)
        
        # Baselines for Acute (60 ug/m3) and Spike (120 ug/m3)
        base_val_pers_acute = self.baseline_forecaster.generate_hotspot_persistence_forecasts(df_val, threshold=60.0)
        base_val_sn_acute = self.baseline_forecaster.generate_hotspot_seasonal_naive_forecasts(df_val, threshold=60.0)
        base_test_pers_acute = self.baseline_forecaster.generate_hotspot_persistence_forecasts(df_test, threshold=60.0)
        base_test_sn_acute = self.baseline_forecaster.generate_hotspot_seasonal_naive_forecasts(df_test, threshold=60.0)
        
        base_val_pers_spike = self.baseline_forecaster.generate_hotspot_persistence_forecasts(df_val, threshold=120.0)
        base_val_sn_spike = self.baseline_forecaster.generate_hotspot_seasonal_naive_forecasts(df_val, threshold=120.0)
        base_test_pers_spike = self.baseline_forecaster.generate_hotspot_persistence_forecasts(df_test, threshold=120.0)
        base_test_sn_spike = self.baseline_forecaster.generate_hotspot_seasonal_naive_forecasts(df_test, threshold=120.0)
        
        predictions_val = df_val[["station", "datetime"]].copy()
        predictions_test = df_test[["station", "datetime"]].copy()
        
        all_metrics: Dict[str, Any] = {
            "acute_excursion_60": {"validation": {}, "test": {}, "station_level": {}},
            "severe_spike_120": {"validation": {}, "test": {}, "station_level": {}}
        }
        
        stations = sorted(df_val["station"].unique().tolist())
        
        target_suites = [
            ("acute_excursion_60", CLASSIFICATION_TARGETS_ACUTE, 60.0, base_val_pers_acute, base_val_sn_acute, base_test_pers_acute, base_test_sn_acute),
            ("severe_spike_120", CLASSIFICATION_TARGETS_SPIKE, 120.0, base_val_pers_spike, base_val_sn_spike, base_test_pers_spike, base_test_sn_spike)
        ]
        
        for suite_name, target_dict, thresh_val, b_val_pers, b_val_sn, b_test_pers, b_test_sn in target_suites:
            for h_key, target_col in target_dict.items():
                y_train = df_train[target_col].values
                y_val = df_val[target_col].values
                y_test = df_test[target_col].values
                
                predictions_val[target_col] = y_val
                predictions_test[target_col] = y_test
                
                # Baseline probabilities (binary 0/1 treated as degenerate probability)
                p_val_pers = b_val_pers[f"pred_hotspot_persistence_{h_key}"].values
                p_val_sn = b_val_sn[f"pred_hotspot_seasonal_naive_{h_key}"].values
                p_test_pers = b_test_pers[f"pred_hotspot_persistence_{h_key}"].values
                p_test_sn = b_test_sn[f"pred_hotspot_seasonal_naive_{h_key}"].values
                
                predictions_val[f"prob_persistence_{suite_name}_{h_key}"] = p_val_pers
                predictions_val[f"prob_seasonal_naive_{suite_name}_{h_key}"] = p_val_sn
                predictions_test[f"prob_persistence_{suite_name}_{h_key}"] = p_test_pers
                predictions_test[f"prob_seasonal_naive_{suite_name}_{h_key}"] = p_test_sn
                
                # Train ML models
                trained = self.train_horizon_classifier(
                    f"{suite_name}_{h_key}",
                    X_train, y_train,
                    X_val, y_val,
                    X_test, y_test
                )
                
                # Determine calibrated threshold tau* on validation predictions
                tau_rf = find_optimal_threshold(y_val, trained["rf"]["val_prob"])
                tau_xgb = find_optimal_threshold(y_val, trained["xgb"]["val_prob"])
                tau_lgb = find_optimal_threshold(y_val, trained["lgb"]["val_prob"])
                
                for m_name in ["rf", "xgb", "lgb"]:
                    predictions_val[f"prob_{m_name}_{suite_name}_{h_key}"] = trained[m_name]["val_prob"]
                    predictions_test[f"prob_{m_name}_{suite_name}_{h_key}"] = trained[m_name]["test_prob"]
                    
                # Compute Network Validation Metrics
                all_metrics[suite_name]["validation"][h_key] = {
                    "persistence": compute_classification_metrics(y_val, p_val_pers, threshold=0.5),
                    "seasonal_naive": compute_classification_metrics(y_val, p_val_sn, threshold=0.5),
                    "random_forest": compute_classification_metrics(y_val, trained["rf"]["val_prob"], threshold=tau_rf),
                    "xgboost": compute_classification_metrics(y_val, trained["xgb"]["val_prob"], threshold=tau_xgb),
                    "lightgbm": compute_classification_metrics(y_val, trained["lgb"]["val_prob"], threshold=tau_lgb),
                    "calibrated_thresholds": {"rf": tau_rf, "xgb": tau_xgb, "lgb": tau_lgb}
                }
                
                # Compute Network Final Test Metrics (evaluated strictly at calibrated tau*)
                all_metrics[suite_name]["test"][h_key] = {
                    "persistence": compute_classification_metrics(y_test, p_test_pers, threshold=0.5),
                    "seasonal_naive": compute_classification_metrics(y_test, p_test_sn, threshold=0.5),
                    "random_forest": compute_classification_metrics(y_test, trained["rf"]["test_prob"], threshold=tau_rf),
                    "xgboost": compute_classification_metrics(y_test, trained["xgb"]["test_prob"], threshold=tau_xgb),
                    "lightgbm": compute_classification_metrics(y_test, trained["lgb"]["test_prob"], threshold=tau_lgb),
                    "calibrated_thresholds_applied": {"rf": tau_rf, "xgb": tau_xgb, "lgb": tau_lgb}
                }
                
                # Compute Station-Level Breakdown on Final Test Split
                all_metrics[suite_name]["station_level"][h_key] = {}
                for st in stations:
                    st_mask_test = (df_test["station"] == st).values
                    y_st_test = y_test[st_mask_test]
                    p_st_pers = p_test_pers[st_mask_test]
                    p_st_sn = p_test_sn[st_mask_test]
                    
                    all_metrics[suite_name]["station_level"][h_key][st] = {
                        "n_total_rows": int(np.sum(st_mask_test)),
                        "n_valid_targets": int(np.sum(~np.isnan(y_st_test))),
                        "persistence": compute_classification_metrics(y_st_test, p_st_pers, threshold=0.5),
                        "seasonal_naive": compute_classification_metrics(y_st_test, p_st_sn, threshold=0.5),
                        "random_forest": compute_classification_metrics(y_st_test, trained["rf"]["test_prob"][st_mask_test], threshold=tau_rf),
                        "xgboost": compute_classification_metrics(y_st_test, trained["xgb"]["test_prob"][st_mask_test], threshold=tau_xgb),
                        "lightgbm": compute_classification_metrics(y_st_test, trained["lgb"]["test_prob"][st_mask_test], threshold=tau_lgb)
                    }
                    
        # Save Predictions
        predictions_val.to_parquet(self.predictions_dir / "classification_predictions_val.parquet", index=False)
        predictions_val.to_csv(self.predictions_dir / "classification_predictions_val.csv", index=False)
        
        predictions_test.to_parquet(self.predictions_dir / "classification_predictions_test.parquet", index=False)
        predictions_test.to_csv(self.predictions_dir / "classification_predictions_test.csv", index=False)
        
        # Save Metrics JSON
        with open(self.predictions_dir / "classification_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)
            
        print("Hotspot classification training and evaluation completed successfully!")
        return all_metrics

if __name__ == "__main__":
    trainer = ClassificationTrainer()
    trainer.run_full_pipeline()
