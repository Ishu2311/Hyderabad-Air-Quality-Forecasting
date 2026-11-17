"""
Feature Scaling Module
Milestone 1: Data Foundation and Preprocessing

Fits scikit-learn StandardScaler strictly on the 2023-2024 training partition.
Transforms features across all splits with zero data leakage.
Exports serialization and human-readable audit parameters.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from ..configs.config import (
        POLLUTANTS,
        WEATHER_VARIABLES,
        PROCESSED_DATA_DIR
    )
except (ImportError, ValueError):
    from configs.config import (
        POLLUTANTS,
        WEATHER_VARIABLES,
        PROCESSED_DATA_DIR
    )

class FeatureScaler:
    """Standardizes continuous features fitted strictly on training data."""

    def __init__(self, feature_columns: Optional[List[str]] = None):
        self.feature_columns = feature_columns if feature_columns else (POLLUTANTS + WEATHER_VARIABLES)
        self.scaler = StandardScaler()
        self.fitted_columns: List[str] = []
        self.is_fitted: bool = False
        self.audit_params: Dict[str, Dict[str, float]] = {}

    def fit(self, train_df: pd.DataFrame) -> "FeatureScaler":
        """
        Fits StandardScaler strictly on training dataframe.
        Excludes validation and test sets completely.
        """
        # Determine available numeric feature columns
        available_cols = [c for c in self.feature_columns if c in train_df.columns]
        if not available_cols:
            raise ValueError(f"None of {self.feature_columns} found in training data.")
            
        self.fitted_columns = available_cols
        
        # Fit scaler on available non-NaN values or dense matrix
        X_train = train_df[self.fitted_columns].copy()
        
        # For columns with NaNs in training, fill with column mean for scaler fitting only
        # to ensure robust mean and std estimates
        X_train_imputed = X_train.fillna(X_train.mean())
        
        self.scaler.fit(X_train_imputed)
        self.is_fitted = True
        
        # Record audit parameters
        self.audit_params = {
            col: {
                "training_mean": round(float(m), 4),
                "training_std": round(float(s), 4)
            }
            for col, m, s in zip(self.fitted_columns, self.scaler.mean_, self.scaler.scale_)
        }
        
        return self

    def transform(self, df: pd.DataFrame, prefix: str = "scaled_") -> pd.DataFrame:
        """Transforms feature columns and returns dataframe with new scaled columns."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform().")
            
        df = df.copy()
        X = df[self.fitted_columns].copy()
        
        # Apply transformation: (x - mean) / std, preserving NaNs where originally present
        X_scaled = (X - self.scaler.mean_) / self.scaler.scale_
        
        for col in self.fitted_columns:
            df[f"{prefix}{col}"] = X_scaled[col]
            
        return df

    def save(self, output_dir: Optional[Union[str, Path]] = None) -> Tuple[Path, Path]:
        """Saves serialized scaler and audit JSON."""
        out_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        
        scaler_file = out_dir / "scaler_train.joblib"
        audit_file = out_dir / "scaler_audit.json"
        
        joblib.dump(self.scaler, scaler_file)
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(self.audit_params, f, indent=2)
            
        return scaler_file, audit_file

    def load(self, scaler_file: Union[str, Path]) -> "FeatureScaler":
        """Loads serialized scaler."""
        self.scaler = joblib.load(scaler_file)
        self.is_fitted = True
        return self
