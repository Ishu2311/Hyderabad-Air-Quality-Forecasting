"""
Feature Store Builder and Pipeline Integration Module
Milestone 2: Multi-Source Feature Engineering

Constructs the standardized, leak-free, hourly feature store:
- Historical autoregressive lags, rolling statistics, and trends
- Open-Meteo meteorological vectors, moisture deficit, and stability indicators
- OpenStreetMap localized road densities and network concentration
- Diurnal congestion windows and traffic pressure interaction features
- Missingness quality indicators and raw observation masks
- Multi-horizon regression and hotspot targets isolated in a dedicated target table
- Chronological partitioning: Train (2023-2024), Val (2025), Test (2026) with 48h boundary guards

Outputs:
- feature_store_primary.parquet & .csv
- feature_store_targets.parquet & .csv
- feature_store_train.parquet & .csv
- feature_store_val.parquet & .csv
- feature_store_test.parquet & .csv
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        TRAIN_START,
        TRAIN_END,
        VAL_START,
        VAL_END,
        TEST_START,
        TEST_END,
        MAX_PREDICTION_HORIZON_HOURS,
        IDENTIFIER_COLUMNS,
        METADATA_AND_QUALITY_COLUMNS,
        PRIMARY_MODEL_PREDICTORS,
        EXPLORATORY_ROAD_PROXY_FEATURES,
        TARGET_COLUMNS
    )
    from ..target_generation.target_builder import TargetBuilder
    from ..validation.chronological_split import ChronologicalSplitter
    from .historical_features import HistoricalFeatureExtractor
    from .weather_features import WeatherFeatureExtractor
    from .osm_road_features import OSMRoadFeatureExtractor
    from .congestion_features import CongestionFeatureExtractor
except (ImportError, ValueError):
    from configs.config import (
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        TRAIN_START,
        TRAIN_END,
        VAL_START,
        VAL_END,
        TEST_START,
        TEST_END,
        MAX_PREDICTION_HORIZON_HOURS,
        IDENTIFIER_COLUMNS,
        METADATA_AND_QUALITY_COLUMNS,
        PRIMARY_MODEL_PREDICTORS,
        EXPLORATORY_ROAD_PROXY_FEATURES,
        TARGET_COLUMNS
    )
    from target_generation.target_builder import TargetBuilder
    from validation.chronological_split import ChronologicalSplitter
    from feature_engineering.historical_features import HistoricalFeatureExtractor
    from feature_engineering.weather_features import WeatherFeatureExtractor
    from feature_engineering.osm_road_features import OSMRoadFeatureExtractor
    from feature_engineering.congestion_features import CongestionFeatureExtractor

class FeatureStoreBuilder:
    """Builds and manages the multi-source hourly spatio-temporal feature store."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir is not None else PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.historical_extractor = HistoricalFeatureExtractor()
        self.weather_extractor = WeatherFeatureExtractor()
        self.osm_extractor = OSMRoadFeatureExtractor()
        self.congestion_extractor = CongestionFeatureExtractor()
        self.target_builder = TargetBuilder()
        self.splitter = ChronologicalSplitter()

    def build_feature_store(
        self,
        input_df: pd.DataFrame,
        attach_targets: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforms base preprocessed observations into full multi-source feature store.
        
        Args:
            input_df: Preprocessed prospective DataFrame containing sensor and weather variables.
            attach_targets: If True, builds and aligns multi-horizon targets.
            
        Returns:
            Tuple of (feature_df, targets_df)
        """
        df = input_df.copy()
        if "station" not in df.columns or "datetime" not in df.columns:
            raise KeyError("DataFrame must contain 'station' and 'datetime' columns.")
            
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values(by=["station", "datetime"]).reset_index(drop=True)
        
        # 1. Historical & Cyclical Features
        df = self.historical_extractor.transform(df, target_col="PM2.5")
        
        # 2. Meteorological Features
        df = self.weather_extractor.transform(df)
        
        # 3. OpenStreetMap Localized Road Features
        df = self.osm_extractor.attach_road_features(df)
        
        # 4. Diurnal Congestion & Traffic Pressure Interactions
        df = self.congestion_extractor.transform(df)
        
        # 5. Chronological Partitioning & Boundary Buffer Flag
        df = self.splitter.add_partition_metadata(df)
        
        targets_df = pd.DataFrame()
        if attach_targets:
            # Targets are derived strictly from raw un-imputed physical sensor observations
            raw_col = "PM2.5_raw" if "PM2.5_raw" in df.columns else "PM2.5"
            df_with_targets = self.target_builder.build_targets(df, pm25_col=raw_col)
            
            # Identify target columns
            target_cols = [c for c in df_with_targets.columns if c.startswith("target_")]
            target_cols_full = ["station", "datetime"] + target_cols
            targets_df = df_with_targets[target_cols_full].copy()
            
            # Attach targets to master feature store for easy slicing
            for col in target_cols:
                df[col] = df_with_targets[col]
                
        return df, targets_df

    def save_feature_store(
        self,
        df: pd.DataFrame,
        targets_df: pd.DataFrame,
        save_csv: bool = True,
        save_parquet: bool = True
    ) -> Dict[str, str]:
        """Saves feature store and partitions to disk in parquet and csv formats."""
        paths = {}
        
        # 1. Primary Feature Store (All columns)
        if save_parquet:
            p_path = self.output_dir / "feature_store_primary.parquet"
            df.to_parquet(p_path, index=False)
            paths["primary_parquet"] = str(p_path)
        if save_csv:
            c_path = self.output_dir / "feature_store_primary.csv"
            df.to_csv(c_path, index=False)
            paths["primary_csv"] = str(c_path)
            
        # 2. Separate Target Store
        if len(targets_df) > 0:
            if save_parquet:
                tp_path = self.output_dir / "feature_store_targets.parquet"
                targets_df.to_parquet(tp_path, index=False)
                paths["targets_parquet"] = str(tp_path)
            if save_csv:
                tc_path = self.output_dir / "feature_store_targets.csv"
                targets_df.to_csv(tc_path, index=False)
                paths["targets_csv"] = str(tc_path)
                
        # 3. Partitioned Stores (Both Full-Flagged and Strictly-Purged)
        for split_name in ["train", "val", "test"]:
            split_df = df[df["partition"] == split_name].copy().reset_index(drop=True)
            # Full partition with boundary buffer flagged
            if save_parquet:
                sp_path = self.output_dir / f"feature_store_{split_name}.parquet"
                split_df.to_parquet(sp_path, index=False)
                paths[f"{split_name}_parquet"] = str(sp_path)
            if save_csv:
                sc_path = self.output_dir / f"feature_store_{split_name}.csv"
                split_df.to_csv(sc_path, index=False)
                paths[f"{split_name}_csv"] = str(sc_path)
                
            # Strictly Purged partition (excluding boundary buffer to prevent any target crossover)
            split_df_purged = split_df[~split_df["is_boundary_buffer"]].copy().reset_index(drop=True)
            if save_parquet:
                spp_path = self.output_dir / f"feature_store_{split_name}_purged.parquet"
                split_df_purged.to_parquet(spp_path, index=False)
                paths[f"{split_name}_purged_parquet"] = str(spp_path)
            if save_csv:
                spc_path = self.output_dir / f"feature_store_{split_name}_purged.csv"
                split_df_purged.to_csv(spc_path, index=False)
                paths[f"{split_name}_purged_csv"] = str(spc_path)
                
        # 4. Summary metadata with exact reconciled schema accounting
        primary_predictors_present = [c for c in df.columns if c in PRIMARY_MODEL_PREDICTORS]
        exploratory_proxies_present = [c for c in df.columns if c in EXPLORATORY_ROAD_PROXY_FEATURES]
        target_cols_present = [c for c in df.columns if c in TARGET_COLUMNS or c.startswith("target_")]
        identifiers_present = [c for c in df.columns if c in IDENTIFIER_COLUMNS]
        metadata_quality_present = [c for c in df.columns if c in METADATA_AND_QUALITY_COLUMNS]

        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "schema_accounting": {
                "identifiers_count": len(identifiers_present),
                "metadata_and_quality_count": len(metadata_quality_present),
                "primary_model_predictors_count": len(primary_predictors_present),
                "exploratory_road_proxies_count": len(exploratory_proxies_present),
                "target_columns_count": len(target_cols_present),
                "total_candidate_predictors_count": len(primary_predictors_present) + len(exploratory_proxies_present),
                "sum_identity_check": (
                    len(identifiers_present)
                    + len(metadata_quality_present)
                    + len(primary_predictors_present)
                    + len(exploratory_proxies_present)
                    + len(target_cols_present)
                ),
                "sum_equals_total": (
                    (
                        len(identifiers_present)
                        + len(metadata_quality_present)
                        + len(primary_predictors_present)
                        + len(exploratory_proxies_present)
                        + len(target_cols_present)
                    ) == len(df.columns)
                )
            },
            "primary_model_predictors": primary_predictors_present,
            "exploratory_road_proxies": exploratory_proxies_present,
            "target_columns": target_cols_present,
            "stations": sorted(df["station"].unique().tolist()),
            "start_time": str(df["datetime"].min()),
            "end_time": str(df["datetime"].max()),
            "partition_counts_full": {
                split: int((df["partition"] == split).sum())
                for split in ["train", "val", "test"]
            },
            "partition_counts_purged": {
                split: int(((df["partition"] == split) & (~df["is_boundary_buffer"])).sum())
                for split in ["train", "val", "test"]
            },
            "boundary_buffer_rows_purged_per_split": {
                split: int(((df["partition"] == split) & (df["is_boundary_buffer"])).sum())
                for split in ["train", "val", "test"]
            }
        }
        sum_path = self.output_dir / "feature_store_summary.json"
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        paths["summary_json"] = str(sum_path)
        
        return paths
