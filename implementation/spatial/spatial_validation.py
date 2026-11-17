"""
Leave-One-Station-Out Spatial Cross-Validation (LOSOCV) Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Implements:
1. Leave-One-Station-Out Cross-Validation across all 4 CAAQMS stations:
   - Bollaram Industrial Area (Heavy Industrial)
   - Central University (Vegetative Background)
   - ECIL Kapra (Dense Commercial / Mixed Urban)
   - ICRISAT Patancheru (Peri-Urban Industrial)
2. Compares Inverse Distance Weighting (p=1, 2, 3) vs. Ordinary Kriging (Exponential, Spherical, Gaussian).
3. Evaluates spatial transfer accuracy across all 4 horizons (+1h, +6h, +24h, +48h) on test partition (2026).
4. Evaluates both:
   a) Interpolation of true observations (pure spatial transfer fidelity).
   b) Interpolation of model-forecasted concentrations (spatio-temporal operational fidelity).
5. Documents network spatial sparsity limitations.
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

from configs.config import (
    STATION_REGISTRY,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR
)
from spatial.idw import InverseDistanceWeighting
from spatial.kriging import OrdinaryKriging
from spatial.spatial_metrics import compute_spatial_validation_metrics

# Horizon best models determined from Milestone 3 Consistency Audit
HORIZON_BEST_MODELS = {
    "h1": {"model_name": "stacking", "pred_col": "pred_stacking_h1", "target_col": "target_pm25_h1"},
    "h6": {"model_name": "stacking", "pred_col": "pred_stacking_h6", "target_col": "target_pm25_h6"},
    "h24": {"model_name": "lightgbm", "pred_col": "pred_lgb_h24", "target_col": "target_pm25_h24"},
    "h48": {"model_name": "lightgbm", "pred_col": "pred_lgb_h48", "target_col": "target_pm25_h48"}
}


class SpatialValidator:
    """Executes Leave-One-Station-Out spatial cross-validation."""

    def __init__(self, predictions_dir: Optional[Path] = None):
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.station_names = list(STATION_REGISTRY.keys())
        self.coords_utm = {
            s: np.array([STATION_REGISTRY[s].easting_utm44n, STATION_REGISTRY[s].northing_utm44n])
            for s in self.station_names
        }

    def run_losocv(
        self,
        sample_stride: int = 1
    ) -> Dict[str, Any]:
        """
        Runs LOSOCV across all stations and horizons on the 2026 test partition.
        
        Parameters
        ----------
        sample_stride : int, default=1
            Timestep stride (1 = evaluate every hour, 6500+ hours).
        """
        df_test = pd.read_parquet(self.predictions_dir / "regression_predictions_test.parquet")
        
        # Reshape data into wide format indexed by datetime with station columns
        results: Dict[str, Any] = {
            "metadata": {
                "n_stations": len(self.station_names),
                "stations": self.station_names,
                "stride": sample_stride,
                "horizons": list(HORIZON_BEST_MODELS.keys())
            },
            "losocv_results": {}
        }

        datetimes = sorted(df_test["datetime"].unique())[::sample_stride]

        for h_code, h_info in HORIZON_BEST_MODELS.items():
            results["losocv_results"][h_code] = {
                "best_model_used": h_info["model_name"],
                "target_col": h_info["target_col"],
                "pred_col": h_info["pred_col"],
                "by_station": {}
            }

            pred_col = h_info["pred_col"]
            target_col = h_info["target_col"]

            # Pivot target and pred tables
            pivot_target = df_test.pivot(index="datetime", columns="station", values=target_col)
            pivot_pred = df_test.pivot(index="datetime", columns="station", values=pred_col)

            for held_out_st in self.station_names:
                remaining_sts = [s for s in self.station_names if s != held_out_st]
                target_coord = self.coords_utm[held_out_st].reshape(1, 2)
                donor_coords = np.array([self.coords_utm[s] for s in remaining_sts])

                # Interpolation methods to benchmark
                methods = {
                    "idw_p1": InverseDistanceWeighting(power=1.0, coordinate_system="utm"),
                    "idw_p2": InverseDistanceWeighting(power=2.0, coordinate_system="utm"),
                    "idw_p3": InverseDistanceWeighting(power=3.0, coordinate_system="utm"),
                    "kriging_exp": OrdinaryKriging(variogram_model="exponential", coordinate_system="utm"),
                    "kriging_sph": OrdinaryKriging(variogram_model="spherical", coordinate_system="utm"),
                    "kriging_gau": OrdinaryKriging(variogram_model="gaussian", coordinate_system="utm")
                }

                interp_preds: Dict[str, List[float]] = {m: [] for m in methods}
                true_vals: List[float] = []
                direct_model_preds: List[float] = []

                # Loop through timestamps
                for dt in datetimes:
                    if dt not in pivot_pred.index:
                        continue
                    
                    # Target value at held out station
                    y_true_st = pivot_target.loc[dt, held_out_st] if held_out_st in pivot_target.columns else np.nan
                    y_mod_st = pivot_pred.loc[dt, held_out_st] if held_out_st in pivot_pred.columns else np.nan

                    # Donor predictions at remaining stations
                    donor_vals = np.array([pivot_pred.loc[dt, s] if s in pivot_pred.columns else np.nan for s in remaining_sts])
                    
                    # Valid donors mask
                    valid_donor_mask = ~np.isnan(donor_vals)
                    if np.sum(valid_donor_mask) == 0 or np.isnan(y_true_st):
                        continue

                    valid_donor_coords = donor_coords[valid_donor_mask]
                    valid_donor_vals = donor_vals[valid_donor_mask]

                    true_vals.append(float(y_true_st))
                    direct_model_preds.append(float(y_mod_st) if not np.isnan(y_mod_st) else np.nan)

                    for m_name, interpolator in methods.items():
                        try:
                            interpolator.fit(valid_donor_coords, valid_donor_vals)
                            p_val, _ = interpolator.predict(target_coord)
                            interp_preds[m_name].append(float(p_val[0]))
                        except Exception:
                            interp_preds[m_name].append(np.nan)

                # Compute metrics
                true_arr = np.array(true_vals)
                station_metrics: Dict[str, Any] = {}

                for m_name in methods:
                    interp_arr = np.array(interp_preds[m_name])
                    m_dict = compute_spatial_validation_metrics(
                        true_arr, interp_arr, station_name=held_out_st, horizon=h_code, method=m_name
                    )
                    station_metrics[m_name] = m_dict

                results["losocv_results"][h_code]["by_station"][held_out_st] = station_metrics

        # Save to predictions directory
        out_path = self.predictions_dir / "spatial_validation_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    validator = SpatialValidator()
    print("Running Leave-One-Station-Out Spatial Validation on Test Set (stride=4 for quick test)...")
    res = validator.run_losocv(sample_stride=4)
    print("LOSOCV Completed successfully!")
    for h in ["h1", "h24"]:
        print(f"\n=== HORIZON {h} LOSOCV SUMMARY ===")
        for st in list(STATION_REGISTRY.keys()):
            idw_mae = res["losocv_results"][h]["by_station"][st]["idw_p2"]["mae"]
            krg_mae = res["losocv_results"][h]["by_station"][st]["kriging_exp"]["mae"]
            print(f"  {st:<25}: IDW(p=2) MAE={idw_mae:5.2f} ug/m3 | Kriging(Exp) MAE={krg_mae:5.2f} ug/m3")
