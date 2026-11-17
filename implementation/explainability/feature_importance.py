"""
Global Feature Importance Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Analyzes and categorizes feature attributions across all 84 approved primary predictors:
1. Feature Category Aggregation:
   - Historical Pollution Dynamics (Lags, rolling means, acceleration)
   - Atmospheric Thermodynamics & Boundary Layer (Temperature, RH, boundary layer height, lapse rate)
   - Wind Vectors & Dispersion Dynamics (u10, v10, ventilation index, stagnation)
   - Chemical Precursors & Co-Pollutants (NO2, SO2, CO, PM10 ratio proxies)
   - Temporal Congestion Windows (Rush-hour diurnal indicators)
   - Cyclical Solar & Calendar Timing (sin/cos hour, day of week)
2. Compares feature importance shift from short-term (+1h) to medium-range (+48h).
3. Generates publication-ready horizontal importance bar charts.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from configs.config import (
    PREDICTIONS_DIR,
    FIGURES_DIR,
    PRIMARY_MODEL_PREDICTORS
)

# Feature Taxonomy Mapping
def classify_feature_category(feat_name: str) -> str:
    """Classifies an 84-predictor feature into its physical/temporal domain."""
    fn = feat_name.lower()
    if any(k in fn for k in ["wind", "u10", "v10", "ventilation", "stagnation", "dispersion"]):
        return "Wind & Dispersion Dynamics"
    elif any(k in fn for k in ["temp", "humidity", "dew", "pressure", "boundary_layer", "lapse", "vapor"]):
        return "Thermodynamics & Boundary Layer"
    elif any(k in fn for k in ["no2", "so2", "co_", "pm10", "ozone", "o3", "ratio"]):
        return "Co-Pollutant / Chemical Ratios"
    elif any(k in fn for k in ["rush_hour", "congestion", "traffic_window", "commute"]):
        return "Temporal Congestion Windows"
    elif any(k in fn for k in ["hour_sin", "hour_cos", "dayofweek", "month", "season", "weekend"]):
        return "Cyclical Calendar & Solar Timing"
    elif any(k in fn for k in ["pm25", "pm2.5", "lag", "rolling", "roll_", "trend", "acceleration", "ewma"]):
        return "Historical PM2.5 Dynamics"
    else:
        return "Other Environmental Predictors"


class FeatureImportanceAnalyzer:
    """Aggregates and visualizes multi-horizon global feature importances."""

    def __init__(
        self,
        predictions_dir: Optional[Path] = None,
        figures_dir: Optional[Path] = None
    ):
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.figures_dir = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def generate_importance_report(self) -> Dict[str, Any]:
        """Loads SHAP summaries across horizons and builds aggregated taxonomy tables."""
        horizons = ["h1", "h6", "h24", "h48"]
        horizon_importances = {}
        category_shares = {}

        for h in horizons:
            json_file = self.predictions_dir / f"shap_summary_{h}.json"
            if not json_file.exists():
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            horizon_importances[h] = data["top_15_features"]

            # Load full NPZ to compute category share
            npz_file = self.predictions_dir / f"shap_values_{h}.npz"
            if npz_file.exists():
                arr = np.load(npz_file, allow_pickle=True)
                shap_vals = arr["shap_values"]
                feats = arr["features"]
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                total_shap = float(np.sum(mean_abs)) if np.sum(mean_abs) > 0 else 1.0

                cat_totals = {}
                for f_name, imp in zip(feats, mean_abs):
                    cat = classify_feature_category(str(f_name))
                    cat_totals[cat] = cat_totals.get(cat, 0.0) + float(imp)

                category_shares[h] = {
                    cat: {"absolute_mean_shap": val, "share_pct": (val / total_shap) * 100.0}
                    for cat, val in cat_totals.items()
                }

        summary = {
            "horizons_analyzed": list(horizon_importances.keys()),
            "top_features_by_horizon": horizon_importances,
            "category_shares_by_horizon": category_shares
        }

        out_json = self.predictions_dir / "global_feature_importance.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def plot_top_features_bar(
        self,
        horizon: str = "h1",
        top_k: int = 15
    ) -> Path:
        """Plots publication-quality horizontal bar chart of top K features by mean |SHAP|."""
        json_file = self.predictions_dir / f"shap_summary_{horizon}.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Missing {json_file}. Run ShapAnalyzer first.")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        top_feats = data["top_15_features"][:top_k][::-1]  # Invert for ascending barh
        names = [d["feature"] for d in top_feats]
        scores = [d["mean_abs_shap"] for d in top_feats]
        categories = [classify_feature_category(n) for n in names]

        # Color mapping by category
        color_map = {
            "Historical PM2.5 Dynamics": "#1f77b4",
            "Thermodynamics & Boundary Layer": "#ff7f0e",
            "Wind & Dispersion Dynamics": "#2ca02c",
            "Co-Pollutant / Chemical Ratios": "#d62728",
            "Temporal Congestion Windows": "#9467bd",
            "Cyclical Calendar & Solar Timing": "#8c564b"
        }
        bar_colors = [color_map.get(c, "#7f7f7f") for c in categories]

        fig, ax = plt.subplots(figsize=(11, 7), dpi=300)
        bars = ax.barh(names, scores, color=bar_colors, edgecolor="black", height=0.65, alpha=0.85)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2.0,
                f"{score:.3f}",
                va="center", ha="left", fontsize=9, fontweight="bold", color="#333333"
            )

        ax.set_title(
            f"Global Feature Attribution (Mean |SHAP| Impact) — Horizon {horizon.upper()}\n"
            f"Primary Model: {data['primary_model'].upper()}",
            fontsize=13, fontweight="bold", pad=12
        )
        ax.set_xlabel(r"Mean Absolute SHAP Value ($E[|\phi_j|]$, $\mu\mathrm{g/m}^3$ impact on PM2.5)", fontsize=10.5)
        ax.set_ylabel("Predictor Variable", fontsize=10.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)

        # Legend for categories
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[cat], ec="black", label=cat)
            for cat in set(categories) if cat in color_map
        ]
        ax.legend(handles=handles, loc="lower right", framealpha=0.9, fontsize=8.5, title="Feature Domain")

        out_path = self.figures_dir / f"shap_global_importance_{horizon}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path
