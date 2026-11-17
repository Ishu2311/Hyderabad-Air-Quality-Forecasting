"""
Local Prediction Explainer and Visualization Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Generates:
1. SHAP Beeswarm Summary Plots (feature value color vs. directional SHAP impact).
2. SHAP Dependence Scatter Plots with interaction coloring.
3. Local Waterfall / Step-down Prediction Decompositions:
   y_hat(x) = E[y] + sum_j phi_j(x)
4. Station-stratified local case study analyses.
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent.parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

from typing import Dict, List, Optional, Tuple, Any
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


class PredictionExplainer:
    """Renders local prediction decompositions, beeswarm summary plots, and dependence charts."""

    def __init__(
        self,
        predictions_dir: Optional[Path] = None,
        figures_dir: Optional[Path] = None
    ):
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else PREDICTIONS_DIR
        self.figures_dir = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.features = PRIMARY_MODEL_PREDICTORS

    def plot_shap_beeswarm(
        self,
        horizon: str = "h1",
        top_k: int = 15
    ) -> Path:
        """
        Renders SHAP summary beeswarm plot for top K features.
        Points represent individual sample predictions: x-axis is SHAP impact, color is feature value.
        """
        npz_file = self.predictions_dir / f"shap_values_{horizon}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing {npz_file}.")

        data = np.load(npz_file, allow_pickle=True)
        shap_vals = data["shap_values"]
        X_exp = data["X_exp"]
        features = data["features"]

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k][::-1]  # Ascending for barh/beeswarm

        fig, ax = plt.subplots(figsize=(11, 8), dpi=300)

        # Plot beeswarm jitter points for each top feature
        y_positions = np.arange(top_k)
        rng = np.random.RandomState(42)

        for pos_idx, f_idx in enumerate(top_idx):
            f_shap = shap_vals[:, f_idx]
            f_val = X_exp[:, f_idx]

            # Normalize feature values to [0, 1] for coloring
            vmin, vmax = np.percentile(f_val, 2), np.percentile(f_val, 98)
            f_norm = (f_val - vmin) / max(vmax - vmin, 1e-6)
            f_norm = np.clip(f_norm, 0.0, 1.0)

            # Add vertical random jitter
            y_jitter = pos_idx + rng.normal(0.0, 0.08, size=len(f_shap))

            scatter = ax.scatter(
                f_shap,
                y_jitter,
                c=f_norm,
                cmap=plt.cm.coolwarm,
                s=22,
                alpha=0.65,
                edgecolors="none"
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([features[idx] for idx in top_idx], fontsize=9.5)
        ax.set_title(
            f"SHAP Beeswarm Summary Plot — Horizon {horizon.upper()}\n"
            f"Directional Impact of Feature Values on Predicted PM2.5",
            fontsize=12.5, fontweight="bold", pad=12
        )
        ax.set_xlabel(r"SHAP Value ($\mu\mathrm{g/m}^3$ impact on PM2.5 prediction)", fontsize=10.5)
        ax.set_ylabel("Predictor Variable", fontsize=10.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        # Colorbar for feature value
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["Low", "Medium", "High"])
        cbar.set_label("Relative Feature Value", fontsize=10)

        out_path = self.figures_dir / f"shap_beeswarm_{horizon}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path

    def plot_shap_dependence(
        self,
        feature_name: str,
        horizon: str = "h1",
        interaction_feature: Optional[str] = None
    ) -> Path:
        """
        Renders SHAP dependence scatter plot for a given physical feature.
        """
        npz_file = self.predictions_dir / f"shap_values_{horizon}.npz"
        if not npz_file.exists():
            raise FileNotFoundError(f"Missing {npz_file}.")

        data = np.load(npz_file, allow_pickle=True)
        shap_vals = data["shap_values"]
        X_exp = data["X_exp"]
        features = list(data["features"])

        if feature_name not in features:
            raise ValueError(f"Feature '{feature_name}' not in predictor set.")

        f_idx = features.index(feature_name)
        x_vals = X_exp[:, f_idx]
        y_shap = shap_vals[:, f_idx]

        fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

        if interaction_feature is not None and interaction_feature in features:
            int_idx = features.index(interaction_feature)
            c_vals = X_exp[:, int_idx]
            vmin, vmax = np.percentile(c_vals, 2), np.percentile(c_vals, 98)
            scatter = ax.scatter(x_vals, y_shap, c=c_vals, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, s=28, alpha=0.7)
            cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(interaction_feature, fontsize=10)
        else:
            ax.scatter(x_vals, y_shap, color="#1f77b4", s=28, alpha=0.7, edgecolors="none")

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_title(
            f"SHAP Dependence Plot: {feature_name} — Horizon {horizon.upper()}",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel(f"Observed Value of {feature_name}", fontsize=10.5)
        ax.set_ylabel(r"SHAP Value ($\mu\mathrm{g/m}^3$ impact on PM2.5)", fontsize=10.5)
        ax.grid(True, linestyle=":", alpha=0.5)

        safe_name = feature_name.replace("/", "_").replace(" ", "_")
        out_path = self.figures_dir / f"shap_dependence_{horizon}_{safe_name}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path

    def explain_single_prediction(
        self,
        sample_idx: int,
        horizon: str = "h1",
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Decomposes an individual sample prediction into expected value, top positive, and top negative drivers.
        """
        npz_file = self.predictions_dir / f"shap_values_{horizon}.npz"
        data = np.load(npz_file, allow_pickle=True)
        shap_vals = data["shap_values"]
        X_exp = data["X_exp"]
        features = list(data["features"])
        expected_val = float(data["expected_value"])

        if sample_idx >= len(shap_vals):
            sample_idx = 0

        sample_shap = shap_vals[sample_idx]
        sample_feat = X_exp[sample_idx]
        predicted_val = expected_val + float(np.sum(sample_shap))

        # Rank positive and negative contributors
        pos_idx = np.where(sample_shap > 0)[0]
        neg_idx = np.where(sample_shap < 0)[0]

        top_pos = sorted(
            [{"feature": features[i], "observed_value": float(sample_feat[i]), "shap_impact": float(sample_shap[i])} for i in pos_idx],
            key=lambda d: d["shap_impact"], reverse=True
        )[:top_n]

        top_neg = sorted(
            [{"feature": features[i], "observed_value": float(sample_feat[i]), "shap_impact": float(sample_shap[i])} for i in neg_idx],
            key=lambda d: d["shap_impact"]
        )[:top_n]

        explanation = {
            "sample_index": int(sample_idx),
            "horizon": horizon,
            "base_expected_value": expected_val,
            "predicted_pm25": predicted_val,
            "net_shap_adjustment": float(np.sum(sample_shap)),
            "top_elevating_factors": top_pos,
            "top_mitigating_factors": top_neg
        }
        return explanation

    def plot_waterfall_explanation(
        self,
        sample_idx: int,
        station_name: str,
        horizon: str = "h1",
        max_display: int = 8
    ) -> Path:
        """
        Renders waterfall step chart decomposing an individual forecast into its constituent feature impacts.
        """
        exp = self.explain_single_prediction(sample_idx, horizon=horizon, top_n=max_display)
        base_val = exp["base_expected_value"]
        pred_val = exp["predicted_pm25"]

        # Interleave top elevating and mitigating factors
        top_factors = exp["top_elevating_factors"][:4] + exp["top_mitigating_factors"][:4]
        top_factors = sorted(top_factors, key=lambda d: abs(d["shap_impact"]), reverse=True)[:max_display][::-1]

        names = [f["feature"] for f in top_factors]
        impacts = [f["shap_impact"] for f in top_factors]

        colors = ["#d62728" if imp > 0 else "#2ca02c" for imp in impacts]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        bars = ax.barh(names, impacts, color=colors, edgecolor="black", height=0.6, alpha=0.85)

        for bar, imp in zip(bars, impacts):
            sign = "+" if imp > 0 else ""
            ax.text(
                bar.get_width() + (0.1 if imp > 0 else -0.1),
                bar.get_y() + bar.get_height() / 2.0,
                f"{sign}{imp:.2f}",
                va="center", ha="left" if imp > 0 else "right",
                fontsize=9, fontweight="bold"
            )

        ax.axvline(0.0, color="black", linestyle="-", linewidth=1.2)
        ax.set_title(
            f"Local Prediction Attribution Waterfall — {station_name} [{horizon.upper()}]\n"
            f"Expected Base Value: {base_val:.1f} " + r"$\mu\mathrm{g/m}^3 \longrightarrow$ " + f"Final Forecast: {pred_val:.1f} " + r"$\mu\mathrm{g/m}^3$",
            fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel(r"SHAP Contribution to PM2.5 ($\mu\mathrm{g/m}^3$)", fontsize=10.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

        out_path = self.figures_dir / f"shap_waterfall_{station_name}_{horizon}_sample{sample_idx}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_path
