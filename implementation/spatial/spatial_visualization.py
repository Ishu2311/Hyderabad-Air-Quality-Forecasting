"""
Spatial Visualization Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Generates publication-quality cartographic figures:
1. Hyderabad Metropolitan Spatial PM2.5 Prediction Surfaces (+1h, +6h, +24h, +48h).
2. Side-by-side IDW vs. Ordinary Kriging Comparison Maps.
3. Kriging Estimation Variance & Spatial Distance Uncertainty Contours.
4. Threshold Hotspot Excursion Zones (>= 60 ug/m3 and >= 120 ug/m3).
5. Explicit scientific disclaimers distinguishing physical station measurements from spatial estimates.
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
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from configs.config import (
    STATION_REGISTRY,
    FIGURES_DIR
)
from spatial.idw import InverseDistanceWeighting
from spatial.kriging import OrdinaryKriging
from spatial.hotspot_detection import HotspotDetector, THRESHOLD_ACUTE_EXCURSION, THRESHOLD_SEVERE_SPIKE

# Hyderabad Spatial Bounds in UTM Zone 44N meters
HYD_UTM_BOUNDS = {
    "easting_min": 202000.0,
    "easting_max": 248000.0,
    "northing_min": 1925000.0,
    "northing_max": 1948000.0
}

# Corresponding WGS84 Reference Coordinates for Hyderabad
# 78.20E to 78.65E, 17.30N to 17.65N


class SpatialVisualizer:
    """Renders cartographic and uncertainty visualizations for Hyderabad PM2.5 forecasting."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        grid_resolution: int = 80
    ):
        self.output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resolution = grid_resolution

        # Create regular interpolation grid in UTM Zone 44N meters
        self.xs = np.linspace(HYD_UTM_BOUNDS["easting_min"], HYD_UTM_BOUNDS["easting_max"], self.resolution)
        self.ys = np.linspace(HYD_UTM_BOUNDS["northing_min"], HYD_UTM_BOUNDS["northing_max"], self.resolution)
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)

        # Station coordinates
        self.station_names = list(STATION_REGISTRY.keys())
        self.station_coords = np.array([
            [STATION_REGISTRY[s].easting_utm44n, STATION_REGISTRY[s].northing_utm44n]
            for s in self.station_names
        ])

    def plot_horizon_prediction_map(
        self,
        station_predictions: Dict[str, float],
        horizon: str,
        method: str = "idw",
        filename_suffix: str = ""
    ) -> Path:
        """
        Renders continuous PM2.5 spatial prediction map for a given forecasting horizon.
        """
        vals = np.array([station_predictions.get(s, np.nan) for s in self.station_names])

        if method == "idw":
            interp = InverseDistanceWeighting(power=2.0, coordinate_system="utm")
            interp.fit(self.station_coords, vals, self.station_names)
            z_grid, dist_grid = interp.interpolate_grid(self.grid_x, self.grid_y)
            method_label = "Inverse Distance Weighting (IDW, p=2.0)"
        else:
            interp = OrdinaryKriging(variogram_model="exponential", coordinate_system="utm")
            interp.fit(self.station_coords, vals, self.station_names)
            z_grid, var_grid = interp.interpolate_grid(self.grid_x, self.grid_y)
            method_label = "Ordinary Kriging (BLUE, Exponential)"

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # Colormap levels from 10 to 100 ug/m3
        levels = np.linspace(15.0, 85.0, 29)
        cmap = plt.cm.YlOrRd

        cf = ax.contourf(
            (self.grid_x - 200000) / 1000.0,
            (self.grid_y - 1920000) / 1000.0,
            z_grid,
            levels=levels,
            cmap=cmap,
            extend="both"
        )
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Predicted $\mathrm{PM}_{2.5}$ Concentration ($\mu\mathrm{g/m}^3$)", fontsize=11, fontweight="bold")

        # Overlay statutory 60 ug/m3 contour line
        cs_acute = ax.contour(
            (self.grid_x - 200000) / 1000.0,
            (self.grid_y - 1920000) / 1000.0,
            z_grid,
            levels=[60.0],
            colors=["black"],
            linewidths=[2.2],
            linestyles=["--"]
        )
        ax.clabel(cs_acute, inline=True, fontsize=10, fmt=r"Statutory NAAQS (60 $\mu\mathrm{g/m}^3$)")

        # Plot physical stations
        for i, s_name in enumerate(self.station_names):
            x_st = (self.station_coords[i, 0] - 200000) / 1000.0
            y_st = (self.station_coords[i, 1] - 1920000) / 1000.0
            pred_val = vals[i]

            ax.scatter(x_st, y_st, color="black", edgecolor="white", s=130, marker="^", zorder=5)
            val_str = f"{pred_val:.1f}" if not np.isnan(pred_val) else "N/A"
            ax.annotate(
                f"{s_name}\n({val_str} " + r"$\mu\mathrm{g/m}^3$)",
                (x_st, y_st),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
                zorder=6
            )

        ax.set_title(
            f"Hyderabad Metropolitan PM2.5 Spatial Forecast [{horizon.upper()}]\n"
            f"Spatial Surface: {method_label}",
            fontsize=13,
            fontweight="bold",
            pad=14
        )
        ax.set_xlabel("UTM Zone 44N Easting Offset (km from 200,000m E)", fontsize=10)
        ax.set_ylabel("UTM Zone 44N Northing Offset (km from 1,920,000m N)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Footnote scientific disclaimer
        fig.text(
            0.5, 0.02,
            "Research Disclaimer: Spatial surface represents continuous mathematical interpolation across 4 physical CAAQMS stations.\n"
            "Interpolated values beyond station proximities reflect exploratory estimates and must not be treated as confirmed ground-level sensors.",
            ha="center", fontsize=7.5, style="italic", color="#444444"
        )

        out_file = self.output_dir / f"spatial_forecast_{horizon}_{method}{filename_suffix}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_file

    def plot_idw_vs_kriging_comparison(
        self,
        station_predictions: Dict[str, float],
        horizon: str
    ) -> Path:
        """
        Renders side-by-side IDW vs Ordinary Kriging comparison map.
        """
        vals = np.array([station_predictions.get(s, np.nan) for s in self.station_names])

        idw = InverseDistanceWeighting(power=2.0, coordinate_system="utm")
        idw.fit(self.station_coords, vals, self.station_names)
        z_idw, _ = idw.interpolate_grid(self.grid_x, self.grid_y)

        krg = OrdinaryKriging(variogram_model="exponential", coordinate_system="utm")
        krg.fit(self.station_coords, vals, self.station_names)
        z_krg, var_krg = krg.interpolate_grid(self.grid_x, self.grid_y)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        levels = np.linspace(15.0, 85.0, 29)
        cmap = plt.cm.YlOrRd

        # Subplot 1: IDW
        cf1 = ax1.contourf(
            (self.grid_x - 200000) / 1000.0, (self.grid_y - 1920000) / 1000.0,
            z_idw, levels=levels, cmap=cmap, extend="both"
        )
        ax1.set_title(f"A. Inverse Distance Weighting (IDW, p=2.0)\nHorizon {horizon.upper()}", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Easting Offset (km)", fontsize=9)
        ax1.set_ylabel("Northing Offset (km)", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Subplot 2: Kriging
        cf2 = ax2.contourf(
            (self.grid_x - 200000) / 1000.0, (self.grid_y - 1920000) / 1000.0,
            z_krg, levels=levels, cmap=cmap, extend="both"
        )
        ax2.set_title(f"B. Ordinary Kriging (Exponential BLUE)\nHorizon {horizon.upper()}", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Easting Offset (km)", fontsize=9)
        ax2.set_ylabel("Northing Offset (km)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)

        for ax in [ax1, ax2]:
            for i, s_name in enumerate(self.station_names):
                x_st = (self.station_coords[i, 0] - 200000) / 1000.0
                y_st = (self.station_coords[i, 1] - 1920000) / 1000.0
                ax.scatter(x_st, y_st, color="black", edgecolor="white", s=90, marker="^", zorder=5)

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(cf2, cax=cbar_ax)
        cbar.set_label(r"Predicted $\mathrm{PM}_{2.5}$ ($\mu\mathrm{g/m}^3$)", fontsize=11, fontweight="bold")

        fig.suptitle(
            f"Spatial Interpolation Method Comparison (IDW vs. Ordinary Kriging) — Horizon {horizon.upper()}",
            fontsize=13, fontweight="bold", y=0.98
        )

        out_file = self.output_dir / f"idw_vs_kriging_comparison_{horizon}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_file

    def plot_spatial_uncertainty_map(
        self,
        station_predictions: Dict[str, float],
        horizon: str
    ) -> Path:
        """
        Renders spatial uncertainty map: Kriging variance and nearest-station distance zones.
        """
        vals = np.array([station_predictions.get(s, np.nan) for s in self.station_names])

        krg = OrdinaryKriging(variogram_model="exponential", coordinate_system="utm")
        krg.fit(self.station_coords, vals, self.station_names)
        _, var_krg = krg.interpolate_grid(self.grid_x, self.grid_y)

        # Compute Euclidean distance in km to nearest reporting station
        valid_coords = self.station_coords[~np.isnan(vals)]
        Ny, Nx = self.grid_x.shape
        flat_pts = np.column_stack([self.grid_x.ravel(), self.grid_y.ravel()])
        dx = flat_pts[:, 0:1] - valid_coords[:, 0:1].T
        dy = flat_pts[:, 1:2] - valid_coords[:, 1:2].T
        min_dists_km = np.min(np.sqrt(dx**2 + dy**2), axis=1).reshape(Ny, Nx) / 1000.0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

        # Plot 1: Kriging Standard Error (sqrt of variance)
        std_err = np.sqrt(np.maximum(var_krg, 0.0))
        cf1 = ax1.contourf(
            (self.grid_x - 200000) / 1000.0, (self.grid_y - 1920000) / 1000.0,
            std_err, levels=20, cmap=plt.cm.Blues
        )
        cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label(r"Kriging Standard Error $\sigma_k$ ($\mu\mathrm{g/m}^3$)", fontsize=10)
        ax1.set_title("A. Theoretical Kriging Estimation Standard Error", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Easting Offset (km)", fontsize=9)
        ax1.set_ylabel("Northing Offset (km)", fontsize=9)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Plot 2: Distance to Nearest Sensor
        cf2 = ax2.contourf(
            (self.grid_x - 200000) / 1000.0, (self.grid_y - 1920000) / 1000.0,
            min_dists_km, levels=[0, 5, 15, 30], cmap=plt.cm.RdYlGn_r
        )
        cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Distance to Nearest Reporting Sensor (km)", fontsize=10)
        ax2.set_title("B. Geometric Monitoring Uncertainty Tiers", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Easting Offset (km)", fontsize=9)
        ax2.set_ylabel("Northing Offset (km)", fontsize=9)
        ax2.grid(True, linestyle=":", alpha=0.5)

        # Mark stations
        for ax in [ax1, ax2]:
            for i, s_name in enumerate(self.station_names):
                x_st = (self.station_coords[i, 0] - 200000) / 1000.0
                y_st = (self.station_coords[i, 1] - 1920000) / 1000.0
                ax.scatter(x_st, y_st, color="black", edgecolor="white", s=100, marker="^", zorder=5)
                ax.annotate(s_name, (x_st, y_st), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

        fig.suptitle(
            f"Spatial Interpolation Uncertainty and Extrapolation Risk Diagnostics — Horizon {horizon.upper()}",
            fontsize=13, fontweight="bold", y=0.98
        )

        out_file = self.output_dir / f"spatial_uncertainty_{horizon}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_file

    def plot_hotspot_excursion_zones(
        self,
        station_predictions: Dict[str, float],
        horizon: str
    ) -> Path:
        """
        Renders hotspot delineation map highlighting acute excursion (>= 60 ug/m3) zones.
        """
        vals = np.array([station_predictions.get(s, np.nan) for s in self.station_names])

        idw = InverseDistanceWeighting(power=2.0, coordinate_system="utm")
        idw.fit(self.station_coords, vals, self.station_names)
        z_idw, dist_grid = idw.interpolate_grid(self.grid_x, self.grid_y)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

        # Binary/categorical hotspot map
        # Level 0: Compliant (< 60)
        # Level 1: Acute Excursion (60 - 120)
        # Level 2: Severe Spike (>= 120)
        hotspot_grid = np.zeros_like(z_idw)
        hotspot_grid[z_idw >= 60.0] = 1.0
        hotspot_grid[z_idw >= 120.0] = 2.0

        cmap = mcolors.ListedColormap(["#d4edda", "#fff3cd", "#f8d7da"])
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        im = ax.imshow(
            hotspot_grid,
            extent=[
                (HYD_UTM_BOUNDS["easting_min"] - 200000) / 1000.0,
                (HYD_UTM_BOUNDS["easting_max"] - 200000) / 1000.0,
                (HYD_UTM_BOUNDS["northing_min"] - 1920000) / 1000.0,
                (HYD_UTM_BOUNDS["northing_max"] - 1920000) / 1000.0
            ],
            origin="lower",
            cmap=cmap,
            norm=norm,
            aspect="auto"
        )

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels([
            r"Compliant (< 60 $\mu\mathrm{g/m}^3$)",
            r"Acute Excursion ($\geq 60\ \mu\mathrm{g/m}^3$)",
            r"Severe Spike ($\geq 120\ \mu\mathrm{g/m}^3$)"
        ], fontsize=9.5)

        # Overlay continuous concentration contour lines
        ax.contour(
            (self.grid_x - 200000) / 1000.0, (self.grid_y - 1920000) / 1000.0,
            z_idw, levels=[60.0, 120.0], colors=["#e0a800", "#c82333"], linewidths=[2.0, 2.5]
        )

        # Stations
        for i, s_name in enumerate(self.station_names):
            x_st = (self.station_coords[i, 0] - 200000) / 1000.0
            y_st = (self.station_coords[i, 1] - 1920000) / 1000.0
            pred_val = vals[i]
            val_str = f"{pred_val:.1f}" if not np.isnan(pred_val) else "N/A"
            ax.scatter(x_st, y_st, color="black", edgecolor="white", s=130, marker="^", zorder=5)
            ax.annotate(
                f"{s_name}\n({val_str} " + r"$\mu\mathrm{g/m}^3$)",
                (x_st, y_st),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85)
            )

        ax.set_title(
            f"Hyderabad Air Quality Hotspot Delineation [{horizon.upper()}]\n"
            f"Threshold-Based Classification & Extrapolation Zones",
            fontsize=13, fontweight="bold", pad=14
        )
        ax.set_xlabel("UTM Zone 44N Easting Offset (km from 200,000m E)", fontsize=10)
        ax.set_ylabel("UTM Zone 44N Northing Offset (km from 1,920,000m N)", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        out_file = self.output_dir / f"hotspot_delineation_{horizon}.png"
        fig.savefig(out_file, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return out_file
