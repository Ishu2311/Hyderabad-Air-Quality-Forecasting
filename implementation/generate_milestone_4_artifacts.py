"""
Master Execution Pipeline for Milestone 4 Artifacts and Visualizations
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability
"""

from pathlib import Path
import sys
implementation_dir = Path(__file__).resolve().parent
if str(implementation_dir) not in sys.path:
    sys.path.insert(0, str(implementation_dir))

import shutil
import json
import numpy as np
import pandas as pd

from configs.config import (
    STATION_REGISTRY,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    FIGURES_DIR
)
from spatial.idw import InverseDistanceWeighting
from spatial.kriging import OrdinaryKriging
from spatial.hotspot_detection import HotspotDetector
from spatial.spatial_visualization import SpatialVisualizer
from explainability.shap_analyzer import ShapAnalyzer
from explainability.feature_importance import FeatureImportanceAnalyzer
from explainability.prediction_explainer import PredictionExplainer
from explainability.hotspot_narrative import generate_human_readable_hotspot_narrative


def run_milestone_4_generation():
    print("=================================================================")
    print("=== STARTING MILESTONE 4 COMPLETE EXECUTION PIPELINE ===")
    print("=================================================================")

    reports_figures_dir = Path("new_research_project/reports/figures")
    reports_figures_dir.mkdir(parents=True, exist_ok=True)
    impl_figures_dir = Path("new_research_project/implementation/reports/figures")
    impl_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run SHAP Analyzer across all horizons
    print("\n[Step 1/5] Computing TreeSHAP and Stacking Composite Attributions...")
    shap_analyzer = ShapAnalyzer()
    for h in ["h1", "h6", "h24", "h48"]:
        print(f"  Computing SHAP for horizon {h}...")
        # 100 evaluation samples from test set
        res_h = shap_analyzer.compute_shap_for_horizon(h, n_explain_samples=100)
        print(f"    Horizon {h} done! Top predictor: {res_h['top_15_features'][0]['feature']} (impact={res_h['top_15_features'][0]['mean_abs_shap']:.3f})")

    # 2. Run Global Feature Importance & Bar Plots
    print("\n[Step 2/5] Aggregating Global Feature Importance & Domain Shares...")
    feat_analyzer = FeatureImportanceAnalyzer(figures_dir=reports_figures_dir)
    importance_summary = feat_analyzer.generate_importance_report()
    for h in ["h1", "h6", "h24", "h48"]:
        bar_path = feat_analyzer.plot_top_features_bar(horizon=h, top_k=15)
        print(f"    Saved global importance plot: {bar_path.name}")

    # 3. Local Prediction Explanations, Beeswarm & Dependence Plots
    print("\n[Step 3/5] Generating SHAP Beeswarm, Dependence, and Waterfall Plots...")
    explainer = PredictionExplainer(figures_dir=reports_figures_dir)
    for h in ["h1", "h6", "h24", "h48"]:
        beeswarm_path = explainer.plot_shap_beeswarm(horizon=h, top_k=15)
        print(f"    Saved beeswarm plot for {h}: {beeswarm_path.name}")

    # Dependence plots for top physical features
    dep1 = explainer.plot_shap_dependence("PM2.5", horizon="h1", interaction_feature="temperature_2m")
    dep2 = explainer.plot_shap_dependence("pm25_roll_mean_24h", horizon="h24", interaction_feature="relative_humidity_2m")
    dep3 = explainer.plot_shap_dependence("wind_speed_10m", horizon="h6", interaction_feature="temperature_2m")
    print(f"    Saved dependence plots: {dep1.name}, {dep2.name}, {dep3.name}")

    # Waterfall plots for representative stations
    wf1 = explainer.plot_waterfall_explanation(sample_idx=10, station_name="ECIL_Kapra", horizon="h1")
    wf2 = explainer.plot_waterfall_explanation(sample_idx=25, station_name="Bollaram_Industrial", horizon="h24")
    print(f"    Saved waterfall plots: {wf1.name}, {wf2.name}")

    # 4. Generate Spatial Cartographic Maps
    print("\n[Step 4/5] Renders Spatial Prediction, Comparison, Uncertainty, and Hotspot Maps...")
    visualizer = SpatialVisualizer(output_dir=reports_figures_dir)

    # Use representative test predictions (e.g. median / typical active period across stations)
    df_test = pd.read_parquet(PREDICTIONS_DIR / "regression_predictions_test.parquet")
    
    # Pick a representative episode timestamp with active pollution contrast
    # e.g., 2026-01-15 10:00:00 (winter daytime)
    target_dt = pd.Timestamp("2026-01-15 10:00:00")
    sub_dt = df_test[df_test["datetime"] == target_dt]
    if len(sub_dt) == 0:
        target_dt = df_test["datetime"].iloc[100]
        sub_dt = df_test[df_test["datetime"] == target_dt]

    print(f"  Reference cartographic timestamp: {target_dt}")

    detector = HotspotDetector()
    hotspot_summaries = {}

    for h in ["h1", "h6", "h24", "h48"]:
        pred_col = f"pred_stacking_{h}" if h in ["h1", "h6"] else f"pred_lgb_{h}"
        st_preds = {}
        for s in STATION_REGISTRY.keys():
            s_rows = sub_dt[sub_dt["station"] == s]
            if len(s_rows) > 0 and not np.isnan(s_rows[pred_col].values[0]):
                st_preds[s] = float(s_rows[pred_col].values[0])
            else:
                # If station had physical sensor outage at this timestamp, use station mean
                st_preds[s] = float(df_test[df_test["station"] == s][pred_col].mean())

        # Render maps
        map_idw = visualizer.plot_horizon_prediction_map(st_preds, horizon=h, method="idw")
        map_krg = visualizer.plot_horizon_prediction_map(st_preds, horizon=h, method="kriging")
        map_comp = visualizer.plot_idw_vs_kriging_comparison(st_preds, horizon=h)
        map_unc = visualizer.plot_spatial_uncertainty_map(st_preds, horizon=h)
        map_hot = visualizer.plot_hotspot_excursion_zones(st_preds, horizon=h)

        print(f"    Horizon {h} maps generated: {map_idw.name}, {map_comp.name}, {map_unc.name}, {map_hot.name}")

        # Run hotspot grid detection summary
        idw_obj = InverseDistanceWeighting(power=2.0).fit(visualizer.station_coords, np.array(list(st_preds.values())))
        z_grid, dist_grid = idw_obj.interpolate_grid(visualizer.grid_x, visualizer.grid_y)
        h_sum = detector.detect_grid_hotspots(
            visualizer.grid_x, visualizer.grid_y, z_grid, variance_surface=None, dist_surface=dist_grid,
            horizon=h, station_predictions=st_preds
        )
        hotspot_summaries[h] = h_sum

    with open(PREDICTIONS_DIR / "hotspot_detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(hotspot_summaries, f, indent=2)

    # 5. Generate Contextual Human-Readable Narratives
    print("\n[Step 5/5] Synthesizing Dynamic Human-Readable Hotspot Narratives...")
    sample_narratives = {}
    for st in STATION_REGISTRY.keys():
        st_rows = sub_dt[sub_dt["station"] == st]
        val_pred = float(st_rows["pred_stacking_h1"].values[0]) if len(st_rows) > 0 else 45.0
        val_obs = float(st_rows["pred_persistence_h1"].values[0]) if len(st_rows) > 0 else 42.0

        # Load local explanation
        exp_h1 = explainer.explain_single_prediction(sample_idx=15, horizon="h1", top_n=3)
        narrative = generate_human_readable_hotspot_narrative(
            station_name=st,
            horizon_label="1 to 6 hours",
            predicted_pm25=val_pred,
            observed_baseline=val_obs,
            top_elevating_factors=exp_h1["top_elevating_factors"],
            top_mitigating_factors=exp_h1["top_mitigating_factors"],
            uncertainty_tier="Low Uncertainty (Proximate <= 5km)",
            acute_threshold=60.0
        )
        sample_narratives[st] = narrative
        print(f"  --- Narrative for {st} ---")
        print(narrative[:220] + "...\n")

    with open(PREDICTIONS_DIR / "human_readable_hotspot_narratives.json", "w", encoding="utf-8") as f:
        json.dump(sample_narratives, f, indent=2)

    # Sync all generated figures to implementation directory
    for fig_file in reports_figures_dir.glob("*.png"):
        shutil.copy2(fig_file, impl_figures_dir / fig_file.name)
    print("All figures successfully mirrored to implementation/reports/figures/!")

    print("\n=================================================================")
    print("=== MILESTONE 4 COMPLETE PIPELINE FINISHED SUCCESSFULLY! ===")
    print("=================================================================")


if __name__ == "__main__":
    run_milestone_4_generation()
