"""
Dashboard Page: 5. Explainability (TreeSHAP)
Presents global and local feature attributions, category transitions across
horizons, SHAP summary/beeswarm figures, and natural language narratives.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.explainability_service import (
    get_top_features_dataframe,
    get_category_shares_dataframe,
    get_station_narrative
)
from dashboard.backend.forecasting_service import get_available_stations
from dashboard.backend.data_loader import get_figure_path
from dashboard.components.disclaimers import render_explainability_disclaimer
from dashboard.styles.theme import render_page_header, apply_plotly_theme, inject_custom_css, render_html


def render_page_explainability():
    """Renders Explainability section."""
    inject_custom_css()

    render_page_header(
        badge="🧠 MODEL INTERPRETABILITY • AXIOMATIC TREESHAP",
        title="Explainability & Feature Attribution Analysis",
        subtitle="Axiomatic Shapley Additive Explanations across 84 Approved Predictors Isolating Atmospheric, Solar, and Autoregressive Drivers."
    )

    # Non-Causal Attribution Warning
    render_explainability_disclaimer()

    # Horizon Selector
    horizon_options = {
        "+1h (Nowcasting Horizon)": "h1",
        "+6h (Sub-Daily Horizon)": "h6",
        "+24h (Statutory 24h Horizon)": "h24",
        "+48h (Medium-Range Horizon)": "h48"
    }
    selected_h_label = st.selectbox(
        "Select Forecasting Horizon to Explain",
        options=list(horizon_options.keys()),
        index=0
    )
    selected_h_key = horizon_options[selected_h_label]

    st.info("ℹ️ **Methodological Disclosure:** SHAP computed on 100 samples per horizon; interpret with caution.")

    st.markdown("---")

    # Top Features & Category Shares
    col_f1, col_f2 = st.columns([3, 2])

    with col_f1:
        st.subheader(f"Top 15 Most Influential Predictors ({selected_h_label})")
        df_top = get_top_features_dataframe(horizon=selected_h_key, top_n=15)

        if not df_top.empty:
            category_colors = {
                "Historical PM2.5 Dynamics": "#06b6d4",
                "Thermodynamics & Boundary Layer": "#10b981",
                "Solar Geometry & Diurnal Time": "#f59e0b",
                "Wind Dynamics (Orthogonal Vectors)": "#3b82f6",
                "Atmospheric Stagnation": "#8b5cf6"
            }

            fig_bar = px.bar(
                df_top,
                x="Mean |SHAP| (ug/m3)",
                y="Feature Name",
                orientation="h",
                color="Feature Category",
                color_discrete_map=category_colors,
                labels={"Feature Name": "Predictor Feature", "Mean |SHAP| (ug/m3)": "Mean Absolute SHAP (µg/m³)"}
            )
            apply_plotly_theme(fig_bar, height=460, title="Global Feature Importance: Mean |SHAP| Impact on Predicted PM2.5")
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, width='stretch')

    with col_f2:
        st.subheader("Attribution Share by Domain")
        df_shares = get_category_shares_dataframe(horizon=selected_h_key)

        if not df_shares.empty:
            fig_pie = px.pie(
                df_shares,
                names="Feature Group",
                values="Attribution Share (%)",
                hole=0.45,
                color_discrete_sequence=["#06b6d4", "#10b981", "#f59e0b", "#3b82f6", "#8b5cf6"]
            )
            apply_plotly_theme(fig_pie, height=460, title="Relative Information Share (%)")
            fig_pie.update_layout(
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1)
            )
            st.plotly_chart(fig_pie, width='stretch')

    st.markdown("---")

    # The Core Temporal Information Shift
    render_html(
        """
        <div class="env-card">
            <div class="env-card-header">
                <div class="env-card-title">⏳ The Temporal Transition: From Autoregression to Synoptic Meteorology</div>
                <span class="badge badge-teal">Core Research Discovery</span>
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.65;">
                <p>
                    A central scientific finding of this investigation is empirical proof of how particulate drivers transition 
                    as the forecast horizon expands from nowcasting (<strong>+1h</strong>) to medium-range advisory (<strong>+48h</strong>):
                </p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 12px;">
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #06b6d4;">
                        <div style="color: #38bdf8; font-weight: 700; font-size: 13px;">+1h Nowcasting</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            <strong>79.6% Autoregressive Share.</strong> Immediate sensor readings and 3h rolling averages dominate due to aerosol inertia.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #10b981;">
                        <div style="color: #34d399; font-weight: 700; font-size: 13px;">+6h Sub-Daily</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            <strong>62.3% Autoregressive.</strong> Solar zenith angles (9.2%) and boundary layer height (10.0%) grow as diurnal heating activates.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #f59e0b;">
                        <div style="color: #fbbf24; font-weight: 700; font-size: 13px;">+24h Planning</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            <strong>12.2% Thermodynamics.</strong> Surface pressure and thermal inversions become primary determinants of 24h accumulation.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #8b5cf6;">
                        <div style="color: #a78bfa; font-weight: 700; font-size: 13px;">+48h Advisory</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            <strong>16.8% Thermodynamics.</strong> Autoregressive nowcasts completely decay; macro-scale weather forces ventilation vs stagnation.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    st.markdown("---")

    # High-Resolution Publication Plots
    st.subheader("Publication-Grade TreeSHAP Visualizations")
    st.caption("SHAP beeswarm distributions and partial dependence surfaces.")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        beeswarm_fig = get_figure_path(f"shap_beeswarm_{selected_h_key}.png")
        if beeswarm_fig:
            st.image(str(beeswarm_fig), caption=f"SHAP Beeswarm Distribution ({selected_h_label})", width='stretch')
        else:
            st.info("Beeswarm figure artifact not found.")

    with col_p2:
        dep_name = "shap_dependence_h1_PM2.5.png" if selected_h_key == "h1" else (
            "shap_dependence_h6_wind_speed_10m.png" if selected_h_key == "h6" else "shap_dependence_h24_pm25_roll_mean_24h.png"
        )
        dep_fig = get_figure_path(dep_name)
        if dep_fig:
            st.image(str(dep_fig), caption=f"Non-Linear Feature Dependence: {dep_name}", width='stretch')
        else:
            st.info("Dependence plot artifact not found.")

    st.markdown("---")

    # Station-Wise Narrative Explanations
    st.subheader("Automated Natural Language Prediction Narratives")
    st.caption("Synthesizes local TreeSHAP contributions into actionable, stakeholder-ready diagnostic narratives.")

    available_stations = get_available_stations()
    narrative_st = st.selectbox(
        "Select Monitoring Station for Local Narrative",
        options=available_stations,
        index=0
    )

    narrative_text = get_station_narrative(narrative_st)
    render_html(
        f"""
        <div class="env-card" style="border-left: 4px solid #10b981;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 14px; font-weight: 700; color: #34d399;">
                    Diagnostic Narrative: {narrative_st}
                </span>
                <span class="badge badge-green">TreeSHAP Synthesis</span>
            </div>
            <p style="color: #f1f5f9; font-size: 13.5px; line-height: 1.65; margin: 0;">
                {narrative_text.replace(chr(10), '<br>')}
            </p>
        </div>
        """
    )

    # Show station waterfall plot if available
    waterfall_fig_name = "shap_waterfall_Bollaram_Industrial_h24_sample25.png" if "Bollaram" in narrative_st else (
        "shap_waterfall_ECIL_Kapra_h1_sample10.png" if "ECIL" in narrative_st else "shap_waterfall_ECIL_Kapra_h1_sample10.png"
    )
    waterfall_fig = get_figure_path(waterfall_fig_name)
    if waterfall_fig:
        st.image(str(waterfall_fig), caption=f"Local SHAP Waterfall Attribution ({waterfall_fig_name})", width='stretch')

    st.markdown("---")

    # Document 25-tree RF approximation
    st.markdown(
        r"""
        > [!NOTE]
        > **Methodological Clarification (TreeSHAP Ensemble Formulation):**  
        > For horizons +1h and +6h, feature attribution represents an exact linear composite of the Stacking ensemble 
        > ($\phi_{\text{stack}} = \sum_m \beta_m \phi_m$). To ensure computational tractability over 84 features, the Random Forest 
        > constituent estimator was sub-sampled to 25 trees. As verified in `MILESTONE_4_INDEPENDENT_AUDIT.md`, this yields an average 
        > attribution difference of only **0.075 µg/m³** compared to the full 100-tree model—well below sensor measurement precision (±1.0 µg/m³).
        """
    )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Explainability - Hyderabad PM2.5",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    from dashboard.components.disclaimers import render_sidebar_safeguards
    render_sidebar_safeguards()
    render_page_explainability()
