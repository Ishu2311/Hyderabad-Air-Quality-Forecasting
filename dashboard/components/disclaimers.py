"""
Dashboard Components: Standardized Research Disclaimers & Safeguards
Enforces transparent academic communication and prevents misinterpretation.
Light modern styling for live environmental platform.
"""

import streamlit as st
from dashboard.components.html_utils import render_html


def render_sidebar_safeguards():
    """Renders mandatory scientific limitations directly in the sidebar."""
    with st.sidebar.expander("🛡️ Scientific Safeguards & Limits", expanded=False):
        render_html(
            """
            <div style="font-size: 11.5px; color: #475569; line-height: 1.55;">
                <p style="margin-bottom: 8px;"><strong>1. Sparse Geometry:</strong> 4 CAAQMS continuous stations for a 1,600 km² domain.</p>
                <p style="margin-bottom: 8px;"><strong>2. Spatial Transfer Penalty:</strong> LOSOCV macro MAE is 11.70 µg/m³ (11.43 µg/m³ pooled) vs 3.01 µg/m³ point error (+388% penalty). IDW only slightly beats plain donor average (11.43 vs 12.13 µg/m³).</p>
                <p style="margin-bottom: 8px;"><strong>3. Weather Reanalysis:</strong> Open-Meteo ERA5 variables reflect retrospective reanalysis, not operational NWP forecasts.</p>
                <p style="margin-bottom: 8px;"><strong>4. Quarantined Road Proxies:</strong> 19 exploratory road proxies excluded from primary models.</p>
                <p style="margin-bottom: 8px;"><strong>5. Severe Event Evaluability:</strong> 120 µg/m³ spike has only 1 test-set positive (not statistically evaluable).</p>
                <p style="margin-bottom: 8px;"><strong>6. Bollaram Seasonality:</strong> Bollaram valid 2026 data only in Jan–Feb and Aug–Sep (36% coverage due to sensor hardware outages); test statistics are seasonally confounded.</p>
                <p style="margin-bottom: 0;"><strong>7. Physical Units:</strong> Values are raw PM2.5 in µg/m³, not unitless AQI.</p>
            </div>
            """
        )


def render_spatial_safeguard_banner():
    """Renders banner explaining uncertainty tiers on spatial pages."""
    render_html(
        """
        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-left: 4px solid #e11d48; 
                    border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="font-size: 16px;">⚠️</span>
                <span style="font-size: 13.5px; font-weight: 700; color: #9f1239; letter-spacing: 0.02em; text-transform: uppercase;">
                    Mandatory Spatial Interpolation Disclosure
                </span>
            </div>
            <p style="color: #475569; margin: 0; font-size: 13px; line-height: 1.55;">
                Continuous pollution fields are generated via Inverse Distance Weighting (IDW, <em>p=2.0</em>) from 4 fixed CAAQMS physical stations. 
                Over <strong>34.9% of the metropolitan area</strong> lies beyond 15 km of any physical sensor (High Extrapolation Tier). 
                Interpolated concentrations in unmonitored sectors reflect distance decay and must 
                <strong>NEVER</strong> be interpreted as neighbourhood ground truth.
            </p>
        </div>
        """
    )


def render_explainability_disclaimer():
    """Renders warning against causal claims from SHAP attributions."""
    render_html(
        """
        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; 
                    border-radius: 12px; padding: 14px 18px; margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                <span style="font-size: 14px;">ℹ️</span>
                <span style="font-size: 13px; font-weight: 700; color: #166534;">Statistical Attribution vs. Physical Causality</span>
            </div>
            <p style="color: #475569; margin: 0; font-size: 12.5px; line-height: 1.5;">
                TreeSHAP feature importance values quantify statistical predictive associations within recursive decision partitions. 
                They demonstrate model information dependency, not chemical kinetics or direct tailpipe causality.
                <em>SHAP computed on 100 samples per horizon; interpret with caution.</em>
            </p>
        </div>
        """
    )
