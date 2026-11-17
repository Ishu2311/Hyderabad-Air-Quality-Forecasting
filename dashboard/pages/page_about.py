"""
Dashboard Page: 7. About the Research
Presents research motivation, scholarly gap analysis, formal objectives,
core scientific contributions, technologies used, and future roadmap.
Clean light theme styling with academic accreditation.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.styles.theme import render_page_header, inject_custom_css, render_html


def render_page_about():
    """Renders About the Research section."""
    inject_custom_css()

    render_page_header(
        badge="🎓 UNIVERSITY CAPSTONE RESEARCH • DEFENSE 2026",
        title="About the Research & Scientific Contributions",
        subtitle="Spatio-Temporal Air Quality Modeling, Transferability Auditing, and Explainable Machine Learning in Greater Hyderabad."
    )

    # 1. Problem Statement
    render_html(
        """
        <div class="env-card">
            <div class="env-card-header">
                <div class="env-card-title">🚨 1. Problem Statement & Urban Context</div>
                <span class="badge badge-rose">Metropolitan Challenge</span>
            </div>
            <p style="color: #334155; font-size: 13.5px; line-height: 1.65; margin: 0 0 10px 0;">
                Rapid demographic growth, expanding vehicular density, and extensive bulk-drug pharmaceutical production 
                have positioned ambient fine particulate matter (<strong>PM<sub>2.5</sub></strong>) as a severe chronic health crisis across 
                the <strong>Greater Hyderabad Metropolitan Area</strong>.
            </p>
            <p style="color: #64748b; font-size: 13px; line-height: 1.6; margin: 0;">
                However, continuous regulatory ambient monitoring in Hyderabad is restricted to only <strong>four physical Continuous Ambient 
                Air Quality Monitoring Stations (CAAQMS)</strong> across a metropolitan jurisdiction spanning over <strong>1,600 km²</strong> 
                and home to more than 10 million residents. This spatial sparsity creates an urgent research and operational challenge: 
                <em>how can municipal authorities and public health planners anticipate localized air pollution excursions when over 80% of urban wards 
                possess no continuous regulatory ground sensor?</em>
            </p>
        </div>
        """
    )

    st.markdown("---")

    # 2. Scholarly Gap Analysis
    st.subheader("2. Identification of the Scholarly Research Gap")
    st.markdown(
        """
        A rigorous forensic survey of contemporary published machine learning literature on Indian urban air quality modeling identified three persistent methodological flaws:
        """
    )

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        render_html(
            """
            <div class="env-card" style="height: 100%; border-top: 3px solid #ef4444;">
                <div style="font-size: 14px; font-weight: 700; color: #dc2626; margin-bottom: 6px;">
                    1. The Spatial Transfer Blindspot
                </div>
                <div style="font-size: 12.5px; color: #475569; line-height: 1.55;">
                    The vast majority of published papers report deceptively high R² > 0.90 by training and testing models at the exact same physical sensor locations. 
                    They completely fail to evaluate <strong>spatial generalizability</strong> to unmonitored sectors, concealing severe spatial degradation penalties.
                </div>
            </div>
            """
        )

    with col_g2:
        render_html(
            """
            <div class="env-card" style="height: 100%; border-top: 3px solid #f59e0b;">
                <div style="font-size: 14px; font-weight: 700; color: #d97706; margin-bottom: 6px;">
                    2. Unverified Spatial Proxies
                </div>
                <div style="font-size: 12.5px; color: #475569; line-height: 1.55;">
                    Previous attempts to incorporate road and highway features relied on unverified static assumptions rather than verified spatial geometries. 
                    This creates illusory feature importance that fails when audited against genuine OpenStreetMap shapefiles.
                </div>
            </div>
            """
        )

    with col_g3:
        render_html(
            """
            <div class="env-card" style="height: 100%; border-top: 3px solid #0f766e;">
                <div style="font-size: 14px; font-weight: 700; color: #0f766e; margin-bottom: 6px;">
                    3. Temporal Boundary Leakage
                </div>
                <div style="font-size: 12.5px; color: #475569; line-height: 1.55;">
                    Standard random cross-validation or unbuffered rolling calculations leak future target values into past features. 
                    This study implements strict chronological partitioning with <strong>48-hour boundary purging buffers</strong>.
                </div>
            </div>
            """
        )

    st.markdown("---")

    # 3. Formal Research Objectives & Core Contributions
    col_obj1, col_obj2 = st.columns(2)

    with col_obj1:
        render_html(
            """
            <div class="env-card" style="height: 100%;">
                <div class="env-card-header">
                    <div class="env-card-title">🎯 Formal Research Objectives</div>
                    <span class="badge badge-teal">Objectives</span>
                </div>
                <div style="font-size: 12.5px; color: #475569; line-height: 1.6;">
                    <p>• <strong>Primary Objective:</strong> Formulate, audit, and benchmark a leak-free multi-source machine learning pipeline for multi-horizon PM2.5 forecasting (+1h, +6h, +24h, +48h) and continuous spatial hotspot mapping across Greater Hyderabad.</p>
                    <p>• <strong>Temporal Skill Quantification:</strong> Benchmark gradient-boosted decision trees and stacking ensembles against Zero-Order Persistence and Seasonal-Naive baselines on identical sample masks.</p>
                    <p>• <strong>Spatial Honesty Assessment:</strong> Execute Leave-One-Station-Out Cross-Validation (LOSOCV) to empirically quantify the true spatial degradation penalty (+388% error increase).</p>
                    <p>• <strong>Axiomatic Interpretability:</strong> Formulate TreeSHAP attributions and natural-language prediction narratives to distinguish predictive associations from physical causality.</p>
                </div>
            </div>
            """
        )

    with col_obj2:
        render_html(
            """
            <div class="env-card" style="height: 100%;">
                <div class="env-card-header">
                    <div class="env-card-title">🏆 Key Research Contributions</div>
                    <span class="badge badge-green">Novelty</span>
                </div>
                <div style="font-size: 12.5px; color: #475569; line-height: 1.6;">
                    <p>• <strong>Empirical Spatial Penalty Discovery:</strong> Quantifies that point nowcasting precision (3.01 µg/m³ MAE) degrades to 11.70 µg/m³ macro MAE (11.43 µg/m³ pooled) under spatial transfer across sparse urban geometry.</p>
                    <p>• <strong>The Temporal Transition Mechanism:</strong> Proves empirically via TreeSHAP that aerosol inertia dominates short horizons (79.6% at +1h), while synoptic thermodynamics govern medium horizons (16.8% at +48h).</p>
                    <p>• <strong>Methodological Quarantine of Road Proxies:</strong> Establishes scientific precedent by actively quarantining 19 unverified static road metrics rather than reporting artificial improvements.</p>
                    <p>• <strong>Fully Audited Reproducibility:</strong> End-to-end automated test suite verifying exact numerical integrity with 64/64 automated tests passing.</p>
                </div>
            </div>
            """
        )

    st.markdown("---")

    # 4. Technologies Used
    st.subheader("4. Scientific Computing & Technology Stack")
    render_html(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;">
            <span class="badge badge-teal">Python 3.13</span>
            <span class="badge badge-teal">Streamlit 1.52</span>
            <span class="badge badge-blue">Scikit-Learn</span>
            <span class="badge badge-green">LightGBM</span>
            <span class="badge badge-green">XGBoost</span>
            <span class="badge badge-blue">TreeSHAP</span>
            <span class="badge badge-amber">PyKrige</span>
            <span class="badge badge-teal">Pydeck & Mapbox</span>
            <span class="badge badge-blue">Plotly Express</span>
            <span class="badge badge-blue">Pandas & NumPy</span>
            <span class="badge badge-teal">CPCB CAAQMS In-Situ Telemetry</span>
            <span class="badge badge-blue">Open-Meteo ERA5 Reanalysis</span>
            <span class="badge badge-green">Pytest Test Harness (64/64 Passing)</span>
        </div>
        """
    )

    st.markdown("---")

    # 5. Limitations & Future Scope
    st.subheader("5. Research Limitations & Future Roadmap")
    render_html(
        """
        <div class="env-card">
            <div class="env-card-header">
                <div class="env-card-title">🚀 Roadmap for Future Expansion</div>
                <span class="badge badge-blue">Future Work</span>
            </div>
            <div style="font-size: 13px; color: #475569; line-height: 1.65;">
                <p>• <strong>Dense Low-Cost Sensor Calibration:</strong> Overcoming the 388% spatial transfer penalty requires integrating high-density low-cost optical particulate sensors (e.g. 50+ PurpleAir or Plantower nodes calibrated against reference CAAQMS stations).</p>
                <p>• <strong>Operational NWP Forecast Integration:</strong> Transitioning from retrospective ERA5 reanalysis to operational numerical weather prediction feeds (e.g. NCMRWF or WRF-Chem operational forecast vintages).</p>
                <p>• <strong>Dynamic Mobility APIs:</strong> Incorporating real-time mobility or live congestion index feeds to capture localized congestion and freight corridors dynamically.</p>
                <p>• <strong>Aerosol Chemical Speciation:</strong> Tracking chemical aerosol species (SO₄²⁻, NO₃⁻, elemental and organic carbon) to isolate biomass combustion from industrial emissions.</p>
            </div>
        </div>
        """
    )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="About the Research - Hyderabad Air Forecast",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    from dashboard.components.disclaimers import render_sidebar_safeguards
    render_sidebar_safeguards()
    render_page_about()
