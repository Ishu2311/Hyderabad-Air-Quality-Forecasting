"""
Streamlit Application Entrypoint
Multi-Source Spatio-Temporal PM2.5 Forecasting and Hotspot Analysis in Hyderabad
Using Explainable Machine Learning
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root (new_research_project) and repo root are on sys.path
_DASHBOARD_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DASHBOARD_DIR.parent
_REPO_ROOT = _PROJECT_ROOT.parent

for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Page configuration
st.set_page_config(
    page_title="Hyderabad Air Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme & High-End Design System Injection (Light IQAir style)
from dashboard.styles.theme import inject_custom_css, render_html
inject_custom_css()

# Import Components & Pages
from dashboard.components.sidebar import render_sidebar
from dashboard.pages.page_overview import render_page_overview
from dashboard.pages.page_observations import render_page_observations
from dashboard.pages.page_forecasting import render_page_forecasting
from dashboard.pages.page_spatial import render_page_spatial
from dashboard.pages.page_explainability import render_page_explainability
from dashboard.pages.page_methodology import render_page_methodology
from dashboard.pages.page_about import render_page_about


def main():
    """Main routing and execution logic."""
    try:
        # Render Sidebar
        selected_page = render_sidebar()

        # Route to appropriate page
        if selected_page == "1. Project Overview":
            render_page_overview()
        elif selected_page == "2. Air Quality Observations":
            render_page_observations()
        elif selected_page == "3. PM2.5 Forecasting":
            render_page_forecasting()
        elif selected_page == "4. Spatial Hotspot Mapping":
            render_page_spatial()
        elif selected_page == "5. Explainability":
            render_page_explainability()
        elif selected_page == "6. Methodology & Data Quality":
            render_page_methodology()
        elif selected_page == "7. About the Research":
            render_page_about()
        else:
            st.error(f"Unknown page section: {selected_page}")

        # Persistent Academic Research Footer (IQAir / Academic Capstone style)
        render_html(
            """
            <div class="academic-footer">
                <div class="academic-footer-title">
                    Multi-Source Spatio-Temporal PM2.5 Forecasting and Hotspot Analysis in Hyderabad Using Explainable Machine Learning
                </div>
                <div class="academic-footer-meta">
                    <strong>Research prototype, not for regulatory or public health use.</strong><br>
                    Student Researcher: <strong>[YOUR NAME]</strong> • Project Guide: <strong>[GUIDE NAME]</strong> • Institution: <strong>[COLLEGE]</strong><br>
                    CPCB CAAQMS Continuous In-Situ Telemetry | Open-Meteo ERA5 Atmospheric Dynamics | Tree Ensembles | TreeSHAP Attribution.
                </div>
            </div>
            """
        )

    except Exception as e:
        st.error(f"Application Runtime Error: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
