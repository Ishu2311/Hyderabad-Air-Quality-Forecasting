"""
Dashboard Components: Sidebar Navigation & Metadata
Controls section switching, data freshness indicators, and persistent academic disclosures.
Light modern styling for live environmental platform.
"""

import sys
from pathlib import Path
from typing import Tuple
import streamlit as st

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.components.disclaimers import render_sidebar_safeguards
from dashboard.components.html_utils import render_html


PAGES = [
    "1. Project Overview",
    "2. Air Quality Observations",
    "3. PM2.5 Forecasting",
    "4. Spatial Hotspot Mapping",
    "5. Explainability",
    "6. Methodology & Data Quality",
    "7. About the Research"
]

PAGE_ICONS = {
    "1. Project Overview": "📊",
    "2. Air Quality Observations": "📈",
    "3. PM2.5 Forecasting": "🔮",
    "4. Spatial Hotspot Mapping": "🗺️",
    "5. Explainability": "🧠",
    "6. Methodology & Data Quality": "🔬",
    "7. About the Research": "🎓"
}


def _format_page(x: str) -> str:
    """Formats page name with icon without double prefixing."""
    icon = PAGE_ICONS.get(x, "")
    if icon and not x.startswith(icon):
        return f"{icon}  {x}"
    return x


def render_sidebar() -> str:
    """
    Renders a modern light sidebar navigation and project metadata card.
    
    Returns
    -------
    selected_page : str
        The chosen navigation section.
    """
    # 1. Platform Brand Card
    render_html(
        """
        <div class="sidebar-brand-card">
            <div class="sidebar-brand-icon">🌤️</div>
            <div class="sidebar-brand-title">Hyderabad Air Forecast</div>
            <div class="sidebar-brand-tag">Spatio-Temporal ML Lab</div>
        </div>
        """,
        container=st.sidebar
    )
    
    # 2. Navigation Menu
    render_html(
        "<p style='font-size: 11px; font-weight: 700; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 8px 4px;'>Navigation</p>",
        container=st.sidebar
    )
    
    # Support URL query parameter ?page=1..7 for direct deep linking and verification
    default_idx = 0
    query_page = st.query_params.get("page", None)
    if query_page is not None:
        try:
            page_int = int(query_page)
            if 1 <= page_int <= len(PAGES):
                default_idx = page_int - 1
        except ValueError:
            for i, p in enumerate(PAGES):
                if query_page.lower() in p.lower():
                    default_idx = i
                    break

    # Navigation Radio
    selected_page = st.sidebar.radio(
        "Navigation",
        options=PAGES,
        index=default_idx,
        format_func=_format_page,
        label_visibility="collapsed"
    )
    
    # 3. Dataset Freshness & System Status
    render_html(
        """
        <div class="sidebar-status-box">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-size: 12px; font-weight: 700; color: #0f172a;">Telemetry Status</span>
                <span style="display: inline-flex; align-items: center; font-size: 11px; font-weight: 700; color: #059669;">
                    <span class="pulse-dot"></span> 2026 Test Replay
                </span>
            </div>
            <div style="font-size: 12px; color: #475569; line-height: 1.6;">
                <div>• <strong>Domain:</strong> Greater Hyderabad (40×40 km)</div>
                <div>• <strong>Time Window:</strong> 2023-01-01 to 2026-09-30</div>
                <div>• <strong>Volume:</strong> 131,424 station-hours</div>
                <div>• <strong>Stations:</strong> 4 CAAQMS continuous</div>
                <div>• <strong>Evaluation:</strong> Out-of-Time 2026 Test</div>
            </div>
        </div>
        """,
        container=st.sidebar
    )
    
    # 4. Mandatory Scientific Disclosures & Safeguards
    render_sidebar_safeguards()
    
    # 5. University Capstone Accreditation
    render_html(
        """
        <div style="text-align: center; margin-top: 24px; padding-top: 14px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #94a3b8;">
            Academic Research Prototype<br>
            <strong>Hyderabad Air Quality Modeling</strong> • 2026
        </div>
        """,
        container=st.sidebar
    )
    
    return selected_page
