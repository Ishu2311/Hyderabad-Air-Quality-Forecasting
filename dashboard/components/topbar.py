"""
Dashboard Components: Top Navigation Bar & Global Controls
Provides the top bar with site name, live replay status, and global Station/Horizon selectors.
"""

from typing import Tuple, List, Optional
import streamlit as st
from dashboard.backend.forecasting_service import get_available_stations
from dashboard.components.html_utils import render_html

HORIZON_OPTIONS = {
    "+1h (Nowcast)": "h1",
    "+6h (Sub-Daily)": "h6",
    "+24h (Planning)": "h24",
    "+48h (Advisory)": "h48"
}


def render_top_bar() -> Tuple[str, str]:
    """
    Renders top bar with site brand, live replay badge, station selector, and horizon selector.
    
    Returns
    -------
    selected_station : str
        Currently selected CPCB station.
    selected_horizon : str
        Currently selected forecast horizon key (e.g. 'h1', 'h6', 'h24', 'h48').
    """
    stations = get_available_stations()
    
    # Initialize session state if not set
    if "selected_station" not in st.session_state:
        st.session_state.selected_station = stations[1]  # Default to Central University
    if "selected_horizon" not in st.session_state:
        st.session_state.selected_horizon = "h1"

    col_brand, col_st, col_hz = st.columns([4, 3, 3])
    
    with col_brand:
        render_html(
            """
            <div style="padding: 4px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 24px;">🌤️</span>
                    <div>
                        <div style="font-size: 19px; font-weight: 800; color: #0f172a; line-height: 1.2;">
                            Hyderabad Air Forecast
                        </div>
                        <div style="display: inline-flex; align-items: center; gap: 5px; font-size: 11px; font-weight: 700; color: #059669; text-transform: uppercase;">
                            <span class="pulse-dot"></span> 2026 Test Period Replay
                        </div>
                    </div>
                </div>
            </div>
            """
        )
        
    with col_st:
        current_st_idx = stations.index(st.session_state.selected_station) if st.session_state.selected_station in stations else 0
        sel_station = st.selectbox(
            "Monitoring Station",
            options=stations,
            index=current_st_idx,
            key="topbar_station_select",
            label_visibility="collapsed"
        )
        st.session_state.selected_station = sel_station

    with col_hz:
        hz_labels = list(HORIZON_OPTIONS.keys())
        current_hz_label = [k for k, v in HORIZON_OPTIONS.items() if v == st.session_state.selected_horizon]
        current_hz_idx = hz_labels.index(current_hz_label[0]) if current_hz_label else 0
        sel_hz_label = st.selectbox(
            "Forecast Horizon",
            options=hz_labels,
            index=current_hz_idx,
            key="topbar_horizon_select",
            label_visibility="collapsed"
        )
        st.session_state.selected_horizon = HORIZON_OPTIONS[sel_hz_label]
        
    render_html("<hr style='margin: 12px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;'>")
    
    return st.session_state.selected_station, st.session_state.selected_horizon
