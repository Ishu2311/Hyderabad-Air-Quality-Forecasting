"""
Dashboard Page: 2. Air Quality Observations
Explores historical CPCB CAAQMS physical monitoring telemetry, multi-pollutant
time series, diurnal and seasonal profiles, and sensor uptime accounting.
"""

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.data_loader import load_historical_observations, load_cpcb_stations_metadata
from dashboard.backend.forecasting_service import get_available_stations
from dashboard.styles.theme import render_page_header, apply_plotly_theme, inject_custom_css, render_html


def render_page_observations():
    """Renders Air Quality Observations section."""
    inject_custom_css()

    render_page_header(
        badge="📈 IN-SITU GROUND TRUTH • CPCB CAAQMS ARCHIVE",
        title="Air Quality Observations & Telemetry Explorer",
        subtitle="Exploration of Physical Continuous Ground-Truth Telemetry across 4 Regulatory Stations in Greater Hyderabad (2023–2026)."
    )

    render_html(
        """
        <div class="alert-box alert-info">
            <span style="font-size: 16px;">ℹ️</span>
            <div>
                <strong>Archival Telemetry Standard:</strong> Measurements reflect physical continuous air quality readings 
                recorded by the 4 CPCB CAAQMS stations in Hyderabad. This offline research repository contains 
                <strong>131,424 hourly station records</strong> used for model calibration and out-of-time benchmarking.
            </div>
        </div>
        """
    )

    available_stations = get_available_stations()

    # Sleek Filter Card
    render_html(
        """
        <div style="font-size: 14px; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
            🎛️ Observation Filter Controls
        </div>
        """
    )

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        selected_station = st.selectbox(
            "Select Monitoring Station",
            options=["All Stations (Comparison)"] + available_stations,
            index=0
        )
    with col_ctrl2:
        pollutant_options = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
        selected_pollutant = st.selectbox(
            "Select Primary Pollutant",
            options=pollutant_options,
            index=0
        )
    with col_ctrl3:
        year_filter = st.selectbox(
            "Select Observation Window",
            options=["Full Archive (2023–2026)", "2026 (Test Set)", "2025 (Validation Set)", "2023–2024 (Training Set)"],
            index=1
        )

    # Load dataset
    with st.spinner("Loading observation records from feature store..."):
        filter_st = None if selected_station == "All Stations (Comparison)" else selected_station
        df_obs = load_historical_observations(station=filter_st, pollutants=pollutant_options)

    if df_obs.empty:
        st.error("No observation records available for the selected parameters.")
        return

    # Date Filtering
    if "2026" in year_filter:
        df_obs = df_obs[(df_obs["datetime"] >= "2026-01-01") & (df_obs["datetime"] <= "2026-09-30")]
    elif "2025" in year_filter:
        df_obs = df_obs[(df_obs["datetime"] >= "2025-01-01") & (df_obs["datetime"] <= "2025-12-31")]
    elif "2023–2024" in year_filter:
        df_obs = df_obs[(df_obs["datetime"] >= "2023-01-01") & (df_obs["datetime"] <= "2024-12-31")]

    st.markdown("---")

    # Interactive Time Series Plot
    st.subheader(f"Hourly {selected_pollutant} Concentration Time Series ({year_filter})")

    if selected_pollutant in df_obs.columns:
        fig = px.line(
            df_obs,
            x="datetime",
            y=selected_pollutant,
            color="station",
            labels={"datetime": "Date & Time (IST)", selected_pollutant: f"{selected_pollutant} Concentration (µg/m³)"},
            color_discrete_map={
                "Bollaram Industrial Area": "#f43f5e",
                "Central University": "#10b981",
                "ECIL Kapra": "#3b82f6",
                "ICRISAT Patancheru": "#14b8a6"
            }
        )

        # Standard thresholds
        if selected_pollutant == "PM2.5":
            fig.add_hline(
                y=60.0,
                line_dash="dash",
                line_color="#f59e0b",
                annotation_text="24h NAAQS Limit (60 µg/m³)",
                annotation_position="top left",
                annotation_font_color="#f59e0b"
            )
            fig.add_hline(
                y=120.0,
                line_dash="dot",
                line_color="#ef4444",
                annotation_text="Severe Spike Trigger (120 µg/m³)",
                annotation_position="top left",
                annotation_font_color="#ef4444"
            )

        apply_plotly_theme(fig, height=450, title=f"Continuous Physical Telemetry: {selected_pollutant}")
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning(f"Pollutant '{selected_pollutant}' has no recorded values in this partition.")

    st.markdown("---")

    # Diurnal Profile Analysis & Telemetry Availability
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.subheader("Diurnal Cycle Profile (Hourly Mean)")
        st.caption("Averaged across 24 hours of the day showing morning and evening diurnal peaks.")

        df_obs["hour"] = df_obs["datetime"].dt.hour
        diurnal_df = df_obs.groupby(["station", "hour"])[selected_pollutant].mean().reset_index()

        fig_diurnal = px.line(
            diurnal_df,
            x="hour",
            y=selected_pollutant,
            color="station",
            markers=True,
            labels={"hour": "Hour of Day (0–23 IST)", selected_pollutant: f"Mean {selected_pollutant} (µg/m³)"},
            color_discrete_map={
                "Bollaram Industrial Area": "#f43f5e",
                "Central University": "#10b981",
                "ECIL Kapra": "#3b82f6",
                "ICRISAT Patancheru": "#14b8a6"
            }
        )

        fig_diurnal.add_vrect(x0=8, x1=11, fillcolor="rgba(245, 158, 11, 0.15)", line_width=0, annotation_text="Morning Peak (08–11)", annotation_font_color="#f59e0b")
        fig_diurnal.add_vrect(x0=18, x1=22, fillcolor="rgba(239, 68, 68, 0.15)", line_width=0, annotation_text="Evening Peak (18–22)", annotation_font_color="#ef4444")

        apply_plotly_theme(fig_diurnal, height=380, title=f"Diurnal Profile: {selected_pollutant}")
        st.plotly_chart(fig_diurnal, width='stretch')

    with col_d2:
        st.subheader("Sensor Uptime & Telemetry Accounting")
        st.caption("Distribution of valid recorded hours vs sensor hardware outages.")

        total_hours = len(df_obs["datetime"].unique())
        uptime_records = []
        for s_name in df_obs["station"].unique():
            st_sub = df_obs[df_obs["station"] == s_name]
            valid_count = int(st_sub[selected_pollutant].notna().sum())
            null_count = int(st_sub[selected_pollutant].isna().sum())
            uptime_pct = round((valid_count / max(1, (valid_count + null_count))) * 100.0, 1)
            mean_val = round(float(st_sub[selected_pollutant].mean()), 2) if valid_count > 0 else np.nan
            max_val = round(float(st_sub[selected_pollutant].max()), 2) if valid_count > 0 else np.nan

            uptime_records.append({
                "Station": s_name,
                "Valid Hours": f"{valid_count:,}",
                "Missing / Outage": f"{null_count:,}",
                "Uptime Rate": f"{uptime_pct}%",
                f"Mean {selected_pollutant}": f"{mean_val} µg/m³" if not np.isnan(mean_val) else "—",
                f"Peak {selected_pollutant}": f"{max_val} µg/m³" if not np.isnan(max_val) else "—"
            })

        st.table(pd.DataFrame(uptime_records))

        render_html(
            """
            <div style="font-size: 11.5px; color: #94a3b8; line-height: 1.5; padding: 6px 0;">
                <em>Forensic Note:</em> Bollaram Industrial Area has valid 2026 PM2.5 observations recorded only in Jan–Feb and 
                Aug–Sep (36% coverage / 1,875 valid hours due to hardware sensor outages), so its test mean (41.85 µg/m³) 
                and under-prediction bias are seasonally confounded, whereas Central University (5,743), ECIL Kapra (5,849), and 
                ICRISAT Patancheru (5,606) maintained continuous uptime.
            </div>
            """
        )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Air Quality Observations - Hyderabad PM2.5",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    from dashboard.components.disclaimers import render_sidebar_safeguards
    render_sidebar_safeguards()
    render_page_observations()
