"""
Dashboard Page: 1. Project Overview & Live Air Quality Replay
Modern Live Air-Quality (IQAir / AQI.in style) Environmental Intelligence Platform.
Displays Hero Card with CPCB Breakpoints, 4-Horizon Forecast Tiles,
Interactive Plotly Time-Series with Replay Slider, Pydeck OpenStreetMap,
Stations Now List, and Scientific Disclaimers.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.data_loader import (
    load_cpcb_stations_metadata,
    load_historical_observations,
    load_regression_predictions,
    load_regression_metrics,
    load_spatial_validation_results
)
from dashboard.components.topbar import render_top_bar
from dashboard.styles.theme import (
    inject_custom_css,
    apply_plotly_theme,
    get_cpcb_category,
    render_html,
    CPCB_CATEGORIES
)


def hex_to_rgb(hex_code: str, alpha: int = 220) -> List[int]:
    """Converts hex color to RGBA list for Pydeck rendering."""
    clean = hex_code.lstrip("#")
    r, g, b = tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


def render_page_overview():
    """Renders Project Overview & Live Air Quality Replay (IQAir style)."""
    inject_custom_css()

    # 1. Top Bar: Site Name, Page Context, Station Selector, and Horizon Selector
    selected_station, selected_horizon = render_top_bar()

    # Load Data (Cached)
    df_preds = load_regression_predictions()
    metrics_data = load_regression_metrics()
    stations_meta = load_cpcb_stations_metadata()

    if df_preds.empty:
        st.error("No prediction data available in data/predictions/regression_predictions_test.parquet.")
        return

    # Filter to selected station
    st_preds = df_preds[df_preds["station"] == selected_station].sort_values("datetime").reset_index(drop=True)
    if st_preds.empty:
        st_preds = df_preds.sort_values("datetime").reset_index(drop=True)

    min_dt = st_preds["datetime"].min().to_pydatetime()
    max_dt = st_preds["datetime"].max().to_pydatetime()

    # 2. Time Replay Controls (Replaying 2026 Test Window as "Now")
    render_html(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <div style="font-size: 13px; font-weight: 700; color: #0f172a; text-transform: uppercase; letter-spacing: 0.04em;">
                ⏱️ 2026 Test Period Replay Slider ("Live" Snapshot Simulator)
            </div>
            <div style="font-size: 12px; color: #64748b;">
                Simulating operational continuous inference across the 2026 chronological evaluation partition
            </div>
        </div>
        """
    )

    # Default to 2026-01-15 12:00:00 where all stations have full uptime
    default_target = datetime(2026, 1, 15, 12, 0)
    if default_target < min_dt or default_target > max_dt:
        default_target = min_dt + (max_dt - min_dt) // 2

    # Date / Time Picker Row
    col_date, col_hour = st.columns([3, 2])
    with col_date:
        sel_date = st.date_input(
            "Select Evaluation Date",
            value=default_target.date(),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            key="replay_date_input"
        )
    with col_hour:
        sel_hour = st.slider(
            "Hour of Day (IST)",
            min_value=0,
            max_value=23,
            value=default_target.hour,
            format="%02d:00",
            key="replay_hour_slider"
        )

    active_dt = datetime.combine(sel_date, datetime.min.time()) + timedelta(hours=sel_hour)
    active_ts = pd.to_datetime(active_dt)

    # Locate current slice row
    cur_match = st_preds[st_preds["datetime"] == active_ts]
    if cur_match.empty:
        # Nearest timestamp fallback
        nearest_idx = (st_preds["datetime"] - active_ts).abs().idxmin()
        cur_row = st_preds.iloc[nearest_idx]
        actual_time_used = cur_row["datetime"]
    else:
        cur_row = cur_match.iloc[0]
        actual_time_used = cur_row["datetime"]

    # Extract Current In-Situ Observation at time t (equal to pred_persistence_h1)
    current_pm25 = cur_row.get("pred_persistence_h1", np.nan)
    if np.isnan(current_pm25):
        # Fallback to target of previous hour if available
        current_pm25 = cur_row.get("target_pm25_h1", np.nan)

    cpcb_cat = get_cpcb_category(current_pm25)
    pm25_display = f"{current_pm25:.1f}" if not np.isnan(current_pm25) else "—"

    # Best Model Test MAE Lookup from JSON metrics
    base_m = metrics_data.get("base_models", {}).get("test", {})
    stack_m = metrics_data.get("stacking", {}).get("test", {})

    mae_h1 = stack_m.get("h1", {}).get("mae", 3.011)
    mae_h6 = stack_m.get("h6", {}).get("mae", 5.987)
    mae_h24 = base_m.get("h24", {}).get("lightgbm", {}).get("mae", 6.668)
    mae_h48 = base_m.get("h48", {}).get("lightgbm", {}).get("mae", 7.930)

    # 3. Hero Card: Large PM2.5, Unit, Category Label, Plain Message, and Category Color Bar
    station_meta = stations_meta.get(selected_station, {})
    typology_str = station_meta.get("station_type", "CAAQMS Reference Station")
    ward_str = station_meta.get("ward", "Hyderabad")

    render_html(
        f"""
        <div class="iqair-hero-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div>
                    <div class="hero-station-title">📍 {selected_station} • {ward_str}</div>
                    <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">
                        Typology: <strong>{typology_str}</strong> • Lat: {station_meta.get('latitude', 17.5):.4f}° N, Lon: {station_meta.get('longitude', 78.4):.4f}° E
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748b;">Replay Evaluation Timestamp</div>
                    <div style="font-size: 14px; font-weight: 700; color: #0f172a;">
                        {pd.to_datetime(actual_time_used).strftime('%A, %d %b %Y • %H:%M IST')}
                    </div>
                </div>
            </div>

            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; margin-top: 10px;">
                <div>
                    <div class="hero-pm25-container">
                        <span class="hero-pm25-val" style="color: {cpcb_cat['color']};">{pm25_display}</span>
                        <span class="hero-pm25-unit">µg/m³</span>
                    </div>
                    <div class="hero-cat-badge" style="background: {cpcb_cat['bg_light']}; color: {cpcb_cat['text_color']}; border: 1px solid {cpcb_cat['color']};">
                        ● {cpcb_cat['name']} Air Quality
                    </div>
                    <div class="hero-plain-msg">
                        {cpcb_cat['message']}
                    </div>
                </div>
            </div>

            <div class="cpcb-bar-container">
                <div class="cpcb-bar-labels">
                    <span>Good (0-30)</span>
                    <span>Satisfactory (31-60)</span>
                    <span>Moderate (61-90)</span>
                    <span>Poor (91-120)</span>
                    <span>Very Poor (121-250)</span>
                    <span>Severe (250+)</span>
                </div>
                <div class="cpcb-spectrum-track">
                    <div class="cpcb-indicator" style="left: {cpcb_cat['percentage']}%;"></div>
                </div>
            </div>
        </div>
        """
    )

    # 4. Forecast Tiles (+1h, +6h, +24h, +48h)
    render_html(
        """
        <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin: 24px 0 12px 0;">
            🔮 Multi-Horizon Predictive Forecasts
        </div>
        """
    )

    tile_cols = st.columns(4)

    horizons_spec = [
        {
            "label": "+1h Nowcast",
            "model_name": "Stacking Ensemble",
            "pred_col": "pred_stacking_h1",
            "target_col": "target_pm25_h1",
            "test_mae": mae_h1,
            "target_time": actual_time_used + pd.Timedelta(hours=1)
        },
        {
            "label": "+6h Sub-Daily",
            "model_name": "Stacking Ensemble",
            "pred_col": "pred_stacking_h6",
            "target_col": "target_pm25_h6",
            "test_mae": mae_h6,
            "target_time": actual_time_used + pd.Timedelta(hours=6)
        },
        {
            "label": "+24h Planning",
            "model_name": "LightGBM Regressor",
            "pred_col": "pred_lgb_h24",
            "target_col": "target_pm25_h24",
            "test_mae": mae_h24,
            "target_time": actual_time_used + pd.Timedelta(hours=24)
        },
        {
            "label": "+48h Advisory",
            "model_name": "LightGBM Regressor",
            "pred_col": "pred_lgb_h48",
            "target_col": "target_pm25_h48",
            "test_mae": mae_h48,
            "target_time": actual_time_used + pd.Timedelta(hours=48)
        }
    ]

    for col, spec in zip(tile_cols, horizons_spec):
        pred_v = cur_row.get(spec["pred_col"], np.nan)
        act_v = cur_row.get(spec["target_col"], np.nan)
        pred_cat = get_cpcb_category(pred_v)
        
        pred_str = f"{pred_v:.1f}" if not np.isnan(pred_v) else "—"
        act_str = f"Observed: {act_v:.1f} µg/m³" if not np.isnan(act_v) else "Observed: Pending / Out of Range"

        with col:
            render_html(
                f"""
                <div class="forecast-tile-card">
                    <div>
                        <div class="forecast-tile-header">
                            <span class="forecast-horizon-pill">{spec['label']}</span>
                            <span class="forecast-model-name" title="{spec['model_name']}">{spec['model_name']}</span>
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">
                            Target: {spec['target_time'].strftime('%d %b, %H:%M')}
                        </div>
                        <div class="forecast-val-row">
                            <span class="forecast-pred-val" style="color: {pred_cat['color']};">{pred_str}</span>
                            <span style="font-size: 14px; font-weight: 600; color: #64748b;">µg/m³</span>
                        </div>
                        <div style="display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: {pred_cat['bg_light']}; color: {pred_cat['text_color']}; margin-bottom: 6px;">
                            {pred_cat['name']}
                        </div>
                        <div class="forecast-actual-pill">
                            {act_str}
                        </div>
                    </div>
                    <div class="forecast-mae-subtext">
                        Typical Error: <strong>±{spec['test_mae']:.2f} µg/m³</strong> (Test MAE)
                    </div>
                </div>
                """
            )

    render_html("<hr style='margin: 28px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 5. Interactive Plotly Chart: Last 3 Days History + Next 24h Forecast
    render_html(
        """
        <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            📈 Historical Context & Multi-Horizon Forecast Trajectory
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
            Window: Prior 72 hours in-situ observations + Forward 24-hour predictive horizon at selected timestamp.
        </div>
        """
    )

    chart_start = actual_time_used - pd.Timedelta(hours=72)
    chart_end = actual_time_used + pd.Timedelta(hours=24)

    chart_slice = st_preds[(st_preds["datetime"] >= chart_start) & (st_preds["datetime"] <= chart_end)].copy()

    if not chart_slice.empty:
        fig_ts = go.Figure()

        # Historical In-Situ Ground Truth (up to time t)
        hist_slice = chart_slice[chart_slice["datetime"] <= actual_time_used]
        if not hist_slice.empty:
            fig_ts.add_trace(go.Scatter(
                x=hist_slice["datetime"],
                y=hist_slice["pred_persistence_h1"],
                name="Historical PM2.5 (CAAQMS)",
                mode="lines+markers",
                marker=dict(size=4, color="#0f766e"),
                line=dict(color="#0f766e", width=2.5),
                hovertemplate="%{x|%d %b %H:%M}<br>In-Situ PM2.5: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Forward Ground Truth (if recorded)
        fut_slice = chart_slice[chart_slice["datetime"] > actual_time_used]
        if not fut_slice.empty and "pred_persistence_h1" in fut_slice.columns:
            fig_ts.add_trace(go.Scatter(
                x=fut_slice["datetime"],
                y=fut_slice["pred_persistence_h1"],
                name="Subsequent Actual PM2.5 (Evaluation)",
                mode="lines",
                line=dict(color="#94a3b8", width=2, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>Subsequent Ground Truth: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Forward Model Forecast (From cur_row targets and model predictions)
        # We can construct a forward profile at +1h, +6h, +24h
        fwd_times = [
            actual_time_used,
            actual_time_used + pd.Timedelta(hours=1),
            actual_time_used + pd.Timedelta(hours=6),
            actual_time_used + pd.Timedelta(hours=24)
        ]
        fwd_values = [
            current_pm25,
            cur_row.get("pred_stacking_h1", np.nan),
            cur_row.get("pred_stacking_h6", np.nan),
            cur_row.get("pred_lgb_h24", np.nan)
        ]
        valid_fwd_mask = [not np.isnan(v) for v in fwd_values]
        valid_fwd_times = [t for t, m in zip(fwd_times, valid_fwd_mask) if m]
        valid_fwd_vals = [v for v, m in zip(fwd_values, valid_fwd_mask) if m]

        if len(valid_fwd_vals) > 1:
            fig_ts.add_trace(go.Scatter(
                x=valid_fwd_times,
                y=valid_fwd_vals,
                name="Predictive Forecast (Best Model)",
                mode="lines+markers",
                marker=dict(size=8, color="#f59e0b", symbol="diamond"),
                line=dict(color="#f59e0b", width=3, dash="solid"),
                hovertemplate="%{x|%d %b %H:%M}<br>Model Forecast: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Persistence Flat Baseline
        if not np.isnan(current_pm25):
            fig_ts.add_trace(go.Scatter(
                x=[actual_time_used, actual_time_used + pd.Timedelta(hours=24)],
                y=[current_pm25, current_pm25],
                name="Zero-Order Persistence Baseline",
                mode="lines",
                line=dict(color="#cbd5e1", width=1.5, dash="dash"),
                hovertemplate="Persistence: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Vertical line for "Now" Replay Point
        fig_ts.add_vline(
            x=actual_time_used,
            line_width=2,
            line_dash="dash",
            line_color="#0f172a",
            annotation_text="Replay Timestamp",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#0f172a")
        )

        # CPCB Standard Threshold lines (60 µg/m³ 24h standard)
        fig_ts.add_hline(
            y=60.0,
            line_width=1.5,
            line_dash="dot",
            line_color="#f97316",
            annotation_text="NAAQS 24h Standard (60 µg/m³)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#f97316")
        )

        apply_plotly_theme(fig_ts, height=360, title=f"PM2.5 Trajectory ({selected_station})")
        fig_ts.update_layout(
            xaxis_title="Timeline (IST)",
            yaxis_title="PM2.5 Concentration (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ts, width='stretch')

    render_html("<hr style='margin: 28px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 6. Real Map & Stations Now Grid
    col_map, col_stations = st.columns([3, 2])

    with col_map:
        render_html(
            """
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
                🗺️ Real-Time Regional Network Basemap
            </div>
            """
        )

        # Mandatory Banner
        render_html(
            """
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-left: 4px solid #f59e0b; 
                        border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 12px; color: #92400e;">
                ⚠️ <strong>Interpolated from 4 stations; a regional estimate, not neighbourhood ground truth.</strong>
            </div>
            """
        )

        # Build stations map dataframe for the active snapshot
        map_records = []
        for s_name, meta in stations_meta.items():
            st_sub = df_preds[(df_preds["station"] == s_name) & (df_preds["datetime"] == actual_time_used)]
            if not st_sub.empty:
                val = st_sub.iloc[0].get("pred_persistence_h1", np.nan)
                if np.isnan(val):
                    val = st_sub.iloc[0].get("target_pm25_h1", np.nan)
            else:
                val = np.nan

            cat = get_cpcb_category(val)
            val_label = f"{val:.1f} µg/m³" if not np.isnan(val) else "Offline"
            rgb = hex_to_rgb(cat["color"], alpha=220)

            map_records.append({
                "name": s_name,
                "lat": meta["latitude"],
                "lon": meta["longitude"],
                "station_type": meta["station_type"],
                "pm25": val,
                "pm25_label": val_label,
                "category": cat["name"],
                "color": rgb
            })

        df_map = pd.DataFrame(map_records)

        # Pydeck Map with OpenStreetMap Carto Style Basemap
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_radius=2000,
            get_fill_color="color",
            get_line_color=[15, 23, 42, 200],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        )

        text_layer = pdk.Layer(
            "TextLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_text="pm25_label",
            get_size=13,
            get_color=[15, 23, 42, 255],
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18],
            font_family="Inter, sans-serif",
            font_weight="bold"
        )

        view_state = pdk.ViewState(
            latitude=17.49,
            longitude=78.40,
            zoom=10,
            pitch=0
        )

        deck = pdk.Deck(
            layers=[station_layer, text_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{name}</b><br>{station_type}<br>PM2.5: <b>{pm25_label}</b> ({category})",
                "style": {"backgroundColor": "#ffffff", "color": "#0f172a", "fontSize": "12px", "borderRadius": "8px", "padding": "8px"}
            },
            map_style="light"
        )
        st.pydeck_chart(deck, height=420)

    with col_stations:
        render_html(
            """
            <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
                🏛️ Stations Now (Snapshot Telemetry)
            </div>
            <div style="font-size: 12.5px; color: #64748b; margin-bottom: 12px;">
                4 CPCB continuous ambient air quality monitoring stations in Hyderabad.
            </div>
            """
        )

        for rec in map_records:
            s_name = rec["name"]
            is_active = (s_name == selected_station)
            border_css = "border: 2px solid #0f766e;" if is_active else "border: 1px solid #e2e8f0;"
            bg_css = "#f0fdfa" if is_active else "#ffffff"
            
            # Bollaram Hardware outage honest note
            bollaram_note = ""
            if "Bollaram" in s_name:
                bollaram_note = "<div style='font-size: 10.5px; color: #94a3b8; margin-top: 4px;'>* Valid 2026 data only in Jan–Feb & Aug–Sep (36% coverage due to sensor hardware outages).</div>"

            render_html(
                f"""
                <div style="background: {bg_css}; {border_css} border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 13.5px; font-weight: 700; color: #0f172a;">{s_name}</div>
                            <div style="font-size: 11.5px; color: #64748b;">{rec['station_type']}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 13px; font-weight: 800; background: {get_cpcb_category(rec['pm25'])['bg_light']}; color: {get_cpcb_category(rec['pm25'])['text_color']};">
                                {rec['pm25_label']}
                            </div>
                        </div>
                    </div>
                    {bollaram_note}
                </div>
                """
            )

    render_html("<hr style='margin: 28px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 7. Model Performance Benchmark Scorecard
    render_html(
        """
        <div style="font-size: 16px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            🏆 Model Benchmark Scorecard (2026 Out-of-Time Test Window)
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
            Benchmarked against Zero-Order Persistence Baseline on identical sample masks.
        </div>
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            label="Nowcast (+1h MAE)",
            value="3.01 µg/m³",
            delta="~5% over Persistence (3.18)",
            delta_color="inverse",
            help="Stacking Ensemble (3.011 µg/m³) vs XGBoost (3.031 µg/m³). Difference is small and not statistically significant."
        )
    with c2:
        st.metric(
            label="Sub-Daily (+6h MAE)",
            value="5.99 µg/m³",
            delta="~20% over Persistence (7.48)",
            delta_color="inverse",
            help="Stacking MAE 5.987 µg/m³; LightGBM 6.001 µg/m³; XGBoost 6.032 µg/m³. Stacking performs comparably."
        )
    with c3:
        st.metric(
            label="Planning (+24h MAE)",
            value="6.67 µg/m³",
            delta="~6-7% over Persistence (7.19)",
            delta_color="inverse",
            help="LightGBM Regressor (6.668 µg/m³) utilizes diurnal cycle and atmospheric thermodynamics."
        )
    with c4:
        st.metric(
            label="Advisory (+48h MAE)",
            value="7.93 µg/m³",
            delta="~3-4% over Persistence (8.25)",
            delta_color="inverse",
            help="LightGBM Regressor (7.930 µg/m³). Synoptic boundary layer and pressure forces ventilation/stagnation."
        )
    with c5:
        st.metric(
            label="Spatial LOSOCV MAE",
            value="11.70 µg/m³",
            delta="+388% Spatial Penalty",
            delta_color="off",
            help="Leave-One-Station-Out macro MAE is 11.70 µg/m³ (pooled is 11.43 µg/m³, n=19,073). IDW slightly beats plain donor average (11.43 vs 12.13 µg/m³)."
        )

    # 8. Methodological Honesty Card
    render_html(
        """
        <div class="env-card" style="margin-top: 24px; border-left: 4px solid #0f766e;">
            <div class="env-card-header">
                <div class="env-card-title">🛡️ Academic Safeguards & Methodological Honesty</div>
                <span class="badge badge-teal">Audited Standards</span>
            </div>
            <div style="font-size: 13px; color: #475569; line-height: 1.65;">
                <p style="margin-bottom: 8px;">
                    <strong>• Evaluation Sample Consistency:</strong> ML models are evaluated strictly on rows where the persistence baseline is valid. 
                    Actual MAE gains over persistence: <strong>+1h ~5%</strong>, <strong>+6h ~20%</strong>, <strong>+24h ~6-7%</strong>, <strong>+48h ~3-4%</strong>.
                </p>
                <p style="margin-bottom: 8px;">
                    <strong>• Model Stacking Equivalence:</strong> Stacking performs comparably to LightGBM/XGBoost; differences are small and not consistently significant (+1h MAE 3.011 vs 3.031 for XGBoost; at +6h stacking differences are marginal).
                </p>
                <p style="margin-bottom: 8px;">
                    <strong>• The 388% Spatial Penalty:</strong> While point forecasting achieves <strong>3.01 µg/m³ MAE</strong> at monitored stations, spatial transfer error jumps to <strong>11.70 µg/m³ macro MAE</strong> (<strong>11.43 µg/m³ pooled MAE</strong> across n=19,073). IDW only slightly beats a plain average of donor stations (11.43 vs 12.13 µg/m³).
                </p>
                <p style="margin-bottom: 8px;">
                    <strong>• Bollaram Seasonal Confounding:</strong> Bollaram has valid 2026 data only in Jan–Feb and Aug–Sep (36% coverage due to sensor hardware outages); its test statistics are seasonally confounded.
                </p>
                <p style="margin-bottom: 0;">
                    <strong>• Severe Spike Evaluability:</strong> The severe spike threshold (≥ 120 µg/m³) has only 1 positive in the test set, making it <em>not statistically evaluable</em>.
                </p>
            </div>
        </div>
        """
    )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Project Overview - Hyderabad Air Forecast",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    render_page_overview()
