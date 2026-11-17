"""
Dashboard Page: 3. PM2.5 Forecasting
Visualizes multi-horizon point forecasting models (+1h, +6h, +24h, +48h)
against ground-truth observations and persistence/seasonal-naive baselines.
"""

import sys
from pathlib import Path
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

from dashboard.backend.forecasting_service import (
    get_available_stations,
    get_horizon_metadata,
    get_forecast_timeseries,
    calculate_slice_metrics
)
from dashboard.components.metrics_cards import render_metric_row
from dashboard.styles.theme import render_page_header, apply_plotly_theme, inject_custom_css, render_html


def render_page_forecasting():
    """Renders PM2.5 Forecasting section."""
    inject_custom_css()

    render_page_header(
        badge="🔮 PREDICTIVE MODELING • 2026 TEST BENCHMARK",
        title="Multi-Horizon PM2.5 Concentration Forecasting",
        subtitle="Evaluation of Tree Ensembles and Stacking Meta-Regressors against CPCB CAAQMS In-Situ Ground Truth."
    )

    render_html(
        """
        <div class="alert-box alert-info">
            <span style="font-size: 16px;">📐</span>
            <div>
                <strong>Physical Concentration Standard:</strong> All targets and model forecasts are expressed strictly in 
                <strong>micrograms per cubic meter (µg/m³)</strong> of ambient PM<sub>2.5</sub>. 
                They do NOT represent unitless Air Quality Index (AQI) categories.
            </div>
        </div>
        """
    )

    available_stations = get_available_stations()

    # Sleek Selection Filter Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
    with col_ctrl1:
        selected_station = st.selectbox(
            "Select Monitoring Station",
            options=available_stations,
            index=0
        )
    with col_ctrl2:
        horizon_labels = {
            "+1h (Nowcasting Horizon)": "h1",
            "+6h (Sub-Daily Operational)": "h6",
            "+24h (Statutory 24h Planning)": "h24",
            "+48h (Medium-Range Advisory)": "h48"
        }
        selected_h_label = st.selectbox(
            "Select Forecast Horizon",
            options=list(horizon_labels.keys()),
            index=0
        )
        selected_h_key = horizon_labels[selected_h_label]
    with col_ctrl3:
        window_options = {
            "Entire 2026 Test Window": (None, None),
            "Winter Peak (Jan 1 – Feb 15)": ("2026-01-01", "2026-02-15"),
            "Pre-Monsoon / Summer (Mar 1 – Apr 30)": ("2026-03-01", "2026-04-30"),
            "Monsoon Season (Jul 1 – Aug 31)": ("2026-07-01", "2026-08-31"),
            "Recent Evaluation Slice (Sep 1 – Sep 28)": ("2026-09-01", "2026-09-28")
        }
        selected_window_label = st.selectbox(
            "Select Date Window",
            options=list(window_options.keys()),
            index=1
        )
        start_date, end_date = window_options[selected_window_label]

    # Model Metadata Card
    model_meta = get_horizon_metadata(selected_h_key)
    render_html(
        f"""
        <div class="env-card" style="padding: 14px 18px; margin-top: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-size: 13px; font-weight: 700; color: #38bdf8;">
                    Selected Forecasting Horizon: {selected_h_label}
                </span>
                <span class="badge badge-teal">{model_meta['name']}</span>
            </div>
            <p style="color: #475569; font-size: 12.5px; margin: 0; line-height: 1.5;">
                {model_meta['description']}
            </p>
        </div>
        """
    )

    # Fetch Data
    with st.spinner("Retrieving synchronized prediction time series..."):
        df_forecast = get_forecast_timeseries(
            station=selected_station,
            horizon=selected_h_key,
            start_date=start_date,
            end_date=end_date
        )

    if df_forecast.empty:
        st.error(f"No prediction records available for {selected_station} at horizon {selected_h_label}.")
        return

    # Calculate Slice Metrics
    target_col = "Actual PM2.5 (ug/m3)"
    pred_col = [c for c in df_forecast.columns if "Forecast (" in c][0]
    pers_col = "Persistence Baseline"

    y_true = df_forecast[target_col].values
    y_pred = df_forecast[pred_col].values
    y_pers = df_forecast[pers_col].values if pers_col in df_forecast.columns else None

    metrics = calculate_slice_metrics(y_true, y_pred)
    base_metrics = calculate_slice_metrics(y_true, y_pers) if y_pers is not None else None

    st.subheader(f"Evaluation Metrics on Selected Window ({metrics['count']:,} Valid Samples)")
    render_metric_row(
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        bias=metrics["bias"],
        r2=metrics["r2"],
        baseline_mae=base_metrics["mae"] if base_metrics else None
    )

    st.markdown("---")

    # Time Series Plot: Actual vs Forecast vs Baselines
    st.subheader(f"Ground Truth vs. Model Forecast Trajectory ({selected_station})")

    fig = go.Figure()

    # Actual PM2.5
    fig.add_trace(go.Scatter(
        x=df_forecast["datetime"],
        y=df_forecast[target_col],
        mode="lines",
        name="Actual Ground Truth (CAAQMS)",
        line=dict(color="#10b981", width=2.4)
    ))

    # Model Forecast
    fig.add_trace(go.Scatter(
        x=df_forecast["datetime"],
        y=df_forecast[pred_col],
        mode="lines",
        name=pred_col,
        line=dict(color="#06b6d4", width=2.2)
    ))

    # Persistence Baseline
    if pers_col in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast["datetime"],
            y=df_forecast[pers_col],
            mode="lines",
            name="Persistence Baseline",
            line=dict(color="#94a3b8", width=1.4, dash="dot"),
            opacity=0.65
        ))

    # Seasonal-Naive Baseline
    if "Seasonal Naive (24h)" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=df_forecast["datetime"],
            y=df_forecast["Seasonal Naive (24h)"],
            mode="lines",
            name="Seasonal Naive (24h)",
            line=dict(color="#f59e0b", width=1.2, dash="dash"),
            opacity=0.55
        ))

    # Threshold horizontal lines
    fig.add_hline(
        y=60.0,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="24h NAAQS (60 µg/m³)",
        annotation_position="bottom left",
        annotation_font_color="#f59e0b"
    )
    fig.add_hline(
        y=120.0,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text="Severe Spike Trigger (120 µg/m³)",
        annotation_position="bottom left",
        annotation_font_color="#ef4444"
    )

    apply_plotly_theme(fig, height=460, title=f"PM2.5 Trajectory: {selected_station} ({selected_h_label})")
    fig.update_layout(
        xaxis_title="Date & Time (IST)",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

    # Residual Analysis & Scatter Plot
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("Parity Plot (Predicted vs. Actual)")
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if np.sum(valid_mask) > 0:
            scatter_df = pd.DataFrame({
                "Actual": y_true[valid_mask],
                "Predicted": y_pred[valid_mask]
            })
            max_val = max(float(scatter_df["Actual"].max()), float(scatter_df["Predicted"].max()), 80.0)

            fig_scatter = px.scatter(
                scatter_df,
                x="Actual",
                y="Predicted",
                opacity=0.5,
                color_discrete_sequence=["#06b6d4"],
                labels={"Actual": "Ground Truth PM2.5 (µg/m³)", "Predicted": "Predicted PM2.5 (µg/m³)"}
            )
            fig_scatter.add_shape(
                type="line",
                x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="#ef4444", width=1.8, dash="dash")
            )
            apply_plotly_theme(fig_scatter, height=380, title="Parity Plot (1:1 Reference Line in Red)")
            st.plotly_chart(fig_scatter, width='stretch')

    with col_r2:
        st.subheader("Forecast Residual Error Distribution")
        if np.sum(valid_mask) > 0:
            residuals = y_pred[valid_mask] - y_true[valid_mask]
            fig_hist = px.histogram(
                x=residuals,
                nbins=40,
                color_discrete_sequence=["#3b82f6"],
                labels={"x": "Prediction Error: (Predicted - Actual) in µg/m³"}
            )
            fig_hist.add_vline(x=0.0, line_dash="solid", line_color="#10b981", line_width=2.0)
            apply_plotly_theme(fig_hist, height=380, title=f"Residual Histogram (Mean Bias = {metrics['bias']:+.2f} µg/m³)")
            st.plotly_chart(fig_hist, width='stretch')


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="PM2.5 Forecasting - Hyderabad PM2.5",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    from dashboard.components.disclaimers import render_sidebar_safeguards
    render_sidebar_safeguards()
    render_page_forecasting()
