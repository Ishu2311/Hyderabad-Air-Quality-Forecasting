"""
Dashboard Components: Metric Display Cards
Renders cleanly formatted, responsive KPI cards for model evaluation.
"""

from typing import Optional
import streamlit as st


def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
):
    """Renders a standard Streamlit metric with custom styling."""
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )


def render_metric_row(
    mae: float,
    rmse: float,
    bias: float,
    r2: float,
    baseline_mae: Optional[float] = None
):
    """Renders 4 side-by-side KPI columns for model evaluation."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_str = f"{mae - baseline_mae:.2f} vs Base" if baseline_mae is not None else None
        render_kpi_card(
            label="Mean Absolute Error (MAE)",
            value=f"{mae:.2f} ug/m3",
            delta=delta_str,
            delta_color="inverse",
            help_text="Average absolute deviation between predictions and physical CAAQMS targets."
        )
        
    with col2:
        render_kpi_card(
            label="Root Mean Squared Error (RMSE)",
            value=f"{rmse:.2f} ug/m3",
            help_text="Square root of mean squared error, penalizing extreme forecast misses."
        )
        
    with col3:
        bias_str = f"{bias:+.2f} ug/m3"
        render_kpi_card(
            label="Mean Systematic Bias",
            value=bias_str,
            help_text="Mean signed error. Positive indicates systematic overestimation; negative indicates underestimation."
        )
        
    with col4:
        r2_color = "normal" if r2 > 0 else "off"
        render_kpi_card(
            label="Variance Explained (R2)",
            value=f"{r2:.3f}",
            help_text="Coefficient of determination relative to sample mean benchmark."
        )
