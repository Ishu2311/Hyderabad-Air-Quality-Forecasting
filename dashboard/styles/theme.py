"""
Theme and Styling System for Hyderabad PM2.5 Research Platform
Modern Live Air-Quality (IQAir / AQI.in style) Design System.
Defines design tokens, CPCB AQI breakpoints, CSS injection, responsive card primitives,
and Plotly light styling.
"""

from typing import Optional, Dict, Any, Tuple
import streamlit as st
import plotly.graph_objects as go
from dashboard.components.html_utils import render_html

# Design Tokens (Light Theme with Deep Teal Accent)
PRIMARY_TEAL = "#0f766e"
SECONDARY_TEAL = "#14b8a6"
ACCENT_EMERALD = "#10b981"
ACCENT_LIME = "#84cc16"
ACCENT_CYAN = "#06b6d4"
ACCENT_BLUE = "#3b82f6"
ACCENT_AMBER = "#f59e0b"
ACCENT_ORANGE = "#f97316"
ACCENT_ROSE = "#ef4444"
ACCENT_CRIMSON = "#7f1d1d"

BG_PAGE = "#f8fafc"
BG_CARD = "#ffffff"
BG_CARD_ALT = "#f1f5f9"
BG_DARK = "#0f172a"
BORDER_SUBTLE = "#e2e8f0"
TEXT_MAIN = "#0f172a"
TEXT_BODY = "#334155"
TEXT_MUTED = "#64748b"

# Canonical CPCB PM2.5 Category Breakpoints & Health Advisories
# Standard: 24-hour average in µg/m³
CPCB_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Good": {
        "min": 0,
        "max": 30,
        "color": "#10b981",
        "bg_light": "#ecfdf5",
        "text_color": "#065f46",
        "message": "Air quality is good. Minimal or no health risk for the general population."
    },
    "Satisfactory": {
        "min": 31,
        "max": 60,
        "color": "#84cc16",
        "bg_light": "#f7fee7",
        "text_color": "#3f6212",
        "message": "Air quality is acceptable; minor breathing discomfort to sensitive individuals."
    },
    "Moderate": {
        "min": 61,
        "max": 90,
        "color": "#f59e0b",
        "bg_light": "#fffbeb",
        "text_color": "#92400e",
        "message": "Air quality is moderate; breathing discomfort to people with lungs, asthma and heart diseases."
    },
    "Poor": {
        "min": 91,
        "max": 120,
        "color": "#f97316",
        "bg_light": "#fff7ed",
        "text_color": "#9a3412",
        "message": "Air quality is poor; breathing discomfort to most people upon prolonged outdoor exposure."
    },
    "Very Poor": {
        "min": 121,
        "max": 250,
        "color": "#ef4444",
        "bg_light": "#fef2f2",
        "text_color": "#991b1b",
        "message": "Air quality is very poor; respiratory illness likely upon prolonged exposure."
    },
    "Severe": {
        "min": 250,
        "max": 1000,
        "color": "#7f1d1d",
        "bg_light": "#450a0a",
        "text_color": "#fee2e2",
        "message": "Air quality is severe; healthy individuals affected, serious impacts on vulnerable groups."
    }
}


def get_cpcb_category(pm25_val: Optional[float]) -> Dict[str, Any]:
    """
    Resolves the CPCB category metadata, color, and plain-language advisory
    for a given raw PM2.5 concentration in µg/m³.
    """
    if pm25_val is None or (isinstance(pm25_val, float) and (pm25_val < 0 or pm25_val != pm25_val)):
        return {
            "name": "Unavailable",
            "color": "#94a3b8",
            "bg_light": "#f1f5f9",
            "text_color": "#475569",
            "message": "Sensor telemetry currently offline or missing.",
            "percentage": 0.0
        }
    
    val = float(pm25_val)
    for cat_name, info in CPCB_CATEGORIES.items():
        if info["min"] <= val <= info["max"]:
            # Calculate spectrum position (0% at 0, 100% at 250+)
            pct = min(100.0, max(0.0, (val / 250.0) * 100.0))
            return {
                "name": cat_name,
                "color": info["color"],
                "bg_light": info["bg_light"],
                "text_color": info["text_color"],
                "message": info["message"],
                "percentage": pct
            }
            
    # Fallback for > 1000
    return {
        "name": "Severe",
        "color": CPCB_CATEGORIES["Severe"]["color"],
        "bg_light": CPCB_CATEGORIES["Severe"]["bg_light"],
        "text_color": CPCB_CATEGORIES["Severe"]["text_color"],
        "message": CPCB_CATEGORIES["Severe"]["message"],
        "percentage": 100.0
    }


def inject_custom_css():
    """Injects high-end, modern live air-quality website CSS styles (IQAir / AQI.in style)."""
    render_html(
        """
        <style>
            /* Google Fonts import */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

            /* Global Typography & Light Background */
            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            .main, .stApp {
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }

            /* Sleek Headings */
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Inter', sans-serif !important;
                font-weight: 700 !important;
                letter-spacing: -0.02em !important;
                color: #0f172a !important;
            }

            /* Hide default Streamlit chrome */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {
                background: rgba(248, 250, 252, 0.85) !important;
                backdrop-filter: blur(8px) !important;
            }

            /* Top Bar Container */
            .iqair-topbar {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 16px 24px;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }

            .iqair-brand-title {
                font-size: 20px;
                font-weight: 800;
                color: #0f172a;
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .iqair-live-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: #ecfdf5;
                color: #059669;
                border: 1px solid #a7f3d0;
                border-radius: 20px;
                padding: 3px 10px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
            }

            /* Hero Card (IQAir / AQI.in style) */
            .iqair-hero-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 24px 28px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
                position: relative;
                overflow: hidden;
            }

            .hero-station-title {
                font-size: 14px;
                font-weight: 600;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 6px;
            }

            .hero-pm25-container {
                display: flex;
                align-items: baseline;
                gap: 12px;
                margin: 10px 0;
            }

            .hero-pm25-val {
                font-size: 64px;
                font-weight: 800;
                line-height: 1;
                font-feature-settings: "tnum";
            }

            .hero-pm25-unit {
                font-size: 20px;
                font-weight: 600;
                color: #64748b;
            }

            .hero-cat-badge {
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 700;
                margin-bottom: 12px;
            }

            .hero-plain-msg {
                font-size: 14px;
                color: #334155;
                line-height: 1.5;
                margin-bottom: 16px;
                font-weight: 500;
            }

            /* CPCB Multi-Color Spectrum Bar */
            .cpcb-bar-container {
                margin-top: 14px;
                padding-top: 12px;
                border-top: 1px solid #f1f5f9;
            }

            .cpcb-bar-labels {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #64748b;
                font-weight: 600;
                margin-bottom: 6px;
            }

            .cpcb-spectrum-track {
                height: 10px;
                border-radius: 5px;
                background: linear-gradient(90deg, 
                    #10b981 0%, #10b981 12%, 
                    #84cc16 12%, #84cc16 24%, 
                    #f59e0b 24%, #f59e0b 36%, 
                    #f97316 36%, #f97316 48%, 
                    #ef4444 48%, #ef4444 80%, 
                    #7f1d1d 80%, #7f1d1d 100%);
                position: relative;
                width: 100%;
            }

            .cpcb-indicator {
                position: absolute;
                top: -4px;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #ffffff;
                border: 3px solid #0f172a;
                transform: translateX(-50%);
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
                transition: left 0.3s ease;
            }

            /* Forecast Tiles (+1h, +6h, +24h, +48h) */
            .forecast-tile-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 18px 20px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }

            .forecast-tile-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }

            .forecast-tile-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 8px;
            }

            .forecast-horizon-pill {
                font-size: 12px;
                font-weight: 700;
                color: #0f766e;
                background: #ccfbf1;
                padding: 2px 8px;
                border-radius: 6px;
            }

            .forecast-model-name {
                font-size: 11px;
                color: #64748b;
                font-weight: 600;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .forecast-val-row {
                display: flex;
                align-items: baseline;
                gap: 6px;
                margin: 8px 0;
            }

            .forecast-pred-val {
                font-size: 28px;
                font-weight: 800;
                line-height: 1.1;
                color: #0f172a;
            }

            .forecast-actual-pill {
                font-size: 11px;
                font-weight: 600;
                color: #475569;
                background: #f1f5f9;
                padding: 3px 8px;
                border-radius: 6px;
                margin-top: 4px;
            }

            .forecast-mae-subtext {
                font-size: 11.5px;
                color: #64748b;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #f1f5f9;
            }

            /* Responsive Standard Cards */
            .env-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 20px 24px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            }

            .env-card-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid #f1f5f9;
                padding-bottom: 12px;
                margin-bottom: 14px;
            }

            .env-card-title {
                font-size: 16px;
                font-weight: 700;
                color: #0f172a;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* Metric Cards Overrides */
            div[data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1px solid #e2e8f0 !important;
                border-top: 3px solid #0f766e !important;
                border-radius: 12px !important;
                padding: 14px 18px !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            }

            div[data-testid="stMetric"] label {
                color: #64748b !important;
                font-size: 12px !important;
                font-weight: 600 !important;
                letter-spacing: 0.02em !important;
                text-transform: uppercase !important;
            }

            div[data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-size: 26px !important;
                font-weight: 800 !important;
                font-family: 'Inter', sans-serif !important;
            }

            /* Sidebar Light Redesign */
            section[data-testid="stSidebar"] {
                background: #ffffff !important;
                border-right: 1px solid #e2e8f0 !important;
            }

            section[data-testid="stSidebar"] div.block-container {
                padding-top: 1.8rem;
                padding-left: 1.2rem;
                padding-right: 1.2rem;
            }

            .sidebar-brand-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 16px;
                margin-bottom: 20px;
                text-align: center;
            }

            .sidebar-brand-icon {
                font-size: 28px;
                margin-bottom: 4px;
                display: inline-block;
            }

            .sidebar-brand-title {
                color: #0f172a;
                font-size: 16px;
                font-weight: 800;
                letter-spacing: -0.01em;
                margin: 0 0 2px 0;
            }

            .sidebar-brand-tag {
                color: #0f766e;
                font-size: 11px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin: 0;
            }

            .sidebar-status-box {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 12px 14px;
                margin-top: 16px;
            }

            .pulse-dot {
                display: inline-block;
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background-color: #10b981;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: pulse-ring 2s infinite cubic-bezier(0.455, 0.03, 0.515, 0.955);
                margin-right: 6px;
            }

            @keyframes pulse-ring {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }

            /* Alert Boxes */
            .alert-box {
                border-radius: 12px;
                padding: 14px 18px;
                margin-bottom: 18px;
                display: flex;
                gap: 12px;
                align-items: flex-start;
                font-size: 13.5px;
                line-height: 1.55;
            }

            .alert-info {
                background: #f0f9ff;
                border-left: 4px solid #0284c7;
                border: 1px solid #bae6fd;
                border-left-width: 4px;
                color: #0369a1;
            }

            .alert-warning {
                background: #fffbeb;
                border-left: 4px solid #d97706;
                border: 1px solid #fde68a;
                border-left-width: 4px;
                color: #92400e;
            }

            .alert-danger {
                background: #fef2f2;
                border-left: 4px solid #dc2626;
                border: 1px solid #fecaca;
                border-left-width: 4px;
                color: #991b1b;
            }

            /* Badges & Tags */
            .badge {
                display: inline-block;
                padding: 3px 9px;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.02em;
            }

            .badge-teal { background: #ccfbf1; color: #0f766e; border: 1px solid #99f6e4; }
            .badge-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
            .badge-blue { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
            .badge-amber { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
            .badge-rose { background: #ffe4e6; color: #be123c; border: 1px solid #fecdd3; }

            /* Academic Platform Footer */
            .academic-footer {
                margin-top: 48px;
                padding: 24px 20px;
                border-top: 1px solid #e2e8f0;
                text-align: center;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            }

            .academic-footer-title {
                color: #0f172a;
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 6px;
            }

            .academic-footer-meta {
                color: #64748b;
                font-size: 12px;
                line-height: 1.6;
            }

            /* Tables in Light Mode */
            div[data-testid="stTable"] {
                border-radius: 10px;
                overflow: hidden;
                border: 1px solid #e2e8f0;
            }

            table {
                background: #ffffff !important;
                color: #1e293b !important;
                font-size: 13px !important;
            }

            thead tr th {
                background: #f8fafc !important;
                color: #0f172a !important;
                font-weight: 700 !important;
                border-bottom: 2px solid #e2e8f0 !important;
                padding: 10px 14px !important;
            }

            tbody tr td {
                padding: 10px 14px !important;
                border-bottom: 1px solid #f1f5f9 !important;
            }

            tbody tr:hover {
                background-color: #f8fafc !important;
            }
        </style>
        """
    )


def apply_plotly_theme(
    fig: go.Figure,
    height: Optional[int] = None,
    title: Optional[str] = None
) -> go.Figure:
    """Applies high-end light theme aesthetic (IQAir style) to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            family="Inter, sans-serif",
            color="#334155",
            size=12
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#e2e8f0",
            font=dict(family="Inter, sans-serif", color="#0f172a", size=12)
        )
    )

    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(family="Inter, sans-serif", size=15, color="#0f172a")
            )
        )

    if height:
        fig.update_layout(height=height)

    # Clean axes gridlines
    fig.update_xaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        linecolor="#cbd5e1"
    )
    fig.update_yaxes(
        gridcolor="#f1f5f9",
        zerolinecolor="#e2e8f0",
        linecolor="#cbd5e1"
    )

    return fig


def render_page_header(
    badge: str,
    title: str,
    subtitle: str
):
    """Renders a standardized executive hero header for platform pages in light theme."""
    render_html(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; 
                    padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
                    border-left: 4px solid #0f766e;">
            <div style="display: inline-flex; align-items: center; gap: 6px; background: #ccfbf1; 
                        color: #0f766e; border-radius: 16px; padding: 3px 10px; font-size: 11px; 
                        font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 8px;">
                {badge}
            </div>
            <h1 style="font-size: 24px; font-weight: 800; color: #0f172a; margin: 0 0 6px 0; line-height: 1.3;">
                {title}
            </h1>
            <p style="color: #64748b; font-size: 13.5px; line-height: 1.5; margin: 0;">
                {subtitle}
            </p>
        </div>
        """
    )
