"""
Hyderabad Air Forecast - Single-Page Interactive Research Demo
IQAir / AQI.in style Live Environmental Intelligence Website Replay.

Replays the 2026 out-of-time test period as 'now' to demonstrate multi-source
spatio-temporal PM2.5 forecasting, CPCB category evaluation, and regional cartography.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

# Configure Streamlit page
st.set_page_config(
    page_title="Hyderabad Air Forecast | Live Replay Demo",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Setup paths relative to project root (new_research_project)
DEMO_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEMO_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "predictions"
PARQUET_PATH = DATA_DIR / "regression_predictions_test.parquet"
REG_METRICS_PATH = DATA_DIR / "regression_metrics.json"
STACK_METRICS_PATH = DATA_DIR / "stacking_regression_metrics.json"

# Station coordinates and typology from implementation config
try:
    from implementation.configs.config import STATION_REGISTRY
    STATION_INFO: Dict[str, Dict[str, Any]] = {
        name: {
            "lat": float(meta.latitude),
            "lon": float(meta.longitude),
            "typology": str(meta.typology),
            "station_id": str(meta.station_id)
        }
        for name, meta in STATION_REGISTRY.items()
    }
except Exception:
    STATION_INFO = {
        "Bollaram Industrial Area": {
            "lat": 17.54, "lon": 78.35,
            "typology": "Heavy Industrial / Pharmaceutical Corridor",
            "station_id": "HYD_BOL"
        },
        "Central University": {
            "lat": 17.46, "lon": 78.33,
            "typology": "Institutional Campus / Vegetative Background",
            "station_id": "HYD_UOH"
        },
        "ECIL Kapra": {
            "lat": 17.47, "lon": 78.57,
            "typology": "Dense Commercial / Mixed Urban Residential",
            "station_id": "HYD_ECL"
        },
        "ICRISAT Patancheru": {
            "lat": 17.51, "lon": 78.27,
            "typology": "Peri-Urban / Agricultural-Industrial Boundary",
            "station_id": "HYD_ICR"
        }
    }

STATIONS: List[str] = list(STATION_INFO.keys())

# CPCB PM2.5 Breakpoints & Health Advisories in ONE Dict (24-hour standard in µg/m³)
CPCB_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "Good": {
        "max": 30.0,
        "color": "#10B981", # Green
        "bg_light": "#ECFDF5",
        "text_color": "#065F46",
        "message": "Air quality is good. Minimal or no physical health risk for the general population."
    },
    "Satisfactory": {
        "max": 60.0,
        "color": "#84CC16", # Lime
        "bg_light": "#F7FEE7",
        "text_color": "#3F6212",
        "message": "Air quality is acceptable; minor breathing discomfort to sensitive individuals."
    },
    "Moderate": {
        "max": 90.0,
        "color": "#F59E0B", # Amber
        "bg_light": "#FFFBEB",
        "text_color": "#92400E",
        "message": "Air quality is moderate; breathing discomfort to people with asthma and heart diseases."
    },
    "Poor": {
        "max": 120.0,
        "color": "#F97316", # Orange
        "bg_light": "#FFF7ED",
        "text_color": "#9A3412",
        "message": "Air quality is poor; breathing discomfort to most people upon prolonged outdoor exposure."
    },
    "Very Poor": {
        "max": 250.0,
        "color": "#EF4444", # Red
        "bg_light": "#FEF2F2",
        "text_color": "#991B1B",
        "message": "Air quality is very poor; respiratory illness likely upon prolonged exposure."
    },
    "Severe": {
        "max": float("inf"),
        "color": "#7F1D1D", # Dark Crimson
        "bg_light": "#450A0A",
        "text_color": "#FEE2E2",
        "message": "Air quality is severe; healthy individuals affected, serious impacts on vulnerable groups."
    },
    "No data": {
        "max": -1.0,
        "color": "#94A3B8",
        "bg_light": "#F1F5F9",
        "text_color": "#475569",
        "message": "Sensor telemetry is unavailable or offline for this period."
    }
}


def render_html(html: str):
    """
    Renders raw HTML cleanly without triggering Markdown code block formatting.
    Strips each line, drops empty lines and lines starting with '<!--',
    joins the rest with a single space, and calls st.markdown(cleaned, unsafe_allow_html=True).
    """
    lines = [line.strip() for line in html.splitlines()]
    cleaned_lines = [
        line for line in lines
        if line and not line.startswith("<!--")
    ]
    cleaned = " ".join(cleaned_lines)
    st.markdown(cleaned, unsafe_allow_html=True)


def get_cpcb_category(val: Optional[float]) -> Dict[str, Any]:
    """
    Resolves CPCB category metadata, color, and health message for a raw PM2.5 value.
    Classifies with upper-bound comparisons only:
    <= 30: Good
    <= 60: Satisfactory
    <= 90: Moderate
    <= 120: Poor
    <= 250: Very Poor
    otherwise: Severe
    NaN / None / < 0: No data
    """
    if val is None or pd.isna(val) or val < 0:
        cat_name = "No data"
        pct = 0.0
    else:
        v = float(val)
        if v <= 30.0:
            cat_name = "Good"
        elif v <= 60.0:
            cat_name = "Satisfactory"
        elif v <= 90.0:
            cat_name = "Moderate"
        elif v <= 120.0:
            cat_name = "Poor"
        elif v <= 250.0:
            cat_name = "Very Poor"
        else:
            cat_name = "Severe"
        pct = min(100.0, max(0.0, (v / 250.0) * 100.0))

    info = CPCB_CATEGORIES[cat_name]
    return {
        "name": cat_name,
        "color": info["color"],
        "bg_light": info["bg_light"],
        "text_color": info["text_color"],
        "message": info["message"],
        "percentage": pct
    }


def hex_to_rgb(hex_code: str, alpha: int = 220) -> List[int]:
    """Converts hex color code to RGBA list for Pydeck rendering."""
    clean = hex_code.lstrip("#")
    r, g, b = tuple(int(clean[i:i+2], 16) for i in (0, 2, 4))
    return [r, g, b, alpha]


@st.cache_data
def load_prediction_dataset() -> pd.DataFrame:
    """Loads 2026 out-of-time test predictions parquet."""
    if not PARQUET_PATH.exists():
        st.error(f"Prediction dataset not found at: {PARQUET_PATH}")
        return pd.DataFrame()
    df = pd.read_parquet(PARQUET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values(["station", "datetime"]).reset_index(drop=True)


@st.cache_data
def load_test_error_metrics() -> Dict[str, float]:
    """Loads typical test MAE errors from regression_metrics.json and stacking_regression_metrics.json."""
    maes = {
        "h1": 3.011,
        "h6": 5.987,
        "h24": 6.668,
        "h48": 7.930
    }
    try:
        if STACK_METRICS_PATH.exists():
            with open(STACK_METRICS_PATH, "r", encoding="utf-8") as f:
                sm = json.load(f)
                test_sm = sm.get("test", {})
                if "h1" in test_sm and "mae" in test_sm["h1"]:
                    maes["h1"] = float(test_sm["h1"]["mae"])
                if "h6" in test_sm and "mae" in test_sm["h6"]:
                    maes["h6"] = float(test_sm["h6"]["mae"])
        if REG_METRICS_PATH.exists():
            with open(REG_METRICS_PATH, "r", encoding="utf-8") as f:
                bm = json.load(f)
                test_bm = bm.get("test", {})
                if "h24" in test_bm and "lightgbm" in test_bm["h24"]:
                    maes["h24"] = float(test_bm["h24"]["lightgbm"]["mae"])
                if "h48" in test_bm and "lightgbm" in test_bm["h48"]:
                    maes["h48"] = float(test_bm["h48"]["lightgbm"]["mae"])
    except Exception:
        pass
    return maes


# Human-readable feature name mapping for SHAP explanations
FEATURE_NAME_MAP: Dict[str, str] = {
    "PM2.5": "Current PM2.5 (in-situ)",
    "pm25_lag_1": "PM2.5 1 hour earlier",
    "pm25_lag_2": "PM2.5 2 hours earlier",
    "pm25_lag_3": "PM2.5 3 hours earlier",
    "pm25_lag_6": "PM2.5 6 hours earlier",
    "pm25_lag_12": "PM2.5 12 hours earlier",
    "pm25_lag_24": "PM2.5 24 hours earlier",
    "pm25_roll_mean_3h": "3h rolling mean PM2.5",
    "pm25_roll_mean_6h": "6h rolling mean PM2.5",
    "pm25_roll_mean_12h": "12h rolling mean PM2.5",
    "pm25_roll_mean_24h": "24h rolling mean PM2.5",
    "pm25_roll_min_24h": "24h minimum PM2.5",
    "pm25_roll_max_24h": "24h maximum PM2.5",
    "pm25_roll_std_6h": "6h PM2.5 variability (std)",
    "pm25_roll_std_24h": "24h PM2.5 variability (std)",
    "pm25_trend_diff_1h": "1h PM2.5 trend (diff)",
    "pm25_trend_diff_24h": "24h PM2.5 trend (diff)",
    "pm25_acceleration_1h": "1h PM2.5 acceleration",
    "PM10": "In-situ PM10 concentration",
    "pm10_lag_1": "PM10 1 hour earlier",
    "pm25_pm10_ratio": "PM2.5 / PM10 Ratio",
    "NO2": "Nitrogen Dioxide (NO2)",
    "no2_lag_1": "NO2 1 hour earlier",
    "SO2": "Sulfur Dioxide (SO2)",
    "CO": "Carbon Monoxide (CO)",
    "Ozone": "Ground-Level Ozone (O3)",
    "temperature_2m": "Air Temperature (2m)",
    "temp_lag_1": "Temperature 1 hour earlier",
    "relative_humidity_2m": "Relative Humidity (2m)",
    "rh_lag_1": "Relative humidity 1 hour earlier",
    "dew_point_temp": "Dew point temperature",
    "surface_pressure": "Surface atmospheric pressure",
    "pressure_tendency_3h": "3h barometric pressure tendency",
    "act_vapor_pressure_ea": "Actual vapor pressure (ea)",
    "boundary_layer_height": "Planetary boundary layer height",
    "wind_speed_10m": "Wind speed (10m)",
    "wind_speed_roll_mean_6h": "6h rolling mean wind speed",
    "wind_speed_roll_mean_24h": "24h rolling mean wind speed",
    "u_component_of_wind_10m": "Zonal wind (U-component)",
    "v_component_of_wind_10m": "Meridional wind (V-component)",
    "hour_sin": "Diurnal cycle (sine)",
    "hour_cos": "Diurnal cycle (cosine)",
    "month_cos": "Seasonal cycle (cosine)",
    "is_morning_peak": "Morning traffic peak indicator",
    "is_monsoon": "Monsoon season indicator",
}


@st.cache_data
def load_global_feature_importance() -> Dict[str, Any]:
    """Loads global feature importance and category shares from JSON."""
    path = DATA_DIR / "global_feature_importance.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


@st.cache_data
def load_shap_summary(horizon: str = "h1") -> Dict[str, Any]:
    """Loads SHAP summary JSON for a specific forecast horizon."""
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    path = DATA_DIR / f"shap_summary_h{clean_h}.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def inject_custom_styles():
    """Injects modern light theme CSS (IQAir / AQI.in style) with Inter font and accent #0F766E."""
    render_html(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }

            .main, .stApp {
                background-color: #f8fafc !important;
                color: #0f172a !important;
            }

            /* Hide default Streamlit menu and footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header[data-testid="stHeader"] {
                background: rgba(248, 250, 252, 0.85) !important;
                backdrop-filter: blur(6px) !important;
            }

            /* Responsive Top Bar */
            .top-bar-container {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 16px 22px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            }

            .top-brand-title {
                font-size: 22px;
                font-weight: 800;
                color: #0f172a;
                letter-spacing: -0.02em;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .replay-status-pill {
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
                margin-top: 4px;
            }

            .pulse-dot {
                display: inline-block;
                width: 7px;
                height: 7px;
                border-radius: 50%;
                background-color: #10b981;
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
                animation: pulse-ring 2s infinite cubic-bezier(0.455, 0.03, 0.515, 0.955);
            }

            @keyframes pulse-ring {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
                70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }

            /* Hero Card */
            .iqair-hero-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 24px 28px;
                margin-bottom: 20px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }

            .hero-pm25-num {
                font-size: 62px;
                font-weight: 800;
                line-height: 1;
                font-feature-settings: "tnum";
            }

            .hero-pm25-unit {
                font-size: 20px;
                font-weight: 600;
                color: #64748b;
                margin-left: 6px;
            }

            /* CPCB Multi-Color Spectrum Track */
            .cpcb-track-container {
                margin-top: 14px;
                padding-top: 12px;
                border-top: 1px solid #f1f5f9;
            }

            .cpcb-track-labels {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #64748b;
                font-weight: 600;
                margin-bottom: 6px;
            }

            .cpcb-spectrum-bar {
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

            .cpcb-indicator-pin {
                position: absolute;
                top: -4px;
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: #ffffff;
                border: 3px solid #0f172a;
                transform: translateX(-50%);
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.25);
            }

            /* Forecast Cards */
            .forecast-tile-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 16px 18px;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: transform 0.15s ease;
            }

            .forecast-tile-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            }

            .forecast-horizon-badge {
                font-size: 12px;
                font-weight: 700;
                color: #0f766e;
                background: #ccfbf1;
                padding: 2px 8px;
                border-radius: 6px;
            }

            .forecast-val-large {
                font-size: 26px;
                font-weight: 800;
                line-height: 1.1;
                margin: 6px 0;
            }

            .forecast-actual-badge {
                font-size: 11px;
                font-weight: 600;
                color: #475569;
                background: #f1f5f9;
                padding: 3px 8px;
                border-radius: 6px;
                margin-top: 6px;
            }

            .forecast-mae-note {
                font-size: 11px;
                color: #64748b;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #f1f5f9;
            }

            /* Responsive standard white container card */
            .content-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 20px 24px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            }

            /* Footer */
            .demo-footer {
                margin-top: 36px;
                padding: 22px 20px;
                border-top: 1px solid #e2e8f0;
                text-align: center;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            }
        </style>
        """
    )


def main():
    """Main rendering pipeline for the single-page demo application."""
    inject_custom_styles()

    df_preds = load_prediction_dataset()
    test_maes = load_test_error_metrics()

    if df_preds.empty:
        st.error("Prediction data could not be loaded. Please ensure data files exist.")
        return

    min_datetime = df_preds["datetime"].min().to_pydatetime()
    max_datetime = df_preds["datetime"].max().to_pydatetime()

    # 1. Top Bar: Title, Station Selector, and Replay Time Controls
    render_html(
        """
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; 
                    padding: 16px 22px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div class="top-brand-title">
                        <span>🌤️</span> Hyderabad Air Forecast
                    </div>
                    <div class="replay-status-pill">
                        <span class="pulse-dot"></span> 2026 Out-of-Time Test Window Replay
                    </div>
                </div>
                <div style="font-size: 12px; color: #64748b; text-align: right;">
                    <strong>Offline Research Simulation</strong> • Replaying chronological test period as "now".<br>
                    <em>Notice: This is an empirical research replay, not a live regulatory sensor telemetry feed.</em>
                </div>
            </div>
        </div>
        """
    )

    col_st, col_date, col_hour = st.columns([3, 2, 2])

    with col_st:
        selected_station = st.selectbox(
            "Select Monitoring Station",
            options=STATIONS,
            index=1,  # Default to Central University
            key="demo_station_select"
        )

    with col_date:
        # Default date: 14 Jan 2026 where all 4 stations have telemetry
        default_date = date(2026, 1, 14)
        if default_date < min_datetime.date() or default_date > max_datetime.date():
            default_date = min_datetime.date()

        selected_date = st.date_input(
            "Replay Date (2026)",
            value=default_date,
            min_value=min_datetime.date(),
            max_value=max_datetime.date(),
            key="demo_date_input"
        )

    with col_hour:
        # Default hour: 08:00
        selected_hour = st.slider(
            "Hour of Day (IST)",
            min_value=0,
            max_value=23,
            value=8,
            format="%02d:00",
            key="demo_hour_slider"
        )

    # Active evaluation timestamp driving all page elements
    active_dt = datetime.combine(selected_date, datetime.min.time()) + timedelta(hours=selected_hour)
    active_ts = pd.to_datetime(active_dt)

    # Filter station slice
    st_df = df_preds[df_preds["station"] == selected_station].sort_values("datetime").reset_index(drop=True)
    cur_match = st_df[st_df["datetime"] == active_ts]

    if not cur_match.empty:
        cur_row = cur_match.iloc[0]
        actual_time_used = cur_row["datetime"]
    else:
        # Nearest row fallback to prevent any crash
        nearest_idx = (st_df["datetime"] - active_ts).abs().idxmin()
        cur_row = st_df.iloc[nearest_idx]
        actual_time_used = cur_row["datetime"]

    # Observed PM2.5 at time t (equals pred_persistence_h1)
    observed_pm25 = cur_row.get("pred_persistence_h1", np.nan)
    if pd.isna(observed_pm25):
        # Fallback to target of preceding hour if present
        observed_pm25 = cur_row.get("target_pm25_h1", np.nan)

    cpcb_cat = get_cpcb_category(observed_pm25)
    pm25_text = f"{observed_pm25:.1f}" if not pd.isna(observed_pm25) else "No data"

    st_meta = STATION_INFO.get(selected_station, {})
    typology_text = st_meta.get("typology", "Urban Monitoring Station")

    # 2. Hero Card: Large PM2.5, Unit, Category Label, Plain Message, and Scale Bar
    render_html(
        f"""
        <div class="iqair-hero-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 13px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em;">
                        📍 {selected_station}
                    </div>
                    <div style="font-size: 13px; color: #475569; margin: 3px 0 10px 0;">
                        Typology: <strong>{typology_text}</strong> • Lat: {st_meta.get('lat', 17.5):.4f}° N, Lon: {st_meta.get('lon', 78.4):.4f}° E
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 12px; font-weight: 600; color: #64748b;">Selected Replay Time</div>
                    <div style="font-size: 14px; font-weight: 700; color: #0f172a;">
                        {pd.to_datetime(actual_time_used).strftime('%A, %d %b %Y • %H:%M IST')}
                    </div>
                </div>
            </div>

            <div style="display: flex; align-items: baseline; gap: 4px; margin: 8px 0 12px 0;">
                <span class="hero-pm25-num" style="color: {cpcb_cat['color']};">{pm25_text}</span>
                <span class="hero-pm25-unit">µg/m³</span>
            </div>

            <div style="display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; 
                        background: {cpcb_cat['bg_light']}; color: {cpcb_cat['text_color']}; border: 1px solid {cpcb_cat['color']}; margin-bottom: 10px;">
                ● {cpcb_cat['name']} Air Quality
            </div>

            <div style="font-size: 13.5px; color: #334155; line-height: 1.5; font-weight: 500; margin-bottom: 16px;">
                {cpcb_cat['message']}
            </div>

            <div class="cpcb-track-container">
                <div class="cpcb-track-labels">
                    <span>Good (0–30)</span>
                    <span>Satisfactory (31–60)</span>
                    <span>Moderate (61–90)</span>
                    <span>Poor (91–120)</span>
                    <span>Very Poor (121–250)</span>
                    <span>Severe (250+)</span>
                </div>
                <div class="cpcb-spectrum-bar">
                    <div class="cpcb-indicator-pin" style="left: {cpcb_cat['percentage']}%;"></div>
                </div>
            </div>
        </div>
        """
    )

    # 3. Forecast Tiles for +1h, +6h, +24h, +48h
    render_html(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin: 24px 0 12px 0;">
            🔮 Multi-Horizon PM2.5 Forecast Tiles
        </div>
        """
    )

    tile_specs = [
        {
            "horizon": "+1h",
            "title": "+1h (Nowcast)",
            "model": "Stacking Ensemble",
            "pred_col": "pred_stacking_h1",
            "target_col": "target_pm25_h1",
            "mae": test_maes.get("h1", 3.011),
            "target_dt": actual_time_used + pd.Timedelta(hours=1)
        },
        {
            "horizon": "+6h",
            "title": "+6h (Sub-Daily)",
            "model": "Stacking Ensemble",
            "pred_col": "pred_stacking_h6",
            "target_col": "target_pm25_h6",
            "mae": test_maes.get("h6", 5.987),
            "target_dt": actual_time_used + pd.Timedelta(hours=6)
        },
        {
            "horizon": "+24h",
            "title": "+24h (Planning)",
            "model": "LightGBM Regressor",
            "pred_col": "pred_lgb_h24",
            "target_col": "target_pm25_h24",
            "mae": test_maes.get("h24", 6.668),
            "target_dt": actual_time_used + pd.Timedelta(hours=24)
        },
        {
            "horizon": "+48h",
            "title": "+48h (Advisory)",
            "model": "LightGBM Regressor",
            "pred_col": "pred_lgb_h48",
            "target_col": "target_pm25_h48",
            "mae": test_maes.get("h48", 7.930),
            "target_dt": actual_time_used + pd.Timedelta(hours=48)
        }
    ]

    t_cols = st.columns(4)

    for col, spec in zip(t_cols, tile_specs):
        p_val = cur_row.get(spec["pred_col"], np.nan)
        t_val = cur_row.get(spec["target_col"], np.nan)

        p_cat = get_cpcb_category(p_val)
        pred_label = f"{p_val:.1f} µg/m³" if not pd.isna(p_val) else "No data"
        obs_label = f"Observed: {t_val:.1f} µg/m³" if not pd.isna(t_val) else "Observed: No data"

        with col:
            render_html(
                f"""
                <div class="forecast-tile-card">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span class="forecast-horizon-badge">{spec['title']}</span>
                            <span style="font-size: 11px; color: #64748b; font-weight: 600;">{spec['model']}</span>
                        </div>
                        <div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">
                            Target: {spec['target_dt'].strftime('%d %b, %H:%M')}
                        </div>
                        <div class="forecast-val-large" style="color: {p_cat['color']};">
                            {pred_label}
                        </div>
                        <div style="display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; background: {p_cat['bg_light']}; color: {p_cat['text_color']}; margin-bottom: 6px;">
                            {p_cat['name']}
                        </div>
                        <div class="forecast-actual-badge">
                            {obs_label}
                        </div>
                    </div>
                    <div class="forecast-mae-note">
                        Typical Error: <strong>±{spec['mae']:.2f} µg/m³</strong> (Test MAE)
                    </div>
                </div>
                """
            )

    render_html("<hr style='margin: 26px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 4. Plotly Chart: Previous 3 Days + Next 24 Hours
    render_html(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            📈 Interactive Station Trajectory & Multi-Horizon Forecast
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 12px;">
            Window: Previous 72 hours observed in-situ PM2.5 plus the next 24 hours predictive forecast.
        </div>
        """
    )

    chart_start = actual_time_used - pd.Timedelta(hours=72)
    chart_end = actual_time_used + pd.Timedelta(hours=24)

    slice_df = st_df[(st_df["datetime"] >= chart_start) & (st_df["datetime"] <= chart_end)].copy()

    if not slice_df.empty:
        fig = go.Figure()

        # Prior 72 hours observed ground truth
        past_slice = slice_df[slice_df["datetime"] <= actual_time_used]
        if not past_slice.empty:
            fig.add_trace(go.Scatter(
                x=past_slice["datetime"],
                y=past_slice["pred_persistence_h1"],
                name="Historical PM2.5 (CAAQMS)",
                mode="lines+markers",
                marker=dict(size=4, color="#0f766e"),
                line=dict(color="#0f766e", width=2.4),
                hovertemplate="%{x|%d %b %H:%M}<br>Observed PM2.5: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Subsequent actual values (evaluation comparison)
        future_slice = slice_df[slice_df["datetime"] > actual_time_used]
        if not future_slice.empty and "pred_persistence_h1" in future_slice.columns:
            fig.add_trace(go.Scatter(
                x=future_slice["datetime"],
                y=future_slice["pred_persistence_h1"],
                name="Subsequent Actual PM2.5",
                mode="lines",
                line=dict(color="#94a3b8", width=1.8, dash="dot"),
                hovertemplate="%{x|%d %b %H:%M}<br>Subsequent Observed: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Model forecast points (+1h, +6h, +24h)
        f_times = [
            actual_time_used,
            actual_time_used + pd.Timedelta(hours=1),
            actual_time_used + pd.Timedelta(hours=6),
            actual_time_used + pd.Timedelta(hours=24)
        ]
        f_values = [
            observed_pm25,
            cur_row.get("pred_stacking_h1", np.nan),
            cur_row.get("pred_stacking_h6", np.nan),
            cur_row.get("pred_lgb_h24", np.nan)
        ]

        valid_times = [t for t, v in zip(f_times, f_values) if not pd.isna(v)]
        valid_vals = [v for v in f_values if not pd.isna(v)]

        if len(valid_vals) > 1:
            fig.add_trace(go.Scatter(
                x=valid_times,
                y=valid_vals,
                name="Model Predictive Forecast",
                mode="lines+markers",
                marker=dict(size=8, color="#f59e0b", symbol="diamond"),
                line=dict(color="#f59e0b", width=2.8),
                hovertemplate="%{x|%d %b %H:%M}<br>Forecast: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Persistence baseline
        if not pd.isna(observed_pm25):
            fig.add_trace(go.Scatter(
                x=[actual_time_used, actual_time_used + pd.Timedelta(hours=24)],
                y=[observed_pm25, observed_pm25],
                name="Zero-Order Persistence Baseline",
                mode="lines",
                line=dict(color="#cbd5e1", width=1.5, dash="dash"),
                hovertemplate="Persistence: <b>%{y:.1f} µg/m³</b><extra></extra>"
            ))

        # Replay timestamp marker line
        fig.add_vline(
            x=actual_time_used,
            line_width=2,
            line_dash="dash",
            line_color="#0f172a",
            annotation_text="Replay 'Now'",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#0f172a", family="Inter")
        )

        # Statutory 24-hour standard line
        fig.add_hline(
            y=60.0,
            line_width=1.5,
            line_dash="dot",
            line_color="#f97316",
            annotation_text="NAAQS 24h Standard (60 µg/m³)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#f97316", family="Inter")
        )

        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis_title="Timeline (IST)",
            yaxis_title="PM2.5 Concentration (µg/m³)",
            hovermode="x unified",
            height=370,
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#cbd5e1")
        fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#cbd5e1")

        st.plotly_chart(fig, width='stretch')

    render_html("<hr style='margin: 26px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 5. Model Explainability: Why did the model predict this?
    render_html(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            🧠 Why did the model predict this? (Model Explainability)
        </div>
        <div style="font-size: 13px; color: #64748b; margin-bottom: 12px;">
            Global feature attributions derived using TreeSHAP across the approved multi-source predictor set.
        </div>
        """
    )

    gfi_data = load_global_feature_importance()
    top_features_by_h = gfi_data.get("top_features_by_horizon", {})
    cat_shares_by_h = gfi_data.get("category_shares_by_horizon", {})

    col_exp_ctrl, col_exp_empty = st.columns([2, 3])
    with col_exp_ctrl:
        selected_explain_h = st.radio(
            "Select Forecast Horizon for Attribution:",
            options=["+1h", "+6h", "+24h", "+48h"],
            index=0,
            horizontal=True,
            key="demo_explain_horizon_radio"
        )

    clean_explain_h = selected_explain_h.lower().replace("+", "")
    shap_summary = load_shap_summary(clean_explain_h)
    n_explain_samples = shap_summary.get("n_explain_samples", 100)

    # Global attribution notice card
    render_html(
        f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #0f766e; border-radius: 8px; 
                    padding: 10px 14px; margin: 10px 0 16px 0; font-size: 12.5px; color: #334155; line-height: 1.5;">
            <strong>Global Attribution Notice:</strong> These visualizations show overall model dependency across the test partition, 
            not a local explanation for the single hour selected on the replay slider.<br>
            <span style="color: #64748b;">SHAP is computed on a sample of {n_explain_samples} observations per horizon (from the JSON); 
            read these as general patterns, not exact values.</span>
        </div>
        """
    )

    col_feat_chart, col_cat_chart = st.columns(2)

    with col_feat_chart:
        render_html(
            f"""
            <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                Top 10 Predictors for {selected_explain_h} Forecast
            </div>
            """
        )

        top_feats = top_features_by_h.get(clean_explain_h, [])
        if not top_feats and "top_15_features" in shap_summary:
            top_feats = shap_summary.get("top_15_features", [])

        if top_feats:
            top_10 = top_feats[:10]
            y_labels = [
                FEATURE_NAME_MAP.get(item.get("feature", ""), item.get("feature", "").replace("_", " ").title())
                for item in reversed(top_10)
            ]
            x_vals = [item.get("mean_abs_shap", 0.0) for item in reversed(top_10)]
            raw_feats = [item.get("feature", "") for item in reversed(top_10)]

            fig_top10 = go.Figure(go.Bar(
                x=x_vals,
                y=y_labels,
                orientation="h",
                marker=dict(
                    color="#0f766e",
                    line=dict(color="#115e59", width=1)
                ),
                hovertemplate="<b>%{y}</b> (<i>%{customdata}</i>)<br>Mean |SHAP|: <b>%{x:.3f} µg/m³</b><extra></extra>",
                customdata=raw_feats
            ))
            fig_top10.update_layout(
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=360,
                margin=dict(l=10, r=20, t=10, b=40),
                font=dict(family="Inter, sans-serif", size=11, color="#334155"),
                xaxis=dict(title="Mean |SHAP| (µg/m³)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                yaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1", tickfont=dict(size=11))
            )
            st.plotly_chart(fig_top10, width="stretch")
        else:
            st.info(f"Top feature importance data is not available for {selected_explain_h}.")

        render_html(
            f"""
            <div style="font-size: 12px; color: #64748b; line-height: 1.45; margin-top: 4px;">
                <em>Caption: Top 10 predictors with the highest overall impact on PM2.5 predictions for {selected_explain_h}. 
                Inertial lags dominate immediate nowcasting, while meteorology and regional dispersion expand their influence at extended horizons.</em>
            </div>
            """
        )

    with col_cat_chart:
        render_html(
            """
            <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 6px;">
                Predictive Category Share Transition (+1h to +48h)
            </div>
            """
        )

        category_order = [
            "Historical PM2.5 Dynamics",
            "Co-Pollutant / Chemical Ratios",
            "Thermodynamics & Boundary Layer",
            "Wind & Dispersion Dynamics",
            "Cyclical Calendar & Solar Timing",
            "Other Environmental Predictors"
        ]

        category_palette = {
            "Historical PM2.5 Dynamics": "#0F766E",        # Deep Teal
            "Co-Pollutant / Chemical Ratios": "#06B6D4",   # Cyan
            "Thermodynamics & Boundary Layer": "#F59E0B",  # Amber / Orange
            "Wind & Dispersion Dynamics": "#3B82F6",       # Blue
            "Cyclical Calendar & Solar Timing": "#8B5CF6", # Purple
            "Other Environmental Predictors": "#94A3B8"   # Slate Gray
        }

        horizons_list = ["h1", "h6", "h24", "h48"]
        h_display = ["+1h", "+6h", "+24h", "+48h"]

        if cat_shares_by_h:
            fig_cats = go.Figure()
            for cat in category_order:
                y_shares = []
                hover_txts = []
                for h_code, h_lbl in zip(horizons_list, h_display):
                    c_info = cat_shares_by_h.get(h_code, {}).get(cat, {"share_pct": 0.0, "absolute_mean_shap": 0.0})
                    pct = c_info.get("share_pct", 0.0)
                    abs_val = c_info.get("absolute_mean_shap", 0.0)
                    y_shares.append(pct)
                    hover_txts.append(f"<b>{cat}</b> ({h_lbl})<br>Attribution Share: <b>{pct:.1f}%</b><br>Mean |SHAP|: <b>{abs_val:.2f} µg/m³</b>")

                fig_cats.add_trace(go.Bar(
                    name=cat,
                    x=h_display,
                    y=y_shares,
                    marker_color=category_palette.get(cat, "#64748b"),
                    hovertemplate="%{text}<extra></extra>",
                    text=hover_txts
                ))

            fig_cats.update_layout(
                barmode="stack",
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                height=360,
                margin=dict(l=10, r=20, t=10, b=40),
                font=dict(family="Inter, sans-serif", size=11, color="#334155"),
                yaxis=dict(title="SHAP Attribution Share (%)", range=[0, 100], gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                xaxis=dict(title="Forecast Horizon", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=9.5)
                )
            )
            st.plotly_chart(fig_cats, width="stretch")
        else:
            st.info("Feature category share data is not available.")

        render_html(
            """
            <div style="font-size: 12px; color: #64748b; line-height: 1.45; margin-top: 4px;">
                <em>Caption: Evolution of predictive dependency across forecast horizons. The models rely heavily on recent PM2.5 at +1h (~80%) 
                and transition towards weather and thermodynamics (over 16% boundary layer/stability) at +48h.</em>
            </div>
            """
        )

    render_html("<hr style='margin: 26px 0; border: none; border-top: 1px solid #e2e8f0;'>")

    # 6. Map: Pydeck OpenStreetMap Basemap & 4 Stations Colored by Category
    render_html(
        """
        <div style="font-size: 17px; font-weight: 700; color: #0f172a; margin-bottom: 4px;">
            🗺️ Spatial Network Basemap & Active Telemetry
        </div>
        """
    )

    # Assemble map dataframe for all 4 stations at the active replay timestamp
    map_rows = []
    for s_name in STATIONS:
        s_meta = STATION_INFO.get(s_name, {})
        match = df_preds[(df_preds["station"] == s_name) & (df_preds["datetime"] == actual_time_used)]

        if not match.empty:
            val = match.iloc[0].get("pred_persistence_h1", np.nan)
            if pd.isna(val):
                val = match.iloc[0].get("target_pm25_h1", np.nan)
        else:
            val = np.nan

        cat = get_cpcb_category(val)
        val_str = f"{val:.1f} µg/m³" if not pd.isna(val) else "No data"
        rgba = hex_to_rgb(cat["color"], alpha=220)

        map_rows.append({
            "name": s_name,
            "lat": s_meta.get("lat", 17.5),
            "lon": s_meta.get("lon", 78.4),
            "typology": s_meta.get("typology", "Urban Station"),
            "pm25": val,
            "pm25_str": val_str,
            "category": cat["name"],
            "color": rgba
        })

    df_map = pd.DataFrame(map_rows)

    col_map_view, col_stations_view = st.columns([3, 2])

    with col_map_view:
        # Pydeck Scatterplot and Label Layers
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_radius=2200,
            get_fill_color="color",
            get_line_color=[15, 23, 42, 220],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True
        )

        label_layer = pdk.Layer(
            "TextLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_text="pm25_str",
            get_size=13,
            get_color=[15, 23, 42, 255],
            get_alignment_baseline="'bottom'",
            get_pixel_offset=[0, -18],
            font_family="Inter, sans-serif",
            font_weight="bold"
        )

        view_state = pdk.ViewState(
            latitude=17.50,
            longitude=78.42,
            zoom=10,
            pitch=0
        )

        deck = pdk.Deck(
            layers=[scatter_layer, label_layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{name}</b><br>{typology}<br>PM2.5: <b>{pm25_str}</b> ({category})",
                "style": {"backgroundColor": "#ffffff", "color": "#0f172a", "fontSize": "12px", "borderRadius": "8px", "padding": "8px"}
            },
            map_style="light"
        )

        st.pydeck_chart(deck, height=420)

    with col_stations_view:
        render_html(
            """
            <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-bottom: 8px;">
                Network Stations at Selected Time
            </div>
            """
        )

        for item in map_rows:
            is_cur = (item["name"] == selected_station)
            card_border = "border: 2px solid #0f766e;" if is_cur else "border: 1px solid #e2e8f0;"
            card_bg = "#f0fdfa" if is_cur else "#ffffff"

            render_html(
                f"""
                <div style="background: {card_bg}; {card_border} border-radius: 12px; padding: 12px 14px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 13.5px; font-weight: 700; color: #0f172a;">{item['name']}</div>
                            <div style="font-size: 11px; color: #64748b;">{item['typology']}</div>
                        </div>
                        <div>
                            <span style="display: inline-block; padding: 4px 10px; border-radius: 8px; font-size: 12.5px; font-weight: 800; background: {get_cpcb_category(item['pm25'])['bg_light']}; color: {get_cpcb_category(item['pm25'])['text_color']};">
                                {item['pm25_str']}
                            </span>
                        </div>
                    </div>
                </div>
                """
            )

        # 7. One small note on Bollaram
        render_html(
            """
            <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 10px 12px; margin-top: 10px; font-size: 11.5px; color: #92400e; line-height: 1.5;">
                ℹ️ <strong>Bollaram has valid 2026 data only in Jan-Feb and Aug-Sep.</strong>
            </div>
            """
        )

    # 7. Research Prototype Footer
    render_html(
        """
        <div style="margin-top: 32px; padding: 24px 16px; border-top: 1px solid #e2e8f0; text-align: center; color: #64748b; font-size: 13px; line-height: 1.6;">
            <div><strong>Hyderabad Air Forecast · Capstone research prototype</strong></div>
            <div>Not for regulatory or public health use · Replays 2026 test data, not a live feed</div>
            <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Data: CPCB CAAQMS observations · Open-Meteo weather</div>
        </div>
        """
    )


if __name__ == "__main__":
    main()
