"""
Dashboard Backend: Explainability Service
Extracts TreeSHAP feature rankings, category shares, natural language narratives,
and ensures non-causal statistical interpretation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.backend.data_loader import load_global_feature_importance, load_hotspot_narratives, load_shap_values, get_figure_path


CATEGORY_DESCRIPTIONS = {
    "Historical PM2.5 Dynamics": {
        "description": "Autoregressive lags (1h, 2h, 3h, 6h, 12h, 24h), rolling window statistics (3h, 6h, 12h, 24h), and short-term trends.",
        "physical_mechanism": "Captures urban aerosol persistence and background accumulation within the nocturnal atmospheric boundary layer."
    },
    "Thermodynamics & Boundary Layer": {
        "description": "Temperature at 2m, relative humidity, surface pressure, atmospheric stagnation index, and boundary layer height.",
        "physical_mechanism": "Controls vertical dispersion volume, nocturnal thermal inversions, and psychrometric hygroscopic aerosol swelling."
    },
    "Wind & Dispersion Dynamics": {
        "description": "10m wind speed, U/V wind orthogonal vectors, wind direction sine/cosine harmonics, and gust metrics.",
        "physical_mechanism": "Drives horizontal advective clearance and regional pollutant transport across metropolitan sectors."
    },
    "Co-Pollutant / Chemical Ratios": {
        "description": "In-situ physical observations of PM10, NO, NO2, SO2, CO, Ozone, and PM2.5/PM10 fine-fraction ratios.",
        "physical_mechanism": "Serves as chemical fingerprints for secondary aerosol precursors, combustion signatures, and industrial plumes."
    },
    "Cyclical Calendar & Solar Timing": {
        "description": "Diurnal hour sine/cosine cycles, day of week indicators, solar zenith angles, and congestion window indicators.",
        "physical_mechanism": "Correlates with diurnal anthropogenic activity cycles, rush hour peaks, and photochemical diurnal forcing."
    },
    "Other Environmental Predictors": {
        "description": "Soil temperature, surface solar radiation downwards, and precipitation.",
        "physical_mechanism": "Wet deposition scavenging and ground-level radiative heating."
    }
}


def get_top_features_dataframe(horizon: str = "h1", top_n: int = 15) -> pd.DataFrame:
    """Returns top ranked features with mean absolute SHAP attributions."""
    gfi = load_global_feature_importance()
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    
    top_list = gfi.get("top_features_by_horizon", {}).get(h_key, [])
    if not top_list:
        return pd.DataFrame()
        
    records = []
    for item in top_list[:top_n]:
        feat = item.get("feature", "")
        # Classify category
        cat = "Historical PM2.5 Dynamics"
        if any(w in feat for w in ["wind", "u_wind", "v_wind"]):
            cat = "Wind & Dispersion Dynamics"
        elif any(w in feat for w in ["temp", "humidity", "pressure", "stagnation", "boundary"]):
            cat = "Thermodynamics & Boundary Layer"
        elif any(w in feat for w in ["NO", "SO2", "CO", "Ozone", "PM10"]):
            cat = "Co-Pollutant / Chemical Ratios"
        elif any(w in feat for w in ["hour", "day", "solar", "window", "peak"]):
            cat = "Cyclical Calendar & Solar Timing"
            
        records.append({
            "Rank": item.get("rank"),
            "Feature Name": feat,
            "Mean |SHAP| (ug/m3)": round(float(item.get("mean_abs_shap", 0.0)), 3),
            "Feature Category": cat
        })
        
    return pd.DataFrame(records)


def get_category_shares_dataframe(horizon: str = "h1") -> pd.DataFrame:
    """Returns category contribution shares for the chosen horizon."""
    gfi = load_global_feature_importance()
    clean_h = horizon.lower().replace("+", "").replace("h", "")
    h_key = f"h{clean_h}"
    
    shares = gfi.get("category_shares_by_horizon", {}).get(h_key, {})
    if not shares:
        return pd.DataFrame()
        
    records = []
    for cat_name, data in shares.items():
        records.append({
            "Feature Group": cat_name,
            "Attribution Share (%)": round(float(data.get("share_pct", 0.0)), 2),
            "Mean |SHAP| (ug/m3)": round(float(data.get("absolute_mean_shap", 0.0)), 3),
            "Physical Mechanism": CATEGORY_DESCRIPTIONS.get(cat_name, {}).get("physical_mechanism", "")
        })
        
    df = pd.DataFrame(records)
    return df.sort_values("Attribution Share (%)", ascending=False).reset_index(drop=True)


def get_station_narrative(station: str) -> str:
    """Retrieves human-readable narrative explanation for a station."""
    narratives = load_hotspot_narratives()
    return narratives.get(station, f"No detailed narrative explanation recorded for {station}.")
