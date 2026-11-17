"""
Human-Readable Hotspot Narrative Generation Module
Milestone 4: Spatial Hotspot Mapping, Interpolation and SHAP Explainability

Dynamically translates numerical model predictions, SHAP attribution values,
and spatial uncertainty diagnostics into rigorous, contextual natural-language narratives.

Academic Integrity Requirements:
- Explains top positive (elevating) and negative (mitigating) contributing features.
- Distinguishes statistical association from physical causation.
- Explicitly avoids asserting traffic or industrial activity caused the episode unless supported by external data.
- Includes operational uncertainty and monitoring density limitations.
"""

from typing import Dict, List, Optional, Any
import numpy as np


def generate_human_readable_hotspot_narrative(
    station_name: str,
    horizon_label: str,
    predicted_pm25: float,
    observed_baseline: float,
    top_elevating_factors: List[Dict[str, Any]],
    top_mitigating_factors: List[Dict[str, Any]],
    uncertainty_tier: str = "Low Uncertainty (Proximate <= 5km)",
    acute_threshold: float = 60.0
) -> str:
    """
    Dynamically generates contextual academic natural-language explanation for an air quality forecast.
    """
    is_excursion = predicted_pm25 >= acute_threshold
    status_str = (
        f"is predicted to experience elevated PM2.5 concentrations exceeding the statutory 24-hour NAAQS benchmark ({acute_threshold:.0f} ug/m3)"
        if is_excursion
        else f"is predicted to remain within the statutory 24-hour NAAQS benchmark ({acute_threshold:.0f} ug/m3)"
    )

    # Narrative opening
    opening = (
        f"{station_name} {status_str} over the next {horizon_label} "
        f"(forecast: {predicted_pm25:.1f} ug/m3, compared to a recent baseline of {observed_baseline:.1f} ug/m3)."
    )

    # Decompose top elevating drivers
    if top_elevating_factors:
        elev_descriptions = []
        for f in top_elevating_factors[:3]:
            feat = f["feature"]
            shap_val = f["shap_impact"]
            obs_val = f.get("observed_value")

            if "lag_1h" in feat:
                elev_descriptions.append(f"high recent persistence from the preceding hour (+{shap_val:.1f} ug/m3)")
            elif "rolling_mean" in feat:
                elev_descriptions.append(f"accumulated diurnal pollutant loading across prior hours (+{shap_val:.1f} ug/m3)")
            elif "stagnation" in feat or "wind_speed" in feat:
                elev_descriptions.append(f"reduced horizontal ventilation and low boundary-layer wind dispersion (+{shap_val:.1f} ug/m3)")
            elif "boundary_layer" in feat:
                elev_descriptions.append(f"shallower nocturnal boundary-layer height suppressing vertical mixing (+{shap_val:.1f} ug/m3)")
            elif "rush_hour" in feat or "congestion" in feat:
                elev_descriptions.append(f"temporal diurnal peak windows associated with active urban activity (+{shap_val:.1f} ug/m3)")
            elif "relative_humidity" in feat:
                elev_descriptions.append(f"elevated atmospheric humidity favoring secondary aerosol hygroscopic growth (+{shap_val:.1f} ug/m3)")
            elif "temperature" in feat:
                elev_descriptions.append(f"local thermodynamic inversion conditions (+{shap_val:.1f} ug/m3)")
            else:
                elev_descriptions.append(f"{feat} (+{shap_val:.1f} ug/m3)")

        elevating_str = "The upward elevation in predicted PM2.5 is primarily driven by " + ", ".join(elev_descriptions) + "."
    else:
        elevating_str = "No dominant upward drivers were identified."

    # Decompose top mitigating drivers
    if top_mitigating_factors:
        mit_descriptions = []
        for f in top_mitigating_factors[:2]:
            feat = f["feature"]
            shap_val = f["shap_impact"]

            if "wind" in feat or "dispersion" in feat:
                mit_descriptions.append(f"active atmospheric ventilation mitigating concentration build-up ({shap_val:.1f} ug/m3)")
            elif "boundary_layer" in feat or "temp" in feat:
                mit_descriptions.append(f"convective vertical dilution into a deeper boundary layer ({shap_val:.1f} ug/m3)")
            elif "trend" in feat or "ewma" in feat:
                mit_descriptions.append(f"negative historical trajectory in recent telemetry ({shap_val:.1f} ug/m3)")
            else:
                mit_descriptions.append(f"{feat} ({shap_val:.1f} ug/m3)")

        mitigating_str = "Downward mitigation is exerted by " + ", ".join(mit_descriptions) + "."
    else:
        mitigating_str = "Mitigating atmospheric factors are minimal during this forecast window."

    # Scientific disclaimer & uncertainty contextualization
    scientific_context = (
        "Methodological Note: Feature contributions are quantified via TreeSHAP and reflect statistical "
        "predictive associations rather than proven direct physical causation. Diurnal window variables "
        "capture historical cyclical correlations but do not constitute direct vehicle telemetry measurements."
    )

    uncertainty_note = (
        f"Spatial Interpretation: This prediction reflects {uncertainty_tier}. Spatial interpolation beyond "
        "physical monitoring locations carries inherent interpolation variance and should be interpreted alongside "
        "the 4-station network geometry limitations."
    )

    full_narrative = f"{opening}\n\n{elevating_str}\n\n{mitigating_str}\n\n{scientific_context}\n\n{uncertainty_note}"
    return full_narrative
