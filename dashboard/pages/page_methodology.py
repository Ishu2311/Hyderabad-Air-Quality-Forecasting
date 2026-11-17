"""
Dashboard Page: 6. Methodology & Data Quality
Presents feature engineering taxonomy, leakage audits, temporal buffering,
spatial cross-validation accounting, and scientific quarantine decisions.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

# Ensure project root (new_research_project) and repo root are on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _PROJECT_ROOT.parent
for _p in [str(_PROJECT_ROOT), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dashboard.styles.theme import render_page_header, inject_custom_css, render_html


def render_page_methodology():
    """Renders Methodology and Data Quality section."""
    inject_custom_css()

    render_page_header(
        badge="🔬 RIGOROUS SCIENTIFIC PROTOCOL • ZERO LEAKAGE AUDIT",
        title="Methodology, Data Quality & Validation Audit",
        subtitle="Forensic Verification of Zero Target Leakage, 48-Hour Boundary Buffering, 84 Approved Features, and Spatial LOSOCV."
    )

    # 1. Preprocessing & Missing Value Protocol
    render_html(
        """
        <div class="env-card">
            <div class="env-card-header">
                <div class="env-card-title">🛡️ 1. Ingestion & Missing Value Protocols</div>
                <span class="badge badge-teal">Leak-Free Preprocessing</span>
            </div>
            <div style="font-size: 13.5px; color: #cbd5e1; line-height: 1.65;">
                <p>To guarantee prospective real-world deployment validity and prevent lookahead leakage:</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 10px;">
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #10b981;">
                        <div style="color: #34d399; font-weight: 700; font-size: 13px;">Short Gaps (≤ 3 hours)</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            Forward-filled strictly using past valid physical observations (zero future lookahead).
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #06b6d4;">
                        <div style="color: #38bdf8; font-weight: 700; font-size: 13px;">Medium Gaps (4–24 hours)</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            Imputed via month-hour diurnal tables fit strictly on the training partition (2023–2024).
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #f59e0b;">
                        <div style="color: #fbbf24; font-weight: 700; font-size: 13px;">Extended Outages (> 24 hours)</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            Preserved strictly as missing (<code>NaN</code>). Target values are <strong>never imputed with synthetic constants</strong>.
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 12px 14px; border-radius: 8px; border-left: 3px solid #8b5cf6;">
                        <div style="color: #a78bfa; font-weight: 700; font-size: 13px;">Physical Bounds Sanitization</div>
                        <div style="font-size: 12px; color: #94a3b8; margin-top: 4px;">
                            Strict sensor physical limits (0.0 ≤ PM<sub>2.5</sub> ≤ 1,000.0 µg/m³). Negative values sanitized to quality flags.
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
    )

    st.markdown("---")

    # 2. Feature Architecture: 84 Approved vs 19 Quarantined
    st.subheader("2. Feature Store Architecture: 84 Approved Predictors vs. 19 Quarantined Proxies")
    st.markdown(
        """
        The project established an exhaustive 173-column schema identity:
        
        $$\\text{Total Columns (173)} = \\text{Identifiers (3)} + \\text{Metadata/Flags (51)} + \\text{Primary Predictors (84)} + \\text{Quarantined Road Proxies (19)} + \\text{Targets (16)}$$
        """
    )

    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        render_html(
            """
            <div class="env-card" style="height: 100%;">
                <div class="env-card-header">
                    <div class="env-card-title">✅ Approved Primary Predictors (84 Features)</div>
                    <span class="badge badge-teal">Verified Ingestion</span>
                </div>
                <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.6;">
                    <p>• <strong>Historical Autoregressive Dynamics (23 features):</strong> Autoregressive lags (1h, 2h, 3h, 6h, 12h, 24h), rolling statistics (mean, std, min, max across 3h, 6h, 12h, 24h windows), and 1h/3h/24h acceleration trends.</p>
                    <p>• <strong>Atmospheric Thermodynamics & Boundary Layer (18 features):</strong> 2m temperature, relative humidity, surface pressure, pressure tendencies, boundary layer height, dew point depression.</p>
                    <p>• <strong>Wind & Dispersion Dynamics (14 features):</strong> 10m wind speed, U/V orthogonal components, wind direction sine/cosine harmonics, wind shear, gust velocity.</p>
                    <p>• <strong>In-Situ Co-Pollutants (18 features):</strong> PM10, NO, NO2, SO2, CO, Ozone telemetry, and fine-fraction PM2.5/PM10 ratio.</p>
                    <p>• <strong>Cyclical Calendar & Solar Dynamics (11 features):</strong> Diurnal hour harmonics, day-of-week, solar zenith angles, morning/evening rush-hour congestion flags.</p>
                </div>
            </div>
            """
        )

    with col_feat2:
        render_html(
            """
            <div class="env-card" style="height: 100%; border-left: 4px solid #ef4444;">
                <div class="env-card-header">
                    <div class="env-card-title">🚫 Quarantined Road Proxies (19 Features)</div>
                    <span class="badge badge-rose">Scientific Quarantine</span>
                </div>
                <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.6;">
                    <div class="alert-box alert-danger" style="margin-bottom: 12px; padding: 10px 14px;">
                        <span>⚠️</span>
                        <div>
                            <strong>Forensic Audit Finding:</strong> An independent audit discovered that highway-distance, 
                            road-density, and intersection metrics represented unverified static assumptions rather than verified 
                            shapefile calculations.
                        </div>
                    </div>
                    <p style="color: #94a3b8;">
                        <strong>Remediation Decision:</strong> All 19 derived road proxy metrics were <strong>strictly quarantined</strong> 
                        and excluded from all primary regression and classification model pipelines. 
                        The models rely exclusively on verified CPCB physical sensor telemetry and Open-Meteo ERA5 atmospheric reanalysis.
                    </p>
                </div>
            </div>
            """
        )

    st.markdown("---")

    # 3. Temporal Boundary Purging
    st.subheader("3. Temporal Boundary Purging & Zero-Leakage Chronology")
    st.markdown(
        """
        To eliminate forward-looking target leakage across partition boundaries, an explicit **48-hour boundary buffer** was enforced:
        
        - **Training Set:** `2023-01-01 00:00:00` to `2024-12-31 23:00:00` ($70,176$ station-hours).
        - **Validation Set:** `2025-01-01 00:00:00` to `2025-12-31 23:00:00` ($35,040$ station-hours).
        - **Locked Test Set:** `2026-01-01 00:00:00` to `2026-09-30 23:00:00` ($26,208$ station-hours).
        - **Purged Boundary Samples:** Exactly 192 samples at the end of Train and 192 samples at the end of Validation (48 hours $\times$ 4 stations) whose multi-step forward target windows crossed into the subsequent partition were completely deleted.
        """
    )

    st.markdown("---")

    # 4. Spatial Validation & LOSOCV Audit
    st.subheader("4. Spatial Validation (Leave-One-Station-Out Cross-Validation)")
    st.markdown(
        """
        To rigorously evaluate spatial generalizability beyond monitored sites, a Leave-One-Station-Out Cross-Validation (LOSOCV) 
        protocol was implemented across the 2026 test partition:
        """
    )

    losocv_records = [
        {"Held-Out Station": "Bollaram Industrial Area", "Valid Test Hours": "1,875", "Observed Mean": "41.85 µg/m³", "IDW Estimate": "30.34 µg/m³", "MAE": "13.14 µg/m³", "Bias": "-11.52 µg/m³", "Spatial R²": "+0.507"},
        {"Held-Out Station": "Central University", "Valid Test Hours": "5,743", "Observed Mean": "19.22 µg/m³", "IDW Estimate": "31.22 µg/m³", "MAE": "12.84 µg/m³", "Bias": "+12.00 µg/m³", "Spatial R²": "+0.025"},
        {"Held-Out Station": "ECIL Kapra", "Valid Test Hours": "5,849", "Observed Mean": "30.02 µg/m³", "IDW Estimate": "27.60 µg/m³", "MAE": "11.67 µg/m³", "Bias": "-2.42 µg/m³", "Spatial R²": "-1.527"},
        {"Held-Out Station": "ICRISAT Patancheru", "Valid Test Hours": "5,606", "Observed Mean": "27.10 µg/m³", "IDW Estimate": "26.88 µg/m³", "MAE": "9.16 µg/m³", "Bias": "-0.22 µg/m³", "Spatial R²": "+0.535"},
        {"Held-Out Station": "Network Macro-Average", "Valid Test Hours": "19,073 Total", "Observed Mean": "29.55 µg/m³", "IDW Estimate": "29.01 µg/m³", "MAE": "11.703 µg/m³", "Bias": "-0.538 µg/m³", "Spatial R²": "-0.115"}
    ]
    st.table(pd.DataFrame(losocv_records))

    st.markdown(
        r"""
        #### Key Findings from the Spatial Audit:
        1. **Sample Accounting & Seasonal Confounding:** Bollaram has valid 2026 data only in Jan-Feb and Aug-Sep (36% coverage); its test statistics are seasonally confounded. Central University ($N=5,743$), ECIL Kapra ($N=5,849$), and ICRISAT ($N=5,606$) maintained continuous uptime, yielding **19,073 total station evaluations** across **1,479 synchronous complete snapshots**.
        2. **Macro-Average vs. Pooled Formulation:** The reported IDW MAE of **$11.703\,\mu\text{g/m}^3$** is an unweighted macro-average across the 4 stations:
           $$\text{Macro MAE} = \frac{13.14 + 12.84 + 11.67 + 9.16}{4} = 11.7025 \approx 11.703\,\mu\text{g/m}^3$$
           The pooled (sample-weighted) network MAE is **$11.429\,\mu\text{g/m}^3$** ($n=19,073$). Note that IDW only slightly beats a plain average of donor stations (**$11.43$ vs. $12.13\,\mu\text{g/m}^3$**).
        3. **The 388% Spatial Penalty:** Point forecasting achieves **$3.01\,\mu\text{g/m}^3$ MAE**, but spatial transfer error jumps to **$11.703\,\mu\text{g/m}^3$**. This is caused by severe micro-climatic typology mismatch (forested university enclave vs chemical manufacturing corridor) and the geographic isolation of ECIL Kapra ($24.6\,\text{km}$ east of other sensors).
        4. **Ordinary Kriging Ill-Conditioning:** With only 3 donor stations per LOSOCV fold (3 distance pairs), theoretical variogram estimation is statistically underdetermined. The Kriging matrix condition number frequently exceeds $10^{12}$, requiring Moore-Penrose pseudo-inversion. Ordinary Kriging was therefore classified strictly as an **experimental diagnostic**, while IDW was designated the primary reporting surface.
        """
    )

    st.markdown("---")

    # 5. Methodological Scope & Core Limitations
    st.subheader("5. Methodological Scope & Explicit Limitations")
    render_html(
        """
        <div class="env-card" style="border-left: 4px solid #f59e0b;">
            <div class="env-card-header">
                <div class="env-card-title">🛡️ Transparent Academic Disclosures</div>
                <span class="badge badge-amber">Limitations</span>
            </div>
            <div style="font-size: 13px; color: #475569; line-height: 1.65;">
                <p>• <strong>Network Geometry (4 Stations):</strong> In-situ ground truth is restricted to 4 CAAQMS reference stations within Greater Hyderabad. Continuous spatial interpolation must be treated as a regional estimate, not neighbourhood ground truth.</p>
                <p>• <strong>Geographic Domain (One City):</strong> All models and baselines are calibrated specifically on the Greater Hyderabad metropolitan domain (1,600 km²); cross-city transferability is not claimed.</p>
                <p>• <strong>No Measured Traffic Data:</strong> Road and highway proxies are quarantined due to unverified geometries. There is no measured traffic data or dynamic vehicular sensor telemetry in the operational feature set.</p>
                <p>• <strong>Severe Spike (≥ 120 µg/m³) Non-Evaluability:</strong> The severe spike class contains only 1 positive event in the entire 2026 chronological test set (and 17 out of 131,424 in raw CPCB data). Severe spike classification results are <em>not statistically evaluable</em>.</p>
                <p>• <strong>Bollaram Hardware Outages:</strong> Bollaram has valid 2026 data only in Jan–Feb and Aug–Sep (36% coverage due to sensor hardware outages); its test statistics are seasonally confounded.</p>
            </div>
        </div>
        """
    )


if __name__ == "__main__":
    try:
        st.set_page_config(
            page_title="Methodology & Data Quality - Hyderabad PM2.5",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass
    from dashboard.components.disclaimers import render_sidebar_safeguards
    render_sidebar_safeguards()
    render_page_methodology()
