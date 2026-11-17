# Implementation Codebase & Research Engineering Architecture

**Project Title:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad Using Multi-Source Environmental Data and Explainable Machine Learning  
**Milestones Completed:**
- **Milestone 1:** Data Foundation and Preprocessing (100% PASSED, 21/21 Automated Tests)
- **Milestone 2:** Multi-Source Feature Engineering (100% PASSED, 31/31 Automated Tests across full suite)

---

## 1. Directory and Module Architecture

The codebase inside `new_research_project/implementation/` is organized into modular, testable, and strictly leak-free packages:

```
new_research_project/implementation/
├── __init__.py
├── README.md                           # Master implementation documentation & CLI guide
├── run_pipeline.py                     # Master CLI script executing Milestones 1 and 2
├── configs/
│   ├── __init__.py
│   └── config.py                       # Configuration dataclasses, thresholds, OSM proxies, paths
├── data_ingestion/
│   ├── __init__.py
│   ├── cpcb_ingestion.py               # Ingestion & validation of physical CPCB CAAQMS observations
│   ├── weather_ingestion.py            # Ingestion & validation of Open-Meteo weather reanalysis
│   └── dataset_loader.py               # Multimodal alignment on [station, datetime] & geodetic metadata
├── preprocessing/
│   ├── __init__.py
│   ├── gap_analyzer.py                 # Contiguous missing-block analyzer (<=3h, 4-24h, >24h)
│   ├── prospective_imputer.py          # Strict prospective imputer (train-fitted diurnal, past-only ffill)
│   └── scaler.py                       # StandardScaler fitted strictly on 2023-2024 training partition
├── target_generation/
│   ├── __init__.py
│   └── target_builder.py               # Multi-horizon (+1h, +6h, +24h, +48h) & hotspot targets
├── feature_engineering/
│   ├── __init__.py
│   ├── historical_features.py          # Autoregressive lags (1,2,3,6,12,24h), rolling stats, trends, cyclical
│   ├── weather_features.py             # Wind vector (u, v), psychrometric (VPD, dew point), stability proxies
│   ├── osm_road_features.py            # Localized roadway metrics in projected UTM 44N (EPSG:32644)
│   ├── congestion_features.py          # Diurnal rush hours, exploratory freight window, interaction proxies
│   ├── feature_store_builder.py        # Master orchestrator assembling hourly Parquet & CSV feature store
│   └── feature_dictionary.py           # Machine-readable JSON and Markdown feature catalog generator
├── validation/
│   ├── __init__.py
│   ├── chronological_split.py          # Partitions 2023-2024 (train), 2025 (val), 2026 (test) with 48h buffer
│   └── integrity_checker.py            # Automated mathematical, continuity, and non-leakage auditor
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures and synthetic test series
│   ├── test_ingestion.py               # 7 tests: duplicates, negatives, aliases, continuity, counts
│   ├── test_missing_values.py          # 5 tests: masks, gap categorizer, ffill past-only, NaNs preserved
│   ├── test_target_generation.py       # 4 tests: multi-horizon shift, N_valid >= 16 rule, boundaries
│   ├── test_partitioning.py            # 2 tests: partition dimensions, chronological ordering, buffer guard
│   ├── test_leakage_safety.py          # 2 tests: scaler and diurnal lookup invariance to future perturbations
│   ├── test_reproducibility.py         # 1 test: deterministic target and preprocessing reproducibility
│   └── test_feature_engineering.py     # 10 tests: lags, past-only rolling, wind u/v, OSM, congestion, decoupling
└── reports/
    ├── __init__.py
    ├── report_generator.py             # Markdown report generation engine
    ├── DATA_QUALITY_REPORT.md          # Comprehensive data quality audit
    ├── STATION_OBSERVATIONS_REPORT.md  # Station-wise empirical counts vs audited specs
    ├── MISSINGNESS_REPORT.md           # Contiguous gap lengths and telemetry breakdown
    ├── TARGET_GENERATION_REPORT.md     # Multi-horizon regression & hotspot target summary
    ├── TEST_RESULTS_REPORT.md          # Complete pytest execution results (31/31 passed)
    ├── WEATHER_PROVENANCE_REPORT.md    # Atmospheric reanalysis provenance & point-in-time protocol
    ├── OSM_PROCESSING_REPORT.md        # OpenStreetMap roadway metrics in projected UTM Zone 44N
    ├── TRAQID_FEASIBILITY_REPORT.md    # Technical feasibility & quarantine assessment of TRAQID
    ├── LEAKAGE_AUDIT_REPORT.md         # Exhaustive 8-point non-leakage verification checklist
    └── FEATURE_DICTIONARY.md           # Complete inventory and specifications for all 153 features
```

---

## 2. Multi-Source Feature Engineering Highlights

### 2.1 Historical PM2.5 & Co-Pollutant Features
- **Autoregressive Lags:** $y_{i, t-1}, y_{i, t-2}, y_{i, t-3}, y_{i, t-6}, y_{i, t-12}, y_{i, t-24}$.
- **Past-Only Rolling Statistics:** Rolling means over 3h, 6h, 12h, 24h; rolling standard deviations over 6h and 24h; rolling min/max over 24h. Aggregations use strictly past observations $[t-w+1 : t]$.
- **Historical Trend Dynamics:** 1h, 3h, 24h differences ($y_t - y_{t-k}$), acceleration $[y_t - y_{t-1}] - [y_{t-1} - y_{t-2}]$, and ratio to historical 24h mean.
- **Continuous Cyclical Time:** $\sin$ and $\cos$ transformations for hour of day (period 24), day of week (period 7), and month of year (period 12), plus IMD seasonal indicators.
- **Chemical Ratios:** Fine-to-coarse ratio $\text{PM}_{2.5} / (\text{PM}_{10} + \varepsilon)$ and combustion ratio $\text{NO} / (\text{NO}_2 + \varepsilon)$.

### 2.2 Open-Meteo Meteorological Reanalysis Features
- **Trigonometric Wind Vectors:**
  $$u = -U_{10\text{m}} \cdot \sin(\theta), \quad v = -U_{10\text{m}} \cdot \cos(\theta), \quad \sqrt{u^2 + v^2} = U_{10\text{m}}$$
- **Psychrometric Variables:** Saturation vapor pressure ($e_s$), actual vapor pressure ($e_a$), Vapor Pressure Deficit ($\text{VPD} = e_s - e_a$), and dew point temperature ($T_{\text{dew}}$) via Magnus formulation.
- **Boundary-Layer Stability:** Stagnation flag ($U_{10\text{m}} < 1.5\,\text{m/s}$ and precipitation $\le 0.01\,\text{mm}$), ventilation index proxy, and accumulated precipitation sums (6h, 24h).
- **Point-in-Time Availability Protocol:** All weather predictors condition strictly on observations $\le t$. Future reanalysis weather ($t+h$) is never supplied as an input feature.

### 2.3 OpenStreetMap Roadway Proxies in Projected UTM Zone 44N
- Projected CRS: **EPSG:32644 (WGS 84 / UTM Zone 44N)**.
- Circular Buffers: 1 km radius ($3.1416\,\text{km}^2$) and 3 km radius ($28.2743\,\text{km}^2$).
- Localized Roadway Metrics: Primary road length ($L_{\text{primary}}$), secondary road length ($L_{\text{secondary}}$), total road density, primary concentration ratio, Euclidean distance to nearest national highway corridor ($D_{\text{highway}}$), and multi-leg intersection density ($N_{\text{intersections}}$).
- Methodological Qualifier: Explicitly labeled as static spatial proxies for land-use intensity with 4 spatial degrees of freedom, NOT measured vehicle volume.

### 2.4 Congestion Windows & Spatio-Temporal Interactions
- Diurnal Congestion Windows: Morning peak (08:00–11:00), Evening peak (17:00–21:00), Weekend indicator, and Weekday peak.
- Exploratory Freight Restriction Window (23:00–05:00): Implemented as an exploratory proxy assumption pending official Hyderabad Police gazette audit.
- Traffic Pressure Proxies: Non-linear interaction features multiplying localized road density by active temporal congestion indicators.

### 2.5 TRAQID Quarantine Assessment
- Formal investigation confirmed author disclaimer: CSV timestamps are synthetically regenerated 5-second resampled index counters that do not reflect wall-clock capture time.
- Zero GPS coordinates are present in the public CSV schema.
- TRAQID is strictly quarantined as an auxiliary qualitative/computer-vision case study and excluded from the continuous CPCB hourly feature store.

---

## 3. How to Execute Pipeline and Tests

### Run Automated Test Suite (31 Automated Tests)
```bash
python -m pytest new_research_project/implementation/tests/ -v
```

### Run Milestone 1 Pipeline (Data Foundation & Preprocessing)
```bash
python new_research_project/implementation/run_pipeline.py --milestone 1
```

### Run Milestone 2 Pipeline (Multi-Source Feature Engineering)
```bash
python new_research_project/implementation/run_pipeline.py --milestone 2
```

### Run All Milestones End-to-End
```bash
python new_research_project/implementation/run_pipeline.py --milestone all
```
