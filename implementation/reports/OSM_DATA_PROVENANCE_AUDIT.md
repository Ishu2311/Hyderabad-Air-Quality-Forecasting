# OpenStreetMap (OSM) Road-Data Provenance and Verification Audit

**Document Code:** `AUDIT-M2-OSM-PROVENANCE`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Audit Dimension:** Data Provenance, Geometric Extraction & Verification  
**Audit Classification:** **FAIL / UNVERIFIED AS MEASURED OSM RESULTS (RE-CLASSIFIED AS ASSUMED BENCHMARK PROXIES)**  

---

## 1. Executive Summary & Audit Mandate

This audit investigates the scientific provenance and empirical validity of the road-network metrics reported in Milestone 2:
- Road lengths by class ($L_{\text{primary}}, L_{\text{secondary}}$ in 1km and 3km buffers).
- Road densities ($\text{km}/\text{km}^2$).
- Primary road concentration ratios.
- Highway corridor proximity ($D_{\text{highway}}$).
- Multi-leg intersection density ($N_{\text{intersections}}$).

### High-Level Finding
An exhaustive search of the project workspace and network execution logs confirms:
1. **No local OSM raw vector geometries** (e.g., `.osm`, `.pbf`, `.geojson`, `.shp`) ever existed in the repository.
2. The Overpass API extraction query was **never executed successfully** due to server-side rate-limiting and connection read timeouts.
3. The reported station road metrics in `configs/config.py` were **hardcoded literature-derived approximations** rather than empirically measured quantities computed from downloaded OpenStreetMap geometries.
4. **Corrective Action Taken:** The codebase, feature store, and documentation have been strictly re-annotated to classify these values as **ASSUMED literature spatial proxies** with 4 degrees of freedom, completely removing any claim that they represent measured OpenStreetMap geometry extractions or real-time vehicular traffic volume.

---

## 2. Investigation of Overpass API Query Failure

To independently verify the Overpass API extraction pipeline, multiple Overpass endpoints and mirrors were tested using Python `requests` with standard HTTP headers:

```python
mirrors = [
    'https://overpass.kumi.systems/api/interpreter',
    'https://lz4.overpass-api.de/api/interpreter',
    'https://overpass-api.de/api/interpreter'
]
query = '[out:json][timeout:10];node(around:500, 17.54, 78.35);out count;'
```

### Empirical Execution Output
- `https://overpass.kumi.systems/api/interpreter`:
  `HTTPSConnectionPool(host='overpass.kumi.systems', port=443): Read timed out. (read timeout=10)`
- `https://lz4.overpass-api.de/api/interpreter`:
  `Status: 406 Not Acceptable (Server-side bot blocking / Header rejection)`
- `https://overpass-api.de/api/interpreter`:
  `Status: 406 Not Acceptable (Server-side bot blocking / Header rejection)`

### Root Cause Analysis
1. **Public Overpass Rate Limiting & Anti-Scraping Rules:** Overpass public instances actively throttle automated script requests that do not authenticate via registered academic API keys or that originate from datacenter IP blocks.
2. **Read Timeout:** Saturated public servers queue requests beyond 10–20 second timeouts, causing connection abandonment.
3. **Absence of Local Geometry Cache:** Because no offline `.osm.pbf` file (such as a Geofabrik Telangana regional extract) was bundled in the original project ZIP, the pipeline lacked an offline fallback for vector geometry calculations.

---

## 3. Tracing the Origin of Reported Road Metrics

Inspection of the codebase confirms that the metrics were defined directly in `implementation/configs/config.py` under `OSM_STATION_ROAD_PROXIES`:

```python
OSM_STATION_ROAD_PROXIES = {
    "Bollaram Industrial Area": OSMRoadMetrics(primary_length_1km=2.45, secondary_length_1km=4.10, ...),
    "Central University": OSMRoadMetrics(primary_length_1km=1.20, secondary_length_1km=2.80, ...),
    "ECIL Kapra": OSMRoadMetrics(primary_length_1km=4.80, secondary_length_1km=8.60, ...),
    "ICRISAT Patancheru": OSMRoadMetrics(primary_length_1km=3.60, secondary_length_1km=3.10, ...)
}
```

### Provenance Audit Findings
- **Were these values calculated from downloaded OSM geometries?** **NO.**
- **Source of Values:** The values were synthesized from manual GIS estimations, published transportation surveys for Greater Hyderabad, and highway corridor distance estimations (e.g., ICRISAT's direct proximity to NH-65, ECIL's dense commercial road grid, and Central University's campus buffer).
- **Audit Verdict:** Classifying these values as "OpenStreetMap extractions" was scientifically inaccurate. They represent **pre-calibrated spatial land-use intensity hypotheses**.

---

## 4. Geometric & Methodological Guardrails Enforced

To correct this provenance issue and ensure academic defensibility:

1. **Re-Classification as Assumed Spatial Proxies:**
   In `configs/config.py`, the dataclass `OSMRoadMetrics` now includes:
   ```python
   data_provenance: str = "Assumed literature proxy (unverified by raw OSM geometry download)"
   ```
2. **Automated Audit Unit Test Added:**
   Test `test_osm_data_provenance_explicit_assumed_label` asserts that all station proxies carry the explicit `"Assumed"` qualification and verifies that between-station variance is preserved.
3. **Documentation Updates:**
   `FEATURE_DICTIONARY.md` and `OSM_PROCESSING_REPORT.md` now explicitly declare:
   *"Static roadway metrics provide 4 unique spatial degrees of freedom across stations. They are assumed literature-derived spatial proxies and must NEVER be described as measured OSM geometries or empirical vehicular flow."*

---

## 5. Audit Item Verdict

| Verification Item | Specification | Observed State | Classification |
| :--- | :--- | :--- | :--- |
| **Local OSM Geometry Existence** | Raw `.pbf` or `.osm` files in repo | 0 files found | **FAIL** |
| **Overpass Query Reproducibility** | Live API extractions reproducible | HTTP 406 / Read Timeouts | **FAIL** |
| **Road Metric Provenance** | Empirically computed from vectors | Hardcoded in `config.py` | **FAIL (Re-classified as ASSUMED)** |
| **Scientific Labeling & Disclaimer** | Explicitly labeled as assumed proxy | Fully updated in code & docs | **PASS (Remediated)** |
| **Spatial Degrees of Freedom Guard** | Explicitly restricted to N=4 | Enforced across reports | **PASS** |

---

*Report prepared as part of Milestone 2 Independent Scientific Verification.*
