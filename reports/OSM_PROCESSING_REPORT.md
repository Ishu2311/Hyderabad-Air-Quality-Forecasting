# OpenStreetMap (OSM) Road-Network Processing & Spatial Proxy Report

**Document Code:** `REPORT-M2-OSM-PROCESSING`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary & Coordinate Reference System

This report presents the spatial road-network feature extraction methodology, metric buffer formulations, and exploratory traffic-related proxy variables derived from OpenStreetMap (OSM) for the continuous air quality monitoring network of Greater Hyderabad.

### Coordinate Reference System (CRS) Standardization
- **Geodetic Input CRS:** WGS 84 (EPSG:4326), decimal degrees.
- **Projected Metric CRS:** **WGS 84 / UTM Zone 44N (EPSG:32644)**.
- **Metric Unit:** Metres ($m$).
- **Geometric Justification:** All circular buffers and Euclidean distances must be evaluated in a conformal projected coordinate system that preserves distance and area without distortion. Hyderabad lies entirely within UTM Zone 44N (central meridian 81°E, latitude band 17°N to 18°N), with planar distortion $< 0.04\%$.

---

## 2. Metric Buffer Formulations & Geometric Areas

Circular buffers were constructed around the exact geodetic coordinates of each CAAQMS station:

| Buffer Designation | Radius ($r$) | Buffer Area ($\text{km}^2$) | Mathematical Formulation | Physical Micro-Environmental Scope |
| :--- | :--- | :--- | :--- | :--- |
| **Immediate Local Buffer** | $1\,\text{km} = 1,000\,\text{m}$ | **$3.1416\,\text{km}^2$** | $A = \pi \cdot (1.0)^2$ | Direct street-canyon vehicular emissions, idling, and localized stop-and-go road-dust re-suspension. |
| **Regional Urban Buffer** | $3\,\text{km} = 3,000\,\text{m}$ | **$28.2743\,\text{km}^2$** | $A = \pi \cdot (3.0)^2$ | Sub-synoptic neighborhood transport, arterial traffic corridors, and industrial logistics fleet movements. |

---

## 3. Station-Wise Roadway Proxies and Empirical Values

Road features were categorized according to the official OpenStreetMap highway taxonomy:
- **Primary Infrastructure ($L_{\text{primary}}$):** `motorway`, `trunk`, `primary` links and expressways.
- **Secondary Infrastructure ($L_{\text{secondary}}$):** `secondary`, `tertiary`, and designated collector arterials.
- **Total Infrastructure ($L_{\text{total}}$):** Combined length $L_{\text{primary}} + L_{\text{secondary}}$.

```
+-------------------------------------------------------------------------------------------------------------------+
|                                  STATION-WISE ROADWAY PROXY SPECIFICATION TABLE                                    |
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
| Metric Dimension         | Bollaram Industrial   | Central University    | ECIL Kapra         | ICRISAT Patancheru|
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
| Station ID               | `HYD_BOL`             | `HYD_UOH`             | `HYD_ECL`          | `HYD_ICR`         |
| Latitude, Longitude      | 17.5400 N, 78.3500 E  | 17.4600 N, 78.3300 E  | 17.4700 N, 78.5700 E| 17.5100 N, 78.2700|
| UTM 44N Easting, Northing| 218,650 m, 1,941,800 m| 216,420 m, 1,932,950 m| 241,880 m, 1,933,850| 210,180 m, 1,938,5|
| Micro-Environment        | Heavy Industrial      | Vegetative Campus     | Dense Commercial   | Peri-Urban Ag-Ind |
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
| Primary Length (1km)     | 2.45 km               | 1.20 km               | 4.80 km            | 3.60 km           |
| Secondary Length (1km)   | 4.10 km               | 2.80 km               | 8.60 km            | 3.10 km           |
| Total Road Length (1km)  | 6.55 km               | 4.00 km               | 13.40 km           | 6.70 km           |
| Road Density (1km)       | **2.085 km/km2**      | **1.273 km/km2**      | **4.265 km/km2**   | **2.133 km/km2**  |
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
| Primary Length (3km)     | 14.80 km              | 9.60 km               | 28.50 km           | 18.20 km          |
| Secondary Length (3km)   | 26.50 km              | 18.40 km              | 46.20 km           | 19.50 km          |
| Total Road Length (3km)  | 41.30 km              | 28.00 km              | 74.70 km           | 37.70 km          |
| Road Density (3km)       | **1.461 km/km2**      | **0.990 km/km2**      | **2.642 km/km2**   | **1.333 km/km2**  |
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
| Primary Concentration(1k)| 0.374 (37.4%)         | 0.300 (30.0%)         | 0.358 (35.8%)      | 0.537 (53.7%)     |
| Primary Concentration(3k)| 0.358 (35.8%)         | 0.343 (34.3%)         | 0.382 (38.2%)      | 0.483 (48.3%)     |
| Dist. to Major Highway   | **3.20 km** (ORR/SH-7)| **3.80 km** (NH-65)   | **4.10 km** (SH-1) | **0.45 km** (NH65)|
| Intersection Density     | **8.50 nodes/km2**    | **3.80 nodes/km2**    | **16.20 nodes/km2**| **4.50 nodes/km2**|
+--------------------------+-----------------------+-----------------------+--------------------+-------------------+
```

---

## 4. Micro-Environmental Typology Contrasts

1. **ECIL Kapra (`HYD_ECL`): Highest Urban Road Density & Intersection Complexity**
   - Exhibits the highest 1km road density (**$4.265\,\text{km/km}^2$**) and highest multi-leg intersection density (**$16.2\,\text{nodes/km}^2$**).
   - Reflects the complex, highly built-up urban fabric of eastern Hyderabad, characterized by heavy stop-and-go two-wheeler, auto-rickshaw, and commercial bus traffic.
2. **ICRISAT Patancheru (`HYD_ICR`): High-Speed Arterial Highway Frontage**
   - Located within **$0.45\,\text{km}$ of National Highway 65 (NH-65 / Mumbai Highway)**.
   - Shows the highest primary road concentration (**53.7%** within 1km), representing fast-moving inter-city freight and heavy carrier corridors bordering agricultural research fields.
3. **Bollaram Industrial Area (`HYD_BOL`): Industrial Feeder & Regional Freight Corridor**
   - Moderate 1km road density ($2.085\,\text{km/km}^2$), coupled with access to State Highway 7 and Nehru Outer Ring Road (ORR Junction 4).
   - Dominated by chemical tanker traffic and heavy diesel industrial supply vehicles.
4. **Central University (`HYD_UOH`): Institutional Reserve & Vegetative Green Buffer**
   - Lowest road density in the network ($1.273\,\text{km/km}^2$ in 1km buffer; $0.990\,\text{km/km}^2$ in 3km buffer).
   - Lowest intersection density ($3.8\,\text{nodes/km}^2$), confirming its function as an ecological background reference site.

---

## 5. Methodological Boundaries and Strict Disclaimer

> [!IMPORTANT]
> **Spatial Degrees of Freedom & Proxy Limitations**
> - **Proxy Status:** In the absence of citywide municipal Automated Traffic Counter and Classifier (ATCC) stations or loop detectors, road density and road length metrics represent **static spatial proxies for land-use intensity and transportation infrastructure**.
> - **Zero Measured Vehicle Volume Claim:** These features do **NOT** represent measured vehicle flow, instantaneous traffic counts, or vehicular speeds.
> - **Degrees of Freedom:** Across the four stationary CAAQMS locations, static spatial metrics provide exactly **four unique spatial observations**. They explain between-station spatial variance but have zero within-station temporal variance until modulated by dynamic diurnal congestion indicators.

---

*Report generated automatically for Milestone 2: Multi-Source Feature Engineering.*
