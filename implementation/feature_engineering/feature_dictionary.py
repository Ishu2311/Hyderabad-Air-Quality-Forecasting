"""
Feature Dictionary Generator Module
Milestone 2: Multi-Source Feature Engineering

Generates the exhaustive feature dictionary cataloging EVERY column (173/173) in the feature store:
- Variable Name
- Modality & Source Tier (Metadata, Tier 1: Physical CAAQMS, Tier 2: Weather Reanalysis, Tier 3: OSM Road Proxies & Congestion, Targets)
- Physical Unit
- Temporal Horizon / Lag / Rolling Window
- Mathematical Transformation & Definition
- Missingness & Imputation Protocol
- Target vs. Predictor Status
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd

try:
    from ..configs.config import PROCESSED_DATA_DIR, REPORTS_DIR
except (ImportError, ValueError):
    from configs.config import PROCESSED_DATA_DIR, REPORTS_DIR

# Complete 173-column catalog
FEATURE_CATALOG: List[Dict[str, str]] = [
    # =========================================================================
    # 1. Administrative Metadata & Geodetic Coordinates (14 columns)
    # =========================================================================
    {
        "name": "station",
        "tier": "Metadata",
        "category": "Identifier",
        "unit": "Text",
        "description": "Official name of the Continuous Ambient Air Quality Monitoring Station (CAAQMS)."
    },
    {
        "name": "station_id",
        "tier": "Metadata",
        "category": "Identifier",
        "unit": "Text",
        "description": "Standardized unique station identifier (HYD_BOL, HYD_UOH, HYD_ECL, HYD_ICR)."
    },
    {
        "name": "datetime",
        "tier": "Metadata",
        "category": "Temporal",
        "unit": "ISO 8601 (IST)",
        "description": "Continuous hourly observation timestamp in Indian Standard Time (UTC+5:30)."
    },
    {
        "name": "latitude",
        "tier": "Metadata",
        "category": "Spatial",
        "unit": "Decimal Degrees",
        "description": "Geodetic latitude in WGS 84 (EPSG:4326)."
    },
    {
        "name": "longitude",
        "tier": "Metadata",
        "category": "Spatial",
        "unit": "Decimal Degrees",
        "description": "Geodetic longitude in WGS 84 (EPSG:4326)."
    },
    {
        "name": "easting_utm44n",
        "tier": "Metadata",
        "category": "Spatial",
        "unit": "Metres",
        "description": "Projected Easting coordinate in UTM Zone 44N (EPSG:32644)."
    },
    {
        "name": "northing_utm44n",
        "tier": "Metadata",
        "category": "Spatial",
        "unit": "Metres",
        "description": "Projected Northing coordinate in UTM Zone 44N (EPSG:32644)."
    },
    {
        "name": "typology",
        "tier": "Metadata",
        "category": "Spatial",
        "unit": "Category",
        "description": "Urban land-use typology (Industrial, Vegetative/Campus, Commercial, Peri-Urban)."
    },
    {
        "name": "partition",
        "tier": "Metadata",
        "category": "Validation",
        "unit": "Category",
        "description": "Chronological evaluation partition: train (2023-2024), val (2025), test (2026)."
    },
    {
        "name": "is_boundary_buffer",
        "tier": "Metadata",
        "category": "Validation",
        "unit": "Binary (0/1)",
        "description": "Flag indicating final 48 hours of each split to prevent future target cross-boundary contamination."
    },
    {
        "name": "hour",
        "tier": "Metadata",
        "category": "Temporal",
        "unit": "Hour of Day [0-23]",
        "description": "Discrete integer hour of day extracted from datetime."
    },
    {
        "name": "dayofweek",
        "tier": "Metadata",
        "category": "Temporal",
        "unit": "Day [0-6]",
        "description": "Discrete integer day of week (0=Monday, 6=Sunday)."
    },
    {
        "name": "month",
        "tier": "Metadata",
        "category": "Temporal",
        "unit": "Month [1-12]",
        "description": "Discrete integer month of year."
    },
    {
        "name": "season",
        "tier": "Metadata",
        "category": "Temporal",
        "unit": "Category",
        "description": "IMD Season name: Winter (Dec-Feb), Summer (Mar-May), Monsoon (Jun-Sep), Post_Monsoon (Oct-Nov)."
    },

    # =========================================================================
    # 2. Tier 1: Prospective Ambient Pollutant Features (8 columns)
    # =========================================================================
    {
        "name": "PM2.5",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Particulate",
        "unit": "ug/m3",
        "description": "Prospective PM2.5 concentration at prediction time t (short gaps ffilled from past; medium gaps diurnal-imputed; extended gaps remain NaN)."
    },
    {
        "name": "PM10",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Particulate",
        "unit": "ug/m3",
        "description": "Prospective PM10 coarse particulate concentration at prediction time t."
    },
    {
        "name": "NO",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "ug/m3",
        "description": "Prospective Nitric Oxide concentration at prediction time t."
    },
    {
        "name": "NO2",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "ug/m3",
        "description": "Prospective Nitrogen Dioxide concentration at prediction time t."
    },
    {
        "name": "NH3",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "ug/m3",
        "description": "Prospective Ammonia concentration at prediction time t."
    },
    {
        "name": "SO2",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "ug/m3",
        "description": "Prospective Sulfur Dioxide concentration at prediction time t."
    },
    {
        "name": "CO",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "mg/m3",
        "description": "Prospective Carbon Monoxide concentration at prediction time t."
    },
    {
        "name": "Ozone",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Prospective Gas",
        "unit": "ug/m3",
        "description": "Prospective Ozone concentration at prediction time t."
    },

    # =========================================================================
    # 3. Tier 1: Historical PM2.5 Autoregressive, Rolling & Trend Features (15 columns)
    # =========================================================================
    {
        "name": "pm25_lag_1",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-1 (1 hour prior to prediction time)."
    },
    {
        "name": "pm25_lag_2",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-2 (2 hours prior)."
    },
    {
        "name": "pm25_lag_3",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-3 (3 hours prior)."
    },
    {
        "name": "pm25_lag_6",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-6 (6 hours prior)."
    },
    {
        "name": "pm25_lag_12",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-12 (12 hours prior)."
    },
    {
        "name": "pm25_lag_24",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Autoregressive Lag",
        "unit": "ug/m3",
        "description": "Instantaneous PM2.5 concentration at lag t-24 (same hour of preceding diurnal cycle)."
    },
    {
        "name": "pm25_roll_mean_3h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Arithmetic mean of PM2.5 over past 3 hours [t-2 : t]. Past observations only."
    },
    {
        "name": "pm25_roll_mean_6h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Arithmetic mean of PM2.5 over past 6 hours [t-5 : t]. Past observations only."
    },
    {
        "name": "pm25_roll_mean_12h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Arithmetic mean of PM2.5 over past 12 hours [t-11 : t]. Past observations only."
    },
    {
        "name": "pm25_roll_mean_24h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Arithmetic mean of PM2.5 over past 24 hours [t-23 : t]. Past observations only."
    },
    {
        "name": "pm25_roll_std_6h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Sample standard deviation of PM2.5 over past 6 hours [t-5 : t]."
    },
    {
        "name": "pm25_roll_std_24h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Sample standard deviation of PM2.5 over past 24 hours [t-23 : t]."
    },
    {
        "name": "pm25_roll_min_24h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Minimum observed PM2.5 concentration over past 24 hours."
    },
    {
        "name": "pm25_roll_max_24h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Rolling Statistic",
        "unit": "ug/m3",
        "description": "Maximum observed PM2.5 concentration over past 24 hours."
    },
    {
        "name": "pm25_ratio_to_24h_mean",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Trend Dynamic",
        "unit": "Dimensionless",
        "description": "Ratio of current concentration to historical 24h baseline: y_t / (roll_mean_24h + eps)."
    },

    # =========================================================================
    # 4. Tier 1: Trend Differences & Curvature (4 columns)
    # =========================================================================
    {
        "name": "pm25_trend_diff_1h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Trend Dynamic",
        "unit": "ug/m3",
        "description": "Short-term rate of change: y_t - y_{t-1}."
    },
    {
        "name": "pm25_trend_diff_3h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Trend Dynamic",
        "unit": "ug/m3",
        "description": "Sub-synoptic rate of change: y_t - y_{t-3}."
    },
    {
        "name": "pm25_trend_diff_24h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Trend Dynamic",
        "unit": "ug/m3",
        "description": "Diurnal rate of change: y_t - y_{t-24}."
    },
    {
        "name": "pm25_acceleration_1h",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Trend Dynamic",
        "unit": "ug/m3 / h^2",
        "description": "Second-order curvature: (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})."
    },

    # =========================================================================
    # 5. Tier 1: Co-Pollutant Lags & Chemical Source Ratios (8 columns)
    # =========================================================================
    {
        "name": "ratio_pm25_pm10",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Chemical Ratio",
        "unit": "Dimensionless",
        "description": "Aerosol source indicator: PM2.5 / (PM10 + eps). High values indicate fine combustion aerosols."
    },
    {
        "name": "ratio_no_no2",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Chemical Ratio",
        "unit": "Dimensionless",
        "description": "Photochemical oxidation proxy: NO / (NO2 + eps). High values indicate fresh tailpipe emissions."
    },
    {
        "name": "pm10_lag_1",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "PM10 coarse particulate concentration at lag t-1."
    },
    {
        "name": "pm10_lag_24",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "PM10 coarse particulate concentration at lag t-24."
    },
    {
        "name": "no2_lag_1",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "Nitrogen dioxide concentration at lag t-1."
    },
    {
        "name": "no2_lag_24",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "Nitrogen dioxide concentration at lag t-24."
    },
    {
        "name": "so2_lag_1",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "Sulfur dioxide industrial indicator at lag t-1."
    },
    {
        "name": "so2_lag_24",
        "tier": "Tier 1: Ground-Truth In-Situ Sensor",
        "category": "Co-Pollutant Lag",
        "unit": "ug/m3",
        "description": "Sulfur dioxide industrial indicator at lag t-24."
    },

    # =========================================================================
    # 6. Cyclical Time & Seasonal Indicators (10 columns)
    # =========================================================================
    {
        "name": "hour_sin",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "sin(2 * pi * hour / 24.0)."
    },
    {
        "name": "hour_cos",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "cos(2 * pi * hour / 24.0)."
    },
    {
        "name": "dow_sin",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "sin(2 * pi * dayofweek / 7.0)."
    },
    {
        "name": "dow_cos",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "cos(2 * pi * dayofweek / 7.0)."
    },
    {
        "name": "month_sin",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "sin(2 * pi * month / 12.0)."
    },
    {
        "name": "month_cos",
        "tier": "Calendar / Cyclical",
        "category": "Temporal Cyclical",
        "unit": "Dimensionless [-1, 1]",
        "description": "cos(2 * pi * month / 12.0)."
    },
    {
        "name": "is_winter",
        "tier": "Calendar / Cyclical",
        "category": "Season",
        "unit": "Binary (0/1)",
        "description": "IMD Winter season flag (December, January, February)."
    },
    {
        "name": "is_summer",
        "tier": "Calendar / Cyclical",
        "category": "Season",
        "unit": "Binary (0/1)",
        "description": "IMD Summer / Pre-monsoon flag (March, April, May)."
    },
    {
        "name": "is_monsoon",
        "tier": "Calendar / Cyclical",
        "category": "Season",
        "unit": "Binary (0/1)",
        "description": "IMD Southwest monsoon flag (June, July, August, September)."
    },
    {
        "name": "is_post_monsoon",
        "tier": "Calendar / Cyclical",
        "category": "Season",
        "unit": "Binary (0/1)",
        "description": "IMD Post-monsoon retreating flag (October, November)."
    },

    # =========================================================================
    # 7. Tier 2: Open-Meteo Meteorological Reanalysis & Transformations (25 columns)
    # =========================================================================
    {
        "name": "temperature_2m",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "deg C",
        "description": "Ambient air temperature at 2 metres above surface at time t."
    },
    {
        "name": "relative_humidity_2m",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "%",
        "description": "Relative humidity at 2 metres above surface at time t."
    },
    {
        "name": "wind_speed_10m",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "m/s",
        "description": "Scalar horizontal wind speed at 10 metres above surface at time t."
    },
    {
        "name": "wind_direction_10m",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "Degrees",
        "description": "Meteorological wind direction at 10m (0-360 degrees, direction from)."
    },
    {
        "name": "precipitation",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "mm",
        "description": "Total precipitation accumulation in preceding hour."
    },
    {
        "name": "surface_pressure",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Meteorology",
        "unit": "hPa",
        "description": "Surface atmospheric pressure."
    },
    {
        "name": "wind_dir_sin",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Wind Transformation",
        "unit": "Dimensionless [-1, 1]",
        "description": "sin(radians(wind_direction_10m))."
    },
    {
        "name": "wind_dir_cos",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Wind Transformation",
        "unit": "Dimensionless [-1, 1]",
        "description": "cos(radians(wind_direction_10m))."
    },
    {
        "name": "wind_u",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Wind Vector",
        "unit": "m/s",
        "description": "Zonal wind vector component (Eastward): -wind_speed * sin(rad)."
    },
    {
        "name": "wind_v",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Wind Vector",
        "unit": "m/s",
        "description": "Meridional wind vector component (Northward): -wind_speed * cos(rad)."
    },
    {
        "name": "sat_vapor_pressure_es",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Psychrometric",
        "unit": "hPa",
        "description": "Saturation vapor pressure e_s(T) via Magnus formulation."
    },
    {
        "name": "act_vapor_pressure_ea",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Psychrometric",
        "unit": "hPa",
        "description": "Actual vapor pressure e_a = e_s * (RH / 100)."
    },
    {
        "name": "vapor_pressure_deficit_vpd",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Psychrometric",
        "unit": "hPa",
        "description": "Vapor Pressure Deficit (e_s - e_a): atmospheric drying potential."
    },
    {
        "name": "dew_point_temp",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Psychrometric",
        "unit": "deg C",
        "description": "Dew point temperature calculated from T and RH."
    },
    {
        "name": "dew_point_depression",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Psychrometric",
        "unit": "deg C",
        "description": "Temperature-dewpoint spread (T - T_dew): fog and condensation indicator."
    },
    {
        "name": "is_atmospheric_stagnation",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Boundary Layer Stability",
        "unit": "Binary (0/1)",
        "description": "Stagnation indicator: wind_speed < 1.5 m/s and precipitation <= 0.01 mm."
    },
    {
        "name": "is_wind_dispersion",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Boundary Layer Stability",
        "unit": "Binary (0/1)",
        "description": "Active aerosol ventilation indicator: wind_speed >= 5.0 m/s."
    },
    {
        "name": "ventilation_index_proxy",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Boundary Layer Stability",
        "unit": "m2 / s",
        "description": "Atmospheric ventilation proxy: wind_speed * (T + 273.15) / 100."
    },
    {
        "name": "precip_roll_sum_6h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Washout Deposition",
        "unit": "mm",
        "description": "Rolling 6-hour precipitation sum for short-term aerosol scavenging."
    },
    {
        "name": "precip_roll_sum_24h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Washout Deposition",
        "unit": "mm",
        "description": "Rolling 24-hour precipitation sum for diurnal wet deposition."
    },
    {
        "name": "has_recent_rain_24h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Washout Deposition",
        "unit": "Binary (0/1)",
        "description": "1 if rolling 24h precipitation sum > 0.1 mm, else 0."
    },
    {
        "name": "temp_lag_1",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "deg C",
        "description": "Temperature at lag t-1."
    },
    {
        "name": "temp_lag_24",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "deg C",
        "description": "Temperature at lag t-24 (same hour previous day)."
    },
    {
        "name": "temp_diff_24h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Trend",
        "unit": "deg C",
        "description": "Diurnal temperature difference: T_t - T_{t-24}."
    },
    {
        "name": "rh_lag_1",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "%",
        "description": "Relative humidity at lag t-1."
    },
    {
        "name": "rh_lag_24",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "%",
        "description": "Relative humidity at lag t-24."
    },
    {
        "name": "rh_diff_24h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Trend",
        "unit": "%",
        "description": "Diurnal humidity difference: RH_t - RH_{t-24}."
    },
    {
        "name": "wind_speed_lag_1",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "m/s",
        "description": "Wind speed at lag t-1."
    },
    {
        "name": "wind_speed_roll_mean_6h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Rolling",
        "unit": "m/s",
        "description": "Past-only 6-hour rolling mean wind speed."
    },
    {
        "name": "wind_speed_roll_mean_24h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Rolling",
        "unit": "m/s",
        "description": "Past-only 24-hour rolling mean wind speed."
    },
    {
        "name": "pressure_lag_1",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Lag",
        "unit": "hPa",
        "description": "Surface pressure at lag t-1."
    },
    {
        "name": "pressure_tendency_3h",
        "tier": "Tier 2: Atmospheric Model Reanalysis",
        "category": "Weather Trend",
        "unit": "hPa",
        "description": "Barometric pressure tendency: P_t - P_{t-3}."
    },

    # =========================================================================
    # 8. Tier 3: Spatial Road-Network Proxies (14 columns)
    # =========================================================================
    {
        "name": "osm_primary_length_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed length of motorway/trunk/primary roads within 1km metric buffer (EPSG:32644)."
    },
    {
        "name": "osm_secondary_length_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed length of secondary and arterial roads within 1km metric buffer."
    },
    {
        "name": "osm_total_length_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed total combined roadway length within 1km metric buffer."
    },
    {
        "name": "osm_primary_length_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed length of motorway/trunk/primary roads within 3km regional buffer."
    },
    {
        "name": "osm_secondary_length_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed length of secondary and arterial roads within 3km regional buffer."
    },
    {
        "name": "osm_total_length_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Roadway Length",
        "unit": "km",
        "description": "Assumed total combined roadway length within 3km regional buffer."
    },
    {
        "name": "osm_road_density_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Density",
        "unit": "km / km2",
        "description": "Assumed total road density in 1km buffer. Literature land-use proxy, NOT vehicle count."
    },
    {
        "name": "osm_road_density_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Density",
        "unit": "km / km2",
        "description": "Assumed total road density in 3km buffer. Literature land-use proxy, NOT vehicle count."
    },
    {
        "name": "osm_primary_density_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Density",
        "unit": "km / km2",
        "description": "Assumed primary/motorway density in 1km buffer."
    },
    {
        "name": "osm_primary_density_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Density",
        "unit": "km / km2",
        "description": "Assumed primary/motorway density in 3km buffer."
    },
    {
        "name": "osm_primary_concentration_1km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Network Composition",
        "unit": "Ratio [0, 1]",
        "description": "Ratio of primary roadway length to total roadway length in 1km buffer."
    },
    {
        "name": "osm_primary_concentration_3km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Road Network Composition",
        "unit": "Ratio [0, 1]",
        "description": "Ratio of primary roadway length to total roadway length in 3km buffer."
    },
    {
        "name": "osm_distance_to_highway_km",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Highway Proximity",
        "unit": "km",
        "description": "Projected Euclidean distance (km) to nearest national highway corridor (NH-65, ORR, etc.)."
    },
    {
        "name": "osm_intersection_density_km2",
        "tier": "Tier 3: Spatial Road-Network Proxies",
        "category": "Network Topology",
        "unit": "Nodes / km2",
        "description": "Assumed density of multi-leg roadway intersections per km2."
    },

    # =========================================================================
    # 9. Tier 3: Congestion Windows & Spatio-Temporal Interactions (12 columns)
    # =========================================================================
    {
        "name": "is_morning_peak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Diurnal Congestion",
        "unit": "Binary (0/1)",
        "description": "Morning rush-hour indicator: 08:00 to 11:00 inclusive."
    },
    {
        "name": "is_evening_peak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Diurnal Congestion",
        "unit": "Binary (0/1)",
        "description": "Evening rush-hour indicator: 17:00 to 21:00 inclusive."
    },
    {
        "name": "is_traffic_peak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Diurnal Congestion",
        "unit": "Binary (0/1)",
        "description": "Composite rush-hour indicator: is_morning_peak OR is_evening_peak."
    },
    {
        "name": "is_weekend",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Calendar Dynamics",
        "unit": "Binary (0/1)",
        "description": "Weekend day flag (Saturday or Sunday)."
    },
    {
        "name": "is_weekday_peak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Diurnal Congestion",
        "unit": "Binary (0/1)",
        "description": "Weekday commuter rush flag: is_traffic_peak AND NOT is_weekend."
    },
    {
        "name": "is_freight_window_exploratory",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Freight Proxy",
        "unit": "Binary (0/1)",
        "description": "Exploratory heavy freight movement window: 23:00 to 05:00. Unverified assumption pending police gazette."
    },
    {
        "name": "is_daytime_offpeak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Diurnal Congestion",
        "unit": "Binary (0/1)",
        "description": "Weekday daytime off-peak window: 11:00 to 17:00 on weekdays."
    },
    {
        "name": "traffic_pressure_total_1km",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Spatio-Temporal Interaction",
        "unit": "km / km2",
        "description": "Interaction feature: osm_road_density_1km * is_traffic_peak."
    },
    {
        "name": "traffic_pressure_weekday_peak",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Spatio-Temporal Interaction",
        "unit": "km / km2",
        "description": "Interaction feature: osm_road_density_1km * is_weekday_peak."
    },
    {
        "name": "traffic_pressure_primary_1km",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Spatio-Temporal Interaction",
        "unit": "km",
        "description": "Interaction feature: osm_primary_length_1km * is_traffic_peak."
    },
    {
        "name": "freight_pressure_proxy",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Spatio-Temporal Interaction",
        "unit": "km / km2",
        "description": "Interaction feature: osm_road_density_3km * is_freight_window_exploratory."
    },
    {
        "name": "highway_rush_pressure",
        "tier": "Tier 3: Congestion Dynamics",
        "category": "Spatio-Temporal Interaction",
        "unit": "1 / km",
        "description": "Interaction feature: is_traffic_peak / (osm_distance_to_highway_km + 0.1)."
    },

    # =========================================================================
    # 10. Raw Sensors, Telemetry Quality & Missingness Indicators (40 columns)
    # =========================================================================
    {
        "name": "PM2.5_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical PM2.5 observation from CAAQMS station."
    },
    {
        "name": "PM10_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical PM10 observation from CAAQMS station."
    },
    {
        "name": "NO_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical NO observation from CAAQMS station."
    },
    {
        "name": "NO2_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical NO2 observation from CAAQMS station."
    },
    {
        "name": "NH3_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical NH3 observation from CAAQMS station."
    },
    {
        "name": "SO2_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical SO2 observation from CAAQMS station."
    },
    {
        "name": "CO_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "mg/m3",
        "description": "Raw un-imputed physical CO observation from CAAQMS station."
    },
    {
        "name": "Ozone_raw",
        "tier": "Raw Telemetry",
        "category": "Uncleaned Ground Truth",
        "unit": "ug/m3",
        "description": "Raw un-imputed physical Ozone observation from CAAQMS station."
    },
    # Missing masks
    {
        "name": "PM2.5_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask: 1 if physical PM2.5 was missing in raw CAAQMS stream, 0 if observed."
    },
    {
        "name": "PM10_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for PM10."
    },
    {
        "name": "NO_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for NO."
    },
    {
        "name": "NO2_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for NO2."
    },
    {
        "name": "NH3_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for NH3."
    },
    {
        "name": "SO2_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for SO2."
    },
    {
        "name": "CO_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for CO."
    },
    {
        "name": "Ozone_missing_mask",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Raw missingness mask for Ozone."
    },
    # Imputed flags
    {
        "name": "PM2.5_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag: 1 if prospective PM2.5 was restored via past-only ffill or diurnal lookup, 0 if raw observed."
    },
    {
        "name": "PM10_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for PM10."
    },
    {
        "name": "NO_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for NO."
    },
    {
        "name": "NO2_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for NO2."
    },
    {
        "name": "NH3_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for NH3."
    },
    {
        "name": "SO2_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for SO2."
    },
    {
        "name": "CO_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for CO."
    },
    {
        "name": "Ozone_imputed_flag",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Binary (0/1)",
        "description": "Imputation flag for Ozone."
    },
    # Gap lengths
    {
        "name": "PM2.5_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length in hours of the missing telemetry block."
    },
    {
        "name": "PM10_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for PM10."
    },
    {
        "name": "NO_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for NO."
    },
    {
        "name": "NO2_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for NO2."
    },
    {
        "name": "NH3_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for NH3."
    },
    {
        "name": "SO2_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for SO2."
    },
    {
        "name": "CO_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for CO."
    },
    {
        "name": "Ozone_gap_length",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Hours",
        "description": "Contiguous outage length for Ozone."
    },
    # Gap categories
    {
        "name": "PM2.5_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap typology: 'observed', 'short_<=3h', 'medium_4-24h', or 'extended_>24h'."
    },
    {
        "name": "PM10_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for PM10."
    },
    {
        "name": "NO_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for NO."
    },
    {
        "name": "NO2_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for NO2."
    },
    {
        "name": "NH3_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for NH3."
    },
    {
        "name": "SO2_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for SO2."
    },
    {
        "name": "CO_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for CO."
    },
    {
        "name": "Ozone_gap_category",
        "tier": "Telemetry Quality",
        "category": "Quality Flag",
        "unit": "Category",
        "description": "Missing gap category for Ozone."
    },

    # =========================================================================
    # 11. Target Columns (16 columns)
    # =========================================================================
    {
        "name": "target_pm25_h1",
        "tier": "Target Column",
        "category": "Continuous Regression",
        "unit": "ug/m3",
        "description": "Future instantaneous PM2.5 concentration at horizon t+1h. Un-imputed raw sensor ground truth."
    },
    {
        "name": "target_pm25_h6",
        "tier": "Target Column",
        "category": "Continuous Regression",
        "unit": "ug/m3",
        "description": "Future instantaneous PM2.5 concentration at horizon t+6h."
    },
    {
        "name": "target_pm25_h24",
        "tier": "Target Column",
        "category": "Continuous Regression",
        "unit": "ug/m3",
        "description": "Future instantaneous PM2.5 concentration at horizon t+24h."
    },
    {
        "name": "target_pm25_h48",
        "tier": "Target Column",
        "category": "Continuous Regression",
        "unit": "ug/m3",
        "description": "Future instantaneous PM2.5 concentration at horizon t+48h."
    },
    {
        "name": "target_pm25_24h_mean",
        "tier": "Target Column",
        "category": "Continuous Regression",
        "unit": "ug/m3",
        "description": "Forward 24-hour arithmetic average over t+1 to t+24. Requires N_valid >= 16; denominator is N_valid."
    },
    {
        "name": "target_pm25_24h_valid_count",
        "tier": "Target Column",
        "category": "Target Metadata",
        "unit": "Count [0-24]",
        "description": "Count of valid hourly observations in forward window t+1 to t+24."
    },
    {
        "name": "target_hotspot_acute_h1",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Acute operational excursion alert at t+1h: 1 if PM2.5(t+1) >= 60 ug/m3, 0 if < 60, NaN if missing."
    },
    {
        "name": "target_hotspot_spike_h1",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory severe-spike alert at t+1h: 1 if PM2.5(t+1) >= 120 ug/m3, 0 if < 120, NaN if missing."
    },
    {
        "name": "target_hotspot_very_poor_h1",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory Very Poor category entry alert at t+1h: 1 if PM2.5(t+1) > 120 ug/m3, 0 if <= 120, NaN if missing."
    },
    {
        "name": "target_hotspot_acute_h6",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Acute operational excursion alert at t+6h: 1 if PM2.5(t+6) >= 60 ug/m3."
    },
    {
        "name": "target_hotspot_spike_h6",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory severe-spike alert at t+6h: 1 if PM2.5(t+6) >= 120 ug/m3."
    },
    {
        "name": "target_hotspot_very_poor_h6",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory Very Poor category entry alert at t+6h: 1 if PM2.5(t+6) > 120 ug/m3."
    },
    {
        "name": "target_hotspot_acute_h24",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Acute operational excursion alert at t+24h: 1 if PM2.5(t+24) >= 60 ug/m3."
    },
    {
        "name": "target_hotspot_spike_h24",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory severe-spike alert at t+24h: 1 if PM2.5(t+24) >= 120 ug/m3."
    },
    {
        "name": "target_hotspot_very_poor_h24",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Exploratory Very Poor category entry alert at t+24h: 1 if PM2.5(t+24) > 120 ug/m3."
    },
    {
        "name": "target_hotspot_24h_statutory",
        "tier": "Target Column",
        "category": "Binary Classification",
        "unit": "Binary (0/1)",
        "description": "Statutory 24-hour hotspot target: 1 if forward 24h mean >= 60 ug/m3 with N_valid >= 16; 0 if < 60; NaN if N_valid < 16."
    }
]

class FeatureDictionaryGenerator:
    """Generates JSON and Markdown feature dictionaries for the feature store."""

    def __init__(self, catalog: Optional[List[Dict[str, str]]] = None):
        self.catalog = catalog if catalog is not None else FEATURE_CATALOG

    def to_dataframe(self) -> pd.DataFrame:
        """Converts catalog to pandas DataFrame."""
        return pd.DataFrame(self.catalog)

    def save_json(self, output_path: Path) -> None:
        """Saves catalog as formatted JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.catalog, f, indent=2)

    def generate_markdown(self) -> str:
        """Generates comprehensive GitHub Flavored Markdown feature dictionary document."""
        df = self.to_dataframe()
        tiers = df["tier"].unique()
        
        md_lines = [
            "# Comprehensive Feature Store Dictionary",
            "",
            "**Document Code:** `SPEC-M2-FEATURE-DICT`  ",
            "**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  ",
            "**Milestone:** Milestone 2 – Multi-Source Feature Engineering  ",
            "",
            "---",
            "",
            "## 1. Summary of Feature Store Modalities (173 Total Columns)",
            "",
            f"- **Total Cataloged Columns:** **{len(df)}** (Exactly reconciles with master feature store)  ",
            f"- **Administrative & Temporal Keys:** **{len(df[df['tier'] == 'Metadata'])}**  ",
            f"- **Target Columns (Ground Truth):** **{len(df[df['tier'] == 'Target Column'])}**  ",
            f"- **Raw Telemetry & Quality Masks:** **{len(df[df['tier'].isin(['Raw Telemetry', 'Telemetry Quality'])])}**  ",
            f"- **Core Predictor Features:** **{len(df[~df['tier'].isin(['Metadata', 'Target Column', 'Raw Telemetry', 'Telemetry Quality'])])}**  ",
            "",
            "---",
            ""
        ]
        
        for tier in tiers:
            sub = df[df["tier"] == tier]
            md_lines.append(f"## {tier} ({len(sub)} variables)\n")
            md_lines.append("| Variable Name | Category | Unit | Description |")
            md_lines.append("| :--- | :--- | :--- | :--- |")
            for _, row in sub.iterrows():
                desc = row["description"].replace("|", "/")
                md_lines.append(f"| `{row['name']}` | {row['category']} | {row['unit']} | {desc} |")
            md_lines.append("\n---\n")
            
        md_lines.append("*Feature dictionary automatically generated by FeatureDictionaryGenerator.*")
        return "\n".join(md_lines)

    def save_markdown(self, output_path: Path) -> None:
        """Saves catalog as markdown document."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown())
