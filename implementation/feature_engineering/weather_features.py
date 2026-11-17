"""
Meteorological Feature Engineering Module
Milestone 2: Multi-Source Feature Engineering

Integrates Open-Meteo atmospheric variables with mathematical consistency:
- 2m Temperature (deg C)
- 2m Relative Humidity (%)
- 10m Wind Speed (m/s)
- 10m Wind Direction (degrees, 0-360)
- Surface Pressure (hPa)
- Precipitation (mm)

Transformations & Formulations:
- Wind directional trigonometry: sin(theta), cos(theta)
- Atmospheric wind vector components:
    u = -wind_speed * sin(theta_rad)  (Zonal / Eastward component)
    v = -wind_speed * cos(theta_rad)  (Meridional / Northward component)
    Satisfies identity: u^2 + v^2 = wind_speed^2
- Psychrometric variables:
    Saturation vapor pressure e_s(T), actual vapor pressure e_a,
    Vapor Pressure Deficit (VPD = e_s - e_a), and dew point temperature T_d
- Boundary-layer dynamics:
    Atmospheric stagnation index (wind < 1.5 m/s, rain == 0)
    Atmospheric ventilation index proxy
    Precipitation washout accumulation (6h, 24h rolling sums)
- Historical weather lags:
    1h and 24h lags for temperature, humidity, and wind speed

POINT-IN-TIME AVAILABILITY PROTOCOL:
All features are computed strictly from observations <= t. Future atmospheric conditions
at t+h are never accessed as input predictors, preventing weather foresight leakage.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class WeatherFeatureExtractor:
    """Extracts meteorological transformations, vector dynamics, and moisture features."""

    def __init__(self):
        pass

    def extract_wind_vector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mathematically consistent wind directional transformations and vector components.
        
        Meteorological wind convention:
        - Wind direction theta is the direction the wind blows FROM (0=North, 90=East, 180=South, 270=West).
        - Zonal wind u: Positive Eastward. When wind blows FROM East (90 deg), u is negative: u = -U * sin(theta).
        - Meridional wind v: Positive Northward. When wind blows FROM North (0 deg), v is negative: v = -U * cos(theta).
        - Identity: sqrt(u^2 + v^2) equals scalar wind speed U.
        """
        df = df.copy()
        if "wind_direction_10m" not in df.columns or "wind_speed_10m" not in df.columns:
            return df
            
        direction_deg = df["wind_direction_10m"]
        speed = df["wind_speed_10m"]
        
        # Radians
        rad = np.radians(direction_deg)
        
        # Directional components on unit circle
        df["wind_dir_sin"] = np.sin(rad)
        df["wind_dir_cos"] = np.cos(rad)
        
        # Orthogonal physical velocity vectors (m/s)
        df["wind_u"] = -speed * np.sin(rad)
        df["wind_v"] = -speed * np.cos(rad)
        
        return df

    def extract_psychrometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts vapor pressure, moisture deficit, and dew point approximations.
        - Saturation vapor pressure e_s(T) via standard Magnus-Tetens formulation.
        - Vapor Pressure Deficit (VPD = e_s - e_a) in hPa.
        - Dew point temperature (T_dew) in deg C.
        """
        df = df.copy()
        if "temperature_2m" not in df.columns or "relative_humidity_2m" not in df.columns:
            return df
            
        t = df["temperature_2m"]
        rh = df["relative_humidity_2m"].clip(lower=1.0, upper=100.0)
        
        # Magnus formula constants (for water, valid -45 C to +60 C)
        a = 17.67
        b = 243.5  # deg C
        
        # Saturation vapor pressure e_s (hPa)
        e_s = 6.112 * np.exp((a * t) / (b + t))
        df["sat_vapor_pressure_es"] = e_s
        
        # Actual vapor pressure e_a (hPa)
        e_a = e_s * (rh / 100.0)
        df["act_vapor_pressure_ea"] = e_a
        
        # Vapor Pressure Deficit (VPD, hPa): measure of atmospheric drying power
        df["vapor_pressure_deficit_vpd"] = np.maximum(0.0, e_s - e_a)
        
        # Dew point temperature (T_dew, deg C)
        alpha = ((a * t) / (b + t)) + np.log(rh / 100.0)
        t_dew = (b * alpha) / (a - alpha)
        df["dew_point_temp"] = t_dew
        
        # Temperature-Dewpoint Spread (depression): indicator of near-surface condensation/fog
        df["dew_point_depression"] = t - t_dew
        
        return df

    def extract_boundary_layer_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts atmospheric stability and washout indicators:
        - Stagnation flag: wind speed < 1.5 m/s and no precipitation.
        - High wind dispersion flag: wind speed > 5.0 m/s.
        - Ventilation index proxy: wind_speed * (T + 273.15) / 100.0
        - Cumulative precipitation washout: rolling 6h and 24h precipitation sums.
        """
        df = df.copy()
        grouped = df.groupby("station")
        
        # Stagnation indicator
        if "wind_speed_10m" in df.columns and "precipitation" in df.columns:
            is_stagnant = (df["wind_speed_10m"] < 1.5) & (df["precipitation"] <= 0.01)
            df["is_atmospheric_stagnation"] = is_stagnant.astype(float)
            
            is_dispersive = df["wind_speed_10m"] >= 5.0
            df["is_wind_dispersion"] = is_dispersive.astype(float)
            
        # Ventilation index proxy
        if "wind_speed_10m" in df.columns and "temperature_2m" in df.columns:
            t_kelvin = df["temperature_2m"] + 273.15
            df["ventilation_index_proxy"] = (df["wind_speed_10m"] * t_kelvin) / 100.0
            
        # Precipitation rolling accumulation (wet deposition washout proxy)
        if "precipitation" in df.columns:
            df["precip_roll_sum_6h"] = grouped["precipitation"].transform(
                lambda s: s.rolling(window=6, min_periods=1).sum()
            )
            df["precip_roll_sum_24h"] = grouped["precipitation"].transform(
                lambda s: s.rolling(window=24, min_periods=1).sum()
            )
            df["has_recent_rain_24h"] = (df["precip_roll_sum_24h"] > 0.1).astype(float)
            
        return df

    def extract_weather_lags_and_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts past-only weather lags and diurnal temperature/pressure differences.
        - Temperature 1h lag, 24h lag, 24h difference
        - Relative humidity 1h lag, 24h lag
        - Wind speed 1h lag, 6h rolling mean, 24h rolling mean
        - Surface pressure 1h lag, 3h barometric tendency (dP/dt)
        """
        df = df.copy()
        grouped = df.groupby("station")
        
        if "temperature_2m" in df.columns:
            df["temp_lag_1"] = grouped["temperature_2m"].shift(1)
            df["temp_lag_24"] = grouped["temperature_2m"].shift(24)
            df["temp_diff_24h"] = df["temperature_2m"] - df["temp_lag_24"]
            
        if "relative_humidity_2m" in df.columns:
            df["rh_lag_1"] = grouped["relative_humidity_2m"].shift(1)
            df["rh_lag_24"] = grouped["relative_humidity_2m"].shift(24)
            df["rh_diff_24h"] = df["relative_humidity_2m"] - df["rh_lag_24"]
            
        if "wind_speed_10m" in df.columns:
            df["wind_speed_lag_1"] = grouped["wind_speed_10m"].shift(1)
            df["wind_speed_roll_mean_6h"] = grouped["wind_speed_10m"].transform(
                lambda s: s.rolling(window=6, min_periods=1).mean()
            )
            df["wind_speed_roll_mean_24h"] = grouped["wind_speed_10m"].transform(
                lambda s: s.rolling(window=24, min_periods=1).mean()
            )
            
        if "surface_pressure" in df.columns:
            df["pressure_lag_1"] = grouped["surface_pressure"].shift(1)
            df["pressure_tendency_3h"] = df["surface_pressure"] - grouped["surface_pressure"].shift(3)
            
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes full meteorological feature engineering pipeline."""
        df = self.extract_wind_vector_features(df)
        df = self.extract_psychrometric_features(df)
        df = self.extract_boundary_layer_dynamics(df)
        df = self.extract_weather_lags_and_trends(df)
        return df
