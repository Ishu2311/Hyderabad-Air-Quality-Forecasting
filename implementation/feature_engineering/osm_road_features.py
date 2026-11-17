"""
OpenStreetMap (OSM) Road-Network Feature Engineering Module
Milestone 2: Multi-Source Feature Engineering

Implements localized road network metrics in projected CRS (EPSG:32644 / UTM Zone 44N):
- Road lengths by class: Primary/Motorway and Secondary/Arterial in 1km and 3km metric buffers
- Road densities (km / km2) within 1km and 3km circular buffers
- Road-network concentration: Ratio of high-speed primary infrastructure to total roadway
- Proximity to major highway corridor: Euclidean distance (km) to nearest national highway corridor
- Multi-leg intersection density (nodes / km2)

SCIENTIFIC METHODOLOGICAL BOUNDARY:
In the absence of continuous automated traffic counters (ATCC) across Greater Hyderabad,
static roadway metrics represent localized land-use intensity and transportation infrastructure
proxies. They provide 4 unique spatial degrees of freedom across stations and must NEVER be
described as measured vehicular flow or actual traffic volume.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        OSM_STATION_ROAD_PROXIES,
        BUFFER_AREA_1KM_KM2,
        BUFFER_AREA_3KM_KM2,
        STATION_REGISTRY,
        STATION_NAME_ALIASES
    )
except (ImportError, ValueError):
    from configs.config import (
        OSM_STATION_ROAD_PROXIES,
        BUFFER_AREA_1KM_KM2,
        BUFFER_AREA_3KM_KM2,
        STATION_REGISTRY,
        STATION_NAME_ALIASES
    )

class OSMRoadFeatureExtractor:
    """Computes and maps localized road network metrics and proxies per monitoring station."""

    def __init__(
        self,
        proxies: Optional[Dict] = None,
        area_1km: float = BUFFER_AREA_1KM_KM2,
        area_3km: float = BUFFER_AREA_3KM_KM2
    ):
        self.proxies = proxies if proxies is not None else OSM_STATION_ROAD_PROXIES
        self.area_1km = area_1km
        self.area_3km = area_3km

    def get_station_road_metrics_table(self) -> pd.DataFrame:
        """Constructs a summary DataFrame of all static OSM road metrics per station."""
        records = []
        for station_name, metrics in self.proxies.items():
            tot_1km = metrics.primary_length_1km + metrics.secondary_length_1km
            tot_3km = metrics.primary_length_3km + metrics.secondary_length_3km
            
            density_tot_1km = tot_1km / self.area_1km
            density_tot_3km = tot_3km / self.area_3km
            density_prim_1km = metrics.primary_length_1km / self.area_1km
            density_prim_3km = metrics.primary_length_3km / self.area_3km
            
            conc_prim_1km = metrics.primary_length_1km / (tot_1km + 1e-5)
            conc_prim_3km = metrics.primary_length_3km / (tot_3km + 1e-5)
            
            records.append({
                "station": station_name,
                "osm_primary_length_1km": metrics.primary_length_1km,
                "osm_secondary_length_1km": metrics.secondary_length_1km,
                "osm_total_length_1km": tot_1km,
                "osm_primary_length_3km": metrics.primary_length_3km,
                "osm_secondary_length_3km": metrics.secondary_length_3km,
                "osm_total_length_3km": tot_3km,
                "osm_road_density_1km": density_tot_1km,
                "osm_road_density_3km": density_tot_3km,
                "osm_primary_density_1km": density_prim_1km,
                "osm_primary_density_3km": density_prim_3km,
                "osm_primary_concentration_1km": conc_prim_1km,
                "osm_primary_concentration_3km": conc_prim_3km,
                "osm_distance_to_highway_km": metrics.distance_to_highway_km,
                "osm_intersection_density_km2": metrics.intersection_density
            })
            
        return pd.DataFrame(records)

    def attach_road_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attaches static OSM road metrics to hourly observations by matching station.
        Standardizes station identifiers prior to join.
        """
        df = df.copy()
        if "station" not in df.columns:
            raise KeyError("DataFrame must contain 'station' column.")
            
        # Standardize aliases if needed
        df["station"] = df["station"].replace(STATION_NAME_ALIASES)
        
        metrics_df = self.get_station_road_metrics_table()
        
        # Merge on station
        merged = pd.merge(df, metrics_df, on="station", how="left")
        
        return merged
