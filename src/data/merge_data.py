"""
Data Merging Module for Hyderabad Air Quality Research
Milestone 1: Data Integration & Integrity Checks

Merges CPCB air pollutant measurements with Open-Meteo meteorological reanalysis.
Preserves exact hourly grid without duplication or record loss.
Saves output to data/interim/merged_raw.csv.
"""

import sys
from pathlib import Path
import pandas as pd

def merge_cpcb_and_weather(cpcb_file: Path, weather_file: Path, output_file: Path):
    print("=" * 80)
    print("MERGING CPCB AND WEATHER REANALYSIS DATASETS")
    print("=" * 80)
    
    if not cpcb_file.exists():
        raise FileNotFoundError(f"CPCB input missing: {cpcb_file}")
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather input missing: {weather_file}")
        
    print(f"Loading CPCB data from: {cpcb_file}")
    df_cpcb = pd.read_csv(cpcb_file)
    print(f"Loading Weather data from: {weather_file}")
    df_weather = pd.read_csv(weather_file)
    
    # Standardize station names
    df_cpcb['station'] = df_cpcb['station'].replace({'Hyderabad': 'Central University'})
    df_weather['station'] = df_weather['station'].replace({'Hyderabad': 'Central University'})
    
    df_cpcb['datetime'] = pd.to_datetime(df_cpcb['datetime'])
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    
    initial_cpcb_rows = len(df_cpcb)
    initial_weather_rows = len(df_weather)
    print(f"CPCB rows: {initial_cpcb_rows:,}")
    print(f"Weather rows: {initial_weather_rows:,}")
    
    # Merge on datetime and station
    merged = pd.merge(df_cpcb, df_weather, on=['datetime', 'station'], how='inner')
    print(f"Merged rows: {len(merged):,}")
    
    assert len(merged) == initial_cpcb_rows == 131424, f"Row count error after merge: {len(merged)}"
    
    # Sort deterministically
    merged = merged.sort_values(by=['station', 'datetime']).reset_index(drop=True)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)
    print(f"Successfully saved merged raw dataset to: {output_file}")
    print("=" * 80)
    return merged

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    cpcb_in = base_dir / "data" / "raw" / "cpcb_hourly_raw.csv"
    weather_in = base_dir / "data" / "raw" / "open_meteo_weather_raw.csv"
    merged_out = base_dir / "data" / "interim" / "merged_raw.csv"
    
    merge_cpcb_and_weather(cpcb_in, weather_in, merged_out)
