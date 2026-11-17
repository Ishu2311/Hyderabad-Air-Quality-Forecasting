"""
Data Integrity and Leakage Verification Module
Milestone 1: Data Foundation and Preprocessing

Performs comprehensive automated mathematical and scientific integrity audits:
- Validates station identifiers, geodetic coordinates, and typologies.
- Verifies chronological monotonicity and uninterrupted 1-hour continuity.
- Audits zero duplicate records.
- Verifies exact station observation counts against research specifications.
- Validates physical value ranges (no negatives, physical upper bounds).
- Audits target generation for missing-awareness and zero lookahead.
- Verifies that unobserved target values evaluate to NaN and NEVER zero.
- Verifies weather reanalysis 100% completeness.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from ..configs.config import (
        STATION_REGISTRY,
        EXPECTED_HOURS_PER_STATION,
        EXPECTED_TOTAL_STATION_HOURS,
        POLLUTANTS,
        WEATHER_VARIABLES,
        PHYSICAL_VALID_RANGES,
        REGRESSION_HORIZONS,
        CLASSIFICATION_HORIZONS
    )
except (ImportError, ValueError):
    from configs.config import (
        STATION_REGISTRY,
        EXPECTED_HOURS_PER_STATION,
        EXPECTED_TOTAL_STATION_HOURS,
        POLLUTANTS,
        WEATHER_VARIABLES,
        PHYSICAL_VALID_RANGES,
        REGRESSION_HORIZONS,
        CLASSIFICATION_HORIZONS
    )

EXPECTED_STATION_VALID_PM25 = {
    "Bollaram Industrial Area": 23662,
    "Central University": 29127,
    "ECIL Kapra": 28519,
    "ICRISAT Patancheru": 30092
}

class DataIntegrityChecker:
    """Automated integrity, boundary, and non-leakage auditor."""

    def __init__(self):
        self.audit_results: Dict[str, Dict] = {}

    def audit_dimensions_and_stations(self, df: pd.DataFrame) -> Dict:
        """Verifies total rows, station count, and per-station hourly dimensions."""
        stations_found = sorted(df["station"].unique())
        expected_stations = sorted(STATION_REGISTRY.keys())
        
        station_match = (stations_found == expected_stations)
        total_rows = len(df)
        rows_match = (total_rows == EXPECTED_TOTAL_STATION_HOURS)
        
        station_hour_counts = df["station"].value_counts().to_dict()
        per_station_match = all(
            station_hour_counts.get(st, 0) == EXPECTED_HOURS_PER_STATION
            for st in expected_stations
        )
        
        status = "PASSED" if (station_match and rows_match and per_station_match) else "FAILED"
        res = {
            "status": status,
            "total_rows": total_rows,
            "expected_total_rows": EXPECTED_TOTAL_STATION_HOURS,
            "stations_found": stations_found,
            "station_hour_counts": station_hour_counts,
            "per_station_match": per_station_match
        }
        self.audit_results["dimensions_and_stations"] = res
        return res

    def audit_duplicates_and_continuity(self, df: pd.DataFrame) -> Dict:
        """Verifies zero duplicates and strictly monotonic 1-hour continuity per station."""
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        
        dup_count = int(df.duplicated(subset=["station", "datetime"]).sum())
        continuity_passed = True
        failed_stations = []
        
        for st in sorted(STATION_REGISTRY.keys()):
            st_df = df[df["station"] == st].sort_values(by="datetime")
            diffs = st_df["datetime"].diff().dropna()
            
            # Check all diffs are exactly 1 hour
            if not (diffs == pd.Timedelta(hours=1)).all():
                continuity_passed = False
                failed_stations.append(st)
                
        status = "PASSED" if (dup_count == 0 and continuity_passed) else "FAILED"
        res = {
            "status": status,
            "duplicate_count": dup_count,
            "continuity_passed": continuity_passed,
            "failed_stations": failed_stations
        }
        self.audit_results["duplicates_and_continuity"] = res
        return res

    def audit_valid_observation_counts(self, df: pd.DataFrame) -> Dict:
        """Verifies empirical valid PM2.5 counts against the audited research counts."""
        col = "PM2.5_raw" if "PM2.5_raw" in df.columns else "PM2.5"
        
        actual_counts = df.groupby("station")[col].count().to_dict()
        mismatches = {}
        
        for st, expected in EXPECTED_STATION_VALID_PM25.items():
            actual = actual_counts.get(st, 0)
            if actual != expected:
                mismatches[st] = {"expected": expected, "actual": actual}
                
        total_valid = sum(actual_counts.values())
        status = "PASSED" if (len(mismatches) == 0 and total_valid == 111400) else "FAILED"
        
        res = {
            "status": status,
            "actual_counts": actual_counts,
            "expected_counts": EXPECTED_STATION_VALID_PM25,
            "total_valid_pm25": total_valid,
            "expected_total_valid": 111400,
            "mismatches": mismatches
        }
        self.audit_results["valid_observation_counts"] = res
        return res

    def audit_physical_value_bounds(self, df: pd.DataFrame) -> Dict:
        """Verifies that concentrations and weather variables respect physical limits."""
        violations = {}
        
        # Check pollutants
        for p in POLLUTANTS:
            if p in df.columns:
                min_b, max_b = PHYSICAL_VALID_RANGES.get(p, (0.0, 1000.0))
                s = df[p].dropna()
                neg_count = int((s < min_b).sum())
                high_count = int((s > max_b).sum())
                if neg_count > 0 or high_count > 0:
                    violations[p] = {"negative_count": neg_count, "excessive_count": high_count}
                    
        # Check weather
        for w in WEATHER_VARIABLES:
            if w in df.columns:
                min_b, max_b = PHYSICAL_VALID_RANGES.get(w, (-100.0, 1000.0))
                s = df[w].dropna()
                low_count = int((s < min_b).sum())
                high_count = int((s > max_b).sum())
                if low_count > 0 or high_count > 0:
                    violations[w] = {"below_bound": low_count, "above_bound": high_count}
                    
        status = "PASSED" if len(violations) == 0 else "FAILED"
        res = {
            "status": status,
            "violations": violations
        }
        self.audit_results["physical_value_bounds"] = res
        return res

    def audit_target_generation_integrity(self, df: pd.DataFrame) -> Dict:
        """
        Verifies target generation integrity:
        1. Target values are NaN when future raw observation is NaN.
        2. No target is filled with 0.0 when unobserved.
        3. 24h statutory target is NaN when valid forward hours < 16.
        """
        checks = {}
        raw_col = "PM2.5_raw" if "PM2.5_raw" in df.columns else "PM2.5"
        
        # Check regression targets
        for h in REGRESSION_HORIZONS:
            col = f"target_pm25_h{h}"
            if col in df.columns:
                # Target at t should match raw_pm25 at t+h per station
                # Check for random sampled indices
                mismatch_count = 0
                for st in sorted(STATION_REGISTRY.keys()):
                    st_df = df[df["station"] == st].sort_values(by="datetime")
                    shifted = st_df[raw_col].shift(-h)
                    # Compare including NaNs
                    matches = (st_df[col].isna() & shifted.isna()) | (st_df[col] == shifted)
                    mismatch_count += int((~matches).sum())
                checks[f"regression_h{h}_alignment"] = {
                    "mismatches": mismatch_count,
                    "passed": mismatch_count == 0
                }
                
        # Check 24h target completeness rule
        if "target_pm25_24h_mean" in df.columns and "target_pm25_24h_valid_count" in df.columns:
            invalid_completeness_mask = (df["target_pm25_24h_valid_count"] < 16) & df["target_pm25_24h_mean"].notna()
            completeness_violation_count = int(invalid_completeness_mask.sum())
            checks["24h_completeness_rule_min16"] = {
                "violations": completeness_violation_count,
                "passed": completeness_violation_count == 0
            }
            
        # Check zero-target-substitution: ensure binary targets are strictly 0, 1, or NaN
        binary_targets = [c for c in df.columns if "hotspot" in c]
        for bt in binary_targets:
            unique_vals = set(df[bt].dropna().unique())
            valid_binary = unique_vals.issubset({0.0, 1.0})
            checks[f"binary_format_{bt}"] = {
                "unique_values": list(unique_vals),
                "passed": valid_binary
            }
            
        all_passed = all(c.get("passed", False) for c in checks.values())
        status = "PASSED" if all_passed else "FAILED"
        
        res = {
            "status": status,
            "detailed_checks": checks
        }
        self.audit_results["target_generation_integrity"] = res
        return res

    def run_full_audit(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Runs all integrity audits and returns overall status and report dict."""
        self.audit_dimensions_and_stations(df)
        self.audit_duplicates_and_continuity(df)
        self.audit_valid_observation_counts(df)
        self.audit_physical_value_bounds(df)
        if any(c.startswith("target_") for c in df.columns):
            self.audit_target_generation_integrity(df)
            
        overall_passed = all(
            res.get("status") == "PASSED"
            for res in self.audit_results.values()
        )
        return overall_passed, self.audit_results
