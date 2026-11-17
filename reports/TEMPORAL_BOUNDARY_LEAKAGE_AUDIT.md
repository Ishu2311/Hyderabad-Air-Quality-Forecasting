# Temporal Boundary Leakage and Target Window Partitioning Audit

**Document Code:** `AUDIT-M2-BOUNDARY-LEAKAGE`  
**Project:** Traffic-Aware Spatio-Temporal PM2.5 Concentration and Hotspot Forecasting in Hyderabad  
**Milestone:** Milestone 2 – Multi-Source Feature Engineering  
**Audit Dimension:** Partition Integrity, Target Shift Windows & Boundary Buffer Enforcement  
**Audit Classification:** **FAIL (ON DEFAULT FILES) / PASS (ON REMEDIATED PURGED DATASETS)**  

---

## 1. Executive Summary & Audit Mandate

In multi-horizon time-series forecasting, prospective targets for prediction origin $t$ are formed by forward shifting future observations:
$$y_{i, t+h} = \text{shift}(-h) \quad \text{for } h \in \{1, 6, 24, 48\}\,\text{hours}$$
When time-series are partitioned into contiguous chronological blocks (Train: 2023–2024, Validation: 2025, Test: 2026), observations near the end of a partition have target timestamps that fall across the boundary into the subsequent partition.

This audit:
1. Identifies the exact sample count crossing partition boundaries for every forecast horizon.
2. Evaluates whether the 48-hour boundary buffer was actually enforced or merely flagged.
3. Provides exact empirical row counts before and after boundary purging.
4. Confirms mathematical non-leakage in the newly generated purged feature store partitions.

---

## 2. Horizon-by-Horizon Boundary Crossing Analysis

For each partition boundary, a sample at timestamp $t$ has target timestamp $t+h$. If $t+h > \text{partition\_end}$, the target observation belongs physically to the next partition:

### 2.1 Training Partition Boundary (Train End: 2024-12-31 23:00:00)
- **Train Station-Hours Total:** 70,176
- **Crossing Samples per Horizon:**

| Horizon ($h$) | Target Shift Formulation | Crossover Condition | Crossing Samples Total | Per Station Count (4 Stations) | Crossover Target Dates in Validation Set |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **+1 Hour** | $y_{i, t+1}$ | $t + 1\text{h} > \text{2024-12-31 23:00}$ | **4 samples** | 1 per station | 2025-01-01 00:00:00 |
| **+6 Hours** | $y_{i, t+6}$ | $t + 6\text{h} > \text{2024-12-31 23:00}$ | **24 samples** | 6 per station | 2025-01-01 00:00 to 05:00 |
| **+24 Hours** | $y_{i, t+24}$ | $t + 24\text{h} > \text{2024-12-31 23:00}$ | **96 samples** | 24 per station | 2025-01-01 00:00 to 23:00 |
| **+48 Hours** | $y_{i, t+48}$ | $t + 48\text{h} > \text{2024-12-31 23:00}$ | **192 samples** | 48 per station | 2025-01-01 00:00 to 2025-01-02 23:00 |

### 2.2 Validation Partition Boundary (Val End: 2025-12-31 23:00:00)
- **Validation Station-Hours Total:** 35,040
- **Crossing Samples per Horizon:**

| Horizon ($h$) | Target Shift Formulation | Crossover Condition | Crossing Samples Total | Per Station Count (4 Stations) | Crossover Target Dates in Test Set |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **+1 Hour** | $y_{i, t+1}$ | $t + 1\text{h} > \text{2025-12-31 23:00}$ | **4 samples** | 1 per station | 2026-01-01 00:00:00 |
| **+6 Hours** | $y_{i, t+6}$ | $t + 6\text{h} > \text{2025-12-31 23:00}$ | **24 samples** | 6 per station | 2026-01-01 00:00 to 05:00 |
| **+24 Hours** | $y_{i, t+24}$ | $t + 24\text{h} > \text{2025-12-31 23:00}$ | **96 samples** | 24 per station | 2026-01-01 00:00 to 23:00 |
| **+48 Hours** | $y_{i, t+48}$ | $t + 48\text{h} > \text{2025-12-31 23:00}$ | **192 samples** | 48 per station | 2026-01-01 00:00 to 2026-01-02 23:00 |

---

## 3. Audit Finding: Flagged vs. Enforced Boundary Buffer

### The Vulnerability in Default Feature Store Files
1. `ChronologicalSplitter.add_partition_metadata` correctly flagged the final 48 hours of each partition with `is_boundary_buffer = True` (exactly 192 rows per partition).
2. However, in `feature_store_builder.py`, the saved partition files:
   - `feature_store_train.parquet` (70,176 rows)
   - `feature_store_val.parquet` (35,040 rows)
   - `feature_store_test.parquet` (26,208 rows)
   **retained the boundary buffer rows by default**.
3. While the column `is_boundary_buffer` allowed post-hoc filtering, a model trained directly on `feature_store_train.parquet` without explicit row filtering would ingest 192 samples whose +48h targets evaluate on 2025 validation observations.
4. **Classification:** **FAIL (on unpurged default files)**. Flagging is insufficient; the pipeline must supply strictly purged datasets where crossing rows are physically removed.

---

## 4. Remediation & Exact Row Counts Before and After Purging

### Code Remediation
`feature_store_builder.py` was updated to explicitly export physically purged partition files alongside full flagged files:
```python
split_df_purged = split_df[~split_df["is_boundary_buffer"]].copy().reset_index(drop=True)
split_df_purged.to_parquet(out_dir / f"feature_store_{split_name}_purged.parquet")
split_df_purged.to_csv(out_dir / f"feature_store_{split_name}_purged.csv")
```

### Exact Row Counts Verification

| Evaluation Partition | Calendar Time Range | Full Station-Hours (Flagged) | Purged Station-Hours (Buffer Removed) | Excluded Boundary Buffer Rows | Max Target Date in Purged Split (+48h) | Boundary Leakage Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Split** | 2023-01-01 to 2024-12-31 | **70,176** | **69,984** | **192** (48h $\times$ 4 stn) | **2024-12-31 23:00** | **ZERO LEAKAGE (100% within Train)** |
| **Validation Split** | 2025-01-01 to 2025-12-31 | **35,040** | **34,848** | **192** (48h $\times$ 4 stn) | **2025-12-31 23:00** | **ZERO LEAKAGE (100% within Val)** |
| **Test Split** | 2026-01-01 to 2026-09-30 | **26,208** | **26,016** | **192** (48h $\times$ 4 stn) | **2026-09-30 23:00** | **ZERO LEAKAGE (100% within Test)** |
| **TOTAL NETWORK** | 2023-01-01 to 2026-09-30 | **131,424** | **130,848** | **576** | `--` | **PASSED** |

### Automated Unit Test Verification
Automated test `test_temporal_boundary_purging_zero_target_crossover` was implemented in `tests/test_feature_engineering.py`. It asserts:
1. `len(df_train_purged) == 69984`
2. `len(df_val_purged) == 34848`
3. For all $h \in \{1, 6, 24, 48\}$: $\max(t_{\text{train}} + h) \le \text{2024-12-31 23:00:00}$
4. For all $h \in \{1, 6, 24, 48\}$: $\max(t_{\text{val}} + h) \le \text{2025-12-31 23:00:00}$
**Status: PASSED.**

---

## 5. Audit Item Verdict

| Verification Item | Specification | Observed State | Classification |
| :--- | :--- | :--- | :--- |
| **Boundary Crossing Identification** | Quantify crossing counts for +1, +6, +24, +48h | 4, 24, 96, 192 rows documented | **PASS** |
| **Boundary Buffer Flagging** | `is_boundary_buffer = True` for last 48h | Exactly 192 rows/split flagged | **PASS** |
| **Default File Exclusion** | Excluded by default in previous files | Included in default files | **FAIL (Remediated)** |
| **Remediated Purged Partitions** | Dedicated purged files exported | `*_purged.parquet/csv` generated | **PASS** |
| **Zero Target Crossover Verification** | Max target time within partition bounds | Verified by automated test | **PASS** |

---

*Report prepared as part of Milestone 2 Independent Scientific Verification.*
