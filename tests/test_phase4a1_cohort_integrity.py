"""
ASTRAEUS Gate 4 Phase 4A.1 Cohort Integrity & Set Arithmetic Test Suite.

Contains 6 focused integrity test functions:
1. test_input_hashes: Cryptographic hashes of all 5 input dependencies.
2. test_raw_label_reconciliation_coverage: Complete 33-label raw BTS reconciliation table (Approved, Quarantined, Unmapped).
3. test_set_arithmetic_consistency: Exact set-theoretic and sequential attrition arithmetic (|A|, |B|, |A ∩ B|, |A ∪ B|, |B \\ A|).
4. test_cohort_accounting_totals: Four distinct cohort accounting totals (Screened: 7,616, Admitted: 7,614) and class support designations (LARGE_SAMPLE_SUPPORT vs NOT_EVALUABLE).
5. test_tde_subclass_reconciliation: Comprehensive TDE subclass breakdown (75 raw, 70 admitted, 64 Train/Val overlaps, 4 spatial exclusions, 2 retained).
6. test_ledger_file_integrity: Ledger row counts (10,988), columns, and partition filtering consistency across Parquet and CSV files.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

EXPECTED_HASHES = {
    "bts_raw_snapshot_csv": "51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed",
    "candidates_10k_parquet": "6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae",
    "data_splits_healpix_json": "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b",
    "gate4a_taxonomy_mapping_yaml": "265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1",
    "gate4a_raw_label_reconciliation_csv": "c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1",
}

PATHS = {
    "bts_raw": Path("data/external_validity_v1/raw/bts/bts_raw_snapshot.csv"),
    "candidates_10k": Path("data/processed/real_survey_10k_v1/candidates_10k.parquet"),
    "healpix_splits": Path("configs/data_splits_healpix.json"),
    "taxonomy_yaml": Path("configs/gate4/gate4a_taxonomy_mapping.yaml"),
    "reconciliation_csv": Path("configs/gate4/gate4a_raw_label_reconciliation.csv"),
    "metrics_json": Path("experiments/gate4a_external_cohort_preregistration/gate4a1_audit_metrics.json"),
    "ledger_parquet": Path("experiments/gate4a_external_cohort_preregistration/gate4a1_event_accounting_ledger.parquet"),
    "ledger_csv": Path("experiments/gate4a_external_cohort_preregistration/gate4a1_event_accounting_ledger.csv"),
    "admitted_csv": Path("experiments/gate4a_external_cohort_preregistration/gate4a1_screened_admitted_candidates.csv"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_input_hashes():
    """Verify cryptographic integrity of all raw datasets and configuration files."""
    assert PATHS["bts_raw"].exists(), "Raw BTS CSV missing"
    assert _sha256(PATHS["bts_raw"]) == EXPECTED_HASHES["bts_raw_snapshot_csv"]

    assert PATHS["candidates_10k"].exists(), "candidates_10k.parquet missing"
    assert _sha256(PATHS["candidates_10k"]) == EXPECTED_HASHES["candidates_10k_parquet"]

    assert PATHS["healpix_splits"].exists(), "HEALPix splits config missing"
    assert _sha256(PATHS["healpix_splits"]) == EXPECTED_HASHES["data_splits_healpix_json"]

    assert PATHS["taxonomy_yaml"].exists(), "Taxonomy mapping YAML missing"
    assert _sha256(PATHS["taxonomy_yaml"]) == EXPECTED_HASHES["gate4a_taxonomy_mapping_yaml"]

    assert PATHS["reconciliation_csv"].exists(), "Reconciliation CSV missing"
    assert _sha256(PATHS["reconciliation_csv"]) == EXPECTED_HASHES["gate4a_raw_label_reconciliation_csv"]


def test_raw_label_reconciliation_coverage():
    """Verify that all 33 raw BTS labels are exhaustively mapped and sum to 10,988 records."""
    df_recon = pd.read_csv(PATHS["reconciliation_csv"])
    assert len(df_recon) == 33, f"Expected 33 raw labels, got {len(df_recon)}"
    assert df_recon["raw_count"].sum() == 10988, f"Expected sum 10,988, got {df_recon['raw_count'].sum()}"

    status_counts = df_recon.groupby("admitted_status")["raw_count"].sum().to_dict()
    assert status_counts["Approved"] == 10927
    assert status_counts["Quarantined"] == 55
    assert status_counts["Unmapped"] == 6

    # Verify no unexpected nulls
    assert df_recon["astraeus_class"].notna().all()
    assert df_recon["normalized_subtype"].notna().all()


def test_set_arithmetic_consistency():
    """Verify exact set-theoretic and sequential attrition arithmetic."""
    assert PATHS["metrics_json"].exists(), "Audit metrics JSON missing"
    with open(PATHS["metrics_json"]) as f:
        metrics = json.load(f)

    arith = metrics["exact_attrition_arithmetic"]
    n_raw = arith["raw_bts_records"]
    size_A = arith["set_A_train_val_overlap"]
    size_B = arith["set_B_spatial_test_buffer"]
    size_inter = arith["set_intersection_A_and_B"]
    size_union = arith["set_union_total_excluded"]
    inc_stage2 = arith["stage2_incremental_spatial_exclusions"]
    n_screened = arith["retained_screened_candidates"]

    # Fundamental set-theoretic identities
    assert n_raw == 10988
    assert size_A == 1712
    assert size_B == 1665
    assert size_inter == 5
    assert size_union == size_A + size_B - size_inter  # 1712 + 1665 - 5 = 3372
    assert size_union == 3372

    # Sequential exclusion identity
    assert inc_stage2 == size_B - size_inter  # 1665 - 5 = 1660
    assert inc_stage2 == 1660
    assert size_A + inc_stage2 == size_union  # 1712 + 1660 = 3372
    assert n_raw - size_union == n_screened  # 10988 - 3372 = 7616
    assert n_screened == 7616


def test_cohort_accounting_totals():
    """Verify the four distinct cohort accounting tiers and class breakdowns."""
    with open(PATHS["metrics_json"]) as f:
        metrics = json.load(f)

    cohort = metrics["four_distinct_cohort_totals"]
    n_screened = cohort["total_1_geometrically_and_identity_screened_candidates"]
    n_quar = cohort["screened_quarantined_or_unclassified_candidates"]
    n_admitted = cohort["total_2_taxonomy_admitted_candidates"]

    assert n_screened == 7616
    assert n_quar == 2  # Exactly 1 '-' and 1 'Other'
    assert n_admitted == 7614
    assert n_screened - n_quar == n_admitted

    # Verify class support designations
    support = metrics["class_support_breakdown"]
    assert support["SN_Ia"]["count"] == 5916
    assert support["SN_Ia"]["support_designation"] == "LARGE_SAMPLE_SUPPORT"

    assert support["SN_CC"]["count"] == 1693
    assert support["SN_CC"]["support_designation"] == "LARGE_SAMPLE_SUPPORT"

    assert support["SLSN"]["count"] == 1
    assert support["SLSN"]["support_designation"] == "NOT_EVALUABLE"

    assert support["TDE"]["count"] == 2
    assert support["TDE"]["support_designation"] == "NOT_EVALUABLE"

    assert support["OTHER_EXTRAGALACTIC"]["count"] == 1
    assert support["OTHER_EXTRAGALACTIC"]["support_designation"] == "NOT_EVALUABLE"

    assert support["AGN_QSO"]["count"] == 1
    assert support["AGN_QSO"]["support_designation"] == "NOT_EVALUABLE"

    assert support["CV_Nova"]["count"] == 0
    assert support["CV_Nova"]["support_designation"] == "NOT_EVALUABLE"

    assert support["Varstar"]["count"] == 0
    assert support["Varstar"]["support_designation"] == "NOT_EVALUABLE"

    # Verify sum equals taxonomy admitted total
    total_class_count = sum(s["count"] for s in support.values())
    assert total_class_count == 7614


def test_tde_subclass_reconciliation():
    """Verify exhaustive reconciliation of all raw TDE records and taxonomy-admitted TDEs."""
    with open(PATHS["metrics_json"]) as f:
        metrics = json.load(f)

    tde = metrics["reconciled_tde_metrics"]

    assert tde["all_raw_tde_string_records"] == 75
    assert tde["taxonomy_admitted_raw_tde_records"] == 70  # TDE (66) + TDE-He (4)
    assert tde["unmapped_raw_tde_records"] == 5  # TDE-H-He (4) + TDE-featureless (1)

    assert tde["all_raw_tde_string_train_val_overlaps"] == 69
    assert tde["all_raw_tde_string_historical_overlap_pct"] == 92.0

    assert tde["taxonomy_admitted_tde_train_val_overlaps"] == 64
    assert tde["taxonomy_admitted_tde_historical_overlap_pct"] == 91.43

    assert tde["taxonomy_admitted_tde_spatial_buffer_exclusions"] == 4
    assert tde["taxonomy_admitted_tde_retained"] == 2

    # Retained candidates verification: 70 - 64 - 4 = 2
    assert 70 - tde["taxonomy_admitted_tde_train_val_overlaps"] - tde["taxonomy_admitted_tde_spatial_buffer_exclusions"] == 2

    retained_ids = {c["ztf_id"] for c in tde["retained_tde_candidates"]}
    assert retained_ids == {"ZTF18aabdajx", "ZTF18aabtxvd"}


def test_ledger_file_integrity():
    """Verify row counts and structural invariants in emitted ledger files."""
    assert PATHS["ledger_parquet"].exists(), "Accounting ledger parquet missing"
    df_ledger = pd.read_parquet(PATHS["ledger_parquet"])
    assert len(df_ledger) == 10988, f"Expected 10,988 rows in full ledger, got {len(df_ledger)}"

    assert "ztf_id" in df_ledger.columns
    assert "is_in_train_val" in df_ledger.columns
    assert "is_in_spatial_buffer" in df_ledger.columns
    assert "is_screened_candidate" in df_ledger.columns
    assert "is_taxonomy_screened_candidate" in df_ledger.columns
    assert "taxonomy_admitted_status" in df_ledger.columns

    assert df_ledger["is_in_train_val"].sum() == 1712
    assert df_ledger["is_in_spatial_buffer"].sum() == 1665
    assert (df_ledger["is_in_train_val"] & df_ledger["is_in_spatial_buffer"]).sum() == 5
    assert df_ledger["is_screened_candidate"].sum() == 7616
    assert df_ledger["is_taxonomy_screened_candidate"].sum() == 7614

    assert PATHS["ledger_csv"].exists(), "Accounting ledger CSV missing"
    df_ledger_csv = pd.read_csv(PATHS["ledger_csv"])
    assert len(df_ledger_csv) == 10988, f"Expected 10,988 rows in ledger CSV, got {len(df_ledger_csv)}"

    assert PATHS["admitted_csv"].exists(), "Screened admitted CSV missing"
    df_admitted = pd.read_csv(PATHS["admitted_csv"])
    assert len(df_admitted) == 7614, f"Expected 7,614 rows in admitted CSV, got {len(df_admitted)}"
    assert (df_admitted["taxonomy_admitted_status"] == "Approved").all()
