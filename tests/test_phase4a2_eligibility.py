"""
ASTRAEUS Gate 4 Phase 4A.2: Evidence & Feature Eligibility Test Suite.

Verifies:
1. Cryptographic hashes of all 7 input dependencies.
2. Spectroscopic evidence ledger row count (7,614) and exact tier breakdown (S1=0, S2=7582, S3=31, S4=1).
3. Traceability of all Tier S2 candidates to published TNSCR/ADS classification bibcodes.
4. Light-curve match ledger row count (7,614) and evaluability breakdown (16 Evaluable, 1 Single-Band, 7,597 Awaiting Broker).
5. Numerical validity and frozen 27-feature interface compatibility for all 16 evaluable candidates.
6. Decoupled and independent auditability across Track A and Track B.
7. Integrity and checksum consistency of the reproducibility manifest.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

EXPECTED_INPUT_HASHES = {
    "bts_raw_snapshot_csv": "51d85f50772fd1412fcc2646f48b4d331b2e07f7023454f89cddd7a8e7e283ed",
    "tns_public_objects_zip": "5d3fa52497de6f24491ebf25715fa9b617a220f662b2fc5102f9baa02e094ad8",
    "candidates_10k_parquet": "6713faee3c2ab47b7a14baa321c540f2b18ddb9b51d6ddcf88dba7008ad214ae",
    "data_splits_healpix_json": "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b",
    "gate4a_taxonomy_mapping_yaml": "265b538f2ad040e15a4b9b0069676c7dbbaec095c9a8b88bbcbe96ad7e8364b1",
    "gate4a_raw_label_reconciliation_csv": "c5c8b06e0e3b065437224513efef33212ae4fdd4404f65c9dca6b7bdc6cef0f1",
    "gate4a1_screened_admitted_candidates_csv": "abd7dc90eebeeb5507334f5f30acddf01824e4c3c6efd2834186ca850456fcb9",
}

PATHS = {
    "bts_raw": Path("data/external_validity_v1/raw/bts/bts_raw_snapshot.csv"),
    "tns_zip": Path("data/external_validity_v1/raw/tns/tns_public_objects.csv.zip"),
    "candidates_10k": Path("data/processed/real_survey_10k_v1/candidates_10k.parquet"),
    "healpix_splits": Path("configs/data_splits_healpix.json"),
    "taxonomy_yaml": Path("configs/gate4/gate4a_taxonomy_mapping.yaml"),
    "reconciliation_csv": Path("configs/gate4/gate4a_raw_label_reconciliation.csv"),
    "admitted_csv": Path("experiments/gate4a_external_cohort_preregistration/gate4a1_screened_admitted_candidates.csv"),
    "spec_ledger_csv": Path("experiments/gate4a2_evidence_and_feature_eligibility/spectroscopic_evidence_ledger.csv"),
    "lc_ledger_parquet": Path("experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.parquet"),
    "lc_ledger_csv": Path("experiments/gate4a2_evidence_and_feature_eligibility/lightcurve_match_ledger.csv"),
    "manifest_json": Path("experiments/gate4a2_evidence_and_feature_eligibility/phase4a2_reproducibility_manifest.json"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_input_hashes():
    """Verify cryptographic checksums for all 7 Phase 4A.2 input dependencies."""
    for key, path in PATHS.items():
        if key in ("spec_ledger_csv", "lc_ledger_parquet", "lc_ledger_csv", "manifest_json"):
            continue
        assert path.exists(), f"Input dependency missing: {path}"

    assert _sha256(PATHS["bts_raw"]) == EXPECTED_INPUT_HASHES["bts_raw_snapshot_csv"]
    assert _sha256(PATHS["tns_zip"]) == EXPECTED_INPUT_HASHES["tns_public_objects_zip"]
    assert _sha256(PATHS["candidates_10k"]) == EXPECTED_INPUT_HASHES["candidates_10k_parquet"]
    assert _sha256(PATHS["healpix_splits"]) == EXPECTED_INPUT_HASHES["data_splits_healpix_json"]
    assert _sha256(PATHS["taxonomy_yaml"]) == EXPECTED_INPUT_HASHES["gate4a_taxonomy_mapping_yaml"]
    assert _sha256(PATHS["reconciliation_csv"]) == EXPECTED_INPUT_HASHES["gate4a_raw_label_reconciliation_csv"]
    assert _sha256(PATHS["admitted_csv"]) == EXPECTED_INPUT_HASHES["gate4a1_screened_admitted_candidates_csv"]


def test_spectroscopic_evidence_ledger():
    """Verify Track A spectroscopic evidence ledger structure, tier distribution, and traceability."""
    assert PATHS["spec_ledger_csv"].exists(), "Spectroscopic evidence ledger missing"
    df_spec = pd.read_csv(PATHS["spec_ledger_csv"])
    assert len(df_spec) == 7614, f"Expected 7,614 rows in spec ledger, got {len(df_spec)}"

    # Required columns
    required_cols = [
        "candidate_id", "ztf_id", "iau_id", "matched_tns_name", "raw_bts_type",
        "normalized_subtype", "astraeus_class", "tns_type", "evidence_tier",
        "class_ads_bibcodes", "discovery_ads_bibcode", "reporting_group",
        "reporters", "discovery_date", "redshift", "tns_object_url",
        "verification_status", "permitted_eval_use",
    ]
    for col in required_cols:
        assert col in df_spec.columns, f"Missing column {col} in spec ledger"

    # Tier breakdown
    tier_counts = df_spec["evidence_tier"].value_counts().to_dict()
    assert tier_counts.get("S1", 0) == 0, "Expected 0 S1 candidates (no local FITS spectra)"
    assert tier_counts.get("S2", 0) == 7582, f"Expected 7,582 S2 candidates, got {tier_counts.get('S2')}"
    assert tier_counts.get("S3", 0) == 31, f"Expected 31 S3 candidates, got {tier_counts.get('S3')}"
    assert tier_counts.get("S4", 0) == 1, f"Expected 1 S4 candidate, got {tier_counts.get('S4')}"

    # Verify that all S2 candidates have non-empty published Class_ADS_bibcodes and valid tns_type
    df_s2 = df_spec[df_spec["evidence_tier"] == "S2"]
    assert (df_s2["class_ads_bibcodes"].str.len() > 0).all()
    assert (df_s2["tns_type"].str.len() > 0).all()
    assert (df_s2["verification_status"] == "VERIFIED_TRACEABLE_REPORT").all()
    assert (df_s2["permitted_eval_use"] == "SECONDARY_TRACEABLE_EVIDENCE").all()

    # Verify both retained TDEs are in Tier S2
    tde_records = df_spec[df_spec["astraeus_class"] == "TDE"]
    assert len(tde_records) == 2
    assert (tde_records["evidence_tier"] == "S2").all()
    assert set(tde_records["ztf_id"]) == {"ZTF18aabdajx", "ZTF18aabtxvd"}


def test_lightcurve_match_ledger():
    """Verify Track B light-curve match ledger structure, counts, and evaluability."""
    assert PATHS["lc_ledger_parquet"].exists(), "Light-curve match ledger parquet missing"
    assert PATHS["lc_ledger_csv"].exists(), "Light-curve match ledger csv missing"

    df_lc = pd.read_parquet(PATHS["lc_ledger_parquet"])
    assert len(df_lc) == 7614, f"Expected 7,614 rows in lc ledger, got {len(df_lc)}"

    df_lc_csv = pd.read_csv(PATHS["lc_ledger_csv"])
    assert len(df_lc_csv) == 7614

    required_cols = [
        "candidate_id", "ztf_id", "iau_id", "astraeus_class",
        "has_local_lightcurve_file", "n_det_g", "n_det_r", "n_det_other",
        "n_det_total", "n_ul_g", "n_ul_r", "n_ul_total", "total_epochs",
        "has_both_bands", "temporal_baseline_days", "median_cadence_days",
        "peak_flux_nmgy", "flux_dynamic_range_nmgy", "feature_interface_compatible",
        "feature_eligibility_status", "missingness_reason",
    ]
    for col in required_cols:
        assert col in df_lc.columns, f"Missing column {col} in lc ledger"

    # Status counts
    status_counts = df_lc["feature_eligibility_status"].value_counts().to_dict()
    assert status_counts.get("FEATURE_EVALUABLE", 0) == 16
    assert status_counts.get("SINGLE_BAND_ONLY", 0) == 1
    assert status_counts.get("AWAITING_BROKER_LIGHTCURVE_INGESTION", 0) == 7597

    # Verify the 16 evaluable candidates
    df_eval = df_lc[df_lc["feature_eligibility_status"] == "FEATURE_EVALUABLE"]
    assert (df_eval["has_local_lightcurve_file"] == True).all()
    assert (df_eval["n_det_total"] >= 5).all()
    assert (df_eval["has_both_bands"] == True).all()
    assert (df_eval["feature_interface_compatible"] == True).all()
    assert (df_eval["peak_flux_nmgy"] > 0.0).all()
    assert (df_eval["temporal_baseline_days"] > 0.0).all()


def test_independent_track_decoupling():
    """Verify that Track A and Track B are independently auditable."""
    df_spec = pd.read_csv(PATHS["spec_ledger_csv"])
    df_lc = pd.read_parquet(PATHS["lc_ledger_parquet"])

    merged = pd.merge(df_spec, df_lc, on="ztf_id")
    assert len(merged) == 7614

    # Crosstab verification
    ct = pd.crosstab(merged["evidence_tier"], merged["feature_eligibility_status"])
    assert "S2" in ct.index
    assert ct.loc["S2", "FEATURE_EVALUABLE"] == 16
    assert ct.loc["S2", "SINGLE_BAND_ONLY"] == 1
    assert ct.loc["S2", "AWAITING_BROKER_LIGHTCURVE_INGESTION"] == 7565


def test_reproducibility_manifest():
    """Verify reproducibility manifest structure, dependencies, and emitted artifact checksums."""
    assert PATHS["manifest_json"].exists(), "Reproducibility manifest missing"
    with open(PATHS["manifest_json"]) as f:
        manifest = json.load(f)

    assert manifest["milestone"] == "ASTRAEUS Gate 4 Phase 4A.2 Evidence & Feature Eligibility"
    assert manifest["track_a_spectroscopic_evidence_summary"]["total_candidates_evaluated"] == 7614
    assert manifest["track_b_lightcurve_feature_eligibility_summary"]["total_candidates_evaluated"] == 7614

    emitted = manifest["emitted_artifacts"]
    assert _sha256(PATHS["spec_ledger_csv"]) == emitted["spectroscopic_evidence_ledger_csv"]["sha256"]
    assert _sha256(PATHS["lc_ledger_parquet"]) == emitted["lightcurve_match_ledger_parquet"]["sha256"]
    assert _sha256(PATHS["lc_ledger_csv"]) == emitted["lightcurve_match_ledger_csv"]["sha256"]
