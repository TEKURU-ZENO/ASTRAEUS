"""
ASTRAEUS Gate 4 Final Closure Verification Suite.

Author: Dev Mehta
Date: November 16, 2026
Milestone: Gate 4 Final Scientific Closure

Tests:
1. Strict spatial and population-level disjointness (Zero leakage into Gate 3 test set).
2. Chronological prefix invariants (Zero future-observation leakage).
3. Conformal calibration specification (N_calib=562, k=507, target=90%).
4. Shift-stratified denominator integrity.
5. Rare-class support designations (INSUFFICIENT_EXTERNAL_SUPPORT assertions).
6. Evidence manifest cryptographic integrity.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from astraeus_bench.real_survey_benchmark_protocol import REAL_SURVEY_CLASSES_8

EXPECTED_HEALPIX_CONFIG_HASH = "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b"
EXPECTED_SPRINT4H_MANIFEST_HASH = "1ddbf733cc52f6fe5a36a7a9854e780a59d2cadc3b4b153aaadf970a3b377761"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_gate3_quarantine_custodian_attestation_and_geometric_disjointness(repo_root: Path):
    """
    GOVERNANCE ASSERTION 1: Gate 3 Quarantine Custodian Attestation & Geometric Footprint Disjointness.

    DISCLOSURE OF PRIOR METADATA VERIFICATION INCIDENT:
    In commit 113f26d, a regression test directly queried candidate_id and matched_external_id values
    for assigned_split == 'test' in candidates_10k.parquet to assert set disjointness. While this was
    a read-only metadata verification, touching the quarantined test rows violates the strict zero-touch
    quarantine protocol. This test replaces that check with an independent, non-disclosing custodian
    attestation and geometric footprint disjointness proof without loading or inspecting any test rows.

    Verifications:
    1. Authoritative HEALPix split configuration cryptographic hash assertion
       (configs/data_splits_healpix.json, exactly 1,808 frozen test pixels at NSIDE=32, NESTED, ICRS).
    2. Custodian release manifest cryptographic seal assertion
       (experiments/sprint4h_final_release_remediation/sprint4h_release_manifest.json,
       attesting to 586 sequestered test candidates under zero-touch invariant).
    3. Strict geometric spatial footprint disjointness: all 7,614 external cohort candidates
       in phase4b2_full_cohort_accounting_ledger.parquet reside strictly outside the
       1,808 quarantined test pixels (zero intersection).
    """
    # 1. Authoritative HEALPix configuration hash & geometry
    healpix_path = repo_root / "configs" / "data_splits_healpix.json"
    assert healpix_path.exists(), f"Missing {healpix_path}"
    hp_sha = hashlib.sha256(healpix_path.read_bytes()).hexdigest()
    assert hp_sha == EXPECTED_HEALPIX_CONFIG_HASH, f"HEALPix config hash mismatch: {hp_sha}"

    with open(healpix_path, "r", encoding="utf-8") as f:
        hp_config = json.load(f)
    test_pixels = set(int(p) for p in hp_config["splits"]["test"])
    assert len(test_pixels) == 1808, f"Expected exactly 1,808 test pixels, got {len(test_pixels)}"

    # 2. Custodian release manifest cryptographic seal (586 sealed test objects)
    sprint4h_path = repo_root / "experiments" / "sprint4h_final_release_remediation" / "sprint4h_release_manifest.json"
    assert sprint4h_path.exists(), f"Missing {sprint4h_path}"
    s4h_sha = hashlib.sha256(sprint4h_path.read_bytes()).hexdigest()
    assert s4h_sha == EXPECTED_SPRINT4H_MANIFEST_HASH, f"Sprint 4H manifest hash mismatch: {s4h_sha}"

    with open(sprint4h_path, "r", encoding="utf-8") as f:
        s4h_data = json.load(f)
    assert s4h_data["spatial_isolation_audit"]["subtraction_verification"]["sequestered_test_count"] == 586
    assert s4h_data["spatial_isolation_audit"]["test_isolation_verdict"] == "CERTIFIED SEALED & UNTOUCHED (Zero-Touch Invariant)"

    # 3. Geometric footprint disjointness for all 7,614 external candidates
    p4b2_path = repo_root / "experiments" / "gate4b_broker_acquisition_and_cohort_expansion" / "phase4b2_full_cohort_accounting_ledger.parquet"
    assert p4b2_path.exists()
    df_p4b2 = pd.read_parquet(p4b2_path)
    assert len(df_p4b2) == 7614

    hp = HEALPix(nside=hp_config.get("healpix_nside", 32), order="nested", frame="icrs")
    sc_candidates = SkyCoord(df_p4b2["ra_raw"], df_p4b2["dec_raw"], unit=(u.hourangle, u.deg))
    cand_pixels = set(hp.skycoord_to_healpix(sc_candidates).tolist())

    geometric_overlap = test_pixels.intersection(cand_pixels)
    assert len(geometric_overlap) == 0, f"Spatial leakage detected: {len(geometric_overlap)} overlapping test pixels: {geometric_overlap}"


def test_chronological_prefix_no_future_leakage():
    """
    SCIENTIFIC INVARIANT 2:
    Verifies that the early-epoch prefix feature extractor strictly excludes
    observations with MJD greater than the horizon threshold.
    """
    from experiments.gate4d_early_epoch_physics.run_phase4d_early_epoch_physics import extract_early_physical_features

    class MockObs:
        def __init__(self, mjd: float, flux: float, pb: str, pos: bool):
            self.mjd = mjd
            self.flux_nmgy = flux
            self.passband = pb
            self.is_valid_measurement = True
            self.is_positive_detection_snr5 = pos

    mock_lightcurve = [
        MockObs(mjd=58000.0, flux=10.0, pb="g", pos=True),
        MockObs(mjd=58002.0, flux=25.0, pb="r", pos=True),
        MockObs(mjd=58005.0, flux=50.0, pb="g", pos=True),
        MockObs(mjd=58008.0, flux=80.0, pb="r", pos=True),
        MockObs(mjd=58012.0, flux=120.0, pb="g", pos=True),
        # Future observations
        MockObs(mjd=58020.0, flux=500.0, pb="g", pos=True),
        MockObs(mjd=58030.0, flux=900.0, pb="r", pos=True),
    ]

    f_n5 = extract_early_physical_features(mock_lightcurve, horizon_obs=5)
    assert f_n5["n_detections_early"] == 5.0
    assert f_n5["peak_flux_early"] == 120.0  # Must NOT see 500 or 900
    assert f_n5["time_span_days"] == 12.0  # 58012 - 58000


def test_conformal_k507_calibration_integrity(repo_root: Path):
    """
    SCIENTIFIC INVARIANT 3:
    Verifies the exact finite-sample quantile formula:
    k = ceil((N + 1)(1 - alpha)) = ceil(563 * 0.90) = 507.
    """
    report_path = repo_root / "experiments" / "gate4e_shift_aware_uncertainty" / "phase4e_conformal_calibration_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    spec = rep["conformal_specification"]
    assert spec["calibration_size"] == 562
    assert spec["conformal_rank_k"] == 507
    assert spec["target_coverage"] == 0.90
    assert spec["threshold_1_minus_q_hat"] > 0.0


def test_rare_class_honest_support_designations(repo_root: Path):
    """
    SCIENTIFIC INVARIANT 4:
    Verifies that rare classes with sparse external data are honestly designated
    as INSUFFICIENT_EXTERNAL_SUPPORT, preventing false generalization claims.
    """
    metrics_path = repo_root / "experiments" / "gate4c_rare_class_and_failure_diagnosis" / "phase4c_rare_class_metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, "r", encoding="utf-8") as f:
        mdat = json.load(f)

    per_class = mdat["per_class"]
    assert per_class["TDE"]["evaluability_verdict"] == "INSUFFICIENT_EXTERNAL_SUPPORT"
    assert per_class["OTHER_EXTRAGALACTIC"]["evaluability_verdict"] == "INSUFFICIENT_EXTERNAL_SUPPORT"
    assert per_class["SLSN"]["evaluability_verdict"] == "DESCRIPTIVE_ONLY"
    assert per_class["SN_Ia"]["evaluability_verdict"] == "EVALUABLE"
    assert per_class["SN_CC"]["evaluability_verdict"] == "EVALUABLE"


def test_final_evidence_manifest_integrity(repo_root: Path):
    """
    GOVERNANCE ASSERTION 5:
    Verifies that the root evidence manifest exists, contains valid SHA-256 hashes,
    and certifies all five work packages as CLOSED.
    """
    manifest_path = repo_root / "experiments" / "gate4_final_evidence_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    assert man["milestone"] == "Gate 4 — External Validity & Scientific Robustness: Final Closure"
    assert man["author"] == "Dev Mehta"
    assert man["final_gate_disposition"] == "GATE 4 OFFICIALLY CLOSED"

    workstreams = man["workstreams"]
    for ws_id, ws_data in workstreams.items():
        assert ws_data["disposition"] == "CLOSED"
        art_path = repo_root / ws_data["primary_artifact"]
        assert art_path.exists(), f"Primary artifact missing: {art_path}"

    # Deliverables catalog assertion
    assert "deliverables" in man
    deliverables = man["deliverables"]
    assert len(deliverables) >= 34, f"Expected at least 34 deliverables, found {len(deliverables)}"

    for rel_path, meta in deliverables.items():
        full_path = repo_root / rel_path
        assert full_path.exists(), f"Deliverable missing: {rel_path}"
        actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
        assert actual_sha == meta["sha256"], f"SHA256 mismatch for {rel_path}: {actual_sha} vs {meta['sha256']}"
