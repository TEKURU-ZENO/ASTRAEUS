"""
ASTRAEUS Phase 5: Scientific Reconciliation & Evidence Integrity Test Suite.
Verifies Workstreams 1 through 7 reconciliation, honest hypothesis falsification testing,
governance invariants, and cryptographic manifest integrity for all 28 assets.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_phase5_final_manifest_cryptographic_seal(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 1: Phase 5 Final Evidence Manifest Cryptographic Integrity.
    Verifies that phase5_final_evidence_manifest.json exists, seals all 28 assets
    (including the two execution scripts), and matches the independent SHA-256 anchor.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    manifest_path = p5_dir / "phase5_final_evidence_manifest.json"
    anchor_path = p5_dir / "phase5_final_evidence_manifest.sha256"

    assert manifest_path.exists(), f"Missing final manifest: {manifest_path}"
    assert anchor_path.exists(), f"Missing final manifest anchor: {anchor_path}"

    # Verify anchor digest matches manifest
    actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    anchor_content = anchor_path.read_text(encoding="utf-8").strip()
    assert actual_manifest_sha in anchor_content

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_identifier"] == "ASTRAEUS-P5-FINAL-CLOSURE-EVIDENCE-001"
    assert data["author"] == "Dev Mehta"
    assert data["governance_invariants"]["gate3_quarantine_test_objects"] == 586
    assert data["governance_invariants"]["gate3_quarantine_test_healpix_pixels"] == 1808

    # Verify all 28 sealed assets (including execution scripts)
    hashes = data["artifact_sha256_digests"]
    assert len(hashes) == 28, f"Expected 28 sealed assets, got {len(hashes)}"
    assert "run_phase5_domain_adaptation.py" in hashes
    assert "run_phase5_uncertainty_and_validation.py" in hashes

    for filename, expected_hash in hashes.items():
        fp = p5_dir / filename
        assert fp.exists(), f"Sealed asset missing from disk: {filename}"
        actual_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"


def test_phase5_ws5_domain_adaptation_results(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 2: Workstream 5 Domain Adaptation Verification.
    Verifies evaluation cohort size N_eval >= 85, CORAL covariance alignment gain,
    falsification of H2 (physical features underperformed raw representations),
    and unmet status of H3 (LOSO retention 84.2% < 90% threshold).
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    results_path = p5_dir / "phase5_domain_adaptation_results.parquet"
    summary_path = p5_dir / "phase5_domain_adaptation_summary.json"

    assert results_path.exists()
    assert summary_path.exists()

    df_results = pd.read_parquet(results_path)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Evaluation size protected
    assert len(df_results) >= 85, f"Evaluation split N={len(df_results)} violates power threshold N >= 85!"
    assert summary["eval_size"] >= 85

    # 2. Experiment 5C: CORAL adaptation improves target Macro-F1 over zero-shot in simulation
    zero_shot_f1 = summary["experiments"]["exp5c_domain_adaptation"]["zero_shot_baseline"]
    coral_f1 = summary["experiments"]["exp5c_domain_adaptation"]["coral_alignment"]
    assert coral_f1 >= zero_shot_f1, f"CORAL {coral_f1} failed to match zero-shot {zero_shot_f1}!"

    # 3. Experiment 5B: Falsification of Hypothesis H2
    delta_f1 = summary["experiments"]["exp5b_physical_vs_raw"]["delta_macro_f1"]
    assert delta_f1 < 0.0, f"Expected negative delta for physical vs raw in this benchmark, got {delta_f1}"
    assert summary["experiments"]["exp5b_physical_vs_raw"]["hypothesis_h2_supported"] is False

    # 4. Experiment 5D: Hypothesis H3 preregistered threshold unmet
    retention = summary["experiments"]["exp5d_loso_cross_survey"]["retention_ratio"]
    assert retention < 0.90, f"Expected retention < 0.90 boundary, got {retention}"
    assert summary["experiments"]["exp5d_loso_cross_survey"]["hypothesis_h3_supported"] is False


def test_phase5_ws6_conformal_calibration_and_shift_detection(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 3: Workstream 6 Uncertainty, Conformal Coverage & Shift Robustness.
    Verifies that empirical coverage in simulation satisfies >= 90%, shift detection AUROC > 0.60 (H4),
    and selective prediction risk curve non-monotonicity is accurately reported (H5 not strictly supported).
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    metrics_path = p5_dir / "phase5_uncertainty_metrics.parquet"
    results_path = p5_dir / "phase5_hypothesis_testing_results.json"

    assert metrics_path.exists()
    assert results_path.exists()

    df_metrics = pd.read_parquet(metrics_path)
    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    # 1. Conformal Coverage Check in Simulation (target 90%)
    coverage = res["conformal_calibration"]["empirical_coverage"]
    assert coverage >= 0.90, f"Conformal coverage {coverage*100:.1f}% below target 90%!"
    assert res["conformal_calibration"]["mean_set_size"] < 3.0
    assert "exchangeability_qualification" in res["conformal_calibration"]

    # 2. Covariate Shift Detection AUROC (Hypothesis H4 - Label-Conditioned Synthetic Demonstration)
    auroc = res["hypothesis_h4_shift_detection"]["misclassification_auroc"]
    assert auroc > 0.60, f"Shift AUROC {auroc} failed threshold > 0.60!"
    assert res["hypothesis_h4_shift_detection"]["supported"] is False
    assert res["hypothesis_h4_shift_detection"]["metric"] == "Label-Conditioned Synthetic Confidence Demonstration"
    assert "NOT_VALID_EVIDENCE" in res["hypothesis_h4_shift_detection"]["disposition"]

    # 3. Selective Abstention Monotonicity (Hypothesis H5)
    r_full = res["hypothesis_h5_selective_abstention"]["risk_full_coverage"]
    r_half = res["hypothesis_h5_selective_abstention"]["risk_half_coverage"]
    assert r_half < r_full, f"Risk at c=0.5 ({r_half}) not lower than c=1.0 ({r_full})!"
    # Strict monotonicity across all deciles is false due to 50% -> 60% dip
    assert res["hypothesis_h5_selective_abstention"]["monotonic_risk_reduction"] is False
    assert res["hypothesis_h5_selective_abstention"]["supported"] is False
    assert "CRITERION_UNMET" in res["hypothesis_h5_selective_abstention"]["disposition"]

    # 4. Hypothesis H2 BCa and Percentile Bootstrap Verification
    h2_boot = res["hypothesis_h2_bootstrap"]
    assert "ci_95_bca" in h2_boot
    assert "ci_95_percentile" in h2_boot
    assert h2_boot["supported"] is False
    assert h2_boot["ci_95_bca"][0] > h2_boot["ci_95_percentile"][0]

    # 5. Exact Paired McNemar Binomial Test
    assert res["mcnemar_paired_test"]["exact_binomial_p"] == 0.25
    assert "asymptotic_power_qualification" in res["mcnemar_paired_test"]


def test_phase5_research_paper_and_governance_invariants(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 4: Publication Manuscript and Governance Invariants.
    Verifies PHASE5_RESEARCH_PAPER.md and PHASE5_FINAL_SCIENTIFIC_CLOSURE_REPORT.md.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    paper_path = p5_dir / "PHASE5_RESEARCH_PAPER.md"
    closure_path = p5_dir / "PHASE5_FINAL_SCIENTIFIC_CLOSURE_REPORT.md"

    assert paper_path.exists()
    assert closure_path.exists()

    paper_text = paper_path.read_text(encoding="utf-8")
    closure_text = closure_path.read_text(encoding="utf-8")

    assert "Dev Mehta" in paper_text
    assert "Dev Mehta" in closure_text
    assert "466c2752808be01e4f41bd8fdb9c0804a57b03b2c02c35e5172ccebda411fce5" in paper_text
    assert "Zero-Touch Scientific Quarantine" in paper_text
    assert "Zero-Touch Scientific Quarantine" in closure_text
    assert "CORAL" in paper_text
    assert "Conformal Prediction" in paper_text
    assert "FALSIFIED IN BENCHMARK" in paper_text
    assert "FALSIFIED IN BENCHMARK" in closure_text
    assert "EMPIRICAL EVALUATION PENDING" in paper_text
    assert "EMPIRICAL EVALUATION PENDING" in closure_text


def test_phase5_all_workstreams_reconciled_status(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 5: Reconciled Workstream 1 through 7 Status Verification.
    Asserts workstreams are honestly classified in the master manifest.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    manifest_path = p5_dir / "phase5_final_evidence_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workstreams = data["sealed_workstreams"]
    for ws_id in ["WS1", "WS2", "WS3", "WS3.1", "WS4", "WS5", "WS6", "WS7"]:
        assert ws_id in workstreams
        status_str = workstreams[ws_id]
        assert any(k in status_str for k in ("CLOSED", "FROZEN", "PENDING", "BENCHMARK", "SYNTHETIC", "RECONCILED", "VALIDATED"))


def test_phase5_ws6_bca_bootstrap_independent_numerical_validation(repo_root: Path):
    """
    MASTER AUDIT ASSERTION 6: Independent Numerical Validation of BCa Bootstrap.
    Verifies that bias correction parameter z0, jackknife acceleration a,
    adjusted quantiles (alpha1, alpha2), and resulting intervals match an
    independent reference calculation from first principles with fixed random seed.
    """
    from scipy import stats
    from sklearn.metrics import f1_score
    from experiments.phase5_cross_survey_generalization.run_phase5_uncertainty_and_validation import compute_bca_interval

    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    df_res = pd.read_parquet(p5_dir / "phase5_domain_adaptation_results.parquet")

    y_true = df_res["true_label"].values
    y_pred_zero = df_res["pred_atlas_zero_shot"].values
    y_pred_raw = df_res["pred_atlas_raw"].values
    n_eval = len(y_true)

    theta_hat = float(f1_score(y_true, y_pred_zero, average="macro", zero_division=0) - f1_score(y_true, y_pred_raw, average="macro", zero_division=0))
    assert abs(theta_hat - (-0.0407925856)) < 1e-5

    # Run fixed seed bootstrap
    np.random.seed(42)
    n_boot = 500
    boot_deltas = np.zeros(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = np.random.choice(n_eval, size=n_eval, replace=True)
        boot_deltas[b] = f1_score(y_true[idx], y_pred_zero[idx], average="macro", zero_division=0) - f1_score(y_true[idx], y_pred_raw[idx], average="macro", zero_division=0)

    # 1. Independent z0
    prop_less = float(np.mean(boot_deltas < theta_hat))
    ref_z0 = float(stats.norm.ppf(prop_less))

    # 2. Independent jackknife acceleration a
    jack_deltas = np.zeros(n_eval, dtype=np.float64)
    all_idx = np.arange(n_eval)
    for i in range(n_eval):
        jack_idx = np.delete(all_idx, i)
        jack_deltas[i] = f1_score(y_true[jack_idx], y_pred_zero[jack_idx], average="macro", zero_division=0) - f1_score(y_true[jack_idx], y_pred_raw[jack_idx], average="macro", zero_division=0)

    u = np.mean(jack_deltas) - jack_deltas
    ref_a = float(np.sum(u ** 3) / (6.0 * (np.sum(u ** 2) ** 1.5)))

    # 3. Independent adjusted quantiles
    z_low = float(stats.norm.ppf(0.025))
    z_high = float(stats.norm.ppf(0.975))
    ref_a1 = float(stats.norm.cdf(ref_z0 + (ref_z0 + z_low) / (1.0 - ref_a * (ref_z0 + z_low))))
    ref_a2 = float(stats.norm.cdf(ref_z0 + (ref_z0 + z_high) / (1.0 - ref_a * (ref_z0 + z_high))))

    ref_bca_low = float(np.percentile(boot_deltas, 100.0 * ref_a1))
    ref_bca_high = float(np.percentile(boot_deltas, 100.0 * ref_a2))

    # Call module implementation
    bca_low, bca_high, perc_low, perc_high = compute_bca_interval(
        theta_hat, boot_deltas, y_true, y_pred_zero, y_pred_raw, alpha=0.05
    )

    assert abs(bca_low - ref_bca_low) < 1e-7
    assert abs(bca_high - ref_bca_high) < 1e-7
    assert bca_high <= 0.0
    assert perc_low < perc_high
