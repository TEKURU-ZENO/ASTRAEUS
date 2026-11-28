"""
ASTRAEUS WS8.12–WS9.12: Independent Evidence Audit, External Custody Service,
and Epistemic Calibration Test Suite.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import math
import pytest
import sqlite3
import numpy as np
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

from astraeus_bench.discovery.astro_benchmarks import (
    ExternalCustodianAuthority,
    PreRegisteredBlindDiscoveryProtocol,
    PreRegistrationCommitment,
    compute_clopper_pearson_interval,
)
from astraeus_models.multiscale.multiscale_simulator import (
    MultiScaleSimulator,
    ExternalReferenceDataset,
)
from astraeus_bench.replication.replication_network import (
    IndependentNumericalKilonovaSolver,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
    SimulatedObservationPoint,
)
from astraeus_models.cross_domain.cross_domain_engine import (
    CrossDomainIntelligenceEngine,
)
from astraeus_agents.discovery.hypothesis_generator import (
    StructuredHypothesis,
    HypothesisPrediction,
)


# ==============================================================================
# 1. External Custodian: Mandatory Disk Persistence & External Key Enforcement
# ==============================================================================

def test_ws9_12_custodian_mandatory_disk_persistence_and_external_key(tmp_path):
    # A. In-memory databases are strictly rejected unless explicitly allowed for testing
    with pytest.raises(ValueError, match="MANDATORY PERSISTENCE VIOLATION"):
        ExternalCustodianAuthority(ledger_db_path=":memory:", allow_ephemeral_for_testing=False)

    with pytest.raises(ValueError, match="MANDATORY PERSISTENCE VIOLATION"):
        ExternalCustodianAuthority(ledger_db_path=None, allow_ephemeral_for_testing=False)

    # B. External key enforcement
    with pytest.raises(ValueError, match="EXTERNAL KEY REQUIRED"):
        ExternalCustodianAuthority(
            custodian_private_key=None,
            ledger_db_path=str(tmp_path / "ledger.db"),
            require_external_key=True,
        )

    # C. Valid initialization with external key and durable disk path
    ext_key = b"0123456789abcdef0123456789abcdef"
    db_file = str(tmp_path / "valid_ledger.db")
    custodian = ExternalCustodianAuthority(
        custodian_id="ESO_EXTERNAL_AUDIT_AUTHORITY",
        custodian_private_key=ext_key,
        ledger_db_path=db_file,
        require_external_key=True,
    )
    assert custodian.custodian_id == "ESO_EXTERNAL_AUDIT_AUTHORITY"
    assert Path(db_file).exists()


# ==============================================================================
# 2. SQLite Triggers Enforce Append-Only Immutability
# ==============================================================================

def test_ws9_12_sqlite_triggers_enforce_append_only_immutability(tmp_path):
    db_file = str(tmp_path / "immutable_ledger.db")
    custodian = ExternalCustodianAuthority(
        custodian_id="TEST_IMMUTABILITY_CUSTODIAN",
        ledger_db_path=db_file,
    )

    commitment = custodian.register_commitment(
        target_id="TARGET_KN_2026_IMMUTABLE",
        hypothesis_id="hyp_immutability_test",
        hypothesis_hash="a" * 64,
        predictions=[{"observable": "g", "pred": 18.0}],
        falsification_criteria=["decay > 3.0"],
    )

    # Direct SQLite connection attempting to mutate or delete ledger records
    raw_conn = sqlite3.connect(db_file)
    cur = raw_conn.cursor()

    # A. Attempting UPDATE must be aborted by SQLite trigger
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY VIOLATION: UPDATE"):
        cur.execute("UPDATE custodian_ledger SET target_id = 'TAMPERED_TARGET' WHERE record_index = 1;")

    # B. Attempting DELETE must be aborted by SQLite trigger
    with pytest.raises(sqlite3.IntegrityError, match="IMMUTABILITY VIOLATION: DELETE"):
        cur.execute("DELETE FROM custodian_ledger WHERE record_index = 1;")

    raw_conn.close()


# ==============================================================================
# 3. Tamper Detection Recomputes Every Payload & Record Hash from Contents
# ==============================================================================

def test_ws9_12_tamper_detection_recomputes_payload_and_record_hashes(tmp_path):
    db_file = str(tmp_path / "tamper_test_ledger.db")
    custodian = ExternalCustodianAuthority(
        custodian_id="TEST_TAMPER_AUDIT",
        ledger_db_path=db_file,
    )

    c1 = custodian.register_commitment("TARGET_A", "hyp_1", "1" * 64, [{"p": 1}], ["f1"])
    custodian.grant_unblinded_access("TARGET_A")
    c2 = custodian.register_commitment("TARGET_B", "hyp_2", "2" * 64, [{"p": 2}], ["f2"])

    # 1. Unmodified ledger passes full audit
    is_valid, notes = custodian.verify_ledger_integrity()
    assert is_valid is True
    assert len(notes) == 0

    # 2. Simulate raw disk tampering (bypassing triggers via table drop/recreate or trigger drop)
    raw_conn = sqlite3.connect(db_file)
    raw_cur = raw_conn.cursor()
    raw_cur.execute("DROP TRIGGER prevent_custodian_ledger_update;")

    # A. Modify payload_json without updating hash -> PAYLOAD TAMPERING detected
    raw_cur.execute("UPDATE custodian_ledger SET payload_json = '{\"tampered\": true}' WHERE record_index = 1;")
    raw_conn.commit()
    raw_conn.close()

    is_valid, notes = custodian.verify_ledger_integrity()
    assert is_valid is False
    assert any("PAYLOAD TAMPERING" in n for n in notes)
    assert any("RECORD HASH MISMATCH" in n for n in notes)

    # B. Access chronology audit fails closed when ledger is tampered
    chrono = custodian.verify_access_chronology("TARGET_A")
    assert chrono.is_valid is False
    assert chrono.hash_chain_verified is False


# ==============================================================================
# 4. Clopper-Pearson Exact Interval: Strict Validation & Zero Wald Fallback
# ==============================================================================

def test_ws9_12_clopper_pearson_input_validation_and_fail_closed():
    # A. Input validations
    with pytest.raises(ValueError, match="Sample size n must be a positive integer"):
        compute_clopper_pearson_interval(0, 0)

    with pytest.raises(ValueError, match="Sample size n must be a positive integer"):
        compute_clopper_pearson_interval(1, -5)

    with pytest.raises(ValueError, match="Success count k must satisfy"):
        compute_clopper_pearson_interval(6, 5)

    with pytest.raises(ValueError, match="Success count k must satisfy"):
        compute_clopper_pearson_interval(-1, 5)

    with pytest.raises(ValueError, match="Confidence level must be strictly within"):
        compute_clopper_pearson_interval(5, 5, confidence=1.0)

    with pytest.raises(ValueError, match="Confidence level must be strictly within"):
        compute_clopper_pearson_interval(5, 5, confidence=0.0)

    with pytest.raises(TypeError, match="must be integers"):
        compute_clopper_pearson_interval(2.5, 5)

    # B. Exact interval values
    assert compute_clopper_pearson_interval(5, 5, 0.95) == (0.4782, 1.0000)
    assert compute_clopper_pearson_interval(4, 5, 0.95) == (0.2836, 0.9949)

    # C. Zero silent fallback: if Beta calculation fails, it MUST fail closed
    with patch("scipy.stats.beta.ppf", side_effect=Exception("SciPy Unavailable")):
        with pytest.raises(RuntimeError, match="EXACT COMPUTATION FAILURE"):
            compute_clopper_pearson_interval(5, 5, 0.95)


# ==============================================================================
# 5. Kasen & Barnes Reference: Standalone File & Zero Fallback on Missing Asset
# ==============================================================================

def test_ws9_12_reference_dataset_provenance_and_zero_fallback(tmp_path):
    sim = MultiScaleSimulator()

    # A. Cleanly loads reference dataset from disk
    ds = sim.load_kasen_barnes_reference_dataset()
    assert ds.dataset_id == "KASEN_BARNES_2019_2D_KN_BENCHMARK"
    assert ds.doi == "10.3847/1538-4357/ab06c2"
    assert len(ds.provenance_sha256) == 64
    assert len(ds.points) == 5

    # B. Physical consistency criteria in physical space (mean <= 5%, max <= 15%)
    rep = sim.validate_against_reference_models()
    assert rep.is_physically_consistent is True
    assert rep.mean_fractional_error_physical <= 0.05
    assert rep.max_fractional_error_physical <= 0.15

    # C. Missing asset strictly FAILS CLOSED (no silent hardcoded fallback)
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="CRITICAL PROVENANCE FAILURE"):
            sim.load_kasen_barnes_reference_dataset()


# ==============================================================================
# 6. Bayesian PPC: Draw-Conditioned Discrepancy & Degenerate Covariance Gate
# ==============================================================================

def test_ws9_12_bayesian_ppc_draw_conditioned_discrepancy():
    engine = ComputationalExperimentEngine(random_seed=42)
    epochs = [0.5, 1.0, 2.0]
    flts = {"g": 4770.0, "r": 6230.0}

    obs = engine.forward_simulate_kilonova(0.04, 0.18, 0.01, epochs, flts, inject_noise=True, photometric_noise_mag=0.03)
    recovery = engine.perform_inverse_parameter_recovery(obs, grid_resolution=15)

    # A. Well-specified check passes with parameter draws and covariance
    ppc = engine.evaluate_parameter_posterior_predictive_check(
        model_name="KN_Conditioned_PPC",
        observed_data=obs,
        posterior_results=recovery,
        epochs_days=epochs,
        filter_wavelengths=flts,
        n_replications=80,
        correlation_coeff=-0.65,
    )
    assert ppc.ppc_passed is True
    assert 0.05 <= ppc.bayesian_p_value <= 0.95
    assert ppc.observed_discrepancy_stat > 0.0

    # B. Degenerate non-positive definite covariance matrix MUST fail closed (no silent fallback)
    invalid_cov = np.array([[0.01, 0.05], [0.05, 0.01]])  # Det = 0.0001 - 0.0025 < 0 (negative eigenvalue)
    with pytest.raises(ValueError, match="POSTERIOR COVARIANCE DEGENERATE"):
        engine.evaluate_parameter_posterior_predictive_check(
            model_name="Degenerate_Cov_Test",
            observed_data=obs,
            posterior_results=recovery,
            epochs_days=epochs,
            filter_wavelengths=flts,
            posterior_covariance=invalid_cov,
        )


# ==============================================================================
# 7. Bayesian PPC: Ensemble Model Misspecification Rejection
# ==============================================================================

def test_ws9_12_bayesian_ppc_misspecification_ensemble():
    engine = ComputationalExperimentEngine(random_seed=123)
    epochs = [0.5, 1.0, 2.0]
    flts = {"g": 4770.0, "r": 6230.0}

    # Evaluate across 3 independent realizations of misspecified shock injection data
    for trial in range(3):
        res = engine.evaluate_model_misspecification_detection(epochs, flts)
        assert res["misspecification_detected"] is True
        assert res["bayesian_p_value"] < 0.05
        assert res["ppc_passed"] is False


# ==============================================================================
# 8. Cross-Domain Transfer: Standalone Disk Benchmark Asset Loading
# ==============================================================================

def test_ws9_12_held_out_cross_domain_disk_asset_verification():
    engine = CrossDomainIntelligenceEngine()

    # A. Evaluates 12 cases loaded directly from external JSON asset
    res = engine.evaluate_held_out_transfer_benchmark(divergence_threshold=3.0)
    assert res["total_benchmark_cases"] == 12
    assert res["true_positives"] == 6
    assert res["true_negatives"] == 6
    assert res["false_positives"] == 0
    assert res["false_negatives"] == 0
    assert res["false_positive_rate"] == 0.0
    assert res["false_negative_rate"] == 0.0
    assert res["accuracy"] == 1.0
    assert res["separation_margin"] > 4.0
    assert res["is_statistically_calibrated"] is True
    assert len(res["benchmark_asset_sha256"]) == 64

    # B. Missing benchmark asset fails closed
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="CRITICAL ASSET FAILURE"):
            engine.evaluate_held_out_transfer_benchmark()


# ==============================================================================
# 9. RK4 Solver Convergence: Regression Statistics & Truncation Bounds
# ==============================================================================

def test_ws9_12_rk4_solver_convergence_regression_statistics():
    solver = IndependentNumericalKilonovaSolver()
    report = solver.run_convergence_study()

    assert report.is_converged is True
    assert 3.80 <= report.asymptotic_convergence_order <= 4.20
    assert report.regression_r_squared >= 0.999
    assert report.order_standard_error < 0.05
    assert report.reference_truncation_error < 1e-6
    assert report.max_relative_discretization_error < 1e-3

    # Energy conservation check
    eb = solver.verify_energy_conservation(dt=900.0)
    assert eb.is_energy_conserved is True
    assert eb.relative_conservation_error < 0.005
