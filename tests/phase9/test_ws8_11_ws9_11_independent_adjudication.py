"""
ASTRAEUS WS8.11–WS9.11: Independent Adjudication, External Reference Authentication,
and Scientific Evidence Certification Test Suite.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

import math
import pytest
import numpy as np
from datetime import datetime, timezone

from astraeus_bench.discovery.astro_benchmarks import (
    ExternalCustodianAuthority,
    PreRegisteredBlindDiscoveryProtocol,
    PreRegistrationCommitment,
    PreRegisteredDiscoveryAudit,
    LightCurvePoint,
    compute_clopper_pearson_interval,
)
from astraeus_models.multiscale.multiscale_simulator import MultiScaleSimulator
from astraeus_bench.replication.replication_network import (
    IndependentNumericalKilonovaSolver,
    ScientificReplicationNetwork,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    SimulatedObservationPoint,
    DataProvenanceTag,
)
from astraeus_models.cross_domain.cross_domain_engine import (
    CrossDomainIntelligenceEngine,
    ScientificDomain,
    DimensionlessInvariants,
)
from astraeus_data.meta_learning.offline_method_optimizer import (
    OfflineMethodOptimizer,
    ScientificMethodConfig,
    MethodVersionStatus,
)
from astraeus_agents.collective.research_collective import (
    ScientificResearchCollective,
    AgentRole,
)
from astraeus_agents.discovery.hypothesis_generator import (
    StructuredHypothesis,
    HypothesisPrediction,
)


# ==============================================================================
# 1. WS8.11: External Custodian Authority & Tamper-Evident Append-Only Ledger
# ==============================================================================

def test_ws9_11_external_custodian_authority_and_tamper_evident_ledger(tmp_path):
    db_file = str(tmp_path / "custodian_audit_ledger.db")
    custodian = ExternalCustodianAuthority(
        custodian_id="INDEPENDENT_ESO_ARCHIVE_CUSTODIAN",
        ledger_db_path=db_file,
    )

    preds = [
        {"observable_name": "optical_mag_g", "filter_id": "g", "predicted_value": 17.5, "uncertainty_range": [17.0, 18.0], "epoch_days": 0.5},
        {"observable_name": "optical_mag_r", "filter_id": "r", "predicted_value": 18.0, "uncertainty_range": [17.5, 18.5], "epoch_days": 1.0},
    ]

    # 1. Register commitment in persistent append-only ledger
    commitment = custodian.register_commitment(
        target_id="PROSPECTIVE_KN_2026B",
        hypothesis_id="hyp_kilonova_model_v1",
        hypothesis_hash="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        predictions=preds,
        falsification_criteria=["Decay rate exceeds 3.0 mag per day"],
    )
    assert commitment.commitment_id == "commit_PROSPECTIVE_KN_2026B_hyp_kilonova_model_v1"
    assert len(commitment.custodian_signature_hex) == 64
    assert custodian.verify_commitment_signature(commitment) is True

    # 2. Access withheld data via custodian token
    unblind_epoch, token_sig = custodian.grant_unblinded_access("PROSPECTIVE_KN_2026B")
    assert unblind_epoch > commitment.committed_at_epoch_sec
    assert len(token_sig) == 64

    # 3. Verify access chronology and immutable hash chain
    chronology = custodian.verify_access_chronology("PROSPECTIVE_KN_2026B")
    assert chronology.is_valid is True
    assert chronology.was_committed_before_unblinding is True
    assert chronology.hash_chain_verified is True
    assert chronology.ledger_records_count == 2
    assert chronology.time_delta_sec > 0.0

    # 4. Access without prior commitment MUST fail closed
    with pytest.raises(ValueError, match="ACCESS DENIED"):
        custodian.grant_unblinded_access("UNREGISTERED_TARGET_2026X")


# ==============================================================================
# 2. WS8.11: Exact Clopper-Pearson Coverage Intervals & Classification
# ==============================================================================

def test_ws9_11_exact_clopper_pearson_coverage_intervals():
    # A. Single 5-point dataset evaluation (binomial CI [0.4782, 1.0000])
    ci_5 = compute_clopper_pearson_interval(5, 5, 0.95)
    assert ci_5[0] == pytest.approx(0.4782, abs=0.001)
    assert ci_5[1] == pytest.approx(1.0000, abs=0.001)

    # B. Extended 25-point benchmark evaluation (k=24/25, binomial CI [0.7965, 0.9989])
    ci_25 = compute_clopper_pearson_interval(24, 25, 0.95)
    assert ci_25[0] >= 0.79
    assert ci_25[1] <= 1.0

    # C. Adjudication with External Custodian
    custodian = ExternalCustodianAuthority(custodian_id="EXT_CUSTODIAN_TEST", allow_ephemeral_for_testing=True)
    protocol = PreRegisteredBlindDiscoveryProtocol(custodian=custodian)

    h = StructuredHypothesis(
        hypothesis_id="hyp_test_coverage",
        phenomenon="Kilonova transient",
        explanation="Radioactive decay",
        mathematical_formulation="L_bol(t) = Q_dot(t)",
        parameters={"M_ej": (0.04, 0.01, 0.10, "M_sun")},
        assumptions=["Causal expansion"],
        predictions=[
            HypothesisPrediction(observable_name="g", predicted_value=17.5, uncertainty_range=(17.0, 18.0), epoch_days_from_trigger=0.5, measurement_condition="VLT FORS2 g-band"),
            HypothesisPrediction(observable_name="g", predicted_value=19.2, uncertainty_range=(18.8, 19.6), epoch_days_from_trigger=1.0, measurement_condition="VLT FORS2 g-band"),
            HypothesisPrediction(observable_name="g", predicted_value=20.5, uncertainty_range=(20.0, 21.0), epoch_days_from_trigger=2.0, measurement_condition="VLT FORS2 g-band"),
            HypothesisPrediction(observable_name="r", predicted_value=18.0, uncertainty_range=(17.5, 18.5), epoch_days_from_trigger=1.0, measurement_condition="VLT FORS2 r-band"),
            HypothesisPrediction(observable_name="r", predicted_value=19.1, uncertainty_range=(18.6, 19.6), epoch_days_from_trigger=2.0, measurement_condition="VLT FORS2 r-band"),
        ],
        alternatives=["hyp_csm"],
        falsification_conditions=["Decay rate exceeds 3.0 mag per day"],
        discriminating_experiment="NIR imaging",
    )
    commitment = protocol.create_pre_registration_commitment("TARGET_COVERAGE_2026", h)
    custodian.grant_unblinded_access("TARGET_COVERAGE_2026")

    obs = [
        LightCurvePoint(mjd=61000.5, filter_id="g", magnitude=17.45, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61001.0, filter_id="g", magnitude=19.22, uncertainty_mag=0.09, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="g", magnitude=20.48, uncertainty_mag=0.10, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61001.0, filter_id="r", magnitude=18.02, uncertainty_mag=0.07, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="r", magnitude=19.12, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
    ]
    audit = protocol.adjudicate_withheld_discovery(commitment, obs)
    assert audit.adjudication_passed is True
    assert audit.coverage_binomial_95_ci[0] == pytest.approx(0.4782, abs=0.001)
    assert audit.access_chronology_verified is True
    assert audit.scientific_classification == "PROSPECTIVE_PREDICTIVE_AGREEMENT_ON_SUPPLIED_TEST_RECORD"


# ==============================================================================
# 3. WS9.11: External Reference Dataset on Disk & Sensitivity Grid
# ==============================================================================

def test_ws9_11_external_reference_dataset_provenance_and_parameter_sensitivity():
    sim = MultiScaleSimulator()

    # A. Loads external JSON file from disk with exact DOI and full provenance
    dataset = sim.load_kasen_barnes_reference_dataset()
    assert dataset.dataset_id == "KASEN_BARNES_2019_2D_KN_BENCHMARK"
    assert dataset.doi == "10.3847/1538-4357/ab06c2"
    assert len(dataset.provenance_sha256) == 64
    assert len(dataset.points) == 5

    # B. Physical relative error in bolometric space [erg/s]
    rep = sim.validate_against_reference_models()
    assert rep.is_physically_consistent is True
    assert rep.mean_fractional_error_physical <= 0.05
    assert rep.mean_dex_discrepancy <= 0.02

    # C. Parameter sensitivity grid evaluation
    sens = sim.validate_parameter_sensitivity_grid()
    assert sens["is_minimum_at_benchmark_configuration"] is True
    assert sens["optimal_parameters"] == {"ejecta_mass_msun": 0.04, "velocity_c": 0.18}
    assert sens["optimal_mean_physical_error"] <= 0.05
    # Off-peak parameter combinations must exhibit larger errors
    off_peak_errs = [
        g["mean_relative_error_physical"] for g in sens["grid_evaluations"]
        if (g["ejecta_mass_msun"], g["velocity_c"]) != (0.04, 0.18)
    ]
    assert all(e > sens["optimal_mean_physical_error"] for e in off_peak_errs)


# ==============================================================================
# 4. WS9.11: RK4 Exact Terminal Time & Richardson Convergence Order
# ==============================================================================

def test_ws9_11_independent_rk4_exact_terminal_time_and_richardson_convergence():
    solver = IndependentNumericalKilonovaSolver()

    # A. Convergence study with identical terminal epoch across all step sizes
    steps = [3600.0, 1800.0, 900.0, 450.0, 225.0]
    conv = solver.run_convergence_study(time_steps_sec=steps)

    assert conv.is_converged is True
    assert conv.terminal_epoch_exact_sec == pytest.approx(3.0 * 86400.0, abs=1e-3)
    assert conv.asymptotic_convergence_order >= 3.80  # Confirms expected O(dt^4) convergence
    assert conv.fine_reference_energy_erg > 0.0
    assert conv.richardson_extrapolated_energy_erg > 0.0
    assert conv.max_relative_discretization_error < 1e-3

    # B. Exact energy conservation balance
    eb = solver.verify_energy_conservation(dt=900.0)
    assert eb.is_energy_conserved is True
    assert eb.relative_conservation_error < 0.005  # < 0.5% error


# ==============================================================================
# 5. WS8.11: Bayesian PPC Parameter Covariance & Misspecification Rejection
# ==============================================================================

def test_ws9_11_ppc_bivariate_covariance_and_misspecification_rejection():
    engine = ComputationalExperimentEngine(random_seed=42)
    epochs = [0.5, 1.0, 2.0]
    flts = {"g": 4770.0, "r": 6230.0}

    # A. Well-specified data with preserved covariance
    obs_valid = engine.forward_simulate_kilonova(0.04, 0.18, 0.01, epochs, flts, inject_noise=True, photometric_noise_mag=0.03)
    recovery_valid = engine.perform_inverse_parameter_recovery(obs_valid, grid_resolution=15)

    ppc_valid = engine.evaluate_parameter_posterior_predictive_check(
        model_name="Kilonova_Covariant_PPC",
        observed_data=obs_valid,
        posterior_results=recovery_valid,
        epochs_days=epochs,
        filter_wavelengths=flts,
        n_replications=80,
        correlation_coeff=-0.65,
    )
    assert ppc_valid.ppc_passed is True
    assert ppc_valid.covariance_structure_propagated is True
    assert 0.05 <= ppc_valid.bayesian_p_value <= 0.95

    # B. Intentionally misspecified data: PPC MUST detect and reject misspecification
    misspec_result = engine.evaluate_model_misspecification_detection(epochs, flts)
    assert misspec_result["misspecification_detected"] is True
    assert misspec_result["bayesian_p_value"] < 0.05


# ==============================================================================
# 6. WS9.11: Pre-Frozen Held-Out Cross-Domain Benchmark (FPR = 0, FNR = 0)
# ==============================================================================

def test_ws9_11_held_out_cross_domain_matrix_fpr_fnr_zero():
    engine = CrossDomainIntelligenceEngine()

    benchmark_report = engine.evaluate_held_out_transfer_benchmark(divergence_threshold=3.0)

    assert benchmark_report["pre_frozen_threshold"] == 3.0
    assert benchmark_report["total_benchmark_cases"] == 12
    assert benchmark_report["true_positives"] == 6
    assert benchmark_report["true_negatives"] == 6
    assert benchmark_report["false_positives"] == 0
    assert benchmark_report["false_negatives"] == 0
    assert benchmark_report["false_positive_rate"] == 0.0
    assert benchmark_report["false_negative_rate"] == 0.0
    assert benchmark_report["accuracy"] == 1.0
    assert benchmark_report["separation_margin"] > 1.0
    assert benchmark_report["is_statistically_calibrated"] is True


# ==============================================================================
# 7. WS9.11: Immutable Optimizer Baseline Provenance & Evaluator Verification
# ==============================================================================

def test_ws9_11_optimizer_immutable_baseline_provenance():
    optimizer = OfflineMethodOptimizer(is_online_stream_active=False)

    def test_evaluator(hyperparams):
        return 80.0 + hyperparams.get("depth", 2) * 2.0

    baseline = ScientificMethodConfig(
        method_id="method_opt_v1",
        method_name="BaselineMethod",
        version="1.0.0",
        hyperparameters={"depth": 2},
        historical_benchmark_scores={"task_1": 84.0},
        evaluator_identity="GenuineTaskEvaluator_v1",
    )
    optimizer.register_operational_method(baseline)
    assert baseline.baseline_provenance_hash is not None

    # Candidate with compatible evaluator promotes legitimately
    candidate_valid = ScientificMethodConfig(
        method_id="method_opt_v1",
        method_name="BaselineMethod",
        version="1.1.0",
        hyperparameters={"depth": 3},
        parent_version="1.0.0",
        evaluator_identity="GenuineTaskEvaluator_v1",
    )
    tasks_valid = [{"task_id": "task_1", "evaluator": test_evaluator, "evaluator_identity": "GenuineTaskEvaluator_v1"}]
    promoted, msg, scores = optimizer.evaluate_candidate_improvement(candidate_valid, tasks_valid)
    assert promoted is True
    assert scores["task_1"] == 86.0

    # Evaluator mismatch fails closed
    tasks_mismatch = [{"task_id": "task_1", "evaluator": test_evaluator, "evaluator_identity": "DifferentEvaluator_v2"}]
    promoted_m, msg_m, _ = optimizer.evaluate_candidate_improvement(candidate_valid, tasks_mismatch)
    assert promoted_m is False
    assert "Evaluator mismatch" in msg_m


# ==============================================================================
# 8. WS9.11: Research Collective AST Symbolic Verification & Likelihood Recheck
# ==============================================================================

def test_ws9_11_collective_ast_symbolic_and_likelihood_verification():
    collective = ScientificResearchCollective()

    # A. AST Symbolic dimensional mismatch detection
    hyp_flawed = StructuredHypothesis(
        hypothesis_id="hyp_flawed_kinetic_shock",
        phenomenon="Shock transient",
        explanation="CSM interaction",
        mathematical_formulation="L_shock = rho_0 * (v_shock^3) * (t^2)",  # Wrong exponent v^3!
        parameters={"v_shock": (0.1, 0.01, 0.5, "c")},
        assumptions=["Conservation"],
        predictions=[HypothesisPrediction(observable_name="g", predicted_value=18.0, uncertainty_range=(17.5, 18.5), epoch_days_from_trigger=1.0, measurement_condition="Optical g-band")],
        alternatives=["hyp_kn"],
        falsification_conditions=["No decay"],
        discriminating_experiment="Spectroscopy",
    )
    result_flawed = collective.deliberate(hyp_flawed, empirical_evidence={"prior_claims": ["claim_1"], "fisher_condition_number": 10.0, "delta_bic": 5.0})
    theorist_critique = next(c for c in result_flawed.agent_critiques if c.agent_role == AgentRole.THEORETICAL_PHYSICIST)
    assert theorist_critique.has_blocking_veto is True
    assert "Dimensional mismatch in shock formulation" in theorist_critique.veto_reason

    # B. Statistician catches falsified Delta-BIC contradicting raw likelihood reconstruction
    hyp_sound = StructuredHypothesis(
        hypothesis_id="hyp_sound_model",
        phenomenon="Kilonova",
        explanation="Decay",
        mathematical_formulation="L_shock = 2*pi*rho_0*(v_shock^5)*(t^2)",
        parameters={"v_shock": (0.1, 0.01, 0.5, "c")},
        assumptions=["Conservation"],
        predictions=[HypothesisPrediction(observable_name="optical_mag_g", predicted_value=18.0, uncertainty_range=(17.5, 18.5), epoch_days_from_trigger=1.0, measurement_condition="Optical g-band")],
        alternatives=["hyp_kn"],
        falsification_conditions=["No decay"],
        discriminating_experiment="Spectroscopy",
    )
    evidence_contradictory = {
        "prior_claims": ["claim_1"],
        "fisher_condition_number": 10.0,
        "delta_bic": 99.0,  # Falsified high Delta-BIC!
        "observations": [
            {"observable_name": "optical_mag_g", "value": 25.0, "uncertainty": 0.1},  # Severe residual: 25 vs 18
        ],
    }
    result_contradictory = collective.deliberate(hyp_sound, empirical_evidence=evidence_contradictory)
    stat_critique = next(c for c in result_contradictory.agent_critiques if c.agent_role == AgentRole.STATISTICIAN)
    assert stat_critique.has_blocking_veto is True
    assert "FALSIFIED_OR_UNRECONCILED_STATISTICAL_EVIDENCE" in stat_critique.veto_reason
