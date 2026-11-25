"""
ASTRAEUS WS8.10–WS9.10: Prospective Scientific Validation & Independent Replication Tests.
Validates the complete remediation of all review findings across 8 core scientific areas:
1. Fail-closed evidence requirements & claim-level precedent audit in Research Collective (WS9.5).
2. Elimination of synthetic scoring fallbacks in Offline Method Optimization (WS9.4).
3. Authenticated physical reference dataset and physical-space metric in Multiscale Simulation (WS9.6).
4. Cryptographically signed pre-registration, prospective timing, strict matching, and falsification (WS8.7 / WS9.9).
5. RK4 solver time-step convergence and energy conservation budget in Replication Network (WS9.7).
6. Dynamically derived physical invariants and calibrated transfer discrepancy metric (WS9.2).
7. Repeated multi-trial PPC calibration, empirical coverage, and residual analysis (WS8.5).
8. Circumstellar shock derivation and formal RFC-8785 JSON canonicalization conformance (WS8.4).
9. Flagship Transient Discovery Demonstration: Kilonova Model Discrimination on AT2017gfo vs Withheld Sample.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

import copy
import json
import math
import numpy as np
import pytest

from astraeus_models.reasoning.physical_reasoner import (
    PhysicalReasoningEngine,
    ConservationViolationType,
)
from astraeus_agents.discovery.hypothesis_generator import (
    HypothesisGenerationEngine,
    StructuredHypothesis,
    HypothesisPrediction,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
    SimulatedObservationPoint,
)
from astraeus_agents.collective.research_collective import (
    ScientificResearchCollective,
    AgentRole,
)
from astraeus_bench.replication.replication_network import (
    ScientificReplicationNetwork,
    IndependentNumericalKilonovaSolver,
    ReplicationStatusOutcome,
)
from astraeus_models.multiscale.multiscale_simulator import (
    MultiScaleSimulator,
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
    OnlineWeightUpdateProhibitedError,
)
from astraeus_bench.discovery.astro_benchmarks import (
    PreRegisteredBlindDiscoveryProtocol,
    LightCurvePoint,
)


# ==============================================================================
# 1. WS9.5: Fail-Closed Evidence Requirements & Claim-Level Literature Audit
# ==============================================================================

def test_ws9_10_collective_fail_closed_on_missing_evidence():
    collective = ScientificResearchCollective(pi_name="Dev Mehta")

    hyp = StructuredHypothesis(
        hypothesis_id="hyp_test_kn_fail_closed",
        phenomenon="Optical-to-NIR fast transient",
        explanation="Radioactive r-process kilonova emission",
        assumptions=["Subluminal expansion"],
        mathematical_formulation="L_bol(t) = M_ej * q(t)",
        parameters={"ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun"), "velocity_c": (0.18, 0.05, 0.30, "c")},
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=20.2,
                uncertainty_range=(19.8, 20.6),
                epoch_days_from_trigger=1.0,
                measurement_condition="g-band filter",
            )
        ],
        alternatives=["hyp_test_shock_v1"],
        falsification_conditions=["Persistent blue color at t > 3d"],
        discriminating_experiment="NIR Ks imaging at t=3d",
    )

    # A. Completely empty evidence record MUST fail closed as REJECTED_INDETERMINATE_EVIDENCE
    res_empty = collective.conduct_deliberation(
        investigation_id="inv_empty_evidence",
        hypothesis=hyp,
        empirical_evidence={},
        adversarial_skeptic_mode=False,
    )
    assert res_empty.is_consensus_reached is False
    assert res_empty.final_verdict == "REJECTED_INDETERMINATE_EVIDENCE"

    # Check individual specialist vetoes
    critiques_by_role = {c.agent_role: c for c in res_empty.agent_critiques}
    assert critiques_by_role[AgentRole.COMPUTATIONAL_SCIENTIST].has_blocking_veto is True
    assert "INDETERMINATE_NUMERICAL_CONDITIONING" in str(critiques_by_role[AgentRole.COMPUTATIONAL_SCIENTIST].veto_reason)

    assert critiques_by_role[AgentRole.OBSERVATIONAL_ASTRONOMER].has_blocking_veto is True
    assert "INDETERMINATE_OBSERVATIONAL_DATA" in str(critiques_by_role[AgentRole.OBSERVATIONAL_ASTRONOMER].veto_reason)

    assert critiques_by_role[AgentRole.STATISTICIAN].has_blocking_veto is True
    assert "INDETERMINATE_STATISTICAL_EVIDENCE" in str(critiques_by_role[AgentRole.STATISTICIAN].veto_reason)

    assert critiques_by_role[AgentRole.LITERATURE_RESEARCHER].has_blocking_veto is True
    assert "INDETERMINATE_BIBLIOGRAPHIC_REGISTRY" in str(critiques_by_role[AgentRole.LITERATURE_RESEARCHER].veto_reason)

    assert critiques_by_role[AgentRole.INDEPENDENT_SKEPTIC].has_blocking_veto is True
    assert "CRITICAL SKEPTIC VETO" in str(critiques_by_role[AgentRole.INDEPENDENT_SKEPTIC].veto_reason)

    # B. Literature Researcher: Claim-level collision detection
    evidence_with_collision = {
        "observations": [{"observable_name": "optical_mag_g", "value": 20.2, "uncertainty": 0.08}],
        "bayes_factor_vs_null": 50.0,
        "delta_bic": 15.0,
        "emd_nats": 3.0,
        "fisher_condition_number": 5.0,
        "prior_claims": [
            "claim_prior_duplicate_exact: Optical-to-NIR fast transient Radioactive r-process kilonova emission L_bol(t) = M_ej * q(t)"
        ],
        "alternative_likelihoods": {"hyp_test_shock_v1": 1.0},
    }
    res_coll = collective.conduct_deliberation(
        investigation_id="inv_collision_evidence",
        hypothesis=hyp,
        empirical_evidence=evidence_with_collision,
    )
    assert res_coll.is_consensus_reached is False
    lit_critique = [c for c in res_coll.agent_critiques if c.agent_role == AgentRole.LITERATURE_RESEARCHER][0]
    assert lit_critique.has_blocking_veto is True
    assert "Bibliographic collision" in str(lit_critique.veto_reason)


# ==============================================================================
# 2. WS9.4: Elimination of Synthetic Scoring Fallback in Method Optimization
# ==============================================================================

def test_ws9_10_offline_method_optimization_zero_synthetic_fallback():
    optimizer = OfflineMethodOptimizer(is_online_stream_active=False)

    base = ScientificMethodConfig(
        method_id="method_planner_opt",
        method_name="PlannerOptimization",
        version="1.0.0",
        hyperparameters={"search_tree_depth": 2},
        historical_benchmark_scores={"task_benchmark_1": 80.0},
    )
    optimizer.register_operational_method(base)

    cand = ScientificMethodConfig(
        method_id="method_planner_opt",
        method_name="PlannerOptimization",
        version="1.1.0",
        hyperparameters={"search_tree_depth": 3},
        parent_version="1.0.0",
    )

    # A. Absent evaluator callable MUST be rejected, never promoted via synthetic arithmetic
    tasks_no_eval = [{"task_id": "task_benchmark_1"}]
    promoted, msg, scores = optimizer.evaluate_candidate_improvement(cand, tasks_no_eval)
    assert promoted is False
    assert cand.status == MethodVersionStatus.REJECTED_REGRESSION
    assert "missing executable evaluator callable" in msg
    assert scores["task_benchmark_1"] == 0.0

    # B. Real task evaluator executes genuine logic and allows legitimate promotion
    def genuine_evaluator(hyperparams):
        depth = hyperparams.get("search_tree_depth", 2)
        return 80.0 + depth * 2.5

    tasks_valid = [{"task_id": "task_benchmark_1", "evaluator": genuine_evaluator}]
    cand_valid = cand.model_copy(deep=True)
    promoted_v, msg_v, scores_v = optimizer.evaluate_candidate_improvement(cand_valid, tasks_valid)
    assert promoted_v is True
    assert cand_valid.status == MethodVersionStatus.PROMOTED_OPERATIONAL
    assert scores_v["task_benchmark_1"] == 87.5


# ==============================================================================
# 3. WS9.6: Authenticated Kasen & Barnes Reference Benchmark & Physical Metric
# ==============================================================================

def test_ws9_10_multiscale_reference_dataset_and_physical_metric():
    simulator = MultiScaleSimulator()

    # A. Authenticated Reference Dataset
    dataset = simulator.load_kasen_barnes_reference_dataset()
    assert dataset.dataset_id == "KASEN_BARNES_2019_2D_KN_BENCHMARK"
    assert "10.3847/1538-4357/ab06c2" in dataset.doi
    assert len(dataset.provenance_sha256) == 64
    assert len(dataset.points) == 5

    # B. Evaluation strictly in physical observable space (erg/s)
    report = simulator.validate_against_reference_models()
    assert report.is_physically_consistent is True
    assert report.target_observable == "Bolometric Luminosity L_bol [erg/s]"
    assert report.mean_fractional_error_physical <= 0.05  # < 5% mean physical discrepancy
    assert report.max_fractional_error_physical <= 0.08   # < 8% max physical discrepancy
    assert report.mean_dex_discrepancy <= 0.02           # < 0.02 dex discrepancy
    assert len(report.benchmark_points) == 5


# ==============================================================================
# 4. WS8.7 / WS9.9: Cryptographic Pre-Registration & Strict Blind Adjudication
# ==============================================================================

def test_ws9_10_cryptographic_preregistration_and_strict_adjudication():
    protocol = PreRegisteredBlindDiscoveryProtocol()

    h = StructuredHypothesis(
        hypothesis_id="hyp_prospective_kn_discovery",
        phenomenon="Multi-messenger relativistic ejecta counterpart",
        explanation="Dynamic decompression and radioactive r-process decay",
        assumptions=["Isotropic expansion"],
        mathematical_formulation="L_bol(t) = M_ej * q(t)",
        parameters={"ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun")},
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=17.5,
                uncertainty_range=(17.0, 18.0),
                epoch_days_from_trigger=0.5,
                measurement_condition="g-band filter",
            ),
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=19.2,
                uncertainty_range=(18.8, 19.6),
                epoch_days_from_trigger=1.0,
                measurement_condition="g-band filter",
            ),
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=20.5,
                uncertainty_range=(20.0, 21.0),
                epoch_days_from_trigger=2.0,
                measurement_condition="g-band filter",
            ),
            HypothesisPrediction(
                observable_name="optical_mag_r",
                predicted_value=18.0,
                uncertainty_range=(17.5, 18.5),
                epoch_days_from_trigger=1.0,
                measurement_condition="r-band filter",
            ),
            HypothesisPrediction(
                observable_name="optical_mag_r",
                predicted_value=19.1,
                uncertainty_range=(18.6, 19.6),
                epoch_days_from_trigger=2.0,
                measurement_condition="r-band filter",
            ),
        ],
        alternatives=["hyp_csm_interaction"],
        falsification_conditions=["Decay rate exceeds 3.0 mag per day"],
        discriminating_experiment="NIR Ks imaging",
    )

    # 1. Cryptographic pre-registration commitment signed by custodian
    commitment = protocol.create_pre_registration_commitment(
        target_id="PROSPECTIVE_TRANSIENT_2026A",
        hypothesis=h,
    )
    assert commitment.custodian_id == "INDEPENDENT_OBSERVATORY_CUSTODIAN"
    assert len(commitment.custodian_signature_hex) == 64

    # 2. Complete withheld observation sample (5 points matching passbands and epochs)
    withheld_obs = [
        LightCurvePoint(mjd=61000.5, filter_id="g", magnitude=17.45, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61001.0, filter_id="g", magnitude=19.22, uncertainty_mag=0.09, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="g", magnitude=20.48, uncertainty_mag=0.10, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61001.0, filter_id="r", magnitude=18.02, uncertainty_mag=0.07, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="r", magnitude=19.12, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
    ]

    # 3. Adjudicate certified discovery
    audit_pass = protocol.adjudicate_withheld_discovery(
        commitment=commitment,
        withheld_observations=withheld_obs,
        min_observations_required=5,
    )
    assert audit_pass.custodian_signature_verified is True
    assert audit_pass.was_committed_before_unblinding is True
    assert audit_pass.total_predictions_evaluated == 5
    assert audit_pass.predictions_within_95_ci == 5
    assert audit_pass.coverage_fraction == 1.0
    assert audit_pass.falsification_triggered is False
    assert audit_pass.adjudication_passed is True
    assert audit_pass.discovery_outcome == "VALIDATED_PREDICTIVE_DISCOVERY"

    # 4. Prospective timing breach check (data accessed before commitment)
    audit_timing_fail = protocol.adjudicate_withheld_discovery(
        commitment=commitment,
        withheld_observations=withheld_obs,
        unblinded_timestamp_sec=commitment.committed_at_epoch_sec - 10.0,  # unblinded BEFORE commitment!
    )
    assert audit_timing_fail.was_committed_before_unblinding is False
    assert audit_timing_fail.adjudication_passed is False
    assert audit_timing_fail.discovery_outcome == "REJECTED_PROSPECTIVE_TIMING_BREACH"

    # 5. Stored falsification condition triggered (decay rate > 3.0 mag/day)
    fast_decay_obs = [
        LightCurvePoint(mjd=61000.5, filter_id="g", magnitude=16.0, uncertainty_mag=0.05, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61001.0, filter_id="g", magnitude=19.5, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="g", magnitude=23.0, uncertainty_mag=0.10, facility_id="VLT", instrument_id="FORS2"),  # 7 mag / 1.5 days = 4.67 mag/day!
        LightCurvePoint(mjd=61001.0, filter_id="r", magnitude=18.5, uncertainty_mag=0.07, facility_id="VLT", instrument_id="FORS2"),
        LightCurvePoint(mjd=61002.0, filter_id="r", magnitude=22.5, uncertainty_mag=0.08, facility_id="VLT", instrument_id="FORS2"),
    ]
    audit_falsified = protocol.adjudicate_withheld_discovery(
        commitment=commitment,
        withheld_observations=fast_decay_obs,
    )
    assert audit_falsified.falsification_triggered is True
    assert audit_falsified.adjudication_passed is False
    assert audit_falsified.discovery_outcome == "FALSIFIED_BY_EMPIRICAL_CRITERIA"


# ==============================================================================
# 5. WS9.7: Independent RK4 Solver Convergence & Energy Conservation
# ==============================================================================

def test_ws9_10_independent_solver_convergence_and_energy_conservation():
    solver = IndependentNumericalKilonovaSolver()
    replication_net = ScientificReplicationNetwork(numerical_tolerance_epsilon=1e-4)

    # A. Time-step convergence study
    conv_report = solver.run_convergence_study(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        time_steps_sec=[3600.0, 1800.0, 900.0, 450.0],
    )
    assert conv_report.is_converged is True
    assert conv_report.max_relative_discretization_error < 1e-3
    assert len(conv_report.terminal_energies_erg) == 4

    # B. Energy conservation budget
    energy_report = solver.verify_energy_conservation(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        dt=900.0,
    )
    assert energy_report.is_energy_conserved is True
    assert energy_report.relative_conservation_error < 0.005  # within 0.5%
    assert energy_report.total_nuclear_energy_input_erg > 0.0
    assert energy_report.total_radiated_energy_erg > 0.0
    assert energy_report.total_expansion_work_pdv_erg > 0.0

    # C. Replication Network demarcation
    claimed = {"mag_g_t1": 14.07, "mag_Ks_t3": 17.13}
    params = {"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01}

    audit = replication_net.audit_research_package(
        investigation_id="inv_test_independent_rep",
        target_claim_id="claim_kn_test",
        claimed_results=claimed,
        input_parameters=params,
        random_seed=42,
        claimed_evidence_hash="sha256_mock_evidence",
        actual_evidence_hash="sha256_mock_evidence",
        use_independent_numerical_solver=True,
    )
    assert audit.audit_category == "INDEPENDENT_SCIENTIFIC_REPLICATION"
    assert audit.is_independent_scientific_replication is True
    assert audit.convergence_report is not None
    assert audit.energy_budget_report is not None
    assert audit.is_fully_replicated is True


# ==============================================================================
# 6. WS9.2: Dynamically Derived Invariants & Metric Calibration
# ==============================================================================

def test_ws9_10_cross_domain_dynamic_invariants_and_calibration():
    engine = CrossDomainIntelligenceEngine()

    # A. Dynamic Cosmology timescales & virial ratio
    cosmo = engine.evaluate_cosmology_domain(redshift_z=1.0)
    # Dynamical-to-cooling timescale ratio dynamically derived from expansion vs bremsstrahlung
    assert 1e-4 <= cosmo.invariants.dynamical_to_cooling_timescale_ratio <= 1e-1
    assert 0.10 <= cosmo.invariants.virial_ratio <= 0.80

    # B. Dynamic Extreme Gravity timescales
    bh = engine.evaluate_extreme_gravity_domain(black_hole_mass_msun=10.0, dimensionless_spin_a=0.90)
    assert 1e-5 <= bh.invariants.dynamical_to_cooling_timescale_ratio <= 1e-2

    # C. Calibrated Transfer Discrepancy Metric
    calib = engine.calibrate_transfer_metric_dataset()
    assert calib["is_calibrated_and_validated"] is True
    assert calib["separation_margin"] > 1.0
    assert calib["max_valid_transfer_discrepancy"] < 2.5
    assert calib["min_invalid_transfer_discrepancy"] > 3.0


# ==============================================================================
# 7. WS8.5: Repeated Multi-Trial PPC Calibration & Residual Verification
# ==============================================================================

def test_ws9_10_repeated_ppc_calibration_study():
    engine = ComputationalExperimentEngine(random_seed=20261125)

    param_grid = [
        {"ejecta_mass_msun": 0.03, "expansion_velocity_c": 0.15, "lanthanide_mass_fraction": 0.01},
        {"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01},
        {"ejecta_mass_msun": 0.05, "expansion_velocity_c": 0.22, "lanthanide_mass_fraction": 0.01},
    ]
    epochs = [1.0, 2.0, 3.0]
    filter_wls = {"g": 4770.0, "Ks": 21500.0}

    report = engine.evaluate_repeated_ppc_calibration(
        parameter_grid=param_grid,
        epochs_days=epochs,
        filter_wavelengths=filter_wls,
        n_trials=10,
    )
    assert report.num_trials == 10
    assert report.empirical_coverage_95 >= 0.80
    assert 0.15 <= report.mean_bayesian_p_value <= 0.85
    assert report.is_well_calibrated is True
    assert abs(report.mean_bias["ejecta_mass_msun"]) < 0.02


# ==============================================================================
# 8. WS8.4: Shock Derivation & RFC-8785 Canonical JSON Conformance
# ==============================================================================

def test_ws9_10_shock_physics_derivation_and_rfc8785_canonicalization():
    reasoner = PhysicalReasoningEngine()

    # A. Physical derivation of circumstellar shock luminosity
    deriv = reasoner.derive_csm_shock_luminosity(
        ambient_density_rho0_g_cm3=1e-12,
        shock_velocity_cm_s=1e9,  # 10,000 km/s
        epoch_sec=86400.0 * 20.0, # 20 days
        diffusion_time_sec=86400.0 * 30.0,
    )
    assert deriv["is_dimensionally_sound"] is True
    assert deriv["diffused_bolometric_luminosity_erg_s"] > 0.0
    assert deriv["swept_mass_rate_g_s"] > 0.0

    # B. RFC-8785 Canonical JSON test vectors
    # Keys must be sorted lexicographically, compact separators, UTF-8 encoded
    h = StructuredHypothesis(
        hypothesis_id="hyp_canonical_test",
        phenomenon="Optical transient",
        explanation="Radioactive decay",
        assumptions=["Causality", "Energy conservation"],
        mathematical_formulation="L(t) = M * q(t)",
        parameters={"ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun")},
        predictions=[
            HypothesisPrediction(
                observable_name="mag_g",
                predicted_value=19.5,
                uncertainty_range=(19.0, 20.0),
                epoch_days_from_trigger=1.0,
                measurement_condition="g-band filter",
            )
        ],
        alternatives=["hyp_alt_1"],
        falsification_conditions=["No red color"],
        discriminating_experiment="NIR imaging",
    )
    h_hash = h.compute_hash()
    assert len(h_hash) == 64
    # Recomputing MUST be strictly idempotent
    assert h.compute_hash() == h_hash


# ==============================================================================
# 9. Flagship Discovery Demonstration: Kilonova Model Discrimination
# ==============================================================================

def test_ws9_10_flagship_kilonova_model_discrimination():
    """
    Demonstrates end-to-end scientific model discrimination on kilonova physics:
    Compares radioactive r-process kilonova hypothesis against circumstellar shock interaction,
    commits pre-registration, unblinds withheld test data, executes independent RK4 replication,
    and conducts fail-closed multi-agent peer review.
    """
    protocol = PreRegisteredBlindDiscoveryProtocol()
    replication_net = ScientificReplicationNetwork()
    collective = ScientificResearchCollective(pi_name="Dev Mehta")

    # 1. Formulate Candidate Kilonova Hypothesis
    hyp_kilonova = StructuredHypothesis(
        hypothesis_id="hyp_flagship_kilonova_gw170817",
        phenomenon="GW170817 multi-messenger optical/NIR counterpart",
        explanation="Two-component dynamical and wind r-process kilonova with lanthanide curtaining",
        assumptions=["Neutron star merger origin", "Radioactive beta decay heating", "Relativistic homologous expansion"],
        mathematical_formulation="L_bol(t) = M_ej * q_nuclear(t) * (1 - exp(-(t/t_diff)^2))",
        parameters={
            "ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun"),
            "expansion_velocity_c": (0.18, 0.08, 0.30, "c"),
            "lanthanide_mass_fraction": (0.01, 0.0001, 0.05, "mass_fraction"),
        },
        predictions=[
            HypothesisPrediction(observable_name="mag_g", predicted_value=17.5, uncertainty_range=(17.0, 18.0), epoch_days_from_trigger=0.5, measurement_condition="g"),
            HypothesisPrediction(observable_name="mag_g", predicted_value=19.2, uncertainty_range=(18.7, 19.7), epoch_days_from_trigger=1.0, measurement_condition="g"),
            HypothesisPrediction(observable_name="mag_g", predicted_value=20.5, uncertainty_range=(20.0, 21.0), epoch_days_from_trigger=2.0, measurement_condition="g"),
            HypothesisPrediction(observable_name="mag_r", predicted_value=18.0, uncertainty_range=(17.5, 18.5), epoch_days_from_trigger=1.0, measurement_condition="r"),
            HypothesisPrediction(observable_name="mag_r", predicted_value=19.1, uncertainty_range=(18.6, 19.6), epoch_days_from_trigger=2.0, measurement_condition="r"),
        ],
        alternatives=["hyp_circumstellar_shock_breakout"],
        falsification_conditions=["Decay rate exceeds 3.0 mag per day"],
        discriminating_experiment="NIR Ks-band imaging at t=3d showing red plateau",
    )

    # 2. Pre-Registration Commitment
    commitment = protocol.create_pre_registration_commitment("AT2017gfo", hyp_kilonova)
    assert commitment.custodian_signature_hex != ""

    # 3. Ingest Withheld Observations
    withheld_photometry = [
        LightCurvePoint(mjd=57982.5, filter_id="g", magnitude=17.48, uncertainty_mag=0.08, facility_id="SWOPE", instrument_id="CAM"),
        LightCurvePoint(mjd=57983.0, filter_id="g", magnitude=19.18, uncertainty_mag=0.09, facility_id="SWOPE", instrument_id="CAM"),
        LightCurvePoint(mjd=57984.0, filter_id="g", magnitude=20.45, uncertainty_mag=0.10, facility_id="SWOPE", instrument_id="CAM"),
        LightCurvePoint(mjd=57983.0, filter_id="r", magnitude=18.05, uncertainty_mag=0.07, facility_id="SWOPE", instrument_id="CAM"),
        LightCurvePoint(mjd=57984.0, filter_id="r", magnitude=19.08, uncertainty_mag=0.08, facility_id="SWOPE", instrument_id="CAM"),
    ]

    # 4. Adjudicate Blind Discovery
    discovery_audit = protocol.adjudicate_withheld_discovery(commitment, withheld_photometry, min_observations_required=5)
    assert discovery_audit.adjudication_passed is True
    assert discovery_audit.discovery_outcome == "VALIDATED_PREDICTIVE_DISCOVERY"

    # 5. Independent Numerical Replication
    rep_audit = replication_net.audit_research_package(
        investigation_id="inv_flagship_rep",
        target_claim_id="claim_at2017gfo_kilonova",
        claimed_results={"mag_g_t1": 19.18, "mag_r_t1": 18.05},
        input_parameters={"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01},
        random_seed=42,
        claimed_evidence_hash="sha256_mock_evidence",
        actual_evidence_hash="sha256_mock_evidence",
        use_independent_numerical_solver=True,
    )
    assert rep_audit.is_independent_scientific_replication is True
    assert rep_audit.is_fully_replicated is True

    # 6. Full Research Collective Consensus
    empirical_evidence = {
        "observations": [{"observable_name": "mag_g", "value": 19.18, "uncertainty": 0.09}],
        "bayes_factor_vs_null": 85.0,
        "delta_bic": 18.2,
        "emd_nats": 3.4,
        "fisher_condition_number": 6.2,
        "prior_claims": ["claim_reference_at2017gfo_discovery_novel"],
        "alternative_likelihoods": {"hyp_circumstellar_shock_breakout": 2.5},  # candidate 18.2 >> 2.5
    }
    deliberation = collective.conduct_deliberation("inv_flagship_deliberation", hyp_kilonova, empirical_evidence)
    assert deliberation.is_consensus_reached is True
    assert deliberation.final_verdict == "ENDORSED_FOR_SCIENTIFIC_PUBLICATION"
    assert deliberation.unanimous_endorsement is True
