"""
ASTRAEUS WS8.9–WS9.9: Independent Scientific Validation & Integrity Closure Tests.
Validates the complete remediation of all 7 critical areas identified by the independent scientific review:
1. Fail-closed physical reasoning and corrected SN2006gy shock physics.
2. Canonical hypothesis hashing and non-trivial manifest digests.
3. Bayesian PPC with true posterior parameter draws and forward propagation.
4. Evidence-driven multi-agent research collective with non-boolean skeptic veto.
5. Non-circular independent numerical ODE solver in replication network.
6. Continuous multiscale physics, Barnes thermalization, and external Kasen/Barnes benchmark.
7. Dynamically derived cross-domain invariants and quantitative negative-transfer detection.
8. Offline method optimization with real executable task evaluations.
9. Withheld discovery pre-registration and blind adjudication.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

import copy
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
)
from astraeus_data.meta_learning.offline_method_optimizer import (
    OfflineMethodOptimizer,
    ScientificMethodConfig,
    OnlineWeightUpdateProhibitedError,
)
from astraeus_bench.discovery.astro_benchmarks import (
    PreRegisteredBlindDiscoveryProtocol,
    LightCurvePoint,
)


# ==============================================================================
# 1. Physical Reasoning: Fail-Closed Unit Path & SN2006gy Shock Formulation
# ==============================================================================

def test_ws8_9_physical_reasoner_fail_closed_and_shock_physics():
    reasoner = PhysicalReasoningEngine()

    # A. Unknown units must strictly fail-closed as INDETERMINATE_UNIT
    valid_lhs, msg_lhs = reasoner.verify_dimensional_consistency(
        lhs_unit="uncataloged_flux_unit",
        rhs_unit_terms=[("cm", 2), ("s", -1)],
    )
    assert not valid_lhs
    assert "INDETERMINATE_UNIT" in msg_lhs

    valid_rhs, msg_rhs = reasoner.verify_dimensional_consistency(
        lhs_unit="erg/s",
        rhs_unit_terms=[("cm", 2), ("unknown_exotic_unit", 1)],
    )
    assert not valid_rhs
    assert "INDETERMINATE_UNIT" in msg_rhs

    # B. Old uncorrected SN2006gy shock expression (mass/time, NOT energy/time)
    # rho_0 * v_shock^3 * t^2 -> (g/cm^3) * (cm/s)^3 * s^2 = g/s = [M T^-1]
    is_old_valid, old_msg = reasoner.verify_dimensional_consistency(
        lhs_unit="erg/s",
        rhs_unit_terms=[("g/cm^3", 1), ("cm/s", 3), ("s", 2)],
    )
    assert not is_old_valid
    assert "Dimensional mismatch" in old_msg

    # C. Corrected SN2006gy shock expression (energy/time)
    # rho_0 * v_shock^5 * t^2 -> (g/cm^3) * (cm/s)^5 * s^2 = g * cm^2 / s^3 = erg/s = [M L^2 T^-3]
    is_corr_valid, corr_msg = reasoner.verify_dimensional_consistency(
        lhs_unit="erg/s",
        rhs_unit_terms=[("g/cm^3", 1), ("cm/s", 5), ("s", 2)],
    )
    assert is_corr_valid
    assert "Dimensional consistency verified" in corr_msg


# ==============================================================================
# 2. Canonical Hypothesis Hashing & Manifest Integrity
# ==============================================================================

def test_ws8_9_canonical_hypothesis_hashing_and_manifest_integrity():
    generator = HypothesisGenerationEngine()

    h = StructuredHypothesis(
        hypothesis_id="hyp_test_integrity_v1",
        phenomenon="Optical-to-NIR fast transient",
        explanation="Radioactive r-process kilonova emission",
        assumptions=["Isotropic expansion"],
        mathematical_formulation="L_bol(t) = M_ej * q(t)",
        parameters={"ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun")},
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=20.5,
                uncertainty_range=(19.8, 21.2),
                epoch_days_from_trigger=1.0,
                measurement_condition="g-band filter",
            )
        ],
        alternatives=["hyp_test_shock_v1"],
        falsification_conditions=["Persistent blue color at t > 3d"],
        discriminating_experiment="NIR Ks imaging at t=3d",
    )

    base_hash = h.compute_hash()
    assert base_hash != ""
    # Ensure neither empty digest nor single-char digest
    assert base_hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert not base_hash.startswith("4b2277")

    # Mutating predictions MUST alter hash
    h_mutated_pred = h.model_copy(deep=True)
    h_mutated_pred.predictions[0].predicted_value = 21.8
    assert h_mutated_pred.compute_hash() != base_hash

    # Mutating alternatives MUST alter hash
    h_mutated_alt = h.model_copy(deep=True)
    h_mutated_alt.alternatives = ["hyp_alternative_magnetar_v2"]
    assert h_mutated_alt.compute_hash() != base_hash


# ==============================================================================
# 3. Bayesian PPC with True Posterior Parameter Draws
# ==============================================================================

def test_ws8_9_bayesian_posterior_predictive_check_with_parameter_draws():
    engine = ComputationalExperimentEngine(random_seed=20261125)

    # Ingest synthetic observations with true parameters M=0.04 Msun, v=0.18c
    epochs = [1.0, 2.0, 3.0, 4.0]
    filter_wls = {"g": 4770.0, "Ks": 21500.0}
    obs = engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        epochs_days=epochs,
        filter_wavelengths=filter_wls,
        provenance=DataProvenanceTag.SIMULATED_TESTBED,
        inject_noise=True,
        photometric_noise_mag=0.05,
    )

    # Recover parameters
    recovery = engine.perform_inverse_parameter_recovery(
        observations=obs,
        grid_resolution=15,
    )

    # Run true posterior draw predictive check
    ppc = engine.evaluate_parameter_posterior_predictive_check(
        model_name="Kilonova_TwoComponent_PosteriorDraws",
        observed_data=obs,
        posterior_results=recovery,
        epochs_days=epochs,
        filter_wavelengths=filter_wls,
        n_replications=100,
    )

    assert ppc.parameter_draws_propagated is True
    assert 0.05 <= ppc.bayesian_p_value <= 0.95
    assert ppc.ppc_passed is True
    assert ppc.empirical_coverage_95 == 1.0


# ==============================================================================
# 4. Evidence-Driven Multi-Agent Research Collective & Evidentiary Skeptic Veto
# ==============================================================================

def test_ws9_9_evidence_driven_multi_agent_research_collective():
    collective = ScientificResearchCollective(pi_name="Dev Mehta")

    h_valid = StructuredHypothesis(
        hypothesis_id="hyp_valid_kilonova_v1",
        phenomenon="Fast optical fading with NIR plateau",
        explanation="Decompression of neutron star matter driving r-process",
        assumptions=["Subluminal expansion", "Energy conservation"],
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
        alternatives=["hyp_competing_magnetar_v1"],
        falsification_conditions=["Radio upper limit < 1 uJy at t=10d"],
        discriminating_experiment="NIR Ks imaging at t=3d",
    )

    # Case A: Decisive empirical evidence supporting candidate over alternative
    evidence_strong = {
        "observations": [{"observable_name": "optical_mag_g", "value": 20.25, "uncertainty": 0.08}],
        "bayes_factor_vs_null": 45.0,
        "delta_bic": 12.5,
        "emd_nats": 3.2,
        "alternative_likelihoods": {"hyp_competing_magnetar_v1": 2.1},  # candidate delta-bic 12.5 >> 2.1
    }

    res_endorse = collective.conduct_deliberation(
        investigation_id="inv_test_consensus",
        hypothesis=h_valid,
        empirical_evidence=evidence_strong,
        adversarial_skeptic_mode=False,
    )
    assert res_endorse.is_consensus_reached is True
    assert res_endorse.is_vetoed_by_skeptic is False
    assert res_endorse.final_verdict == "ENDORSED_FOR_SCIENTIFIC_PUBLICATION"

    # Case B: Evidence-driven Skeptic Veto when an alternative hypothesis fits equally well (Delta-BIC <= 2.0)
    evidence_ambiguous = {
        "observations": [{"observable_name": "optical_mag_g", "value": 20.25, "uncertainty": 0.08}],
        "bayes_factor_vs_null": 15.0,
        "delta_bic": 7.0,
        "alternative_likelihoods": {"hyp_competing_magnetar_v1": 6.5},  # |7.0 - 6.5| = 0.5 <= 2.0!
    }

    res_skeptic_veto = collective.conduct_deliberation(
        investigation_id="inv_test_skeptic_veto",
        hypothesis=h_valid,
        empirical_evidence=evidence_ambiguous,
        adversarial_skeptic_mode=False,
    )
    assert res_skeptic_veto.is_consensus_reached is False
    assert res_skeptic_veto.is_vetoed_by_skeptic is True
    assert "SKEPTIC VETO" in str(res_skeptic_veto.veto_details)
    assert res_skeptic_veto.final_verdict == "REJECTED_OR_BLOCKED_BY_PEER_REVIEW"

    # Case C: Theoretical Physicist Veto on uncorrected dimensional error
    h_bad_dimensions = h_valid.model_copy(deep=True)
    h_bad_dimensions.mathematical_formulation = "L_shock(t) = 0.5 * (4 * pi * rho_0 * v_shock^3 * t^2) / (1 + t / t_diff)"
    h_bad_dimensions.sha256_hash = h_bad_dimensions.compute_hash()

    res_theorist_veto = collective.conduct_deliberation(
        investigation_id="inv_test_theorist_veto",
        hypothesis=h_bad_dimensions,
        empirical_evidence=evidence_strong,
    )
    assert res_theorist_veto.is_consensus_reached is False
    theorist_critique = [c for c in res_theorist_veto.agent_critiques if c.agent_role == AgentRole.THEORETICAL_PHYSICIST][0]
    assert theorist_critique.has_blocking_veto is True
    assert "Dimensional mismatch" in str(theorist_critique.veto_reason)


# ==============================================================================
# 5. Non-Circular Independent Numerical Replication Network
# ==============================================================================

def test_ws9_9_non_circular_independent_numerical_replication():
    replication_net = ScientificReplicationNetwork(numerical_tolerance_epsilon=1e-4)
    independent_solver = IndependentNumericalKilonovaSolver()

    # Claimed values computed from primary forward simulator
    claimed = {
        "mag_g_t1": 14.073,
        "mag_Ks_t3": 17.134,
    }
    input_params = {
        "ejecta_mass_msun": 0.04,
        "expansion_velocity_c": 0.18,
        "lanthanide_mass_fraction": 0.01,
    }

    # A. Execute independent RK4 ODE integration
    ode_results = independent_solver.solve_trajectory(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        epochs_days=[1.0, 3.0],
        filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
    )
    assert "mag_g_t1" in ode_results
    assert "mag_Ks_t3" in ode_results

    # B. Test independent solver replication mode (non-circular physical agreement)
    audit_ode = replication_net.audit_research_package(
        investigation_id="inv_rep_ode",
        target_claim_id="claim_kilonova_gw170817",
        claimed_results=claimed,
        input_parameters=input_params,
        random_seed=42,
        claimed_evidence_hash="sha256_mock_evidence",
        actual_evidence_hash="sha256_mock_evidence",
        use_independent_numerical_solver=True,
    )
    assert audit_ode.is_independent_numerical_solver is True
    assert audit_ode.is_fully_replicated is True
    assert audit_ode.max_relative_discrepancy <= 0.50  # bounded physical discrepancy

    # C. Cryptographic provenance verification with real raw records
    raw_records = [{"obs_id": 1, "mag": 18.84, "filter": "g"}, {"obs_id": 2, "mag": 17.56, "filter": "Ks"}]
    import json, hashlib
    expected_hash = hashlib.sha256(json.dumps(raw_records, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    # Matching raw evidence
    audit_prov_ok = replication_net.audit_research_package(
        investigation_id="inv_rep_prov_ok",
        target_claim_id="claim_kilonova_gw170817",
        claimed_results=claimed,
        input_parameters=input_params,
        random_seed=42,
        claimed_evidence_hash=expected_hash,
        actual_evidence_hash="",
        raw_evidence_records=raw_records,
        use_independent_numerical_solver=False,
    )
    assert audit_prov_ok.provenance_hash_verified is True
    assert audit_prov_ok.replication_status == ReplicationStatusOutcome.REPRODUCED_EXACT

    # Tampered raw evidence
    tampered_records = copy.deepcopy(raw_records)
    tampered_records[0]["mag"] = 12.0
    audit_prov_fail = replication_net.audit_research_package(
        investigation_id="inv_rep_prov_fail",
        target_claim_id="claim_kilonova_gw170817",
        claimed_results=claimed,
        input_parameters=input_params,
        random_seed=42,
        claimed_evidence_hash=expected_hash,
        actual_evidence_hash="",
        raw_evidence_records=tampered_records,
        use_independent_numerical_solver=False,
    )
    assert audit_prov_fail.provenance_hash_verified is False
    assert audit_prov_fail.replication_status == ReplicationStatusOutcome.FAILED_PROVENANCE_MISMATCH
    assert audit_prov_fail.is_fully_replicated is False


# ==============================================================================
# 6. Continuous Multiscale Physics & External Reference Benchmark
# ==============================================================================

def test_ws9_9_multiscale_continuous_uncertainty_and_external_benchmarks():
    simulator = MultiScaleSimulator()

    # A. Continuous opacity transition and Barnes thermalization
    state = simulator.step_multiscale_evolution(
        epoch_days=1.5,
        parameters={"ejecta_mass_msun": 0.04, "velocity_c": 0.18, "electron_fraction_ye": 0.20},
    )
    assert 0.0 < state.thermalization_efficiency <= 1.0
    assert 0.01 <= state.surrogate_uncertainty_fraction <= 0.06
    assert state.is_surrogate_extrapolating is False

    # B. External reference model comparison (Kasen & Barnes 2019)
    report = simulator.validate_against_reference_models()
    assert report.is_physically_consistent is True
    assert report.mean_relative_error < 0.08
    assert report.reference_data_points == 5


# ==============================================================================
# 7. Dynamically Derived Cross-Domain Invariants & Negative Transfer Detection
# ==============================================================================

def test_ws9_9_cross_domain_invariants_and_negative_transfer_measurement():
    cross_engine = CrossDomainIntelligenceEngine()

    # A. Dynamic invariant derivation
    res_transient = cross_engine.evaluate_multi_messenger_domain(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        peak_luminosity_erg_s=1e41,
    )
    assert res_transient.invariants.dynamical_to_cooling_timescale_ratio > 0.0
    assert res_transient.invariants.eddington_ratio > 0.0

    res_exoplanet = cross_engine.evaluate_exoplanet_domain(
        planet_radius_earth=1.0,
        stellar_radius_solar=1.0,
        equilibrium_temp_k=288.0,
    )
    assert res_exoplanet.invariants.gravitational_compactness < 1e-8
    assert res_exoplanet.invariants.dynamical_to_cooling_timescale_ratio > 0.0

    # B. Quantitative Negative Transfer Detection
    transfer_report = cross_engine.evaluate_cross_domain_transfer(
        source_domain=ScientificDomain.MULTI_MESSENGER_TRANSIENTS,
        target_domain=ScientificDomain.EXOPLANETS_ASTROBIOLOGY,
        source_invariants=res_transient.invariants,
        target_invariants=res_exoplanet.invariants,
    )
    assert transfer_report.is_negative_transfer_detected is True
    assert transfer_report.is_transfer_admissible is False
    assert transfer_report.domain_discrepancy_metric > 3.0
    assert len(transfer_report.diagnostic_divergences) >= 2


# ==============================================================================
# 8. Offline Method Optimization with Real Executable Task Evaluations
# ==============================================================================

def test_ws9_9_offline_method_optimization_real_task_execution():
    optimizer = OfflineMethodOptimizer(is_online_stream_active=False)

    baseline = ScientificMethodConfig(
        method_id="method_planner_eig",
        method_name="PredictiveEIGPlanner",
        version="1.0.0",
        hyperparameters={"search_tree_depth": 2, "discount_factor": 0.95},
        historical_benchmark_scores={"task_kilonova_planning": 80.0, "task_supernova_planning": 81.0},
    )
    optimizer.register_operational_method(baseline)

    # Real evaluation callable
    def real_task_evaluator(hyperparams):
        depth = hyperparams.get("search_tree_depth", 2)
        # E.g. simulated real information gain
        return 80.0 + (depth - 2) * 3.5

    tasks = [
        {"task_id": "task_kilonova_planning", "evaluator": real_task_evaluator},
        {"task_id": "task_supernova_planning", "historical_episodes": [{"weight": 1.2}, {"weight": 0.9}]},
    ]

    candidate_improved = ScientificMethodConfig(
        method_id="method_planner_eig",
        method_name="PredictiveEIGPlanner",
        version="1.1.0",
        parent_version="1.0.0",
        hyperparameters={"search_tree_depth": 3, "discount_factor": 0.95},
    )

    promoted, msg, scores = optimizer.evaluate_candidate_improvement(
        candidate=candidate_improved,
        benchmark_tasks=tasks,
        max_allowable_regression_pct=0.50,
    )
    assert promoted is True
    assert scores["task_kilonova_planning"] == 83.5
    assert candidate_improved.regression_detected is False


# ==============================================================================
# 9. Withheld Discovery Pre-Registration Protocol
# ==============================================================================

def test_ws9_9_withheld_discovery_pre_registration_protocol():
    protocol = PreRegisteredBlindDiscoveryProtocol()

    h = StructuredHypothesis(
        hypothesis_id="hyp_blind_eval_at2017gfo",
        phenomenon="GW170817 electromagnetic counterpart",
        explanation="Decompression of cold neutron star matter",
        assumptions=["Spherically symmetric"],
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
                predicted_value=20.2,
                uncertainty_range=(19.8, 20.6),
                epoch_days_from_trigger=2.0,
                measurement_condition="g-band filter",
            ),
        ],
        alternatives=["hyp_csm_v1"],
        falsification_conditions=["No reddening after 48h"],
        discriminating_experiment="NIR Ks imaging",
    )

    # 1. Commit pre-registration before seeing withheld observations
    commitment = protocol.create_pre_registration_commitment(
        target_id="AT2017gfo",
        hypothesis=h,
    )
    assert commitment.commitment_id.startswith("commit_AT2017gfo")
    assert commitment.frozen_hypothesis_hash != ""

    # 2. Unblind withheld test observations
    withheld_obs = [
        LightCurvePoint(mjd=57982.5, filter_id="g", magnitude=17.48, uncertainty_mag=0.08, facility_id="SWOPE", instrument_id="CAM"),
        LightCurvePoint(mjd=57984.0, filter_id="g", magnitude=20.22, uncertainty_mag=0.10, facility_id="SWOPE", instrument_id="CAM"),
    ]

    # 3. Adjudicate
    audit = protocol.adjudicate_withheld_discovery(
        commitment=commitment,
        withheld_observations=withheld_obs,
    )
    assert audit.was_committed_before_unblinding is True
    assert audit.coverage_fraction == 1.0
    assert audit.adjudication_passed is True
    assert audit.discovery_outcome == "VALIDATED_PREDICTIVE_DISCOVERY"
