"""
ASTRAEUS Phase 9 - Generalizable Scientific Research Intelligence Test Suite.
Verifies WS9.1 through WS9.7:
- WS9.1: Scientific World Models (Latent Dynamics & Long-Horizon Forecasting)
- WS9.2: Cross-Domain Intelligence (Multi-Messenger, Cosmology, Exoplanets, Extreme Gravity)
- WS9.3: Autonomous Research Planning (Programme DAGs & Falsification Replanning)
- WS9.4: Self-Improving Scientific Methods (Offline Meta-Learning & Regression Gate)
- WS9.5: Multi-Agent Research Collective (9-Agent Consensus & Skeptic Veto Power)
- WS9.6: Multi-Scale Simulation Intelligence (Nuclear to Hydro to Radiative Transfer)
- WS9.7: Scientific Replication Network (Clean-Room Replay & Cryptographic Provenance)

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

import copy
import pytest
import numpy as np

from astraeus_models.world_models.scientific_world_model import (
    PhysicalDomainState,
    WorldModelTrajectory,
    ScientificWorldModel,
)
from astraeus_models.cross_domain.cross_domain_engine import (
    ScientificDomain,
    DimensionlessInvariants,
    DomainReasoningResult,
    CrossDomainIntelligenceEngine,
)
from astraeus_agents.research.research_planner import (
    ResearchNodeType,
    ResearchNodeStatus,
    ResearchDAGNode,
    ResearchProgrammeDAG,
    AutonomousResearchPlanner,
)
from astraeus_data.meta_learning.offline_method_optimizer import (
    OnlineWeightUpdateProhibitedError,
    MethodVersionStatus,
    ScientificMethodConfig,
    OfflineMethodOptimizer,
)
from astraeus_agents.collective.research_collective import (
    AgentRole,
    AgentCritique,
    CollectiveDeliberationResult,
    ScientificResearchCollective,
)
from astraeus_models.multiscale.multiscale_simulator import (
    ScaleLayer,
    MultiScaleState,
    MultiScaleSimulator,
)
from astraeus_bench.replication.replication_network import (
    ReplicationStatusOutcome,
    ReplicationAuditReport,
    ScientificReplicationNetwork,
)
from astraeus_agents.discovery.hypothesis_generator import (
    StructuredHypothesis,
    HypothesisPrediction,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
)


# ==============================================================================
# WS9.1: Scientific World Models Tests
# ==============================================================================

def test_ws9_1_scientific_world_models_long_horizon_prediction():
    wm = ScientificWorldModel(random_seed=42)

    init_state = PhysicalDomainState(
        epoch_days=0.5,
        state_vector=[1.5e14, 0.15 * 3e10, 8000.0, 5.0, 1e42],
        state_uncertainties=[1e13, 1e8, 200.0, 0.5, 5e40],
    )

    params = {"ejecta_mass_msun": 0.04, "opacity_cm2_g": 1.0}

    # Step simulation
    next_state = wm.step_dynamical_state(init_state, dt_days=0.5, system_parameters=params)
    assert next_state.epoch_days == 1.0
    assert next_state.state_vector[0] > init_state.state_vector[0]  # expanding radius
    assert next_state.conservation_residual < 0.05

    # Multi-day forecast trajectory
    traj = wm.forecast_trajectory(
        target_id="tgt_world_model_01",
        initial_state=init_state,
        horizon_days=5.0,
        step_days=0.5,
        system_parameters=params,
    )
    assert traj.is_physically_bounded
    assert len(traj.states) == 11
    assert traj.max_energy_violation_fraction < 0.05


# ==============================================================================
# WS9.2: Cross-Domain Intelligence Tests
# ==============================================================================

def test_ws9_2_cross_domain_scientific_intelligence():
    engine = CrossDomainIntelligenceEngine()

    # 1. Multi-Messenger Transient Domain
    res_mm = engine.evaluate_multi_messenger_domain(
        ejecta_mass_msun=0.045,
        expansion_velocity_c=0.18,
        peak_luminosity_erg_s=1e42,
    )
    assert res_mm.domain == ScientificDomain.MULTI_MESSENGER_TRANSIENTS
    assert res_mm.is_safe_transfer
    assert res_mm.invariants.eddington_ratio > 0.0

    # 2. Cosmology & Large-Scale Structure Domain
    res_cosmo = engine.evaluate_cosmology_domain(
        redshift_z=1.5,
        h0_km_s_mpc=67.4,
    )
    assert res_cosmo.domain == ScientificDomain.COSMOLOGY_STRUCTURE
    assert res_cosmo.is_safe_transfer
    assert res_cosmo.analytical_predictions["hubble_parameter_hz"] > 67.4
    assert res_cosmo.analytical_predictions["hubble_tension_sigma"] > 4.0

    # 3. Exoplanets & Astrobiology Domain
    res_exo = engine.evaluate_exoplanet_domain(
        planet_radius_earth=1.4,
        stellar_radius_solar=0.8,
        equilibrium_temp_k=300.0,
    )
    assert res_exo.domain == ScientificDomain.EXOPLANETS_ASTROBIOLOGY
    assert res_exo.is_safe_transfer
    assert res_exo.analytical_predictions["transit_depth_ppm"] > 100.0
    assert res_exo.analytical_predictions["atmospheric_scale_height_km"] > 0.0

    # 4. Extreme Gravity & Compact Objects Domain
    res_bh = engine.evaluate_extreme_gravity_domain(
        black_hole_mass_msun=4.15e6,  # Sgr A*
        dimensionless_spin_a=0.92,
        distance_kpc=8.178,
    )
    assert res_bh.domain == ScientificDomain.EXTREME_GRAVITY_COMPACT_OBJECTS
    assert res_bh.is_safe_transfer
    assert res_bh.analytical_predictions["shadow_angular_diameter_microarcsec"] > 20.0
    assert res_bh.analytical_predictions["qnm_frequency_f220_hz"] > 0.0


# ==============================================================================
# WS9.3: Autonomous Research Planning Tests
# ==============================================================================

def test_ws9_3_autonomous_research_planning_and_replanning():
    planner = AutonomousResearchPlanner()

    dag = planner.create_transient_research_programme(
        programme_id="prog_fbot_engine_investigation",
        target_phenomenon="Luminous Fast Blue Optical Transient AT2018cow",
    )
    assert len(dag.nodes) == 6 or len(dag.nodes) == 7
    assert dag.available_gpu_hours == 500.0

    # Initial ready nodes (no dependencies)
    ready = planner.get_ready_nodes(dag)
    assert len(ready) == 2  # The 2 initial hypothesis formulation nodes
    assert all(n.node_type == ResearchNodeType.HYPOTHESIS_FORMULATION for n in ready)

    # Complete the CSM hypothesis formulation node
    dag.nodes["node_hyp_csm"].status = ResearchNodeStatus.COMPLETED

    # Trigger falsification replanning when simulation is rejected by radio limit
    updated_dag = planner.trigger_falsification_replanning(
        dag=dag,
        falsified_node_id="node_hyp_csm",
        falsification_evidence="VLA radio non-detection ruled out dense circumstellar wind.",
    )
    assert updated_dag.replanning_iterations == 1
    assert updated_dag.nodes["node_hyp_csm"].status == ResearchNodeStatus.FALSIFIED_OR_FAILED
    # Verify refined mutant node created
    assert "node_hyp_csm_refined_v2" in updated_dag.nodes
    assert updated_dag.nodes["node_hyp_csm_refined_v2"].status == ResearchNodeStatus.PENDING


# ==============================================================================
# WS9.4: Self-Improving Methods & Offline Meta-Learning Tests
# ==============================================================================

def test_ws9_4_offline_meta_learning_and_regression_detection():
    optimizer = OfflineMethodOptimizer(is_online_stream_active=False)

    base_config = ScientificMethodConfig(
        method_id="method_rolling_horizon_scheduler",
        method_name="RollingHorizonScheduler",
        version="1.0.0",
        hyperparameters={"search_tree_depth": 2, "pruning_factor": 0.8},
        status=MethodVersionStatus.PROMOTED_OPERATIONAL,
        historical_benchmark_scores={"task_kilonova_cadence": 86.6, "task_supernova_triage": 86.6},
    )
    optimizer.register_operational_method(base_config)

    # Real evaluation callable executing algorithmic objective
    def scheduler_benchmark_evaluator(hyperparams):
        depth = hyperparams.get("search_tree_depth", 2)
        pruning = hyperparams.get("pruning_factor", 0.8)
        return 75.0 + depth * 5.0 + pruning * 2.0

    # Candidate 1: Improved depth 3 (positive improvement)
    cand_good = ScientificMethodConfig(
        method_id="method_rolling_horizon_scheduler",
        method_name="RollingHorizonScheduler",
        version="1.1.0",
        hyperparameters={"search_tree_depth": 3, "pruning_factor": 0.85},
        parent_version="1.0.0",
    )
    tasks = [
        {"task_id": "task_kilonova_cadence", "evaluator": scheduler_benchmark_evaluator},
        {"task_id": "task_supernova_triage", "evaluator": scheduler_benchmark_evaluator},
    ]
    promoted, msg, scores = optimizer.evaluate_candidate_improvement(cand_good, tasks)
    assert promoted
    assert cand_good.status == MethodVersionStatus.PROMOTED_OPERATIONAL
    assert not cand_good.regression_detected

    # Candidate 2: Degraded depth 1 (causes regression)
    cand_bad = ScientificMethodConfig(
        method_id="method_rolling_horizon_scheduler",
        method_name="RollingHorizonScheduler",
        version="0.9.0",
        hyperparameters={"search_tree_depth": 1, "pruning_factor": 0.50},
        parent_version="1.1.0",
    )
    promoted_bad, msg_bad, _ = optimizer.evaluate_candidate_improvement(cand_bad, tasks)
    assert not promoted_bad
    assert cand_bad.status == MethodVersionStatus.REJECTED_REGRESSION
    assert cand_bad.regression_detected

    # Governance Covenant: Online streaming alerts MUST NOT update weights
    online_optimizer = OfflineMethodOptimizer(is_online_stream_active=True)
    with pytest.raises(OnlineWeightUpdateProhibitedError):
        online_optimizer.evaluate_candidate_improvement(cand_good, tasks)


# ==============================================================================
# WS9.5: Multi-Agent Research Collective Tests
# ==============================================================================

def test_ws9_5_multi_agent_research_collective_and_skeptic_veto():
    collective = ScientificResearchCollective(pi_name="Dev Mehta")

    hyp = StructuredHypothesis(
        hypothesis_id="hyp_collective_test_kn",
        phenomenon="Fast blue-to-red transient with GW counterpart",
        explanation="Two-component r-process kilonova with lanthanide line curtaining",
        assumptions=["Electron fraction Ye < 0.25 in dynamical ejecta", "Subluminal velocity v=0.2c"],
        mathematical_formulation="L ~ t^-1.3",
        parameters={"ejecta_mass_msun": (0.04, 0.01, 0.08, "M_sun")},
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=19.5,
                uncertainty_range=(19.0, 20.0),
                epoch_days_from_trigger=1.0,
                measurement_condition="g band",
            )
        ],
        discriminating_experiment="NIR spectroscopy",
    )

    # 1. Consensus deliberation with full valid empirical evidence: All 9 agents endorse
    valid_evidence = {
        "gw_strain_snr": 32.0,
        "observations": [{"observable_name": "optical_mag_g", "value": 19.5, "uncertainty": 0.1}],
        "bayes_factor_vs_null": 25.0,
        "delta_bic": 10.0,
        "emd_nats": 2.5,
        "fisher_condition_number": 12.0,
        "prior_claims": ["claim_reference_prior_distinct_1"],
        "alternative_likelihoods": {"hyp_alternative_supernova": 1.5},
    }
    res_consensus = collective.conduct_deliberation(
        investigation_id="inv_collective_001",
        hypothesis=hyp,
        empirical_evidence=valid_evidence,
        adversarial_skeptic_mode=False,
    )
    assert res_consensus.is_consensus_reached
    assert not res_consensus.is_vetoed_by_skeptic
    assert len(res_consensus.agent_critiques) == 9
    assert res_consensus.final_verdict == "ENDORSED_FOR_SCIENTIFIC_PUBLICATION"

    # 2. Adversarial Skeptic Veto: Blocks premature consensus
    res_veto = collective.conduct_deliberation(
        investigation_id="inv_collective_002",
        hypothesis=hyp,
        empirical_evidence=valid_evidence,
        adversarial_skeptic_mode=True,
    )
    assert not res_consensus.is_vetoed_by_skeptic
    assert res_veto.is_vetoed_by_skeptic
    assert not res_veto.is_consensus_reached
    assert res_veto.final_verdict == "REJECTED_OR_BLOCKED_BY_PEER_REVIEW"
    assert "CRITICAL SKEPTIC VETO" in res_veto.veto_details

    # 3. Missing evidence fails closed as INDETERMINATE
    res_indeterminate = collective.conduct_deliberation(
        investigation_id="inv_collective_003",
        hypothesis=hyp,
        empirical_evidence={},
        adversarial_skeptic_mode=False,
    )
    assert not res_indeterminate.is_consensus_reached
    assert res_indeterminate.final_verdict == "REJECTED_INDETERMINATE_EVIDENCE"


# ==============================================================================
# WS9.6: Multi-Scale Simulation Intelligence Tests
# ==============================================================================

def test_ws9_6_multiscale_simulation_intelligence():
    sim = MultiScaleSimulator()

    # In-envelope simulation
    params_nominal = {
        "ejecta_mass_msun": 0.04,
        "velocity_c": 0.18,
        "electron_fraction_ye": 0.20,
    }
    state = sim.step_multiscale_evolution(epoch_days=1.5, parameters=params_nominal)
    assert state.nuclear_specific_heating_erg_s_g > 0.0
    assert state.ejecta_radius_cm > 1e14
    assert state.bolometric_luminosity_erg_s > 1e40
    assert not state.is_surrogate_extrapolating
    assert state.surrogate_uncertainty_fraction <= 0.05

    # Out-of-envelope parameter: triggers surrogate extrapolation warning
    params_extrap = {
        "ejecta_mass_msun": 0.25,  # Above maximum validated 0.10 Msun
        "velocity_c": 0.50,        # Above 0.35 c
        "electron_fraction_ye": 0.20,
    }
    state_extrap = sim.step_multiscale_evolution(epoch_days=1.5, parameters=params_extrap)
    assert state_extrap.is_surrogate_extrapolating
    assert state_extrap.surrogate_uncertainty_fraction >= 0.20


# ==============================================================================
# WS9.7: Scientific Replication Network Tests
# ==============================================================================

def test_ws9_7_scientific_replication_network():
    network = ScientificReplicationNetwork(numerical_tolerance_epsilon=1e-4)
    exp_engine = ComputationalExperimentEngine(random_seed=123)

    # Simulate true ground-truth observables
    true_sim = exp_engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        epochs_days=[1.0, 3.0],
        filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
        provenance=DataProvenanceTag.SIMULATED_TESTBED,
        inject_noise=False,
    )
    claimed_results = {f"mag_{p.filter_id}_t{p.epoch_days:.0f}": p.simulated_mag for p in true_sim}

    # 1. Clean Replication: Exact numerical reproducibility
    report_reproduced = network.audit_research_package(
        investigation_id="inv_rep_001",
        target_claim_id="claim_kn_fiducial",
        claimed_results=claimed_results,
        input_parameters={"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01},
        random_seed=123,
        claimed_evidence_hash="hash_wal_event_001",
        actual_evidence_hash="hash_wal_event_001",
    )
    assert report_reproduced.is_fully_replicated
    assert report_reproduced.provenance_hash_verified
    assert report_reproduced.max_relative_discrepancy <= 1e-4

    # 2. Provenance Mismatch: Detects forged or altered evidence hash
    report_bad_hash = network.audit_research_package(
        investigation_id="inv_rep_002",
        target_claim_id="claim_kn_fiducial",
        claimed_results=claimed_results,
        input_parameters={"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01},
        random_seed=123,
        claimed_evidence_hash="hash_wal_event_001",
        actual_evidence_hash="tampered_unmatched_hash",
    )
    assert not report_bad_hash.is_fully_replicated
    assert report_bad_hash.replication_status == ReplicationStatusOutcome.FAILED_PROVENANCE_MISMATCH

    # 3. Numerical Divergence: Detects irreproducible claimed findings
    falsified_claimed_results = copy.deepcopy(claimed_results)
    falsified_claimed_results["mag_g_t1"] = 15.0  # Claimed 15.0 mag instead of true ~20 mag
    report_diverged = network.audit_research_package(
        investigation_id="inv_rep_003",
        target_claim_id="claim_kn_fiducial",
        claimed_results=falsified_claimed_results,
        input_parameters={"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.18, "lanthanide_mass_fraction": 0.01},
        random_seed=123,
        claimed_evidence_hash="hash_wal_event_001",
        actual_evidence_hash="hash_wal_event_001",
    )
    assert not report_diverged.is_fully_replicated
    assert report_diverged.replication_status == ReplicationStatusOutcome.FAILED_NUMERICAL_DIVERGENCE
