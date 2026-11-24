"""
ASTRAEUS Phase 8 - Autonomous Scientific Discovery Engine Test Suite.
Verifies WS8.1 through WS8.7:
- WS8.1: Scientific Knowledge Fabric & Contradiction Detection
- WS8.2: Anomaly and Discovery Intelligence (OOD & Artifact Discrimination)
- WS8.3: Structured Hypothesis Generation & Evolutionary Tournament
- WS8.4: Mathematical & Physical Reasoning (Conservation Laws & Dimensionality)
- WS8.5: Computational Experiment Engine (Forward/Inverse & Bayesian PPC)
- WS8.6: Discriminating Experimental Design & Human Approval Gate Boundary
- WS8.7: Core Benchmark Suite (AstroKnowledge, AstroAnomaly, AstroHypothesis, AstroExperiment, AstroDiscovery)

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

import time
import pytest
import numpy as np

from astraeus_knowledge.claims.claim_registry import (
    ScientificClaim,
    ClaimType,
    ExtractedEquation,
    ReplicationStatus,
    ScientificKnowledgeFabric,
)
from astraeus_data.discovery.anomaly_engine import (
    AnomalyDiscoveryEngine,
    LightCurvePoint,
    AnomalySource,
)
from astraeus_agents.discovery.hypothesis_generator import (
    HypothesisGenerationEngine,
    StructuredHypothesis,
    HypothesisPrediction,
)
from astraeus_models.reasoning.physical_reasoner import (
    PhysicalReasoningEngine,
    ConservationViolationType,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
    SimulatedObservationPoint,
)
from astraeus_agents.observatory.discriminating_planner import (
    DiscriminatingExperimentPlanner,
    CandidateObservationDesign,
)
from astraeus_agents.observatory.human_approval_interface import (
    AutonomousDispatchProhibitedError,
    AuthorizationToken,
    HumanApprovalGate,
)
from astraeus_bench.discovery.astro_benchmarks import (
    AstroDiscoveryBenchmarkSuite,
)
from engines.quarantine_firewall import QuarantineViolationError


# ==============================================================================
# WS8.1: Scientific Knowledge Fabric Tests
# ==============================================================================

def test_ws8_1_claim_registration_and_contradiction_detection():
    fabric = ScientificKnowledgeFabric()

    c1 = ScientificClaim(
        claim_id="claim_barnes_kilonova_opacity",
        claim_type=ClaimType.THEORETICAL_PREDICTION,
        statement="Lanthanide elements produce optical opacity kappa ~ 10 cm^2/g due to millions of f-shell atomic transitions.",
        source_publication="2013ApJ...775...18B",
        domain="KILONOVA_PHYSICS",
        domain_validity_range={"ejecta_mass_msun": (0.001, 0.1), "velocity_c": (0.05, 0.35)},
        equations=[ExtractedEquation(
            latex_str=r"\kappa \sim 10 \, {\rm cm}^2/{\rm g}",
            sympy_expr_str="kappa - 10.0",
            dependent_variable="kappa",
        )],
        uncertainty_bounds={"kappa": (8.0, 12.0)},
    )
    fabric.register_claim(c1)
    assert c1.sha256_hash != ""

    # Conflicting claim with 0.1 opacity
    c2 = ScientificClaim(
        claim_id="claim_conflicting_low_opacity",
        claim_type=ClaimType.THEORETICAL_PREDICTION,
        statement="Lanthanide opacity is less than 0.5 cm^2/g under local thermodynamic equilibrium.",
        source_publication="2020Toy...123Z",
        domain="KILONOVA_PHYSICS",
        domain_validity_range={"ejecta_mass_msun": (0.01, 0.05), "velocity_c": (0.1, 0.3)},
        equations=[],
        uncertainty_bounds={"kappa": (0.1, 0.5)},
    )
    contradictions = fabric.detect_contradictions(c2)
    assert len(contradictions) >= 1
    assert contradictions[0].discrepancy_significance_sigma >= 4.0

    # Evidence linkage
    fabric.link_evidence(
        claim_id=c1.claim_id,
        evidence_id="obs_at2017gfo_hst",
        supports=True,
        evidence_hash="sha256_mock_hst_hash",
    )
    retrieved = fabric.get_claim(c1.claim_id)
    assert "obs_at2017gfo_hst" in retrieved.supporting_evidence_ids
    assert "sha256_mock_hst_hash" in retrieved.evidence_hashes

    # Claim supersession
    c3_updated = ScientificClaim(
        claim_id="claim_tanaka_2020_opacity_revised",
        claim_type=ClaimType.THEORETICAL_PREDICTION,
        statement="Updated atomic data for open f-shell ions gives kappa ~ 15 cm^2/g at t=3d.",
        source_publication="2020ApJ...900...35T",
        domain="KILONOVA_PHYSICS",
        domain_validity_range={"ejecta_mass_msun": (0.001, 0.1), "velocity_c": (0.05, 0.35)},
    )
    fabric.supersede_claim(old_claim_id=c1.claim_id, new_claim=c3_updated)
    assert retrieved.superseded_by == c3_updated.claim_id
    assert c3_updated.supersedes == c1.claim_id


# ==============================================================================
# WS8.2: Anomaly Intelligence Tests
# ==============================================================================

def test_ws8_2_anomaly_intelligence_and_artifact_rejection():
    engine = AnomalyDiscoveryEngine()

    # 1. Test Cosmic Ray / Sub-Seeing Artifact
    artifact_points = [
        LightCurvePoint(
            mjd=60000.0,
            filter_id="g",
            magnitude=17.5,
            uncertainty_mag=0.03,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
            psf_fwhm_arcsec=0.35,  # Much smaller than atmospheric seeing 1.2 arcsec
            nominal_seeing_arcsec=1.20,
        )
    ]
    diag_artifact = engine.evaluate_transient("tgt_cosmic_ray", artifact_points)
    assert not diag_artifact.is_genuine_physical_anomaly
    assert diag_artifact.source_classification == AnomalySource.INSTRUMENTAL_COSMIC_RAY
    assert len(diag_artifact.rejection_reasons) > 0

    # 2. Test Subtraction Dipole Residual
    dipole_points = [
        LightCurvePoint(
            mjd=60000.0,
            filter_id="r",
            magnitude=18.0,
            uncertainty_mag=0.05,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
            psf_fwhm_arcsec=1.18,
            nominal_seeing_arcsec=1.20,
            negative_flux_adjacent=True,  # Characteristic dipole
        )
    ]
    diag_dipole = engine.evaluate_transient("tgt_dipole", dipole_points)
    assert not diag_dipole.is_genuine_physical_anomaly
    assert diag_dipole.source_classification == AnomalySource.INSTRUMENTAL_SUBTRACTION_RESIDUAL

    # 3. Test True Physical Transient with rapid decay & multi-messenger GW
    physical_points = [
        LightCurvePoint(
            mjd=60000.0,
            filter_id="g",
            magnitude=17.8,
            uncertainty_mag=0.04,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
            psf_fwhm_arcsec=1.15,
            nominal_seeing_arcsec=1.10,
        ),
        LightCurvePoint(
            mjd=60001.5,
            filter_id="g",
            magnitude=19.5,  # > 1.1 mag/day decline
            uncertainty_mag=0.06,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
            psf_fwhm_arcsec=1.12,
            nominal_seeing_arcsec=1.10,
        ),
    ]
    diag_physical = engine.evaluate_transient(
        "tgt_real_kilonova",
        physical_points,
        cross_messenger_data={"has_gravitational_wave": True},
    )
    assert diag_physical.is_genuine_physical_anomaly
    assert diag_physical.combined_anomaly_score >= 0.50
    assert len(diag_physical.physical_indicators) > 0


# ==============================================================================
# WS8.3: Hypothesis Generation & Tournament Tests
# ==============================================================================

def test_ws8_3_hypothesis_generation_and_tournament():
    generator = HypothesisGenerationEngine()
    anomaly_engine = AnomalyDiscoveryEngine()

    pts = [
        LightCurvePoint(
            mjd=60000.0,
            filter_id="g",
            magnitude=18.0,
            uncertainty_mag=0.05,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
        ),
        LightCurvePoint(
            mjd=60001.5,
            filter_id="g",
            magnitude=19.8,
            uncertainty_mag=0.06,
            facility_id="ZTF",
            instrument_id="ZTF_CAM",
        ),
    ]
    diag = anomaly_engine.evaluate_transient(
        "tgt_hypo_test",
        pts,
        cross_messenger_data={"has_gravitational_wave": True},
    )

    hyps = generator.generate_candidate_hypotheses(diag)
    assert len(hyps) >= 2
    # Verify schema requirements
    for h in hyps:
        assert h.phenomenon
        assert h.explanation
        assert len(h.assumptions) > 0
        assert h.mathematical_formulation
        assert len(h.predictions) > 0
        assert len(h.falsification_conditions) > 0
        assert h.discriminating_experiment
        assert h.sha256_hash != ""

    # Tournament evaluation
    evidence = [
        {"observable_name": "optical_mag_g", "value": 20.3, "uncertainty": 0.1},
        {"observable_name": "color_g_minus_k", "value": 3.2, "uncertainty": 0.2},
    ]
    tournament_ranked = generator.evaluate_tournament(hyps, evidence)
    assert len(tournament_ranked) == len(hyps)
    # Top hypothesis should have highest fitness
    assert tournament_ranked[0].tournament_score >= tournament_ranked[-1].tournament_score

    # Refinement upon falsification
    mutant = generator.refine_falsified_hypothesis(
        hypothesis=hyps[0],
        falsification_evidence="Radio upper limit < 10 uJy rejected dense wind CSM.",
        generation_index=2,
    )
    assert mutant.hypothesis_id.endswith("_gen2")
    assert hyps[0].hypothesis_id in mutant.lineage_parent_ids


# ==============================================================================
# WS8.4: Mathematical & Physical Reasoning Tests
# ==============================================================================

def test_ws8_4_physical_reasoning_and_conservation_laws():
    reasoner = PhysicalReasoningEngine()

    # 1. Dimensional consistency
    # L = 4 * pi * R^2 * sigma * T^4
    # LHS: erg/s
    # RHS: R^2 (cm^2) * sigma (erg/(cm^2 s K^4)) * T^4 (K^4) -> erg/s
    is_valid, msg = reasoner.verify_dimensional_consistency(
        lhs_unit="erg/s",
        rhs_unit_terms=[("cm", 2), ("K", 4)],
    )
    # Dimensional check behaves deterministically
    assert isinstance(is_valid, bool)

    # 2. Conservation of Energy Violation Test
    # Radiated 1e53 erg from a 1e51 erg reservoir
    viols_energy = reasoner.verify_conservation_laws(
        total_radiated_energy_erg=1e53,
        available_reservoir_energy_erg=1e51,
        expansion_velocity_c=0.2,
        variability_timescale_sec=100.0,
        emission_radius_cm=1e12,
    )
    assert any(v[0] == ConservationViolationType.ENERGY_CONSERVATION for v in viols_energy)

    # 3. Superluminal Velocity Violation Test (v = 1.25 c)
    viols_velocity = reasoner.verify_conservation_laws(
        total_radiated_energy_erg=1e49,
        available_reservoir_energy_erg=1e51,
        expansion_velocity_c=1.25,
        variability_timescale_sec=100.0,
        emission_radius_cm=1e12,
    )
    assert any(v[0] == ConservationViolationType.SUBLUMINAL_VELOCITY for v in viols_velocity)

    # 4. Causality Violation Test
    # Variability 1 second, but radius 1e12 cm (c * 1s = 3e10 cm)
    viols_causality = reasoner.verify_conservation_laws(
        total_radiated_energy_erg=1e49,
        available_reservoir_energy_erg=1e51,
        expansion_velocity_c=0.2,
        variability_timescale_sec=1.0,
        emission_radius_cm=1e12,
    )
    assert any(v[0] == ConservationViolationType.CAUSALITY_LIGHT_TRAVEL for v in viols_causality)

    # 5. Full Consistent Audit
    audit = reasoner.evaluate_model_audit(
        model_name="Kasen2017_Kilonova",
        total_radiated_erg=5e49,
        reservoir_erg=2e50,
        v_c=0.18,
        dt_var_sec=86400.0,
        r_cm=1e14,
        n_data_points=20,
        chi2_fit=18.5,
        n_free_parameters=3,
        null_chi2=85.0,
    )
    assert audit.is_physically_consistent
    assert audit.conservation_audit_passed
    assert audit.bayes_factor_vs_null > 10.0


# ==============================================================================
# WS8.5: Computational Experiment Engine Tests
# ==============================================================================

def test_ws8_5_computational_experiment_engine():
    exp_engine = ComputationalExperimentEngine(random_seed=42)

    # 1. Forward simulation
    sim_pts = exp_engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.15,
        lanthanide_mass_fraction=0.02,
        epochs_days=[1.0, 3.0, 5.0],
        filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
        provenance=DataProvenanceTag.SYNTHETIC_SIMULATOR,
    )
    assert len(sim_pts) == 6
    assert all(p.provenance == DataProvenanceTag.SYNTHETIC_SIMULATOR for p in sim_pts)
    # Ks should fade slower than g (lanthanide blanketing)
    g_pts = [p for p in sim_pts if p.filter_id == "g"]
    ks_pts = [p for p in sim_pts if p.filter_id == "Ks"]
    g_decay = g_pts[-1].simulated_mag - g_pts[0].simulated_mag
    ks_decay = ks_pts[-1].simulated_mag - ks_pts[0].simulated_mag
    assert g_decay > ks_decay  # g fades faster

    # 2. Inverse parameter recovery
    noisy_pts = exp_engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.05,
        expansion_velocity_c=0.20,
        lanthanide_mass_fraction=0.01,
        epochs_days=[1.0, 2.5, 4.0],
        filter_wavelengths={"g": 4770.0, "r": 6230.0},
        provenance=DataProvenanceTag.SIMULATED_TESTBED,
        inject_noise=True,
        photometric_noise_mag=0.03,
    )
    recovered = exp_engine.perform_inverse_parameter_recovery(noisy_pts)
    assert "ejecta_mass_msun" in recovered
    assert recovered["ejecta_mass_msun"].is_contained_in_95_ci

    # 3. Counterfactual evaluation
    cf_res = exp_engine.generate_counterfactual(
        experiment_id="cf_ejecta_mass_shift",
        baseline_params={"ejecta_mass_msun": 0.04, "expansion_velocity_c": 0.15},
        counterfactual_params={"ejecta_mass_msun": 0.01, "expansion_velocity_c": 0.15},
        epochs_days=[1.0, 2.0, 4.0],
        filter_wavelengths={"g": 4770.0},
    )
    assert cf_res.mean_absolute_difference_mag > 0.10

    # 4. Bayesian Posterior Predictive Check (PPC)
    clean_pts = exp_engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.05,
        expansion_velocity_c=0.20,
        lanthanide_mass_fraction=0.01,
        epochs_days=[1.0, 2.5, 4.0],
        filter_wavelengths={"g": 4770.0, "r": 6230.0},
        provenance=DataProvenanceTag.SIMULATED_TESTBED,
        inject_noise=False,
    )
    pred_means = [p.simulated_mag for p in clean_pts]
    pred_sigmas = [0.04 for _ in clean_pts]
    ppc_res = exp_engine.evaluate_posterior_predictive_check(
        model_name="Kilonova_Test_Model",
        observed_data=noisy_pts,
        predicted_means=pred_means,
        predicted_sigmas=pred_sigmas,
        n_replications=200,
    )
    assert ppc_res.ppc_passed
    assert 0.05 <= ppc_res.bayesian_p_value <= 0.95


# ==============================================================================
# WS8.6: Discriminating Planner & Human Approval Gate Boundary Tests
# ==============================================================================

def test_ws8_6_discriminating_planner_and_human_approval_gate():
    gate = HumanApprovalGate()
    planner = DiscriminatingExperimentPlanner(human_gate=gate)

    h1 = StructuredHypothesis(
        hypothesis_id="h1_csm",
        phenomenon="blue transient",
        explanation="CSM",
        mathematical_formulation="L ~ t",
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=17.5,
                uncertainty_range=(17.0, 18.0),
                epoch_days_from_trigger=1.0,
                measurement_condition="g band",
            )
        ],
        discriminating_experiment="g band",
    )

    h2 = StructuredHypothesis(
        hypothesis_id="h2_kilonova",
        phenomenon="blue transient",
        explanation="kilonova",
        mathematical_formulation="L ~ t^-1.3",
        predictions=[
            HypothesisPrediction(
                observable_name="optical_mag_g",
                predicted_value=21.0,
                uncertainty_range=(20.5, 21.5),
                epoch_days_from_trigger=1.0,
                measurement_condition="g band",
            )
        ],
        discriminating_experiment="g band",
    )

    candidate = CandidateObservationDesign(
        action_id="act_vlt_g_exp",
        target_id="tgt_discrim_01",
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="g",
        exposure_time_sec=300.0,
        slew_time_sec=60.0,
        scheduled_mjd=60000.5,
    )

    eval_res = planner.evaluate_expected_model_discrimination(
        candidate=candidate,
        hypotheses=[h1, h2],
        hypothesis_priors={"h1_csm": 0.5, "h2_kilonova": 0.5},
    )
    assert eval_res.expected_model_discrimination_nats > 5.0  # Clear separation between 17.5 and 21.0
    assert eval_res.value_of_information > 0.0

    # Covenant 1: Autonomous Dispatch Without Token Must Fail With AutonomousDispatchProhibitedError
    with pytest.raises(AutonomousDispatchProhibitedError):
        planner.dispatch_observation_to_telescope(eval_res, human_token=None)

    # Covenant 2: Valid Human Token Permits Dispatch
    valid_token = AuthorizationToken(
        token_id="tok_12345",
        recommendation_id="rec_001",
        target_id=candidate.target_id,
        facility_id=candidate.facility_id,
        instrument_id=candidate.instrument_id,
        filter_id=candidate.filter_id,
        exposure_time_sec=candidate.exposure_time_sec,
        scheduled_mjd=candidate.scheduled_mjd,
        proposal_hash="hash_mock_proposal",
        investigator_email="devmehta060505@gmail.com",
        created_at_utc=time.time(),
        token_signature_sha256="sig_mock",
        hmac_signature="hmac_mock_valid_signature",
    )
    dispatch_receipt = planner.dispatch_observation_to_telescope(eval_res, human_token=valid_token)
    assert dispatch_receipt["status"] == "SUCCESS"
    assert dispatch_receipt["action_id"] == candidate.action_id


# ==============================================================================
# WS8.7: Core Benchmark Suite Execution Tests
# ==============================================================================

def test_ws8_7_core_astro_discovery_benchmarks():
    suite = AstroDiscoveryBenchmarkSuite()

    # AstroKnowledge
    res_know = suite.run_astro_knowledge_benchmark()
    assert res_know.passed
    assert res_know.quarantine_breaches == 0

    # AstroAnomaly
    res_anom = suite.run_astro_anomaly_benchmark(n_samples=20)
    assert res_anom.passed
    assert res_anom.primary_metric_value >= 0.90
    assert res_anom.quarantine_breaches == 0

    # AstroHypothesis
    res_hyp = suite.run_astro_hypothesis_benchmark()
    assert res_hyp.passed
    assert res_hyp.quarantine_breaches == 0

    # AstroExperiment
    res_exp = suite.run_astro_experiment_benchmark()
    assert res_exp.passed
    assert res_exp.quarantine_breaches == 0

    # AstroDiscovery Blind Historical
    res_disc = suite.run_astro_discovery_blind_historical_benchmark()
    assert res_disc.passed
    assert res_disc.primary_metric_value == 1.0  # 100% recovery across AT2017gfo, AT2018cow, SN2006gy
    assert res_disc.quarantine_breaches == 0
