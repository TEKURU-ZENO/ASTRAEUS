"""
ASTRAEUS Phase 10: Operational Scientific Intelligence, Independent Validation
& Final System Integration Test Suite.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import json
import math
import sqlite3
import pytest
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

# Custody & Security (WS10.2)
from astraeus_bench.custody.custodian_service import (
    StandaloneCustodianService,
    CustodianServiceClient,
    EnvironmentExternalKeyProvider,
    SimulatedKMSAuthority,
    SignedStateCheckpoint,
    RollbackViolationError,
)
from astraeus_bench.discovery.astro_benchmarks import ExternalCustodianAuthority

# Experiment Engine & PPC (WS10.5)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    SimulatedObservationPoint,
    DataProvenanceTag,
)
from astraeus_models.multiscale.data.extract_kasen_barnes_2019 import (
    pixel_to_physical_coordinate,
    calculate_asymmetric_uncertainty,
    verify_reference_extraction,
    build_records,
    RAW_12_COORDINATES,
)

# Cross-Domain Intelligence (WS10.5)
from astraeus_models.cross_domain.cross_domain_engine import (
    CrossDomainIntelligenceEngine,
    ScientificDomain,
    DimensionlessInvariants,
)

# Ingestion & Broker Adapters (WS10.1)
from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    IngestionStatus,
)
from astraeus_data.brokers.ztf_broker import ZtfBrokerAdapter, ZtfBrokerConfig
from astraeus_data.brokers.gracedb_broker import GraceDBBrokerAdapter, GraceDBBrokerConfig
from astraeus_data.brokers.archive_adapters import MultiArchiveFederator, GaiaCrossmatchRecord
from astraeus_data.ingestion.alert_replay_engine import (
    ArchivalAlertReplayEngine,
    ReplayEventItem,
)
from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate
from astraeus_data.schemas.gw_notice import GwNoticePacket

# Evidence Fabric & Multimessenger Reasoning (WS10.3 & WS10.4)
from astraeus_agents.discovery.evidence_fabric import (
    UnifiedEvidenceFabric,
    EvidenceNodeType,
)
from astraeus_agents.discovery.multimessenger_reasoner import (
    MultiMessengerInferenceEngine,
    CompetingPhysicalModel,
    JeffreysEvidenceScale,
)

# Operations & Governance (WS10.6 & WS10.7)
from astraeus_agents.governance.learning_governance import (
    LiveIntelligenceGovernanceGate,
    OfflineLearningPipeline,
    OperationalMode,
    OperationalModeViolationError,
    AutonomousDispatchProhibitedError,
)
from astraeus_agents.console.investigator_console import InvestigatorConsole
from engines.quarantine_firewall import QuarantineViolationError


# ==============================================================================
# 1. WS10.2: Standalone Custodian Service & Rollback Prevention
# ==============================================================================

def test_ws10_custodian_service_boundary_and_rollback_detection(tmp_path):
    db_path = str(tmp_path / "standalone_ledger.db")
    key_auth = SimulatedKMSAuthority(key_id="KMS_TEST_KEY_V1")

    service = StandaloneCustodianService(
        custodian_id="TEST_CUSTODIAN_SERVICE",
        key_authority=key_auth,
        ledger_db_path=db_path,
    )
    client = CustodianServiceClient(service=service)

    # A. Register commitment through client boundary
    c1 = client.register_commitment(
        target_id="TARGET_KN_PHASE10",
        hypothesis_id="hyp_p10_alpha",
        hypothesis_hash="1" * 64,
        predictions=[{"observable": "g_mag", "pred": 19.5}],
        falsification_criteria=["decay > 2.0 mag/day"],
    )
    assert c1["custodian_id"] == "TEST_CUSTODIAN_SERVICE"
    assert c1["key_id"] == "KMS_TEST_KEY_V1"

    # B. Unblind access
    t_unblind, sig_unblind = client.grant_unblinded_access("TARGET_KN_PHASE10")
    assert t_unblind > 0
    assert len(sig_unblind) == 64

    # C. Verify ledger integrity: recomputes payload hash, record hash with bound signature, and HMAC
    is_valid, notes = client.verify_ledger_integrity()
    assert is_valid is True
    assert len(notes) == 0

    # D. Rollback detection: simulate database truncation / rollback
    chk1 = client.refresh_and_verify_checkpoint()
    assert chk1.sequence_number == 2

    # If sequence number drops, client raises RollbackViolationError
    bad_service = StandaloneCustodianService(
        custodian_id="ROLLED_BACK_SERVICE",
        key_authority=key_auth,
        ledger_db_path=":memory:",
        allow_ephemeral_for_testing=True,
    )
    bad_client = CustodianServiceClient(service=bad_service)
    bad_client._latest_known_checkpoint = chk1  # simulate client that already observed seq=2

    with pytest.raises(RollbackViolationError, match="LEDGER ROLLBACK DETECTED"):
        bad_client.refresh_and_verify_checkpoint()


# ==============================================================================
# 2. WS10.5: Genuine Posterior Grid Sampling (No Gaussian Approx / No Clipping)
# ==============================================================================

def test_ws10_genuine_joint_posterior_grid_sampling_ppc():
    engine = ComputationalExperimentEngine(random_seed=42)
    epochs = [0.5, 1.0, 2.0]
    flts = {"g": 4770.0, "r": 6230.0}

    # Simulate observation
    obs = engine.forward_simulate_kilonova(
        ejecta_mass_msun=0.04,
        expansion_velocity_c=0.18,
        lanthanide_mass_fraction=0.01,
        epochs_days=epochs,
        filter_wavelengths=flts,
        inject_noise=True,
        photometric_noise_mag=0.03,
    )

    # 1. Inverse parameter recovery populates exact discrete posterior grid
    recovery = engine.perform_inverse_parameter_recovery(obs, grid_resolution=15)
    assert getattr(engine, "last_posterior_grid", None) is not None
    grid = engine.last_posterior_grid
    assert len(grid["normalized_weights"]) == 225
    assert abs(sum(grid["normalized_weights"]) - 1.0) < 1e-6

    # 2. Evaluate PPC using genuine posterior grid draws
    ppc = engine.evaluate_parameter_posterior_predictive_check(
        model_name="KN_Genuine_Grid_PPC",
        observed_data=obs,
        posterior_results=recovery,
        epochs_days=epochs,
        filter_wavelengths=flts,
        n_replications=80,
        posterior_grid=engine.last_posterior_grid,
    )
    assert ppc.ppc_passed is True
    assert 0.05 <= ppc.bayesian_p_value <= 0.95

    # 3. Model misspecification rejection: fit diffusion KN model to shock cooling data
    shock_obs = []
    for ep in epochs:
        for f, wl in flts.items():
            # Power law cooling envelope (steep optical drop)
            mag = 15.0 + 3.5 * math.log10(ep + 0.1) + (0.8 if f == "r" else 0.0)
            shock_obs.append(SimulatedObservationPoint(
                epoch_days=ep,
                filter_id=f,
                effective_wavelength_angstrom=wl,
                simulated_mag=mag,
                uncertainty_mag=0.02,
                provenance=DataProvenanceTag.SIMULATED_TESTBED,
            ))

    shock_recovery = engine.perform_inverse_parameter_recovery(shock_obs, grid_resolution=8)
    ppc_misspec = engine.evaluate_parameter_posterior_predictive_check(
        model_name="KN_Fit_To_Shock_Data",
        observed_data=shock_obs,
        posterior_results=shock_recovery,
        epochs_days=epochs,
        filter_wavelengths=flts,
        n_replications=80,
        posterior_grid=engine.last_posterior_grid,
    )
    # Severe lack of fit correctly rejected by Bayesian PPC
    assert ppc_misspec.bayesian_p_value < 0.05
    assert ppc_misspec.ppc_passed is False


# ==============================================================================
# 3. WS10.5: Digitizer Coordinate Transformation & Analytic Uncertainty Bounds
# ==============================================================================

def test_ws10_reference_dataset_coordinate_calibration_and_uncertainty():
    # A. Axis coordinate transformation: x=184, y=412 -> t=0.5, logL=41.25
    t_test, logl_test = pixel_to_physical_coordinate(x_pixel=184, y_pixel=412)
    assert t_test == 0.5
    assert abs(logl_test - 41.25) < 0.01

    # B. Asymmetric uncertainty propagation for +/- 0.05 dex
    unc = calculate_asymmetric_uncertainty(41.35, uncertainty_dex=0.05)
    # Upper fractional uncertainty: ~12.20%
    assert 0.12 <= unc["fractional_uncertainty_upper"] <= 0.125
    # Lower fractional uncertainty: ~10.87%
    assert 0.105 <= unc["fractional_uncertainty_lower"] <= 0.115

    # C. Dense 12-point reference extraction verification
    records = build_records(RAW_12_COORDINATES)
    assert len(records) == 12
    is_verified, notes = verify_reference_extraction(records)
    assert is_verified is True
    assert len(notes) == 0


# ==============================================================================
# 4. WS10.5: Truly Blind Cross-Domain Benchmark with Distinct Domains
# ==============================================================================

def test_ws10_truly_blind_cross_domain_distinct_domains():
    engine = CrossDomainIntelligenceEngine()
    rep = engine.evaluate_held_out_transfer_benchmark(divergence_threshold=3.0)

    # 1. Total 12 cases evaluated
    assert rep["total_benchmark_cases"] == 12
    assert rep["true_positives"] == 6
    assert rep["true_negatives"] == 6
    assert rep["false_positives"] == 0
    assert rep["false_negatives"] == 0
    assert rep["false_positive_rate"] == 0.0
    assert rep["false_negative_rate"] == 0.0
    assert rep["accuracy"] == 1.0
    assert rep["separation_margin"] > 1.0

    # 2. Verify that distinct source and target domains were evaluated
    distinct_domain_cases = [
        e for e in rep["case_evaluations"]
        if e["source_domain"] != e["target_domain"]
    ]
    # At least 8 of the 12 cases evaluate distinct scientific domains
    assert len(distinct_domain_cases) >= 8


# ==============================================================================
# 5. WS10.1: Production Observation Ingestion & Deterministic Replay
# ==============================================================================

def test_ws10_production_ingestion_and_alert_replay(tmp_path):
    storage_root = tmp_path / "events"
    engine = ScientificEventIngestionEngine(storage_root=storage_root)

    # A. ZTF broker adapter with real packet
    ztf_adapter = ZtfBrokerAdapter(ingestion_engine=engine)
    packet_data = {
        "candid": 12345,
        "objectId": "ZTF26aaa_valid",
        "candidate": {
            "jd": 2460000.5,
            "fid": 1,
            "pid": 12345,
            "diffmaglim": 21.0,
            "isdiffpos": "t",
            "ra": 180.0,
            "dec": 25.0,
            "magpsf": 18.5,
            "sigmapsf": 0.05,
            "drb": 0.95,
        },
        "prv_candidates": [],
    }
    packet = ZtfAlertPacket(**packet_data)
    res_ztf = ztf_adapter.process_alert(packet)
    assert res_ztf.status == IngestionStatus.INGESTED

    # Idempotent deduplication
    res_dup = ztf_adapter.process_alert(packet)
    assert res_dup.status == IngestionStatus.DUPLICATE_SKIPPED

    # B. GraceDB broker adapter
    gw_adapter = GraceDBBrokerAdapter(ingestion_engine=engine)
    notice = GwNoticePacket(
        superevent_id="S261128a",
        notice_type="INITIAL",
        time_utc="2026-11-28T10:00:00Z",
        event_time_mjd=60000.416,
        false_alarm_rate_hz=1e-8,
        far_yr=0.315,
        area_90_deg2=35.0,
        distance_mean_mpc=40.0,
        distance_std_mpc=8.0,
        prob_bns=0.92,
        prob_nsbh=0.03,
        prob_bbh=0.01,
        prob_terrestrial=0.04,
        prob_has_ns=0.99,
        prob_has_remnant=0.95,
        central_ra_deg=197.45,
        central_dec_deg=-23.38,
    )
    res_gw = gw_adapter.process_notice(notice)
    assert res_gw.status == IngestionStatus.INGESTED

    # C. Gate 3 Quarantine Covenant enforcement: coordinates in quarantined pixel
    quarantine_notice = GwNoticePacket(
        superevent_id="S261128_quarantined",
        notice_type="INITIAL",
        time_utc="2026-11-28T10:05:00Z",
        event_time_mjd=60000.420,
        false_alarm_rate_hz=1e-9,
        far_yr=0.0315,
        area_90_deg2=10.0,
        distance_mean_mpc=45.0,
        distance_std_mpc=5.0,
        prob_bns=0.95,
        prob_nsbh=0.02,
        prob_bbh=0.01,
        prob_terrestrial=0.02,
        prob_has_ns=0.99,
        prob_has_remnant=0.98,
        central_ra_deg=67.5,
        central_dec_deg=-87.0758,
    )
    res_q = gw_adapter.process_notice(quarantine_notice)
    assert res_q.status == IngestionStatus.QUARANTINED_COVENANT

    # D. Deterministic archival alert replay
    replay_engine = ArchivalAlertReplayEngine(ingestion_engine=engine)
    manifest = [
        ReplayEventItem(
            sequence_id=1,
            event_type="LVK_GW",
            simulated_delay_sec=0.0,
            payload=notice.model_dump(),
        ),
        ReplayEventItem(
            sequence_id=2,
            event_type="ZTF",
            simulated_delay_sec=180.0,
            payload=packet_data,
        ),
    ]
    summary = replay_engine.replay_manifest(manifest)
    assert summary.total_events_in_manifest == 2
    assert summary.is_replay_deterministic is True
    assert len(summary.replay_stream_hash) == 64


# ==============================================================================
# 6. WS10.3: Unified Evidence Fabric Cryptographic Lineage & Replay
# ==============================================================================

def test_ws10_unified_evidence_fabric_lineage_and_audit():
    fabric = UnifiedEvidenceFabric()

    # 1. Observation
    n_obs = fabric.add_node(
        node_type=EvidenceNodeType.OBSERVATION,
        payload={"target": "AT2026p10", "ra": 197.45, "dec": -23.38},
        author_or_agent="ZTF_ALERT_INGESTION",
    )

    # 2. Measurement
    n_meas = fabric.add_node(
        node_type=EvidenceNodeType.MEASUREMENT,
        payload={"epoch_days": 1.0, "mag_g": 18.2, "mag_r": 17.5},
        parent_node_ids=[n_obs.node_id],
        author_or_agent="PHOTOMETRY_CALIBRATION_PIPELINE",
    )

    # 3. Hypothesis
    n_hyp = fabric.add_node(
        node_type=EvidenceNodeType.HYPOTHESIS,
        payload={"hypothesis": "Radioactive kilonova powered by r-process lanthanides"},
        parent_node_ids=[n_meas.node_id],
        author_or_agent="HYPOTHESIS_GENERATOR",
    )

    # 4. Model & Prediction
    n_pred = fabric.add_node(
        node_type=EvidenceNodeType.PREDICTION,
        payload={"predicted_decay_rate": 1.2, "cooling_timescale": 3.0},
        parent_node_ids=[n_hyp.node_id],
        author_or_agent="MULTISCALE_SIMULATOR",
    )

    # 5. Adjudication Decision
    n_dec = fabric.add_node(
        node_type=EvidenceNodeType.DECISION,
        payload={"verdict": "PROSPECTIVE_PREDICTIVE_AGREEMENT_ON_SUPPLIED_TEST_RECORD", "status": "CERTIFIED"},
        parent_node_ids=[n_pred.node_id],
        author_or_agent="INVESTIGATOR_DEV_MEHTA",
    )

    # Traverse upward lineage
    lineage = fabric.get_ancestors_lineage(n_dec.node_id)
    assert len(lineage) == 5
    assert lineage[0].node_id == n_obs.node_id
    assert lineage[-1].node_id == n_dec.node_id

    # Verify cryptographic integrity
    is_valid, notes = fabric.verify_lineage_integrity(n_dec.node_id)
    assert is_valid is True
    assert len(notes) == 0

    # Tamper detection
    n_meas.payload["mag_g"] = 12.0  # simulate in-memory payload tampering
    is_valid_tampered, notes_tampered = fabric.verify_lineage_integrity(n_dec.node_id)
    assert is_valid_tampered is False
    assert any("PAYLOAD TAMPERING" in n for n in notes_tampered)


# ==============================================================================
# 7. WS10.4: Multimessenger Inference & Bayesian Model Comparison
# ==============================================================================

def test_ws10_multimessenger_reasoner_and_bayesian_model_selection():
    reasoner = MultiMessengerInferenceEngine(random_seed=42)

    # A. 3D Spatial-temporal overlap
    overlap = reasoner.compute_3d_spatial_overlap(
        ra_opt_deg=197.45,
        dec_opt_deg=-23.38,
        gw_centroid_ra=197.40,
        gw_centroid_dec=-23.35,
        gw_area_90_deg2=28.0,
        host_distance_mpc=40.7,
        gw_distance_mean_mpc=40.0,
        gw_distance_std_mpc=6.0,
    )
    assert overlap["angular_separation_deg"] < 0.1
    assert overlap["distance_residual_mpc"] < 1.0
    assert overlap["volumetric_probability_density"] > 0.0

    # B. Time-slide scrambling (1000 slides)
    ts_audit = reasoner.execute_time_slide_scrambling(
        event_time_mjd=60000.0,
        candidate_time_mjd=60000.0001,  # 8.6 seconds delta
        time_coincidence_window_sec=120.0,
        num_slides=1000,
        background_rate_per_sec=1e-7,
    )
    assert ts_audit.num_slides == 1000
    assert ts_audit.is_statistically_significant is True
    assert ts_audit.observed_significance_sigma >= 3.0

    # C. Bayesian Model Selection (Kilonova vs Shock vs Magnetar)
    kilonova_lc = [
        {"epoch_days": 0.5, "mag": 17.0, "mag_err": 0.05},
        {"epoch_days": 1.0, "mag": 17.5, "mag_err": 0.05},
        {"epoch_days": 2.0, "mag": 18.5, "mag_err": 0.06},
        {"epoch_days": 3.0, "mag": 19.3, "mag_err": 0.08},
        {"epoch_days": 5.0, "mag": 20.4, "mag_err": 0.10},
    ]
    report = reasoner.evaluate_competing_physical_models(kilonova_lc)
    assert report.favored_model == CompetingPhysicalModel.RADIOACTIVE_KILONOVA
    assert report.decisive_evidence is True
    runner_up = report.model_comparisons[1]
    assert runner_up.delta_bic >= 10.0


# ==============================================================================
# 8. WS10.6 & WS10.7: Operations Governance & Investigator Console
# ==============================================================================

def test_ws10_operations_governance_and_console_dispatch(tmp_path):
    # A. Governance boundary: weight update blocked in LIVE_INTELLIGENCE mode
    gate = LiveIntelligenceGovernanceGate(mode=OperationalMode.LIVE_INTELLIGENCE)
    with pytest.raises(OperationalModeViolationError, match="ONLINE BACKPROPAGATION PROHIBITED"):
        gate.assert_weight_updates_blocked()

    # B. Offline learning pipeline verifies quarantine firewall
    offline_pipe = OfflineLearningPipeline()
    candidates = [
        {"target_id": "clean_1", "ra_deg": 180.0, "dec_deg": 25.0},
        {"target_id": "quarantined_1", "ra_deg": 67.5, "dec_deg": -87.0758},  # quarantined pixel
    ]
    clean_batch, rejected_cnt = offline_pipe.curate_training_batch(candidates)
    assert len(clean_batch) == 1
    assert rejected_cnt == 1

    # C. Zero Autonomous Telescope Dispatch: requires human HMAC signature
    console = InvestigatorConsole(
        investigator_id="Dev Mehta <devmehta060505@gmail.com>",
        governance_gate=gate,
    )
    lc = [
        {"epoch_days": 0.5, "mag": 17.2, "mag_err": 0.04},
        {"epoch_days": 1.0, "mag": 17.5, "mag_err": 0.05},
        {"epoch_days": 2.0, "mag": 18.6, "mag_err": 0.06},
    ]
    summary = console.register_incoming_event(
        target_id="AT2026_DISCOVERY_CANDIDATE",
        survey_origin="ZTF",
        ra_deg=180.5,
        dec_deg=25.2,
        photometry_points=lc,
    )
    assert summary.favored_model == "RADIOACTIVE_KILONOVA"
    assert summary.follow_up_recommended is True
    assert summary.is_authorized_by_human is False

    # Autonomous attempt to dispatch fails closed
    rec_id = console.active_targets["AT2026_DISCOVERY_CANDIDATE"]["recommendation"].recommendation_id
    with pytest.raises(AutonomousDispatchProhibitedError, match="AUTONOMOUS DISPATCH PROHIBITED"):
        gate.dispatch_telescope_command(rec_id)

    # D. Human approval with investigator private HMAC key succeeds
    investigator_secret = b"investigator_hmac_master_key_2026"
    dispatch_res = console.human_approve_follow_up(
        target_id="AT2026_DISCOVERY_CANDIDATE",
        investigator_hmac_key=investigator_secret,
    )
    assert dispatch_res["dispatch_status"] == "DISPATCH_AUTHORIZED"
    assert dispatch_res["investigator"] == "Dev Mehta <devmehta060505@gmail.com>"
    assert len(dispatch_res["authorization_signature"]) == 64

    # Dashboard render verification
    dash_text = console.render_console_dashboard_text()
    assert "AT2026_DISCOVERY_CANDIDATE" in dash_text
    assert "DISPATCHED" in dash_text
