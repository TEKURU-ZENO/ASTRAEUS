"""
ASTRAEUS Phase 7 - WS7.7 Master Verification Suite:
Durable Scientific Event Memory, Deterministic Replay, IVOA Prov-DM Lineage,
Cross-Facility Correlation, and Human-in-the-Loop Operations.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import os
import sys
import math
import time
import json
import tempfile
import concurrent.futures
import pytest
import numpy as np

from astraeus_data.memory.event_store import (
    DurableScientificEventStore,
    InvestigationCheckpoint,
)
from astraeus_data.memory.replay_engine import (
    DeterministicScientificReplayEngine,
    ReplayVerificationResult,
    InvestigationTimelineAuditResult,
)
from astraeus_data.correlation.cross_facility_correlator import (
    CrossFacilityCorrelationEngine,
    CorrelatedEventCandidate,
    MultiMessengerAssociation,
    MessengerType,
    AssociationGrade,
)
from astraeus_agents.lineage.prov_dm import (
    IVOAProvenanceTracker,
    ProvEntity,
    ProvActivity,
    ProvAgent,
    EntityType,
    ActivityType,
    AgentType,
)
from astraeus_agents.operations.human_gate_ops import (
    HumanOperationsDashboardBackend,
    OperationalAction,
    ProposalDiffSummary,
)
from astraeus_agents.observatory.human_approval_interface import (
    HumanApprovalGate,
    ApprovalStatus,
    AuthorizationToken,
    TokenValidationError,
    AutonomousDispatchProhibitedError,
)
from astraeus_agents.observatory.schedulers import ScheduledObservation, TargetState
from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
    CalibrationStatus,
)
from astraeus_agents.feedback.bayesian_recalibration import (
    BayesianRecalibrationEngine,
)
from astraeus_agents.feedback.posterior_predictive import (
    PosteriorPredictiveChecker,
    HypothesisCriticismResult,
    PosteriorPredictiveAssessment,
)
from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from engines.quarantine_firewall import QuarantineFirewall


# =============================================================================
# 1. WS7.6 FOLLOW-UPS VERIFICATION
# =============================================================================

def test_ws7_6_followup_regime_a_exact_bayes_factor_conversion():
    """
    WS7.6 Follow-up 1:
    Verifies that the analytical conversion of ln BF = 210.0 yields:
        e^210 ~= 1.5916335e91 (not 1.60333e91).
    """
    ln_bf = 210.0
    log10_val = ln_bf / math.log(10.0)
    exponent_10 = int(math.floor(log10_val))
    mantissa = 10.0 ** (log10_val - exponent_10)

    assert exponent_10 == 91
    assert abs(mantissa - 1.5916335) < 1e-5
    # Discrepancy against erroneous 1.60333 is confirmed
    assert abs(mantissa - 1.60333) > 0.01


def test_ws7_6_followup_full_bayesian_ppc_vs_conditional_plugin():
    """
    WS7.6 Follow-up 2:
    Distinguishes the plug-in conditional predictive check from a full Bayesian
    posterior predictive check that explicitly samples parameter uncertainty theta ~ p(theta | y).
    """
    obs_mags = [18.20, 18.60, 19.10]
    obs_errs = [0.08, 0.08, 0.10]
    num_samples = 1000

    # 1. Conditional Plug-in Check: fixes trajectory at point estimate mu_hat
    mu_hat = [18.15, 18.55, 19.05]
    t_rep_plugin = PosteriorPredictiveChecker.simulate_posterior_predictive_replicates(
        predicted_magnitudes=mu_hat,
        total_uncertainties=obs_errs,
        num_replicates=num_samples,
        random_seed=42,
    )
    t_obs_plugin = float(np.sum(((np.array(obs_mags) - np.array(mu_hat)) / np.array(obs_errs)) ** 2))
    p_plugin = float(np.mean(t_rep_plugin >= t_obs_plugin))

    # 2. Full Bayesian Check: draws parameter trajectories mu(theta^(s)) reflecting parameter uncertainty
    rng = np.random.default_rng(42)
    param_scatter = rng.normal(loc=0.0, scale=0.06, size=(num_samples, 1))
    mu_samples = np.array(mu_hat) + param_scatter

    t_rep_full, t_obs_full, p_full = PosteriorPredictiveChecker.simulate_full_posterior_predictive_replicates(
        observed_magnitudes=obs_mags,
        observed_uncertainties=obs_errs,
        parameter_trajectory_samples=mu_samples,
        random_seed=42,
    )

    assert t_rep_full.shape == (num_samples,)
    assert t_obs_full.shape == (num_samples,)
    assert 0.05 < p_plugin < 0.95
    assert 0.05 < p_full < 0.95
    assert np.var(t_obs_full) > 0.0


# =============================================================================
# 2. DURABLE EVENT STORE & WRITE-AHEAD LOG INTEGRITY
# =============================================================================

def test_durable_event_store_acid_and_deduplication():
    """
    Verifies that the SQLite WAL event store handles inserts, deduplication,
    observation superseding, and database integrity checks.
    """
    store = DurableScientificEventStore(db_path=":memory:")

    # 1. Ingest alert
    event = UnifiedScientificEvent(
        event_id="rubin_alert_7701",
        survey_origin="RUBIN",
        ra_deg=150.2500,
        dec_deg=2.1500,
        healpix_pixel_nside32=4500,
        event_time_mjd=60100.25,
        photometry_points=[{"mjd": 60100.25, "mag": 19.4, "err": 0.05, "band": "r"}],
        event_metadata={},
        raw_payload_hash="sha256_alert_payload_12345",
    )
    ok, status = store.store_event(event)
    assert ok is True
    assert status == "STORED"

    # Ingest duplicate: idempotent return
    ok_dup, status_dup = store.store_event(event)
    assert ok_dup is True
    assert status_dup == "ALREADY_STORED"

    # 2. Ingest observations
    obs1 = CalibratedObservation(
        observation_id="obs_7701_1",
        target_id="TGT_7701",
        target_ra_deg=150.25,
        target_dec_deg=2.15,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60101.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="sha256_fits_raw_001",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=19.80,
        apparent_mag_err=0.06,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs1)

    obs2 = CalibratedObservation(
        observation_id="obs_7701_2",
        target_id="TGT_7701",
        target_ra_deg=150.25,
        target_dec_deg=2.15,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60101.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="sha256_fits_raw_002",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=19.78,
        apparent_mag_err=0.04,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs2)
    # Supersede obs1 with obs2
    store.supersede_observation(old_observation_id="obs_7701_1", new_observation_id="obs_7701_2")

    active_obs = store.get_active_observations_for_target("TGT_7701")
    assert len(active_obs) == 1
    assert active_obs[0].observation_id == "obs_7701_2"

    # 3. Verify integrity
    integrity = store.verify_database_integrity()
    assert integrity["is_healthy"] is True
    assert integrity["total_events"] == 1
    assert integrity["total_observations"] == 2
    store.close()


def test_durable_event_store_crash_resilience_and_on_disk_persistence():
    """
    Verifies that on-disk SQLite WAL store persists checkpoints and observations
    across complete session close and reopen cycles (simulating crash recovery).
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = os.path.join(tmp_dir, "astraeus_durable_memory.db")

        # Session 1: Create store, save checkpoint, close
        store1 = DurableScientificEventStore(db_path=db_file)
        chk = store1.store_investigation_checkpoint(
            investigation_id="INV_CRASH_TEST",
            target_id="TGT_CRASH",
            revision_index=0,
            timestamp_mjd=60050.0,
            evidence_hashes=["hash_abc", "hash_def"],
            model_versions={"kilonova": "v2.1", "sn_ia": "v1.4"},
            hypothesis_posteriors={"KILONOVA": 0.65, "SUPERNOVA_IA": 0.35},
            entropy_nats=0.6474,
            parent_state_hash=None,
        )
        assert chk.state_hash is not None
        store1.close()

        # Session 2: Reopen from disk file
        store2 = DurableScientificEventStore(db_path=db_file)
        integrity = store2.verify_database_integrity()
        assert integrity["is_healthy"] is True
        assert integrity["total_checkpoints"] == 1

        loaded_chks = store2.get_investigation_checkpoints("INV_CRASH_TEST")
        assert len(loaded_chks) == 1
        assert loaded_chks[0].state_hash == chk.state_hash
        assert loaded_chks[0].hypothesis_posteriors == {"KILONOVA": 0.65, "SUPERNOVA_IA": 0.35}
        store2.close()


# =============================================================================
# 3. DETERMINISTIC SCIENTIFIC REPLAY ENGINE
# =============================================================================

def test_deterministic_scientific_replay_engine():
    """
    Verifies retrospective state reconstruction:
    Replays sequential Bayesian updating from raw evidence and asserts exact match
    with recorded state hash and posterior probabilities.
    """
    store = DurableScientificEventStore(db_path=":memory:")
    target_id = "TGT_REPLAY_TEST"
    inv_id = "INV_REPLAY_001"

    # Observation 1: Optical detection (Epoch 1.0 day)
    obs1 = CalibratedObservation(
        observation_id="obs_rep_1",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_001",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.50,
        apparent_mag_err=0.06,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs1)

    priors = {"KILONOVA": 0.50, "SUPERNOVA_IA": 0.50}

    up1 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=priors,
        observation=obs1,
        time_since_trigger_days=1.0,
    )

    chk1 = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=1,
        timestamp_mjd=60001.0,
        evidence_hashes=[obs1.raw_data_hash_sha256],
        model_versions={"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"},
        hypothesis_posteriors=up1.updated_posteriors,
        entropy_nats=up1.posterior_entropy_nats,
        parent_state_hash=None,
    )

    # Observation 2: NIR detection (Epoch 2.0 days)
    obs2 = CalibratedObservation(
        observation_id="obs_rep_2",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="HAWK-I",
        filter_id="J_VLT",
        epoch_mjd=60002.0,
        exposure_time_sec=900.0,
        raw_data_hash_sha256="raw_hash_002",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.80,
        apparent_mag_err=0.07,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs2)

    up2 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=up1.updated_posteriors,
        observation=obs2,
        effective_wavelength_angstrom=12500.0,
        time_since_trigger_days=2.0,
    )

    chk2 = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=2,
        timestamp_mjd=60002.0,
        evidence_hashes=[obs1.raw_data_hash_sha256, obs2.raw_data_hash_sha256],
        model_versions={"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"},
        hypothesis_posteriors=up2.updated_posteriors,
        entropy_nats=up2.posterior_entropy_nats,
        parent_state_hash=chk1.state_hash,
    )

    replay_engine = DeterministicScientificReplayEngine(event_store=store)

    rev1_res = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=1,
        initial_priors=priors,
    )
    assert rev1_res.is_verified is True
    assert rev1_res.expected_state_hash == chk1.state_hash
    assert rev1_res.reconstructed_state_hash == chk1.state_hash
    assert rev1_res.max_posterior_absolute_error < 1e-5

    rev2_res = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=2,
        initial_priors=priors,
    )
    assert rev2_res.is_verified is True
    assert rev2_res.expected_state_hash == chk2.state_hash
    assert rev2_res.reconstructed_state_hash == chk2.state_hash
    assert rev2_res.chain_link_valid is True
    assert rev2_res.max_posterior_absolute_error < 1e-5

    full_audit = replay_engine.audit_entire_timeline(investigation_id=inv_id, initial_priors=priors)
    assert full_audit.is_fully_reproducible is True
    assert full_audit.total_revisions_verified == 2
    assert full_audit.merkle_chain_unbroken is True
    store.close()


def test_replay_engine_detects_tampered_evidence():
    """
    Verifies that the Replay Engine catches any alteration in historical observations
    or parent state hashes (Merkle verification failure).
    """
    store = DurableScientificEventStore(db_path=":memory:")
    target_id = "TGT_TAMPER"
    inv_id = "INV_TAMPER"

    obs = CalibratedObservation(
        observation_id="obs_orig",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r_ZTF",
        epoch_mjd=60000.5,
        exposure_time_sec=30.0,
        raw_data_hash_sha256="raw_hash_orig",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=19.00,
        apparent_mag_err=0.05,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs)

    priors = {"KILONOVA": 0.50, "SUPERNOVA_IA": 0.50}
    up = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=priors,
        observation=obs,
    )

    chk = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=1,
        timestamp_mjd=60000.5,
        evidence_hashes=[obs.raw_data_hash_sha256],
        initial_priors=priors,
        hypothesis_posteriors=up.updated_posteriors,
        entropy_nats=up.posterior_entropy_nats,
    )

    forged_hash = "forged_sha256_hash_1234567890abcdef"
    with store._get_connection() as conn:
        conn.execute(
            "UPDATE investigation_checkpoints SET state_hash = ? WHERE checkpoint_id = ?",
            (forged_hash, chk.checkpoint_id),
        )

    replay_engine = DeterministicScientificReplayEngine(event_store=store)
    res = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=1,
        initial_priors=priors,
    )
    assert res.is_verified is False
    assert res.expected_state_hash == forged_hash
    assert res.reconstructed_state_hash != forged_hash
    store.close()


# =============================================================================
# 4. CROSS-FACILITY MULTI-MESSENGER CORRELATION ENGINE
# =============================================================================

def test_cross_facility_multi_messenger_gw_grb_optical_association():
    """
    Verifies that CrossFacilityCorrelationEngine detects physical association between
    a LIGO/Virgo GW trigger, a Fermi prompt GRB trigger (dt=1.7s), and a Rubin optical counterpart.
    """
    engine = CrossFacilityCorrelationEngine()

    gw_event = CorrelatedEventCandidate(
        event_id="GW_S261124gw",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60200.500000,
        ra_deg=100.0,
        dec_deg=20.0,
        localization_error_deg=0.25,
        luminosity_distance_mpc=40.0,
        distance_err_mpc=8.0,
        classification_hint="BNS",
    )

    dt_prompt_days = 1.7 / 86400.0
    grb_event = CorrelatedEventCandidate(
        event_id="FERMI_GRB_261124A",
        facility="FERMI_GBM",
        messenger_type=MessengerType.GAMMA_RAY,
        trigger_mjd=60200.500000 + dt_prompt_days,
        ra_deg=100.02,
        dec_deg=20.01,
        localization_error_deg=1.5,
        classification_hint="SHORT_GRB",
    )

    dt_optical_days = 0.4375
    rubin_event = CorrelatedEventCandidate(
        event_id="RUBIN_AT2026mma",
        facility="RUBIN",
        messenger_type=MessengerType.OPTICAL,
        trigger_mjd=60200.500000 + dt_optical_days,
        ra_deg=100.0003,
        dec_deg=20.0002,
        localization_error_deg=0.0001,
        classification_hint="FAST_BLUE_TRANSIENT",
    )

    assoc_prompt = engine.evaluate_association(gw_event, grb_event)
    assert assoc_prompt is not None
    assert assoc_prompt.is_quarantined is False
    assert assoc_prompt.suggested_transient_class == "BINARY_NEUTRON_STAR_PROMPT_GRB"
    assert abs(assoc_prompt.temporal_delta_seconds - 1.70) < 0.1
    assert assoc_prompt.association_significance_sigma >= 3.0
    assert assoc_prompt.chance_coincidence_probability < 1e-4

    assoc_optical = engine.evaluate_association(gw_event, rubin_event)
    assert assoc_optical is not None
    assert assoc_optical.suggested_transient_class == "KILONOVA_OPTICAL_NIR_COUNTERPART"
    assert assoc_optical.angular_separation_arcsec < 5.0
    assert assoc_optical.log_bayes_factor_association > 5.0


def test_cross_facility_correlation_rejects_unrelated_noise():
    """
    Verifies that alerts outside coincidence windows or with huge angular separations
    are rejected without false positive associations.
    """
    engine = CrossFacilityCorrelationEngine()

    gw_event = CorrelatedEventCandidate(
        event_id="GW_001",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60000.0,
        ra_deg=50.0,
        dec_deg=10.0,
        localization_error_deg=0.5,
    )

    distant_optical = CorrelatedEventCandidate(
        event_id="OPT_DISTANT",
        facility="ZTF",
        messenger_type=MessengerType.OPTICAL,
        trigger_mjd=60000.1,
        ra_deg=95.0,
        dec_deg=10.0,
        localization_error_deg=0.001,
    )
    assert engine.evaluate_association(gw_event, distant_optical) is None

    late_optical = CorrelatedEventCandidate(
        event_id="OPT_LATE",
        facility="RUBIN",
        messenger_type=MessengerType.OPTICAL,
        trigger_mjd=60030.0,
        ra_deg=50.001,
        dec_deg=10.001,
        localization_error_deg=0.001,
    )
    assert engine.evaluate_association(gw_event, late_optical) is None


# =============================================================================
# 5. IVOA PROV-DM LINEAGE TRACKER
# =============================================================================

def test_ivoa_prov_dm_lineage_graph_completeness_and_export():
    """
    Verifies complete IVOA Prov-DM lineage modeling:
    Entities, Activities, Agents, causal relationships, and referential integrity.
    """
    tracker = IVOAProvenanceTracker(investigation_id="INV_PROV_77")

    agent_rubin = ProvAgent(
        agent_id="agent_rubin_broker",
        name="RubinAlertBrokerClient",
        agent_type=AgentType.SOFTWARE_AGENT,
        role="Alert Stream Ingestion",
    )
    tracker.register_agent(agent_rubin)

    ent_alert = tracker.record_entity(
        entity_id="ent_alert_1",
        name="Rubin LSST Alert 882194",
        entity_type=EntityType.ALERT_PACKET,
        payload_hash="sha256_alert_payload_12345",
        attributed_to_agent_id="agent_rubin_broker",
    )

    act_ingest = tracker.record_activity(
        activity_id="act_ingest_1",
        name="Streaming Ingestion and Normalization",
        activity_type=ActivityType.EVENT_INGESTION,
        started_at_utc="2026-11-24T12:00:00Z",
        ended_at_utc="2026-11-24T12:00:01Z",
        associated_agent_id="agent_rubin_broker",
        used_entity_ids=[ent_alert.entity_id],
        parameters={"filter": "r", "snr_threshold": 5.0},
    )

    ent_triage = tracker.record_entity(
        entity_id="ent_triage_decision_1",
        name="High-Urgency Kilonova Triage",
        entity_type=EntityType.TRIAGE_DECISION,
        payload_hash="sha256_triage_payload_67890",
        generated_by_activity_id=act_ingest.activity_id,
        derived_from_entity_ids=[ent_alert.entity_id],
        attributes={"urgency": "HIGH", "classification": "KILONOVA"},
    )

    is_complete, issues = tracker.verify_lineage_completeness()
    assert is_complete is True
    assert len(issues) == 0

    prov_doc = tracker.export_ivoa_prov_json()
    assert "prefix" in prov_doc
    assert "entity" in prov_doc
    assert "activity" in prov_doc
    assert "agent" in prov_doc
    assert "wasGeneratedBy" in prov_doc
    assert "used" in prov_doc
    assert "wasAssociatedWith" in prov_doc
    assert "wasDerivedFrom" in prov_doc
    assert len(prov_doc["entity"]) == 2
    assert len(prov_doc["activity"]) == 1


# =============================================================================
# 6. HUMAN OPERATIONS GATE & TOKEN DISPATCH GOVERNANCE
# =============================================================================

def test_human_operations_gate_token_binding_and_autonomous_prohibition():
    """
    Verifies the operational mission control interface:
    1. Proposal diff summary generation with projected entropy delta.
    2. Recommendation-bound, single-use, expiring AuthorizationToken.
    3. Rejection of autonomous dispatch without valid token (AutonomousDispatchProhibitedError).
    4. Token expiration and single-use invalidation.
    """
    dash = HumanOperationsDashboardBackend()

    obs = ScheduledObservation(
        target_id="TGT_KN_ALPHA",
        facility_id="VLT",
        instrument_id="X-shooter",
        filter_id="UVB_VIS_NIR",
        start_mjd=60100.5,
        end_mjd=60100.52,
        exposure_time_sec=1800.0,
        slew_time_sec=60.0,
        readout_time_sec=30.0,
        airmass=1.2,
        expected_eig_nats=0.825,
        weather_score=0.95,
    )
    target = TargetState(
        target_id="TGT_KN_ALPHA",
        ra_deg=100.0,
        dec_deg=20.0,
        trigger_mjd=60100.0,
        base_magnitude=19.0,
        hypothesis_posteriors={"KILONOVA": 0.55, "SN_IA": 0.45},
    )

    # 1. Submit recommendation
    diff = dash.submit_recommendation_for_review(
        observation=obs,
        target=target,
        scientific_justification="Candidate follow-up",
    )
    assert diff.predicted_eig_nats == 0.825
    assert diff.proposal_hash is not None
    rec_id = diff.recommendation_id

    # 2. Strict Covenant: Autonomous dispatch without token must fail
    with pytest.raises(AutonomousDispatchProhibitedError):
        dash.execute_dispatch(token=None, recommendation_id=rec_id)

    # 3. Human PI reviews and issues token
    status, token = dash.review_and_authorize(
        recommendation_id=rec_id,
        action=OperationalAction.APPROVE,
        authorized_by="devmehta060505@gmail.com",
        ttl_seconds=3600.0,
    )
    assert status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
    assert token is not None
    assert token.is_consumed is False
    assert token.recommendation_id == rec_id
    assert token.facility_id == "VLT"

    # 4. Dispatch with valid token succeeds
    dispatch_res = dash.execute_dispatch(token=token, recommendation_id=rec_id)
    assert dispatch_res["status"] == "DISPATCH_AUTHORIZED"
    assert token.is_consumed is True

    # 5. Token reuse is blocked
    with pytest.raises(TokenValidationError) as exc_reuse:
        dash.execute_dispatch(token=token, recommendation_id=rec_id)
    assert "already been consumed" in str(exc_reuse.value).lower()


# =============================================================================
# 7. GATE 3 QUARANTINE FIREWALL & OFFLINE COVENANT
# =============================================================================

def test_gate3_quarantine_enforcement_across_ws7_7_subsystems():
    """
    Verifies that the Gate 3 Quarantine Firewall blocks restricted coordinates
    across Event Store, Cross-Facility Correlator, and Human Operations Gate.
    """
    quar_pixels = QuarantineFirewall.get_quarantined_pixels()
    quar_pixel = next(iter(quar_pixels))
    hp = QuarantineFirewall.get_healpix()
    sc = hp.healpix_to_skycoord(quar_pixel)
    quarantined_ra = float(sc.ra.deg)
    quarantined_dec = float(sc.dec.deg)

    # 1. Event Store routes to dead-letter
    store = DurableScientificEventStore(db_path=":memory:")
    quar_event = UnifiedScientificEvent(
        event_id="quar_alert_999",
        survey_origin="RUBIN",
        ra_deg=quarantined_ra,
        dec_deg=quarantined_dec,
        healpix_pixel_nside32=quar_pixel,
        event_time_mjd=60000.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="sha256_quar_payload",
    )
    ok, status = store.store_event(quar_event)
    assert ok is False
    assert "QUARANTINED" in status
    dead_letters = store.get_quarantined_records()
    assert len(dead_letters) == 1
    assert dead_letters[0]["entity_id"] == "quar_alert_999"
    store.close()

    # 2. Correlator blocks quarantined candidate
    engine = CrossFacilityCorrelationEngine()
    primary = CorrelatedEventCandidate(
        event_id="safe_primary",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60000.0,
        ra_deg=10.0,
        dec_deg=10.0,
        localization_error_deg=0.5,
    )
    quar_cand = CorrelatedEventCandidate(
        event_id="quar_cand",
        facility="RUBIN",
        messenger_type=MessengerType.OPTICAL,
        trigger_mjd=60000.01,
        ra_deg=quarantined_ra,
        dec_deg=quarantined_dec,
        localization_error_deg=0.001,
    )
    assoc = engine.evaluate_association(primary, quar_cand)
    assert assoc is not None
    assert assoc.is_quarantined is True
    assert "Gate 3 Quarantine Firewall" in assoc.quarantine_reason

    # 3. Operations dashboard blocks recommendation in quarantined pixel
    dash = HumanOperationsDashboardBackend()
    obs_quar = ScheduledObservation(
        target_id="TGT_QUAR",
        facility_id="VLT",
        instrument_id="HAWK-I",
        filter_id="H_VLT",
        start_mjd=60000.1,
        end_mjd=60000.12,
        exposure_time_sec=600.0,
        slew_time_sec=30.0,
        readout_time_sec=15.0,
        airmass=1.1,
        expected_eig_nats=0.5,
        weather_score=0.9,
    )
    target_quar = TargetState(
        target_id="TGT_QUAR",
        ra_deg=quarantined_ra,
        dec_deg=quarantined_dec,
        trigger_mjd=60000.0,
        base_magnitude=20.0,
        hypothesis_posteriors={"KILONOVA": 0.5, "SN_IA": 0.5},
    )
    diff = dash.submit_recommendation_for_review(obs_quar, target_quar)
    assert diff.is_quarantined_coordinate is True

    status, token = dash.review_and_authorize(
        recommendation_id=diff.recommendation_id,
        action=OperationalAction.APPROVE,
    )
    assert status == ApprovalStatus.REJECTED_BY_INVESTIGATOR
    assert token is None


# =============================================================================
# 8. WS7.7.1 DURABILITY, REPLAY & GOVERNANCE HARDENING TESTS
# =============================================================================

def test_historical_replay_survives_observation_supersession():
    """
    WS7.7.1 Audit Finding 1:
    Verifies that historical state replay remains strictly reproducible even after
    an observation used in revision k is subsequently superseded in revision k+1.
    Replay engine retrieves immutable raw evidence by hash rather than filtering current active view.
    """
    store = DurableScientificEventStore(db_path=":memory:")
    target_id = "TGT_SUPERSEDE_TEST"
    inv_id = "INV_SUPERSEDE_001"
    priors = {"KILONOVA": 0.50, "SUPERNOVA_IA": 0.50}

    # Revision 1: Initial quick-reduction observation (obs1)
    obs1 = CalibratedObservation(
        observation_id="obs_quick_1",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_quick_001",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.50,
        apparent_mag_err=0.06,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs1)

    up1 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=priors,
        observation=obs1,
        time_since_trigger_days=1.0,
    )

    chk1 = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=1,
        timestamp_mjd=60001.0,
        evidence_hashes=[obs1.raw_data_hash_sha256],
        initial_priors=priors,
        model_versions={"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"},
        hypothesis_posteriors=up1.updated_posteriors,
        entropy_nats=up1.posterior_entropy_nats,
        parent_state_hash=None,
    )

    # Revision 2: Improved deep pipeline re-reduction supersedes obs1
    obs2 = CalibratedObservation(
        observation_id="obs_deep_refined_2",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_deep_refined_002",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.42,
        apparent_mag_err=0.03,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    ok_sup, sup_status = store.atomic_supersede_observation(
        old_observation_id="obs_quick_1",
        new_observation=obs2,
        superseding_mjd=60001.5,
    )
    assert ok_sup is True
    assert sup_status == "ATOMICALLY_SUPERSEDED"

    # Verify that currently active query returns ONLY obs2
    active_now = store.get_active_observations_for_target(target_id)
    assert len(active_now) == 1
    assert active_now[0].observation_id == "obs_deep_refined_2"

    up2 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=priors,
        observation=obs2,
        time_since_trigger_days=1.0,
    )

    chk2 = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=2,
        timestamp_mjd=60001.5,
        evidence_hashes=[obs2.raw_data_hash_sha256],
        initial_priors=priors,
        model_versions={"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"},
        hypothesis_posteriors=up2.updated_posteriors,
        entropy_nats=up2.posterior_entropy_nats,
        parent_state_hash=chk1.state_hash,
    )

    replay_engine = DeterministicScientificReplayEngine(event_store=store)

    # CRITICAL TEST: Reconstructing revision 1 MUST still succeed by fetching obs1 via hash!
    rev1_res = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=1,
        initial_priors=priors,
    )
    assert rev1_res.is_verified is True
    assert rev1_res.expected_state_hash == chk1.state_hash
    assert rev1_res.reconstructed_state_hash == chk1.state_hash
    assert rev1_res.max_posterior_absolute_error < 1e-5

    # Reconstructing revision 2 matches chk2
    rev2_res = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=2,
        initial_priors=priors,
    )
    assert rev2_res.is_verified is True
    assert rev2_res.expected_state_hash == chk2.state_hash
    assert rev2_res.reconstructed_state_hash == chk2.state_hash

    # Timeline audit must be fully reproducible
    timeline = replay_engine.audit_entire_timeline(investigation_id=inv_id, initial_priors=priors)
    assert timeline.is_fully_reproducible is True
    assert timeline.total_revisions_verified == 2
    store.close()


def test_v2_canonical_state_hash_full_precision_and_config_sensitivity():
    """
    WS7.7.1 Audit Finding 2:
    Verifies that the v2 state hash schema:
    1. Uses 12-digit scientific notation precision to prevent rounding collisions.
    2. Commits cryptographically to initial priors (prior_hash).
    3. Commits to model config hashes and decision metadata.
    """
    priors_base = {"KILONOVA": 0.50, "SUPERNOVA_IA": 0.50}
    models = {"recalibrator": "v1.0"}
    configs = {"recalibrator": "cfg_sha256_1111"}
    evidence = ["hash_evidence_001"]

    # 1. Full precision sensitivity: difference of 1e-8 MUST produce different hashes
    hash_a, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_HASH",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.500000000000, "SUPERNOVA_IA": 0.500000000000},
        entropy_nats=0.69314718056,
    )

    hash_b, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_HASH",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.500000010000, "SUPERNOVA_IA": 0.499999990000},
        entropy_nats=0.69314718056,
    )
    assert hash_a != hash_b, "v2 state hash failed to detect 1e-8 posterior difference!"

    # 2. Prior hash sensitivity
    priors_alt = {"KILONOVA": 0.60, "SUPERNOVA_IA": 0.40}
    hash_prior_diff, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_HASH",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_alt,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.500000000000, "SUPERNOVA_IA": 0.500000000000},
        entropy_nats=0.69314718056,
    )
    assert hash_a != hash_prior_diff

    # 3. Model config sensitivity
    configs_alt = {"recalibrator": "cfg_sha256_2222"}
    hash_cfg_diff, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_HASH",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs_alt,
        hypothesis_posteriors={"KILONOVA": 0.500000000000, "SUPERNOVA_IA": 0.500000000000},
        entropy_nats=0.69314718056,
    )
    assert hash_a != hash_cfg_diff


def test_concurrent_token_dispatch_atomic_single_use():
    """
    WS7.7.1 Audit Finding 4:
    Verifies that persistent SQLite token registry prevents double-dispatch race conditions.
    When 10 concurrent threads attempt to dispatch the same authorization token simultaneously,
    exactly 1 thread succeeds and 9 threads are rejected with TokenValidationError.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "governance_tokens.db")
        event_store = DurableScientificEventStore(db_path=db_path)
        try:
            dash = HumanOperationsDashboardBackend(event_store=event_store)

            obs = ScheduledObservation(
                target_id="TGT_RACE_TEST",
                facility_id="VLT",
                instrument_id="X-shooter",
                filter_id="UVB_VIS_NIR",
                start_mjd=60100.5,
                end_mjd=60100.52,
                exposure_time_sec=1800.0,
                slew_time_sec=60.0,
                readout_time_sec=30.0,
                airmass=1.2,
                expected_eig_nats=0.825,
                weather_score=0.95,
            )
            target = TargetState(
                target_id="TGT_RACE_TEST",
                ra_deg=110.0,
                dec_deg=25.0,
                trigger_mjd=60100.0,
                base_magnitude=19.0,
                hypothesis_posteriors={"KILONOVA": 0.60, "SN_IA": 0.40},
            )

            diff = dash.submit_recommendation_for_review(obs, target, "Race condition test")
            status, token = dash.review_and_authorize(
                recommendation_id=diff.recommendation_id,
                action=OperationalAction.APPROVE,
                authorized_by="devmehta060505@gmail.com",
                ttl_seconds=3600.0,
            )
            assert status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
            assert token is not None
            assert token.hmac_signature is not None

            # Execute 10 concurrent dispatch attempts
            success_count = 0
            failure_count = 0
            exceptions = []

            def attempt_dispatch(worker_id: int):
                try:
                    # Each worker passes the token
                    res = dash.execute_dispatch(token=token, recommendation_id=diff.recommendation_id)
                    return True, res
                except Exception as e:
                    return False, str(e)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(attempt_dispatch, i) for i in range(10)]
                for fut in concurrent.futures.as_completed(futures):
                    ok, res = fut.result()
                    if ok:
                        success_count += 1
                    else:
                        failure_count += 1
                        exceptions.append(res)

            assert success_count == 1, f"Expected exactly 1 successful dispatch, got {success_count}"
            assert failure_count == 9, f"Expected 9 failed dispatches, got {failure_count}"
            for exc_msg in exceptions:
                assert "already been consumed" in exc_msg.lower() or "single-use" in exc_msg.lower()

            # Verify token in persistent store is consumed
            persisted_token = event_store.get_authorization_token(token.token_id)
            assert persisted_token is not None
            assert persisted_token["is_consumed"] == 1
        finally:
            event_store.close()


def test_calibrated_multi_messenger_significance_and_skymap_area():
    """
    WS7.7.1 Audit Finding 3:
    Verifies that CrossFacilityCorrelationEngine:
    1. Uses continuous Gaussian deviance (norm.isf(p_chance)).
    2. Converts 2D skymap 90% credible region area to effective sigma.
    3. Employs instrument-specific background rates.
    4. Categorizes into distinct AssociationGrade levels.
    """
    engine = CrossFacilityCorrelationEngine()

    # 1. Gravitational wave candidate with 90% credible area = 40 deg^2
    gw_candidate = CorrelatedEventCandidate(
        event_id="GW_O4_001",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60300.000000,
        ra_deg=120.0,
        dec_deg=15.0,
        localization_error_deg=1.5,
        skymap_90_cr_area_deg2=40.0,
        luminosity_distance_mpc=45.0,
    )
    sigma_eff = engine.get_effective_sigma_deg(gw_candidate)
    expected_sigma = math.sqrt(40.0 / (2.0 * math.pi * 2.302585))
    assert abs(sigma_eff - expected_sigma) < 1e-4
    assert 1.6 < sigma_eff < 1.7

    # 2. Prompt GRB counterpart inside coincidence window (dt = 0.8s, d_theta = 0.05 deg)
    grb_prompt = CorrelatedEventCandidate(
        event_id="FERMI_GRB_001",
        facility="FERMI_GBM",
        messenger_type=MessengerType.GAMMA_RAY,
        trigger_mjd=60300.000000 + (0.8 / 86400.0),
        ra_deg=120.03,
        dec_deg=15.04,
        localization_error_deg=1.0,
        classification_hint="SHORT_GRB",
    )
    assoc = engine.evaluate_association(gw_candidate, grb_prompt)
    assert assoc is not None
    assert assoc.association_significance_sigma >= 3.0
    assert isinstance(assoc.association_significance_sigma, float)
    assert assoc.association_grade in (
        AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION,
        AssociationGrade.STATISTICALLY_SIGNIFICANT_COINCIDENCE,
    )

    # 3. Background rates test: Optical rate vs GW rate
    p_opt = engine.calculate_calibrated_chance_probability(
        search_radius_deg=0.5,
        time_window_seconds=86400.0,
        candidate_messenger=MessengerType.OPTICAL,
    )
    p_gw = engine.calculate_calibrated_chance_probability(
        search_radius_deg=0.5,
        time_window_seconds=86400.0,
        candidate_messenger=MessengerType.GRAVITATIONAL_WAVE,
    )
    # Optical background transient rate is far higher than LVK GW alert rate
    assert p_opt > p_gw


def test_atomic_supersession_transaction_rollback_on_failure():
    """
    Verifies that atomic_supersede_observation preserves transactional integrity:
    If superseding an invalid or nonexistent observation, the database transaction
    rolls back completely and does not create orphan records.
    """
    store = DurableScientificEventStore(db_path=":memory:")

    obs_initial = CalibratedObservation(
        observation_id="obs_valid_orig",
        target_id="TGT_TX_TEST",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_valid_orig",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=19.00,
        apparent_mag_err=0.05,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs_initial)

    # Attempt to supersede a non-existent observation ID
    obs_new = CalibratedObservation(
        observation_id="obs_new_orphan",
        target_id="TGT_TX_TEST",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.5,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_orphan",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.90,
        apparent_mag_err=0.04,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    ok, status = store.atomic_supersede_observation(
        old_observation_id="NON_EXISTENT_OBS_ID",
        new_observation=obs_new,
        superseding_mjd=60001.5,
    )
    assert ok is False
    assert "not found" in status

    # Verify original observation is still active and orphan was NOT inserted
    active = store.get_active_observations_for_target("TGT_TX_TEST")
    assert len(active) == 1
    assert active[0].observation_id == "obs_valid_orig"
    store.close()


def test_hmac_token_forgery_and_tamper_rejection():
    """
    WS7.7.1 Audit Finding 4:
    Verifies that forged or tampered authorization tokens are immediately rejected by HMAC verification.
    """
    dash = HumanOperationsDashboardBackend()
    obs = ScheduledObservation(
        target_id="TGT_FORGERY_TEST",
        facility_id="VLT",
        instrument_id="HAWK-I",
        filter_id="Ks_VLT",
        start_mjd=60100.5,
        end_mjd=60100.52,
        exposure_time_sec=1200.0,
        slew_time_sec=40.0,
        readout_time_sec=20.0,
        airmass=1.15,
        expected_eig_nats=0.75,
        weather_score=0.92,
    )
    target = TargetState(
        target_id="TGT_FORGERY_TEST",
        ra_deg=150.0,
        dec_deg=5.0,
        trigger_mjd=60100.0,
        base_magnitude=19.5,
        hypothesis_posteriors={"KILONOVA": 0.50, "SN_IA": 0.50},
    )

    diff = dash.submit_recommendation_for_review(obs, target)
    status, token = dash.review_and_authorize(
        recommendation_id=diff.recommendation_id,
        action=OperationalAction.APPROVE,
        authorized_by="devmehta060505@gmail.com",
    )
    assert status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
    assert token is not None

    # Case 1: Tampered HMAC signature
    tampered_sig_token = token.model_copy()
    tampered_sig_token.hmac_signature = "bad_forged_hmac_signature_0000000000000000"
    with pytest.raises(TokenValidationError) as exc_forged:
        dash.execute_dispatch(token=tampered_sig_token, recommendation_id=diff.recommendation_id)
    assert "forgery detected" in str(exc_forged.value).lower()

    # Case 2: Tampered payload with original signature (e.g. attacker changing target or facility)
    tampered_target_token = token.model_copy()
    tampered_target_token.facility_id = "KECK"
    with pytest.raises(TokenValidationError) as exc_mismatch:
        dash.execute_dispatch(token=tampered_target_token, recommendation_id=diff.recommendation_id)
    assert "forgery detected" in str(exc_mismatch.value).lower() or "mismatch" in str(exc_mismatch.value).lower()


# =============================================================================
# 9. WS7.7.2 CRYPTOGRAPHIC GOVERNANCE, REPLAY DETERMINISM & EMPIRICAL VALIDATION
# =============================================================================

def test_ws7_7_2_missing_or_insufficient_hmac_secret_key_fails_closed():
    """
    WS7.7.2-A Critical 1:
    Verifies that HumanApprovalGate fails closed:
    1. If HMAC secret key is missing (neither provided nor in environment), raises KeyError.
    2. If HMAC secret key has insufficient entropy (< 32 bytes), raises ValueError.
    3. Strictly bars any hardcoded default fallback keys.
    """
    orig_env = os.environ.pop("ASTRAEUS_HMAC_SECRET_KEY", None)
    try:
        with pytest.raises(KeyError) as exc_missing:
            HumanApprovalGate(hmac_secret_key=None)
        assert "mandatory hmac secret key missing" in str(exc_missing.value).lower()
        assert "hardcoded default keys are strictly prohibited" in str(exc_missing.value).lower()

        with pytest.raises(ValueError) as exc_entropy:
            HumanApprovalGate(hmac_secret_key=b"too_short_secret_key_16b!")
        assert "entropy insufficient" in str(exc_entropy.value).lower()

        valid_key = b"A" * 32
        gate = HumanApprovalGate(hmac_secret_key=valid_key)
        assert gate.hmac_secret_key == valid_key
    finally:
        if orig_env is not None:
            os.environ["ASTRAEUS_HMAC_SECRET_KEY"] = orig_env


def test_ws7_7_2_process_restart_full_restoration():
    """
    WS7.7.2-A Critical 2:
    Verifies full process restart recovery:
    In Session 1, a recommendation is submitted and approved, issuing an authorization token.
    Both recommendation and token are persisted to SQLite.
    In Session 2 (simulating fresh process reboot), a new Dashboard instance is initialized
    with completely empty memory. It restores the recommendation from SQLite, validates the HMAC
    token, and authorizes physical dispatch without KeyError.
    """
    test_key = b"ASTRAEUS_WS772_RESTART_SECRET_KEY_DEV_MEHTA_2026_ENTROPY"
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "restart_governance.db")

        # Session 1: Create recommendation and issue token
        store_s1 = DurableScientificEventStore(db_path=db_path)
        dash_s1 = HumanOperationsDashboardBackend(event_store=store_s1, hmac_secret_key=test_key)

        obs = ScheduledObservation(
            target_id="TGT_REBOOT_001",
            facility_id="VLT",
            instrument_id="X-shooter",
            filter_id="UVB_VIS_NIR",
            start_mjd=60100.5,
            end_mjd=60100.52,
            exposure_time_sec=1800.0,
            slew_time_sec=60.0,
            readout_time_sec=30.0,
            airmass=1.2,
            expected_eig_nats=0.825,
            weather_score=0.95,
        )
        target = TargetState(
            target_id="TGT_REBOOT_001",
            ra_deg=115.0,
            dec_deg=22.0,
            trigger_mjd=60100.0,
            base_magnitude=19.2,
            hypothesis_posteriors={"KILONOVA": 0.70, "SN_IA": 0.30},
        )

        diff = dash_s1.submit_recommendation_for_review(obs, target, "Restart recovery test proposal")
        rec_id = diff.recommendation_id
        status, token = dash_s1.review_and_authorize(
            recommendation_id=rec_id,
            action=OperationalAction.APPROVE,
            authorized_by="devmehta060505@gmail.com",
            ttl_seconds=3600.0,
        )
        assert status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
        assert token is not None

        persisted_rec = store_s1.get_recommendation(rec_id)
        assert persisted_rec is not None
        assert persisted_rec["target_id"] == "TGT_REBOOT_001"
        assert persisted_rec["approval_status"] == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR.value
        store_s1.close()

        # Session 2: Completely fresh process simulation (empty in-memory state)
        store_s2 = DurableScientificEventStore(db_path=db_path)
        try:
            dash_s2 = HumanOperationsDashboardBackend(event_store=store_s2, hmac_secret_key=test_key)
            assert len(dash_s2.proposals) == 0
            assert len(dash_s2.gate._recommendations) == 0
            assert len(dash_s2.gate._tokens) == 0

            # Execute dispatch: restores both recommendation and token from SQLite!
            res = dash_s2.execute_dispatch(token=token, recommendation_id=rec_id)
            assert res["status"] == "DISPATCH_AUTHORIZED"
            assert res["dispatch_status"] == "DISPATCHED_TO_FACILITY_QUEUE"
            assert res["recommendation_id"] == rec_id

            # Verify single-use policy survives across restart: second attempt rejected!
            with pytest.raises(TokenValidationError) as exc_consumed:
                dash_s2.execute_dispatch(token=token, recommendation_id=rec_id)
            assert "already been consumed" in str(exc_consumed.value).lower()
        finally:
            store_s2.close()


def test_ws7_7_2_exact_binary64_sub_10_to_12_precision_differentiation():
    """
    WS7.7.2-B Important 3:
    Verifies that the canonical float serialization uses exact IEEE 754 binary64 float.hex():
    1. Preserves differences down to hardware machine epsilon (~2.22e-16).
    2. Sub-10^-12 differences produce completely distinct SHA-256 state hashes.
    3. Eliminates decimal rounding collisions.
    """
    priors_base = {"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5}
    models = {"recalibrator": "v1.0"}
    configs = {"recalibrator": "cfg_sha256_1111"}
    evidence = ["hash_evidence_001"]

    hash_base, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_IEEE754",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5},
        entropy_nats=0.6931471805599453,
    )

    # Sub-10^-12 difference (1e-14)
    hash_sub12, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_IEEE754",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.5 + 1.0e-14, "SUPERNOVA_IA": 0.5 - 1.0e-14},
        entropy_nats=0.6931471805599453,
    )
    assert hash_base != hash_sub12, "float.hex() failed to differentiate sub-10^-12 posterior delta!"

    # Machine epsilon test (sys.float_info.epsilon)
    eps = sys.float_info.epsilon
    hash_eps, _ = InvestigationCheckpoint.compute_state_hash(
        target_id="TGT_IEEE754",
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=evidence,
        initial_priors=priors_base,
        model_versions=models,
        model_config_hashes=configs,
        hypothesis_posteriors={"KILONOVA": 0.5 + eps, "SUPERNOVA_IA": 0.5 - eps},
        entropy_nats=0.6931471805599453,
    )
    assert hash_base != hash_eps, "float.hex() failed to differentiate machine epsilon delta!"


def test_ws7_7_2_deterministic_replay_with_tied_observation_epochs():
    """
    WS7.7.2-B Important 3:
    Verifies that when multiple observations occur at the exact same epoch_mjd,
    the replay engine deterministically preserves the exact recorded assimilation sequence
    and uses (epoch_mjd, observation_id) as an immutable tie-breaker.
    """
    store = DurableScientificEventStore(db_path=":memory:")
    target_id = "TGT_TIED_EPOCHS"
    inv_id = "INV_TIED_001"
    priors = {"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5}

    obs_A = CalibratedObservation(
        observation_id="obs_tied_A",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="FORS2",
        filter_id="r_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_tied_A",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.50,
        apparent_mag_err=0.05,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    obs_B = CalibratedObservation(
        observation_id="obs_tied_B",
        target_id=target_id,
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="HAWK-I",
        filter_id="J_VLT",
        epoch_mjd=60001.0,
        exposure_time_sec=600.0,
        raw_data_hash_sha256="raw_hash_tied_B",
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=18.20,
        apparent_mag_err=0.06,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
    )
    store.store_observation(obs_A)
    store.store_observation(obs_B)

    # Assimilate A then B
    up_A = BayesianRecalibrationEngine.recalibrate_posteriors(priors, obs_A, time_since_trigger_days=1.0)
    up_AB = BayesianRecalibrationEngine.recalibrate_posteriors(up_A.updated_posteriors, obs_B, effective_wavelength_angstrom=12500.0, time_since_trigger_days=1.0)

    chk_AB = store.store_investigation_checkpoint(
        investigation_id=inv_id,
        target_id=target_id,
        revision_index=1,
        timestamp_mjd=60001.0,
        evidence_hashes=[obs_A.raw_data_hash_sha256, obs_B.raw_data_hash_sha256],
        initial_priors=priors,
        hypothesis_posteriors=up_AB.updated_posteriors,
        entropy_nats=up_AB.posterior_entropy_nats,
    )

    replay_engine = DeterministicScientificReplayEngine(event_store=store)
    res_AB = replay_engine.reconstruct_checkpoint(
        investigation_id=inv_id,
        target_revision=1,
        initial_priors=priors,
        effective_wavelengths={"r_VLT": 6000.0, "J_VLT": 12500.0},
    )
    assert res_AB.is_verified is True
    assert res_AB.expected_state_hash == chk_AB.state_hash

    # Reversed order B then A produces different state hash
    up_B = BayesianRecalibrationEngine.recalibrate_posteriors(priors, obs_B, effective_wavelength_angstrom=12500.0, time_since_trigger_days=1.0)
    up_BA = BayesianRecalibrationEngine.recalibrate_posteriors(up_B.updated_posteriors, obs_A, time_since_trigger_days=1.0)
    hash_BA, _ = InvestigationCheckpoint.compute_state_hash(
        target_id=target_id,
        revision_index=1,
        parent_state_hash=None,
        evidence_hashes=[obs_B.raw_data_hash_sha256, obs_A.raw_data_hash_sha256],
        initial_priors=priors,
        model_versions={"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"},
        model_config_hashes={"bayesian_recalibration": "default_cfg_hash", "kilonova": "default_cfg_hash"},
        hypothesis_posteriors=up_BA.updated_posteriors,
        entropy_nats=up_BA.posterior_entropy_nats,
    )
    assert chk_AB.state_hash != hash_BA, "Order reversal of tied epoch observations must alter state digest!"
    store.close()


def test_ws7_7_2_multi_messenger_time_slides_and_spatial_scrambling():
    """
    WS7.7.2-C Important 4:
    Verifies empirical false-alarm background rejection:
    1. Time-slide analysis (shifting trigger times by +/- 10 to 100 days) yields zero false physical associations.
    2. Spatial scrambling analysis (shifting coordinates by 15 to 270 deg) yields zero false physical associations.
    3. Confirms significance is formally designated as MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT.
    """
    gw_event = CorrelatedEventCandidate(
        event_id="GW_PHYSICAL_01",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60200.500000,
        ra_deg=100.0,
        dec_deg=20.0,
        localization_error_deg=0.5,
        skymap_90_cr_area_deg2=25.0,
        luminosity_distance_mpc=40.0,
    )

    grb_prompt = CorrelatedEventCandidate(
        event_id="FERMI_PHYSICAL_01",
        facility="FERMI_GBM",
        messenger_type=MessengerType.GAMMA_RAY,
        trigger_mjd=60200.500000 + (1.5 / 86400.0),
        ra_deg=100.01,
        dec_deg=20.01,
        localization_error_deg=1.0,
        classification_hint="SHORT_GRB",
    )

    # 1. Genuine physical association
    assoc_genuine = CrossFacilityCorrelationEngine.evaluate_association(gw_event, grb_prompt)
    assert assoc_genuine is not None
    assert assoc_genuine.significance_type == "MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT"
    assert assoc_genuine.association_grade in (
        AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION,
        AssociationGrade.STATISTICALLY_SIGNIFICANT_COINCIDENCE,
    )

    # 2. Time-slide background analysis (offsets of +/- 10d, 20d, 30d, 50d, 100d)
    ts_res = CrossFacilityCorrelationEngine.perform_time_slide_analysis(gw_event, grb_prompt)
    assert ts_res["total_slides"] == 10
    assert ts_res["false_alarms"] == 0
    assert ts_res["rejection_confirmed"] is True

    # 3. Spatial scrambling analysis (RA shifts of 15 deg to 270 deg)
    ss_res = CrossFacilityCorrelationEngine.perform_spatial_scrambling_analysis(gw_event, grb_prompt)
    assert ss_res["total_scrambles"] == 10
    assert ss_res["false_alarms"] == 0
    assert ss_res["rejection_confirmed"] is True


def test_ws7_7_2_dispatch_fails_without_durable_event_store():
    """
    WS7.7.2-A Critical 1:
    Verifies that physical dispatch fails closed (AutonomousDispatchProhibitedError)
    if the governance gate has no durable SQLite event store attached.
    Volatile in-memory dispatch without durable single-use registration is strictly prohibited.
    """
    test_key = b"ASTRAEUS_DISPATCH_STORE_TEST_KEY_DEV_MEHTA_2026_SECURE"
    gate = HumanApprovalGate(event_store=None, hmac_secret_key=test_key)

    obs = ScheduledObservation(
        target_id="TGT_NO_STORE",
        facility_id="VLT",
        instrument_id="X-shooter",
        filter_id="UVB_VIS_NIR",
        start_mjd=60100.5,
        end_mjd=60100.52,
        exposure_time_sec=1800.0,
        slew_time_sec=60.0,
        readout_time_sec=30.0,
        airmass=1.2,
        expected_eig_nats=0.825,
        weather_score=0.95,
    )
    target = TargetState(
        target_id="TGT_NO_STORE",
        ra_deg=100.0,
        dec_deg=20.0,
        trigger_mjd=60100.0,
        base_magnitude=19.0,
        hypothesis_posteriors={"KILONOVA": 0.60, "SN_IA": 0.40},
    )

    rec = gate.create_recommendation(obs, target, "Test without store")
    token = gate.issue_authorization_token(rec.recommendation_id, "devmehta060505@gmail.com")

    with pytest.raises(AutonomousDispatchProhibitedError) as exc_store:
        gate.dispatch_with_token(token, rec.recommendation_id)
    assert "durable sqlite event store is mandatory" in str(exc_store.value).lower()
