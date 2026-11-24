"""
ASTRAEUS Phase 7 - WS7.7 Master Verification Suite:
Durable Scientific Event Memory, Deterministic Replay, IVOA Prov-DM Lineage,
Cross-Facility Correlation, and Human-in-the-Loop Operations.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import os
import math
import time
import json
import tempfile
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

    forged_hash = "forged_sha256_hash_1234567890abcdef"
    with store._get_connection() as conn:
        conn.execute(
            """
            INSERT INTO investigation_checkpoints (
                checkpoint_id, investigation_id, target_id, revision_index,
                timestamp_mjd, state_hash, parent_state_hash,
                evidence_hashes_json, model_versions_json,
                hypothesis_posteriors_json, entropy_nats,
                decision_metadata_json, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chk_forged",
                inv_id,
                target_id,
                1,
                60000.5,
                forged_hash,
                None,
                json.dumps([obs.raw_data_hash_sha256]),
                json.dumps({"model": "v1"}),
                json.dumps(up.updated_posteriors),
                up.posterior_entropy_nats,
                json.dumps({}),
                "2026-11-24T12:00:00Z",
            ),
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
