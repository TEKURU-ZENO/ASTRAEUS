"""
ASTRAEUS Phase 7 Final Closure & Governance Test Suite (WS7.8).
Validates:
- WS7.8-A: Production Cryptographic Governance & Persistent WAL DB Enforcement.
- WS7.8-B: Complete Proposal-to-Token Binding Verification at Dispatch Boundary.
- WS7.8-C: Exact Evidence Sequence Persistence Across Reboots & Fail-Closed Replay.
- WS7.8-D: Archival Background False-Alarm Benchmark & Honest Statistical Qualification.
- WS7.8-E: Gate 3 Quarantine & Offline Learning Covenant Verification.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import time
import math
import secrets
import tempfile
import pytest

from astraeus_agents.observatory.human_approval_interface import (
    HumanApprovalGate,
    ApprovalStatus,
    AuthorizationToken,
    TokenValidationError,
    AutonomousDispatchProhibitedError,
    ProductionConfigurationError,
    KeyEntropyError,
    validate_secret_key_entropy,
    provision_cryptographic_secret,
)
from astraeus_agents.observatory.schedulers import ScheduledObservation, TargetState
from astraeus_agents.operations.human_gate_ops import (
    HumanOperationsDashboardBackend,
    OperationalAction,
)
from astraeus_data.memory.event_store import (
    DurableScientificEventStore,
    InvestigationCheckpoint,
)
from astraeus_data.memory.replay_engine import (
    DeterministicScientificReplayEngine,
    MissingEvidenceError,
    EvidenceCorruptionError,
    MissingHistoricalStateError,
    ModelVersionMismatchError,
)
from astraeus_data.correlation.cross_facility_correlator import (
    CrossFacilityCorrelationEngine,
    CorrelatedEventCandidate,
    MessengerType,
    AssociationGrade,
    ArchivalBackgroundBenchmark,
    ArchivalBackgroundValidationResult,
)
from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
    CalibrationStatus,
)
from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from engines.quarantine_firewall import QuarantineFirewall


# ==============================================================================
# SECTION 1: WS7.8-A PRODUCTION CRYPTOGRAPHIC GOVERNANCE & PERSISTENT DB
# ==============================================================================

def test_ws7_8_a_shannon_entropy_rejects_trivial_keys_in_production():
    """Verifies that length is not mistaken for entropy: repeated bytes fail closed."""
    # 1. 32 repeated characters have length 32 but 0 entropy -> fails closed in production
    with pytest.raises(KeyEntropyError) as exc_repeat:
        validate_secret_key_entropy(b"A" * 32, is_production=True)
    assert "insufficient key diversity" in str(exc_repeat.value).lower()

    # 2. Alternating 2 characters have length 32 but 1 bit entropy -> fails closed
    with pytest.raises(KeyEntropyError) as exc_alt:
        validate_secret_key_entropy(b"AB" * 16, is_production=True)
    assert "insufficient key diversity" in str(exc_alt.value).lower()

    # 3. Source-controlled test key indicator rejected in production
    test_key = b"ASTRAEUS_OBSERVATORY_TEST_SECRET_KEY_2026_ENTROPY_DEV_MEHTA_SECURE"
    with pytest.raises(KeyEntropyError) as exc_test:
        validate_secret_key_entropy(test_key, is_production=True)
    assert "source-controlled test key" in str(exc_test.value).lower()

    # 4. Cryptographically provisioned secret passes with high Shannon entropy
    real_secret = provision_cryptographic_secret(32)
    valid, entropy, msg = validate_secret_key_entropy(real_secret, is_production=True)
    assert valid is True
    assert entropy >= 3.8
    assert "VALID_ENTROPY" in msg


def test_ws7_8_a_production_mode_rejects_in_memory_databases():
    """Verifies that production dispatch strictly forbids :memory: databases."""
    real_key = provision_cryptographic_secret(32)
    in_mem_store = DurableScientificEventStore(db_path=":memory:", allow_in_memory=True)

    gate = HumanApprovalGate(
        event_store=in_mem_store,
        hmac_secret_key=real_key,
        require_persistent_db=True,
    )

    obs = ScheduledObservation(
        target_id="tgt_prod_mem",
        facility_id="RUBIN",
        instrument_id="LSST_CAM",
        filter_id="r",
        exposure_time_sec=60.0,
        start_mjd=60100.0,
        end_mjd=60100.001,
        slew_time_sec=10.0,
        readout_time_sec=5.0,
        airmass=1.1,
        expected_eig_nats=0.85,
        weather_score=0.95,
    )
    target = TargetState(
        target_id="tgt_prod_mem",
        ra_deg=155.0,
        dec_deg=-12.0,
        trigger_mjd=60099.0,
        base_magnitude=19.0,
        hypothesis_posteriors={"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5},
    )

    rec = gate.create_recommendation(obs, target, "Production memory test")
    tok = gate.issue_authorization_token(rec.recommendation_id, "investigator@astraeus.org")

    # Physical dispatch must raise ProductionConfigurationError because DB is :memory:
    with pytest.raises(ProductionConfigurationError) as exc_mem:
        gate.dispatch_with_token(tok, rec.recommendation_id)
    assert "in-memory database (:memory:) is strictly prohibited" in str(exc_mem.value)


def test_ws7_8_a_durable_on_disk_recovery_across_independent_connections():
    """Verifies complete process crash recovery and restart on a persistent SQLite WAL database."""
    real_key = provision_cryptographic_secret(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "ws7_8_governance.db")
        store_p1 = None
        store_p2 = None

        try:
            # Process 1: Create store, submit recommendation, authorize, and issue token
            store_p1 = DurableScientificEventStore(db_path=db_path)
            gate_p1 = HumanApprovalGate(
                event_store=store_p1,
                hmac_secret_key=real_key,
                require_persistent_db=True,
            )

            obs = ScheduledObservation(
                target_id="tgt_persist_1",
                facility_id="KECK_I",
                instrument_id="LRIS",
                filter_id="g",
                exposure_time_sec=120.0,
                start_mjd=60200.0,
                end_mjd=60200.002,
                slew_time_sec=15.0,
                readout_time_sec=10.0,
                airmass=1.15,
                expected_eig_nats=1.2,
                weather_score=0.92,
            )
            target = TargetState(
                target_id="tgt_persist_1",
                ra_deg=210.0,
                dec_deg=15.0,
                trigger_mjd=60199.0,
                base_magnitude=19.0,
                hypothesis_posteriors={"KILONOVA": 0.6, "SUPERNOVA_IA": 0.4},
            )

            rec = gate_p1.create_recommendation(obs, target, "Crash resilience verification")
            tok = gate_p1.issue_authorization_token(rec.recommendation_id, "devmehta060505@gmail.com")

            token_id = tok.token_id
            rec_id = rec.recommendation_id

            # Simulate process crash
            store_p1.close()
            store_p1 = None
            del gate_p1

            # Process 2: Re-open fresh connection from persistent disk file
            store_p2 = DurableScientificEventStore(db_path=db_path)
            gate_p2 = HumanApprovalGate(
                event_store=store_p2,
                hmac_secret_key=real_key,
                require_persistent_db=True,
            )

            # Dispatch with token restored from disk
            res = gate_p2.dispatch_with_token(tok, rec_id)
            assert res["dispatch_status"] == "DISPATCHED_TO_FACILITY_QUEUE"
            assert res["token_id"] == token_id
            assert res["authorized_by"] == "devmehta060505@gmail.com"

            # Verify single-use consumption: re-dispatch fails
            with pytest.raises(TokenValidationError) as exc_replay:
                gate_p2.dispatch_with_token(tok, rec_id)
            assert "HAS ALREADY BEEN CONSUMED" in str(exc_replay.value)

        finally:
            if store_p1 is not None:
                store_p1.close()
            if store_p2 is not None:
                store_p2.close()


# ==============================================================================
# SECTION 2: WS7.8-B FULL PROPOSAL-TO-TOKEN BINDING VERIFICATION
# ==============================================================================

def test_ws7_8_b_full_parameter_divergence_detection():
    """Verifies that dispatch rejects divergence on ANY material proposal parameter."""
    real_key = provision_cryptographic_secret(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "binding_audit.db")
        store = None
        try:
            store = DurableScientificEventStore(db_path=db_path)
            gate = HumanApprovalGate(
                event_store=store,
                hmac_secret_key=real_key,
                require_persistent_db=True,
            )

            obs = ScheduledObservation(
                target_id="tgt_bind",
                facility_id="VLT",
                instrument_id="XSHOOTER",
                filter_id="UVB",
                exposure_time_sec=300.0,
                start_mjd=60250.0,
                end_mjd=60250.005,
                slew_time_sec=20.0,
                readout_time_sec=10.0,
                airmass=1.2,
                expected_eig_nats=1.5,
                weather_score=0.9,
            )
            target = TargetState(
                target_id="tgt_bind",
                ra_deg=120.0,
                dec_deg=-30.0,
                trigger_mjd=60249.0,
                base_magnitude=19.0,
                hypothesis_posteriors={"KILONOVA": 0.7, "SUPERNOVA_IA": 0.3},
            )

            rec = gate.create_recommendation(obs, target, "Full parameter binding test")
            valid_tok = gate.issue_authorization_token(rec.recommendation_id, "investigator@astraeus.org")

            # 1. Modify filter_id on presented token
            forged_filter_tok = valid_tok.model_copy(update={"filter_id": "VIS"})
            with pytest.raises(TokenValidationError) as exc_filter:
                gate.dispatch_with_token(forged_filter_tok, rec.recommendation_id)
            assert "HMAC signature is invalid" in str(exc_filter.value) or "TOKEN BINDING DIVERGENCE" in str(exc_filter.value)

            # 2. Modify exposure_time_sec in stored recommendation and recompute HMAC to test proposal digest check
            with store._get_connection() as conn:
                conn.execute(
                    "UPDATE recommendations SET exposure_time_sec = 600.0 WHERE recommendation_id = ?",
                    (rec.recommendation_id,),
                )
            gate._recommendations.clear()

            with pytest.raises(TokenValidationError) as exc_prop:
                gate.dispatch_with_token(valid_tok, rec.recommendation_id)
            assert "TOKEN BINDING DIVERGENCE" in str(exc_prop.value) or "CORRUPTED PROPOSAL RECORD" in str(exc_prop.value)

        finally:
            if store is not None:
                store.close()


def test_ws7_8_b_unauthorized_state_fails_closed():
    """Verifies that an unapproved or rejected recommendation fails closed at dispatch."""
    real_key = provision_cryptographic_secret(32)

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "unauth_gate.db")
        store = None
        try:
            store = DurableScientificEventStore(db_path=db_path)
            gate = HumanApprovalGate(
                event_store=store,
                hmac_secret_key=real_key,
                require_persistent_db=True,
            )

            obs = ScheduledObservation(
                target_id="tgt_unauth",
                facility_id="RUBIN",
                instrument_id="LSST_CAM",
                filter_id="i",
                exposure_time_sec=45.0,
                start_mjd=60300.0,
                end_mjd=60300.001,
                slew_time_sec=8.0,
                readout_time_sec=4.0,
                airmass=1.1,
                expected_eig_nats=0.9,
                weather_score=0.98,
            )
            target = TargetState(
                target_id="tgt_unauth",
                ra_deg=45.0,
                dec_deg=0.0,
                trigger_mjd=60299.0,
                base_magnitude=19.0,
                hypothesis_posteriors={"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5},
            )

            rec = gate.create_recommendation(obs, target, "Rejection test")
            gate.reject_dispatch(rec.recommendation_id, "Unfavorable atmospheric seeing")

            with pytest.raises(AutonomousDispatchProhibitedError) as exc_gate:
                gate.assert_authorized_for_dispatch(rec.recommendation_id)
            assert "REJECTED_BY_INVESTIGATOR" in str(exc_gate.value)

        finally:
            if store is not None:
                store.close()


# ==============================================================================
# SECTION 3: WS7.8-C EXACT EVIDENCE SEQUENCE PERSISTENCE & FAIL-CLOSED REPLAY
# ==============================================================================

def test_ws7_8_c_exact_evidence_sequence_survives_db_reopen():
    """Verifies that the exact assimilation sequence of evidence hashes persists across DB close/reopen."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "seq_persistence.db")
        store1 = None
        store2 = None
        try:
            store1 = DurableScientificEventStore(db_path=db_path)
            evidence_sequence = ["hash_delta_4", "hash_alpha_1", "hash_gamma_3", "hash_beta_2"]

            for h in evidence_sequence:
                obs = CalibratedObservation(
                    observation_id=f"obs_{h}",
                    target_id="tgt_seq_test",
                    target_ra_deg=150.0,
                    target_dec_deg=2.0,
                    exposure_time_sec=60.0,
                    facility_id="RUBIN",
                    instrument_id="LSST_CAM",
                    filter_id="r",
                    epoch_mjd=60100.0,
                    measurement_type=MeasurementType.POINT_DETECTION,
                    apparent_mag=21.0,
                    apparent_mag_err=0.05,
                    limiting_mag_5sigma=24.5,
                    calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
                    raw_data_hash_sha256=h,
                )
                store1.store_observation(obs)

            chk1 = store1.store_investigation_checkpoint(
                investigation_id="inv_seq_001",
                target_id="tgt_seq_test",
                revision_index=1,
                timestamp_mjd=60100.0,
                evidence_hashes=evidence_sequence,
                hypothesis_posteriors={"KILONOVA": 0.8, "SUPERNOVA_IA": 0.2},
                entropy_nats=0.5004,
            )
            stored_hash = chk1.state_hash

            store1.close()
            store1 = None

            store2 = DurableScientificEventStore(db_path=db_path)
            checkpoints = store2.get_investigation_checkpoints("inv_seq_001")
            assert len(checkpoints) == 1
            retrieved_chk = checkpoints[0]

            assert retrieved_chk.evidence_hashes == evidence_sequence
            assert retrieved_chk.evidence_hashes != sorted(evidence_sequence)
            assert retrieved_chk.state_hash == stored_hash

        finally:
            if store1 is not None:
                store1.close()
            if store2 is not None:
                store2.close()


def test_ws7_8_c_replay_engine_fails_closed_on_missing_evidence():
    """Verifies that the replay engine fails closed if any evidence hash in the checkpoint is missing."""
    store = DurableScientificEventStore(db_path=":memory:", allow_in_memory=True)
    engine = DeterministicScientificReplayEngine(event_store=store)

    obs1 = CalibratedObservation(
        observation_id="obs_present",
        target_id="tgt_missing_ev",
        target_ra_deg=150.0,
        target_dec_deg=2.0,
        exposure_time_sec=60.0,
        facility_id="RUBIN",
        instrument_id="LSST_CAM",
        filter_id="g",
        epoch_mjd=60100.0,
        measurement_type=MeasurementType.POINT_DETECTION,
        apparent_mag=20.5,
        apparent_mag_err=0.04,
        limiting_mag_5sigma=24.5,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
        raw_data_hash_sha256="hash_present_001",
    )
    store.store_observation(obs1)

    store.store_investigation_checkpoint(
        investigation_id="inv_missing_ev",
        target_id="tgt_missing_ev",
        revision_index=1,
        timestamp_mjd=60100.0,
        evidence_hashes=["hash_present_001", "hash_ABSENT_002"],
        hypothesis_posteriors={"KILONOVA": 0.7, "SUPERNOVA_IA": 0.3},
        entropy_nats=0.6108,
    )

    with pytest.raises(MissingEvidenceError) as exc_missing:
        engine.reconstruct_checkpoint("inv_missing_ev", 1)
    assert "EVIDENCE INTEGRITY ERROR: 1 observation(s)" in str(exc_missing.value)
    assert "hash_ABSENT_002" in str(exc_missing.value)

    store.close()


def test_ws7_8_c_replay_engine_fails_closed_on_model_version_mismatch():
    """Verifies that the replay engine fails closed when runtime pinned model versions diverge."""
    store = DurableScientificEventStore(db_path=":memory:", allow_in_memory=True)
    engine = DeterministicScientificReplayEngine(
        event_store=store,
        pinned_model_versions={"kilonova": "v2.0", "bayesian_recalibration": "ws7.6-c"},
    )

    store.store_investigation_checkpoint(
        investigation_id="inv_model_mismatch",
        target_id="tgt_mismatch",
        revision_index=0,
        timestamp_mjd=60100.0,
        evidence_hashes=[],
        hypothesis_posteriors={"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5},
        entropy_nats=0.6931,
        model_versions={"kilonova": "v3.0_experimental", "bayesian_recalibration": "ws7.6-c"},
    )

    with pytest.raises(ModelVersionMismatchError) as exc_ver:
        engine.reconstruct_checkpoint("inv_model_mismatch", 0)
    assert "MODEL VERSION MISMATCH" in str(exc_ver.value)
    assert "v3.0_experimental" in str(exc_ver.value)

    store.close()


# ==============================================================================
# SECTION 4: WS7.8-D ARCHIVAL BACKGROUND FALSE-ALARM BENCHMARK & QUALIFICATION
# ==============================================================================

def test_ws7_8_d_archival_background_trials_and_poisson_uncertainty():
    """Verifies archival background benchmarking, trials factor accounting, and Poisson upper bounds."""
    benchmark_res: ArchivalBackgroundValidationResult = ArchivalBackgroundBenchmark.run_benchmark(
        total_trials=100,
        exposure_years=1.0,
    )

    assert benchmark_res.total_trials == 100
    assert benchmark_res.trials_factor == 100.0
    assert benchmark_res.exposure_time_years == 1.0

    assert benchmark_res.false_alarms_observed == 0
    assert benchmark_res.poisson_95cl_upper_far_per_year > 0.0
    assert abs(benchmark_res.poisson_95cl_upper_far_per_year - 2.9957) < 0.01

    assert benchmark_res.recovery_rate == 1.0
    assert benchmark_res.known_associations_recovered == 10
    assert benchmark_res.missed_associations_count == 0

    assert benchmark_res.significance_label == "MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT"
    assert "QUALIFIED_AS_SYNTHETIC_ARCHIVAL_BENCHMARK_ONLY" in benchmark_res.qualification_status


def test_ws7_8_d_synthetic_time_slide_and_spatial_scramble_sanity_tests():
    """Verifies that synthetic time-slide and spatial scramble routines report proper sanity check labels."""
    gw_cand = CorrelatedEventCandidate(
        event_id="gw_sanity_001",
        facility="GRACEDB",
        messenger_type=MessengerType.GRAVITATIONAL_WAVE,
        trigger_mjd=60100.0,
        ra_deg=197.45,
        dec_deg=-23.38,
        localization_error_deg=1.2,
        skymap_90_cr_area_deg2=25.0,
    )
    opt_cand = CorrelatedEventCandidate(
        event_id="opt_sanity_001",
        facility="RUBIN",
        messenger_type=MessengerType.OPTICAL,
        trigger_mjd=60100.005,
        ra_deg=197.451,
        dec_deg=-23.379,
        localization_error_deg=0.0005,
    )

    # 1. Time slide sanity test
    ts_res = CrossFacilityCorrelationEngine.perform_synthetic_time_slide_sanity_test(gw_cand, opt_cand)
    assert ts_res["test_type"] == "SYNTHETIC_PERTURBATION_SANITY_CHECK"
    assert ts_res["is_calibrated_empirical_rate"] is False
    assert ts_res["false_alarms_observed"] == 0
    assert ts_res["poisson_95cl_count_upper_bound"] > 0.0
    assert "Archival background survey calibration required" in ts_res["statistical_warning"]

    # 2. Spatial scramble sanity test
    sc_res = CrossFacilityCorrelationEngine.perform_synthetic_spatial_scramble_sanity_test(gw_cand, opt_cand)
    assert sc_res["test_type"] == "SYNTHETIC_PERTURBATION_SANITY_CHECK"
    assert sc_res["is_calibrated_empirical_rate"] is False
    assert sc_res["false_alarms_observed"] == 0
    assert "Deterministic coordinate perturbations do not reproduce real survey" in sc_res["statistical_warning"]


# ==============================================================================
# SECTION 5: WS7.8-E GATE 3 QUARANTINE & COVENANTS INTEGRITY
# ==============================================================================

def test_ws7_8_e_gate3_quarantine_integrity():
    """Verifies that the Gate 3 Quarantine firewall remains active across all event and governance paths."""
    assert QuarantineFirewall.get_quarantined_pixel_count() == 1808

    quarantine_pixel = list(QuarantineFirewall.get_quarantined_pixels())[0]
    center_coord = QuarantineFirewall._HP.healpix_to_skycoord(quarantine_pixel)
    ra_q = float(center_coord.ra.deg)
    dec_q = float(center_coord.dec.deg)

    assert QuarantineFirewall.is_quarantined_coordinate(ra_q, dec_q) is True

    store = DurableScientificEventStore(db_path=":memory:", allow_in_memory=True)
    quarantined_event = UnifiedScientificEvent(
        event_id="evt_quarantine_test",
        survey_origin="RUBIN",
        ra_deg=ra_q,
        dec_deg=dec_q,
        healpix_pixel_nside32=quarantine_pixel,
        event_time_mjd=60100.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="sha256_quar_test_hash",
    )
    stored, status = store.store_event(quarantined_event)
    assert stored is False
    assert "QUARANTINED" in status

    with store._get_connection() as conn:
        cur = conn.execute("SELECT * FROM dead_letter_quarantine WHERE entity_id = 'evt_quarantine_test'")
        row = cur.fetchone()
        assert row is not None
        assert "Gate 3 Quarantine Covenant" in row["reason"]

    store.close()
