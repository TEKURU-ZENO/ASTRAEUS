"""
ASTRAEUS Phase 7 Master Acceptance Scenario - Workstream 7.6:
Closed-Loop Observation Ingestion, Quality Auditing, Bayesian Recalibration & Sequential Replanning.

Comprehensive end-to-end master scenario demonstrating the full closed-loop observational feedback cycle:
1. Alert Ingestion: Fast-fading optical transient at unquarantined coordinates (RA=190.0, Dec=10.0).
2. Tri-Hypothesis Generation: Competing hypotheses (H1: Kilonova, H2: CCSN, H3: TDE/AGN) with prior entropy.
3. WS7.5 Initial Scheduling: Multi-facility schedule generation (Swift UVOT + VLT X-shooter) with Human-in-the-Loop gate.
4. Non-Detection Upper Limit Ingestion: Swift UVOT uvw2 non-detection (m_lim = 21.5 mag) rejecting TDE model via censored likelihood.
5. Quality Auditor Defense: Ingestion of deliberately misleading optical measurement with zero-point drift; caught and blocked.
6. Sequential Dynamic Replanning: Recalculates EIG and regenerates feasible action set, prioritizing VLT NIR.
7. NIR Detection Ingestion: VLT X-shooter calibrated NIR detection (m = 18.2 mag) promoting Kilonova posterior to >0.98.
8. Posterior Predictive Checking: Reduced chi-square and p-value checks verifying Kilonova as STRONGER_EVIDENCE.
9. Reproducible Reporting: Sealed IVOA Prov-DM cryptographic digest and Zero-Touch Quarantine Covenant verification.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import math
import hashlib
import json
from datetime import datetime, timezone

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_agents.observatory.facility_registry import (
    FacilityRegistry,
    WavelengthRegime,
)
from astraeus_agents.observatory.schedulers import (
    TargetState,
    RollingHorizonOptimizer,
    GreedyEIGPlanner,
)
from astraeus_agents.observatory.human_approval_interface import (
    HumanApprovalGate,
    ApprovalStatus,
    AutonomousDispatchProhibitedError,
)
from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
    CalibrationStatus,
)
from astraeus_agents.feedback.measurement_auditor import (
    MeasurementQualityAuditor,
    MeasurementAuditResult,
)
from astraeus_agents.feedback.bayesian_recalibration import (
    BayesianRecalibrationEngine,
)
from astraeus_agents.feedback.evidence_ledger import (
    ObservationEvidenceLedger,
    LedgerEntryStatus,
)
from astraeus_agents.feedback.posterior_predictive import (
    PosteriorPredictiveChecker,
)
from astraeus_agents.feedback.sequential_replanner import (
    SequentialReplanningEngine,
)
from astraeus_agents.reground.models import HypothesisEvaluationStatus


def test_ws7_6_closed_loop_master_acceptance_scenario():
    """
    Executes the complete multi-facility kilonova follow-up, observation ingestion,
    quality auditing, Bayesian recalibration, and sequential replanning pipeline.
    """
    print("\n" + "=" * 80)
    print("ASTRAEUS PHASE 7 - WS7.6 CLOSED-LOOP MASTER ACCEPTANCE SCENARIO")
    print("=" * 80)

    # =========================================================================
    # Step 1: Alert Arrival & Gate 3 Quarantine Clearance
    # =========================================================================
    firewall = QuarantineFirewall()
    target_ra = 180.0
    target_dec = 10.0
    trigger_mjd = 60000.0

    # Verify target coordinates are strictly outside the 1,808 quarantined pixels
    firewall.assert_not_quarantined(target_ra, target_dec)
    print(f"[Stage 1] Transient AT2026kna received at RA={target_ra:.1f}, Dec={target_dec:.1f}. Gate 3 CLEARED.")

    # =========================================================================
    # Step 2: Tri-Hypothesis Formulation & Prior Entropy State
    # =========================================================================
    # Initial competing hypotheses from REGROUND multi-agent debate
    initial_posteriors = {
        "KILONOVA": 0.40,
        "CCSN": 0.35,
        "TDE_AGN": 0.25,
    }
    initial_entropy = BayesianRecalibrationEngine.calculate_entropy(initial_posteriors)
    assert 1.0 < initial_entropy < 1.15
    print(f"[Stage 2] Prior Hypotheses: {initial_posteriors}")
    print(f"          Initial Shannon Entropy: {initial_entropy:.4f} nats.")

    # =========================================================================
    # Step 3: WS7.5 Initial Multi-Facility Schedule Generation & Human Gate
    # =========================================================================
    reg = FacilityRegistry()
    facilities = [reg.get_facility("SWIFT"), reg.get_facility("VLT"), reg.get_facility("ZTF")]

    target = TargetState(
        target_id="tgt_at2026kna",
        ra_deg=target_ra,
        dec_deg=target_dec,
        trigger_mjd=trigger_mjd,
        base_magnitude=17.0,
        hypothesis_posteriors=dict(initial_posteriors),
        is_too=True,
    )

    optimizer = RollingHorizonOptimizer(horizon_depth=2)
    initial_schedule = optimizer.plan_schedule(
        targets=[target],
        facilities=facilities,
        start_mjd=trigger_mjd + 0.5,
        duration_hours=24.0,
        slew_origin_ra_deg=target_ra,
        slew_origin_dec_deg=target_dec,
    )
    assert len(initial_schedule.observations) >= 1
    assert initial_schedule.num_hard_violations == 0

    # Enforce Human-in-the-Loop review: autonomous dispatch without token fails
    human_gate = HumanApprovalGate()
    first_obs = initial_schedule.observations[0]
    rec0 = human_gate.create_recommendation(
        obs=first_obs,
        target=target,
        scientific_justification="Initial multi-facility rapid follow-up for AT2026kna.",
    )
    assert rec0.approval_status == ApprovalStatus.PENDING_HUMAN_REVIEW
    with pytest.raises(AutonomousDispatchProhibitedError):
        human_gate.assert_authorized_for_dispatch(rec0.recommendation_id)

    # Lead investigator Dev Mehta explicitly authorizes dispatch
    rec0_auth = human_gate.authorize_dispatch(
        recommendation_id=rec0.recommendation_id,
        investigator_email="devmehta060505@gmail.com",
    )
    assert rec0_auth.approval_status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
    human_gate.assert_authorized_for_dispatch(rec0.recommendation_id)
    print(f"[Stage 3] Initial schedule {initial_schedule.schedule_id} generated & approved by Lead Investigator.")

    # =========================================================================
    # Step 4: Swift UVOT Non-Detection Upper Limit Ingestion & Censored Recalibration
    # =========================================================================
    ledger = ObservationEvidenceLedger()

    swift_raw = "SWIFT_UVOT_UVW2_EXPOSURE_1200S_EPOCH_60001.0_UPPER_LIMIT_21.5"
    swift_obs = CalibratedObservation(
        observation_id="obs_swift_uvot_001",
        target_id="tgt_at2026kna",
        target_ra_deg=target_ra,
        target_dec_deg=target_dec,
        facility_id="SWIFT",
        instrument_id="UVOT",
        filter_id="uvw2",
        measurement_type=MeasurementType.UPPER_LIMIT_CENSORED,
        epoch_mjd=trigger_mjd + 1.0,
        limiting_mag_5sigma=21.5,
        exposure_time_sec=1200.0,
        airmass=1.0,
        seeing_fwhm_arcsec=0.0,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
        calibration_reference_catalog="Swift_CALDB_v4.9",
        raw_data_hash_sha256=CalibratedObservation.compute_payload_hash(swift_raw),
    )

    # Audit Swift observation
    swift_audit = MeasurementQualityAuditor.audit_observation(swift_obs)
    assert swift_audit.is_valid_for_inference is True
    assert swift_audit.recommended_action == "ACCEPT_FOR_INFERENCE"

    # Ingest into evidence ledger
    ledger_entry1, is_new1 = ledger.ingest_observation(swift_obs, swift_audit)
    assert is_new1 is True
    assert ledger_entry1.audit_status == LedgerEntryStatus.ACTIVE_IN_POSTERIOR

    # Recalibrate posteriors using exact censored survival likelihood
    recal1 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=initial_posteriors,
        observation=swift_obs,
        effective_wavelength_angstrom=2600.0,
        regime=WavelengthRegime.UV,
        time_since_trigger_days=1.0,
    )

    # TDE/AGN model predicted bright UV emission (m ~ 18.2), which is ruled out by non-detection at 21.5 mag
    assert recal1.updated_posteriors["TDE_AGN"] < 0.005
    # Kilonova model predicted heavy UV line blanketing (m ~ 23.1 > 21.5), so it survives
    assert recal1.updated_posteriors["KILONOVA"] > initial_posteriors["KILONOVA"]
    assert recal1.entropy_delta_nats < 0.0  # Uncertainty reduced
    posteriors_t1 = recal1.updated_posteriors
    print(f"[Stage 4] Swift UVOT uvw2 non-detection ingested (m_lim=21.5 mag).")
    print(f"          Updated Posteriors: {posteriors_t1}")
    print(f"          TDE/AGN successfully rejected (P={posteriors_t1['TDE_AGN']:.5f}).")

    # =========================================================================
    # Step 5: Quality Auditor Detects & Quarantines Misleading Measurement
    # =========================================================================
    corrupt_raw = "ANOMALOUS_GROUND_IMAGING_UNVERIFIED_ZP_DRIFT"
    corrupt_obs = CalibratedObservation(
        observation_id="obs_corrupt_test_002",
        target_id="tgt_at2026kna",
        target_ra_deg=target_ra,
        target_dec_deg=target_dec,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=trigger_mjd + 1.2,
        apparent_mag=16.5,
        apparent_mag_err=0.05,
        exposure_time_sec=300.0,
        airmass=1.3,
        seeing_fwhm_arcsec=1.2,
        calibration_status=CalibrationStatus.CALIBRATION_SUSPECT,
        calibration_zero_point=25.0,
        nominal_zero_point=25.80,  # Zero-point drift = 0.80 mag (> 0.35 mag threshold!)
        raw_data_hash_sha256=CalibratedObservation.compute_payload_hash(corrupt_raw),
    )

    corrupt_audit = MeasurementQualityAuditor.audit_observation(corrupt_obs)
    assert corrupt_audit.is_valid_for_inference is False
    assert corrupt_audit.recommended_action == "REJECT_CORRUPTED"
    assert any("Severe zero-point drift" in a for a in corrupt_audit.anomalies_detected)

    ledger_corrupt_entry, is_new_corrupt = ledger.ingest_observation(corrupt_obs, corrupt_audit)
    assert is_new_corrupt is False
    assert ledger_corrupt_entry.audit_status == LedgerEntryStatus.REJECTED_AUDIT_FAILURE

    # Verify active observations in ledger do not include corrupt measurement
    active_obs = ledger.get_active_observations()
    assert len(active_obs) == 1
    assert active_obs[0].observation_id == "obs_swift_uvot_001"
    print(f"[Stage 5] Misleading measurement detected (ZP drift=0.80 mag). REJECTED & BLOCKED from posterior.")

    # =========================================================================
    # Step 6: Sequential Dynamic Replanning
    # =========================================================================
    replanning_engine = SequentialReplanningEngine(
        facility_registry=reg,
        optimizer=optimizer,
    )

    replan_record = replanning_engine.trigger_sequential_replan(
        current_schedule=initial_schedule,
        target=target,
        facilities=facilities,
        triggering_observation=swift_obs,
        updated_posteriors=posteriors_t1,
        current_mjd=trigger_mjd + 1.0,
        remaining_duration_hours=24.0,
    )

    assert replan_record.revision_number == 1
    assert replan_record.entropy_reduction_nats > 0.0
    assert len(replan_record.revision_hash_sha256) == 64
    assert replan_record.optimal_next_observation is not None
    print(f"[Stage 6] Sequential dynamic replanning executed. Schedule revision: {replan_record.new_schedule_id}.")
    print(f"          Next prioritized facility: {replan_record.optimal_next_observation.facility_id} "
          f"({replan_record.optimal_next_observation.filter_id})")

    # Authorize revised schedule
    rec_replan = human_gate.create_recommendation(
        obs=replan_record.optimal_next_observation,
        target=target,
        scientific_justification="Approved revised schedule prioritizing VLT X-shooter NIR characterization.",
    )
    human_gate.authorize_dispatch(
        recommendation_id=rec_replan.recommendation_id,
        investigator_email="devmehta060505@gmail.com",
    )
    human_gate.assert_authorized_for_dispatch(rec_replan.recommendation_id)

    # =========================================================================
    # Step 7: VLT X-Shooter NIR Calibrated Detection Ingestion
    # =========================================================================
    vlt_raw = "VLT_XSHOOTER_NIR_SPEC_CALIBRATED_FITS_60002.0"
    vlt_obs = CalibratedObservation(
        observation_id="obs_vlt_xshooter_003",
        target_id="tgt_at2026kna",
        target_ra_deg=target_ra,
        target_dec_deg=target_dec,
        facility_id="VLT",
        instrument_id="XSHOOTER",
        filter_id="NIR_SPEC",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=trigger_mjd + 2.0,
        apparent_mag=18.20,
        apparent_mag_err=0.06,
        exposure_time_sec=600.0,
        airmass=1.15,
        seeing_fwhm_arcsec=0.75,
        calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
        calibration_reference_catalog="Gaia_DR3_2MASS",
        calibration_zero_point=27.50,
        nominal_zero_point=27.50,
        raw_data_hash_sha256=CalibratedObservation.compute_payload_hash(vlt_raw),
    )

    vlt_audit = MeasurementQualityAuditor.audit_observation(vlt_obs)
    assert vlt_audit.is_valid_for_inference is True
    assert vlt_audit.recommended_action == "ACCEPT_FOR_INFERENCE"

    ledger_entry2, is_new2 = ledger.ingest_observation(vlt_obs, vlt_audit)
    assert is_new2 is True
    assert ledger_entry2.audit_status == LedgerEntryStatus.ACTIVE_IN_POSTERIOR

    # Recalibrate posteriors with VLT NIR observation
    recal2 = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=posteriors_t1,
        observation=vlt_obs,
        effective_wavelength_angstrom=12500.0,
        regime=WavelengthRegime.NIR,
        time_since_trigger_days=2.0,
    )

    final_posteriors = recal2.updated_posteriors
    print(f"[Stage 7] VLT X-shooter NIR detection ingested (m = 18.20 +- 0.06 mag).")
    print(f"          Final Posteriors: {final_posteriors}")

    # Verify Kilonova posterior rises to >0.98
    assert final_posteriors["KILONOVA"] > 0.98
    assert final_posteriors["CCSN"] < 0.02
    assert final_posteriors["TDE_AGN"] < 0.001
    assert recal2.posterior_entropy_nats < 0.10

    # =========================================================================
    # Step 8: Posterior Predictive Checking & Model Criticism
    # =========================================================================
    active_observations = ledger.get_active_observations()
    assessment = PosteriorPredictiveChecker.evaluate_model_fit(
        target_id="tgt_at2026kna",
        hypothesis_posteriors=final_posteriors,
        accumulated_observations=active_observations,
        trigger_mjd=trigger_mjd,
    )

    assert assessment.all_models_misspecified is False
    assert assessment.preferred_hypothesis == "KILONOVA"

    kn_critique = assessment.hypotheses_critiques["KILONOVA"]
    assert kn_critique.is_statistically_consistent is True
    assert kn_critique.goodness_of_fit_p_value >= 0.40
    assert kn_critique.evaluation_status == HypothesisEvaluationStatus.STRONGER_EVIDENCE

    ccsn_critique = assessment.hypotheses_critiques["CCSN"]
    assert ccsn_critique.is_statistically_consistent is False
    assert ccsn_critique.evaluation_status == HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS

    print(f"[Stage 8] Posterior Predictive Check passed:")
    print(f"          KILONOVA: reduced_chi2={kn_critique.reduced_chi2:.2f}, p={kn_critique.goodness_of_fit_p_value:.4f} -> {kn_critique.evaluation_status.value}")
    print(f"          CCSN:     reduced_chi2={ccsn_critique.reduced_chi2:.2f}, p={ccsn_critique.goodness_of_fit_p_value:.4f} -> {ccsn_critique.evaluation_status.value}")

    # =========================================================================
    # Step 9: Sealed Scientific Provenance Report & Covenant Verification
    # =========================================================================
    # Ledger summary
    summary = ledger.get_ledger_summary()
    assert summary["total_observations_recorded"] == 3
    assert summary["active_observations_in_posterior"] == 2
    assert summary["rejected_observations"] == 1

    # Build reproducible IVOA Prov-DM sealed scientific digest
    report_manifest = {
        "scenario": "ASTRAEUS_PHASE7_WS7_6_MASTER_ACCEPTANCE",
        "investigation_target": "tgt_at2026kna",
        "lead_investigator": "Dev Mehta <devmehta060505@gmail.com>",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "quarantine_compliance": {
            "cleared": True,
            "quarantined_pixels_touched": 0,
            "covenant": "GATE3_ZERO_TOUCH_PRESERVED",
        },
        "evidence_ledger_observations": [
            ledger_entry1.entry_id,
            ledger_entry2.entry_id,
        ],
        "rejected_corrupt_observations": [
            ledger_corrupt_entry.entry_id,
        ],
        "final_hypothesis_posteriors": final_posteriors,
        "preferred_hypothesis": assessment.preferred_hypothesis,
        "evaluation_status": kn_critique.evaluation_status.value,
        "replanning_revisions": [r.revision_hash_sha256 for r in replanning_engine.get_revision_history()],
    }

    serialized_manifest = json.dumps(report_manifest, sort_keys=True)
    sealed_digest = hashlib.sha256(serialized_manifest.encode("utf-8")).hexdigest()
    assert len(sealed_digest) == 64
    print(f"[Stage 9] Provenance sealed digest: {sealed_digest}")
    print("=" * 80)
    print("WS7.6 MASTER ACCEPTANCE SCENARIO COMPLETED WITH FULL VERIFICATION")
    print("=" * 80 + "\n")
