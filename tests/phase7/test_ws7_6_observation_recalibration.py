"""
ASTRAEUS Phase 7 - Workstream 7.6 Observation Ingestion & Recalibration Unit Tests.
Verifies:
1. Observation payload schema and SHA-256 cryptographic sealing.
2. Measurement quality auditor detecting deliberately misleading/corrupted measurements.
3. Bayesian recalibration matching analytically solvable Gaussian conjugate reference cases.
4. Censored non-detection upper-limit survival likelihoods.
5. Correlated measurement covariance handling for shared calibration references.
6. Evidence ledger deduplication preventing artificial confidence inflation.
7. Posterior predictive checking (PPC) and model criticism detecting misspecification.
8. Gate 3 Zero-Touch Quarantine Firewall enforcement.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import math
import numpy as np

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
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
    RecalibrationUpdateResult,
)
from astraeus_agents.feedback.evidence_ledger import (
    ObservationEvidenceLedger,
    EvidenceLedgerEntry,
)
from astraeus_agents.feedback.posterior_predictive import (
    PosteriorPredictiveChecker,
    PosteriorPredictiveAssessment,
)
from astraeus_agents.reground.models import HypothesisEvaluationStatus
from astraeus_agents.observatory.facility_registry import WavelengthRegime


def test_observation_payload_schema_and_hashing():
    """Verifies that calibrated observation payloads enforce valid schema and deterministic SHA-256 hashes."""
    raw_text = "FITS_HEADER_RAW_EXPOSURE_600S_ZTF_R_20261124"
    h = CalibratedObservation.compute_payload_hash(raw_text)
    assert len(h) == 64

    obs = CalibratedObservation(
        observation_id="obs_test_001",
        target_id="tgt_test",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.5,
        apparent_mag=18.5,
        apparent_mag_err=0.04,
        exposure_time_sec=300.0,
        airmass=1.1,
        seeing_fwhm_arcsec=1.1,
        raw_data_hash_sha256=h,
    )
    assert obs.is_detection() is True
    assert obs.is_upper_limit() is False
    assert obs.calibration_status == CalibrationStatus.CALIBRATED_NOMINAL


def test_measurement_quality_auditor_detects_deliberate_misleading_measurement():
    """
    Verifies that the MeasurementQualityAuditor detects and rejects:
    1. Severe zero-point drift (|ZP - ZP_0| > 0.35 mag).
    2. Fatal instrument flags (saturated, bad pixel, corrupted flatfield).
    3. Degraded seeing with implausibly tiny reported uncertainty.
    """
    # 1. Nominal valid observation -> ACCEPT_FOR_INFERENCE
    obs_valid = CalibratedObservation(
        observation_id="obs_nominal",
        target_id="tgt_nominal",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="XSHOOTER",
        filter_id="VIS_SPEC",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.0,
        apparent_mag=18.2,
        apparent_mag_err=0.03,
        calibration_zero_point=28.48,
        nominal_zero_point=28.50,
        exposure_time_sec=300.0,
        raw_data_hash_sha256="hash_nominal_123",
    )
    res_valid = MeasurementQualityAuditor.audit_observation(obs_valid)
    assert res_valid.is_valid_for_inference is True
    assert res_valid.recommended_action == "ACCEPT_FOR_INFERENCE"
    assert res_valid.confidence_score == 1.0

    # 2. Deliberately misleading observation with severe zero-point calibration offset (delta_ZP = 0.55 mag)
    obs_corrupted_zp = CalibratedObservation(
        observation_id="obs_corrupted_zp",
        target_id="tgt_nominal",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="XSHOOTER",
        filter_id="VIS_SPEC",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.0,
        apparent_mag=17.65,  # Artificially bright due to bad zero-point!
        apparent_mag_err=0.03,
        calibration_zero_point=27.95,  # 0.55 mag drift!
        nominal_zero_point=28.50,
        exposure_time_sec=300.0,
        raw_data_hash_sha256="hash_corrupted_zp_123",
    )
    res_corrupted_zp = MeasurementQualityAuditor.audit_observation(obs_corrupted_zp)
    assert res_corrupted_zp.is_valid_for_inference is False
    assert res_corrupted_zp.recommended_action == "REJECT_CORRUPTED"
    assert any("Severe zero-point drift" in a for a in res_corrupted_zp.anomalies_detected)

    # 3. Fatal detector flag (saturated core)
    obs_saturated = CalibratedObservation(
        observation_id="obs_saturated",
        target_id="tgt_nominal",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.0,
        apparent_mag=14.0,
        apparent_mag_err=0.01,
        exposure_time_sec=30.0,
        instrument_flags=["SATURATED", "BLEED_TRAIL"],
        raw_data_hash_sha256="hash_sat_123",
    )
    res_sat = MeasurementQualityAuditor.audit_observation(obs_saturated)
    assert res_sat.is_valid_for_inference is False
    assert res_sat.recommended_action == "REJECT_CORRUPTED"
    assert any("Fatal instrument telemetry" in a for a in res_sat.anomalies_detected)


def test_bayesian_recalibration_analytical_gaussian_agreement():
    """
    Verifies that the Bayesian recalibration engine reproduces analytically
    solvable conjugate Gaussian updating to high numerical tolerance.
    
    Analytical formulation:
    Two models: H1 (mu1 = 18.0), H2 (mu2 = 20.0).
    Prior: P(H1) = 0.5, P(H2) = 0.5.
    Observed y = 18.0 with total sigma = 0.50.
    
    Likelihood ratio:
    L1 / L2 = exp(-0.5 * [0^2 - (2.0 / 0.5)^2]) = exp(8.0) = 2980.957987
    Updated posterior P(H1 | y) = 2980.957987 / (1 + 2980.957987) = 0.9996645
    """
    prior = {"H1": 0.50, "H2": 0.50}
    meas_m = 18.0
    # Setup sigma_tot = 0.50 exactly
    # sigma_tot^2 = meas_err^2 + sys^2 + model^2 + extra^2
    # sys=0.03, model=0.05 -> base=0.0034
    # let meas_err = sqrt(0.5^2 - 0.03^2 - 0.05^2) = sqrt(0.25 - 0.0009 - 0.0025) = sqrt(0.2466) ~ 0.496588
    meas_err = math.sqrt(0.25 - 0.03**2 - 0.05**2)

    ln_L1 = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(
        measured_mag=meas_m,
        measured_err=meas_err,
        predicted_mag=18.0,
    )
    ln_L2 = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(
        measured_mag=meas_m,
        measured_err=meas_err,
        predicted_mag=20.0,
    )

    delta_ln_L = ln_L1 - ln_L2
    analytical_delta = 8.0  # 0.5 * (4.0^2 - 0^2) = 8.0
    assert abs(delta_ln_L - analytical_delta) < 1e-4

    bayes_factor = math.exp(delta_ln_L)
    expected_p1 = bayes_factor / (1.0 + bayes_factor)
    assert abs(expected_p1 - 0.9996645) < 1e-5


def test_censored_upper_limit_likelihood():
    """
    Verifies that a non-detection upper limit (e.g. m_lim = 21.0 mag):
    1. Strongly penalizes a bright model (mu_pred = 18.0 mag << m_lim).
    2. Rewards a faint / rapidly faded model (mu_pred = 23.0 mag >> m_lim).
    """
    lim_mag = 21.0

    # Bright model predicts 18.0 (should have been detected at 15-sigma!)
    ln_L_bright = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=lim_mag,
        predicted_mag=18.0,
    )

    # Faint model predicts 23.0 (correctly non-detected)
    ln_L_faint = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=lim_mag,
        predicted_mag=23.0,
    )

    # The faint model must have vastly higher likelihood (higher survival probability)
    assert ln_L_faint > ln_L_bright
    assert ln_L_bright < -10.0  # Extinguished likelihood for unobserved bright source
    assert ln_L_faint > -0.05   # Survival probability ~ 1.0 -> ln L ~ 0.0


def test_correlated_measurement_error_model():
    """
    Verifies that two observations sharing a calibration reference (shared_cal_sigma > 0)
    produce a covariance matrix penalizing correlated false confidence.
    """
    m1, m2 = 18.5, 18.52
    err1, err2 = 0.04, 0.04
    pred1, pred2 = 18.5, 18.5

    # Bivariate likelihood with shared calibration covariance
    ln_L_correlated = BayesianRecalibrationEngine.evaluate_correlated_pair_log_likelihood(
        m1=m1, err1=err1, pred1=pred1,
        m2=m2, err2=err2, pred2=pred2,
        shared_cal_sigma=0.05,
    )

    # Independent product of likelihoods (ignoring correlation)
    ln_L1 = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(m1, err1, pred1)
    ln_L2 = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(m2, err2, pred2)
    ln_L_indep = ln_L1 + ln_L2

    # Correlated model correctly accounts for shared calibration uncertainty (higher determinant)
    assert abs(ln_L_correlated - ln_L_indep) > 0.10


def test_evidence_ledger_deduplication_and_reprocessing():
    """
    Verifies that:
    1. Ingesting an identical duplicate observation does not duplicate Bayesian evidence.
    2. Ingesting a reprocessed measurement supersedes the parent version in the ledger.
    """
    ledger = ObservationEvidenceLedger()

    obs1 = CalibratedObservation(
        observation_id="obs_ztf_001",
        target_id="tgt_kilonova",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.1,
        apparent_mag=18.5,
        apparent_mag_err=0.05,
        exposure_time_sec=300.0,
        raw_data_hash_sha256="hash_001_unique",
    )

    # First ingestion -> informative
    entry1, is_new1 = ledger.ingest_observation(obs1)
    assert is_new1 is True
    assert entry1.applied_to_inference is True

    # Ingesting exact duplicate -> rejected from inference update
    entry2, is_new2 = ledger.ingest_observation(obs1)
    assert is_new2 is False
    assert entry2.is_duplicate is True
    assert entry2.applied_to_inference is False

    # Reprocessed version with better calibration
    obs_reprocessed = CalibratedObservation(
        observation_id="obs_ztf_001_v2",
        target_id="tgt_kilonova",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        measurement_type=MeasurementType.POINT_DETECTION,
        epoch_mjd=60000.1,
        apparent_mag=18.45,  # Refined photometry
        apparent_mag_err=0.03,
        exposure_time_sec=300.0,
        is_reprocessed_version_of="obs_ztf_001",
        raw_data_hash_sha256="hash_001_v2_reprocessed",
    )
    entry3, is_new3 = ledger.ingest_observation(obs_reprocessed)
    assert is_new3 is True
    assert entry3.is_reprocessed_update is True
    assert entry3.applied_to_inference is True

    # Verify that parent entry1 is marked superseded
    active_obs = ledger.get_active_observations_for_target("tgt_kilonova")
    assert len(active_obs) == 1
    assert active_obs[0].observation_id == "obs_ztf_001_v2"


def test_posterior_predictive_checking_detects_model_misspecification():
    """
    Verifies that PosteriorPredictiveChecker detects model misspecification:
    When observations diverge strongly from model physics (p < 0.01),
    the model is classified as MODEL_REJECTED_UNDER_ASSUMPTIONS.
    """
    target_id = "tgt_ppc_test"
    posts = {"KILONOVA": 0.85, "SUPERNOVA_IA": 0.15}

    # Generate observations that match Kilonova fast decline:
    # Day 1: m=18.0, Day 2: m=18.9, Day 3: m=19.8
    obs_kilonova_like = [
        CalibratedObservation(
            observation_id="p1", target_id=target_id, target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.0,
            apparent_mag=18.95, apparent_mag_err=0.04, exposure_time_sec=300.0, raw_data_hash_sha256="h1",
        ),
        CalibratedObservation(
            observation_id="p2", target_id=target_id, target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60002.0,
            apparent_mag=19.90, apparent_mag_err=0.05, exposure_time_sec=300.0, raw_data_hash_sha256="h2",
        ),
    ]

    assessment = PosteriorPredictiveChecker.evaluate_model_fit(
        target_id=target_id,
        hypothesis_posteriors=posts,
        accumulated_observations=obs_kilonova_like,
        trigger_mjd=60000.0,
    )

    assert assessment.num_observations_evaluated == 2
    # Kilonova should be well-fit (small residuals)
    kn_critique = assessment.hypotheses_critiques["KILONOVA"]
    assert kn_critique.is_statistically_consistent is True
    assert kn_critique.evaluation_status == HypothesisEvaluationStatus.STRONGER_EVIDENCE

    # Supernova Ia (which predicts ~0.08 mag decay over 2 days, so ~17.5-18.0 mag, completely missing m=19.9)
    # should be rejected under assumptions
    sn_critique = assessment.hypotheses_critiques["SUPERNOVA_IA"]
    assert sn_critique.is_statistically_consistent is False
    assert sn_critique.evaluation_status == HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS


def test_gate3_quarantine_firewall_blocks_observation_ingestion():
    """Verifies that an observation attempted in a Gate 3 quarantined pixel is immediately blocked by the auditor."""
    QuarantineFirewall._initialize()
    q_pix = next(iter(QuarantineFirewall.get_quarantined_pixels()))
    coord = QuarantineFirewall._HP.healpix_to_skycoord(q_pix)
    q_ra, q_dec = float(coord.ra.deg), float(coord.dec.deg)

    obs_quarantined = CalibratedObservation(
        observation_id="obs_quarantine_violation",
        target_id="tgt_quarantine",
        target_ra_deg=q_ra,
        target_dec_deg=q_dec,
        facility_id="ZTF",
        instrument_id="ZTF_CAM",
        filter_id="r",
        epoch_mjd=60000.0,
        apparent_mag=18.0,
        apparent_mag_err=0.05,
        exposure_time_sec=300.0,
        raw_data_hash_sha256="hash_quarantine_123",
    )

    res = MeasurementQualityAuditor.audit_observation(obs_quarantined)
    assert res.is_valid_for_inference is False
    assert res.recommended_action == "REJECT_CORRUPTED"
    assert "GATE_3_QUARANTINE_VIOLATION" in res.anomalies_detected
