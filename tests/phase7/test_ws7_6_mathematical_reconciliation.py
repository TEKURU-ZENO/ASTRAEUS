"""
ASTRAEUS Phase 7 - WS7.6 Remediation Checkpoint:
Mathematical Reconciliation, Independent Bayes Validation, Spectral Feedback & Governance Hardening.

Independent validation suite addressing all 8 scientific and numerical audit findings:
1. Exact Analytical Conjugate Gaussian Bayes Factor (both high-discrimination and moderate-discrimination regimes).
2. Numerically Stable Log-Survival for Non-Detections (Gaussian deep tail z=-12.5 -> ln L = -81.576; flux vs magnitude).
3. Genuine Monte Carlo Bayesian Posterior Predictive Check (PPC) and degrees-of-freedom calibration.
4. Out-of-Model / Unmodeled Transient Hypothesis preventing premature posterior collapse.
5. VLT X-shooter Spectral Data Model (CalibratedSpectrum) and synthetic passband projection.
6. Evidence Ledger Reprocessing Rollback / Exact Posterior Recomputation and N-dim correlated covariance.
7. Model Misspecification Detection on generating process different from candidate templates.
8. Recommendation-Bound, Single-Use, Expiring Authorization Tokens.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import math
import numpy as np
import scipy.stats as stats
import time

from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    CalibratedSpectrum,
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
    LedgerEntryStatus,
    EvidenceLedgerEntry,
)
from astraeus_agents.feedback.posterior_predictive import (
    PosteriorPredictiveChecker,
    PosteriorPredictiveAssessment,
    HypothesisCriticismResult,
)
from astraeus_agents.observatory.human_approval_interface import (
    HumanApprovalGate,
    ApprovalStatus,
    AutonomousDispatchProhibitedError,
    AuthorizationToken,
    TokenValidationError,
)
from astraeus_data.memory.event_store import DurableScientificEventStore
from astraeus_agents.observatory.schedulers import (
    ScheduledObservation,
    TargetState,
)
from astraeus_agents.reground.models import HypothesisEvaluationStatus
from astraeus_agents.observatory.facility_registry import WavelengthRegime


def test_analytical_conjugate_bayes_factor_exact_reconciliation():
    """
    Deliverable 1: Independently calculates exact conjugate Gaussian Bayes factors
    and verifies engine outputs against published analytical intermediate values.
    """
    # -------------------------------------------------------------------------
    # Regime A: High Astronomical Discrimination (Reported parameter set)
    # H1: mu1 = 18.00, H2: mu2 = 19.50, y = 18.05, sigma_tot = 0.070710678
    # -------------------------------------------------------------------------
    mu1_a = 18.00
    mu2_a = 19.50
    y_a = 18.05
    sigma_tot_a = math.sqrt(0.0050)  # ~0.0707106781

    exact_a = BayesianRecalibrationEngine.calculate_exact_two_model_bayes_factor(
        mu1=mu1_a,
        mu2=mu2_a,
        y=y_a,
        sigma_tot=sigma_tot_a,
    )

    # Verify intermediate values
    assert abs(exact_a["res1"] - 0.05) < 1e-9
    assert abs(exact_a["res2"] - (-1.45)) < 1e-9
    assert abs(exact_a["sq_res1"] - 0.0025) < 1e-9
    assert abs(exact_a["sq_res2"] - 2.1025) < 1e-9
    assert abs(exact_a["numerator"] - 2.1000) < 1e-9
    assert abs(exact_a["denominator"] - 0.0100) < 1e-9
    assert abs(exact_a["ln_bayes_factor"] - 210.000000) < 1e-9
    # Posterior for H1 is virtually 1.0 (difference from 1 is ~6.237e-92)
    assert exact_a["posterior_h1"] > 0.999999999999
    assert exact_a["posterior_h2"] < 1e-90

    # Test engine likelihood ratio evaluation
    ln_L1_a = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(
        measured_mag=y_a,
        measured_err=math.sqrt(max(1e-8, sigma_tot_a**2 - 0.03**2 - 0.05**2)),
        predicted_mag=mu1_a,
    )
    ln_L2_a = BayesianRecalibrationEngine.evaluate_detection_log_likelihood(
        measured_mag=y_a,
        measured_err=math.sqrt(max(1e-8, sigma_tot_a**2 - 0.03**2 - 0.05**2)),
        predicted_mag=mu2_a,
    )
    ln_bf_engine_a = ln_L1_a - ln_L2_a
    assert abs(ln_bf_engine_a - 210.0000) < 1e-4

    # -------------------------------------------------------------------------
    # Regime B: Moderate Discrimination (Numerical BF = 2980.957987 test case)
    # H1: mu1 = 18.00, H2: mu2 = 20.00, y = 18.00, sigma_tot = 0.50000
    # -------------------------------------------------------------------------
    mu1_b = 18.00
    mu2_b = 20.00
    y_b = 18.00
    sigma_tot_b = 0.50

    exact_b = BayesianRecalibrationEngine.calculate_exact_two_model_bayes_factor(
        mu1=mu1_b,
        mu2=mu2_b,
        y=y_b,
        sigma_tot=sigma_tot_b,
    )

    assert abs(exact_b["numerator"] - 4.0000) < 1e-9
    assert abs(exact_b["denominator"] - 0.5000) < 1e-9
    assert abs(exact_b["ln_bayes_factor"] - 8.000000) < 1e-9
    assert abs(exact_b["bayes_factor"] - 2980.95798704) < 1e-5
    assert abs(exact_b["posterior_h1"] - 0.99966464987) < 1e-7
    assert abs(exact_b["posterior_h2"] - 0.00033535013) < 1e-7


def test_stable_gaussian_log_cdf_non_detection_tail_reconciliation():
    """
    Deliverable 2: Validates numerically stable log-survival calculations across deep tails,
    threshold transitions, and flux-space formulations.
    """
    # 1. Deep tail case: mu = 18.5, m_lim = 21.0
    # Case 1A: With model physics uncertainty (sigma_lim = 0.20, sigma_phys = 0.05 -> sigma_tot = sqrt(0.0425) = 0.206155)
    # z = (18.5 - 21.0) / 0.206155 = -12.12678 -> exact ln L = -76.950455
    ln_L_deep_a = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=21.0,
        predicted_mag=18.5,
        sigma_limit_mag=0.20,
    )
    sigma_tot_a = math.sqrt(0.20**2 + 0.05**2)
    exact_ln_L_a = float(stats.norm.logcdf((18.5 - 21.0) / sigma_tot_a))
    assert abs(ln_L_deep_a - exact_ln_L_a) < 1e-5
    assert abs(ln_L_deep_a - (-76.9504547)) < 1e-4

    # Case 1B: Pure sigma_tot = 0.2000 (z = -12.5 exactly)
    # z = (18.5 - 21.0) / 0.20 = -12.5 -> exact ln L = -81.575968
    sigma_limit_pure = math.sqrt(0.20**2 - 0.05**2)
    ln_L_deep_pure = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=21.0,
        predicted_mag=18.5,
        sigma_limit_mag=sigma_limit_pure,
    )
    exact_ln_L_pure = float(stats.norm.logcdf(-12.5))
    assert abs(exact_ln_L_pure - (-81.57596787)) < 1e-5
    assert abs(ln_L_deep_pure - exact_ln_L_pure) < 1e-5
    # Demonstrates that likelihood is ~3.7e-36 (ln L = -81.58), NOT clamped at 10^-12 (ln L = -27.63)
    assert ln_L_deep_pure < -80.0

    # 2. Threshold transition case: mu == m_lim
    # z = 0.0 -> Phi(0) = 0.5 -> ln L = ln(0.5) = -0.693147
    ln_L_thresh = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=21.0,
        predicted_mag=21.0,
        sigma_limit_mag=0.20,
    )
    assert abs(ln_L_thresh - (-0.69314718)) < 1e-5

    # 3. Faint undetectable source case: mu = 24.0, m_lim = 21.0
    # z = (24.0 - 21.0) / 0.20 = +15.0 -> Phi(+15) ~ 1.0 -> ln L ~ 0.0
    ln_L_faint = BayesianRecalibrationEngine.evaluate_censored_upper_limit_log_likelihood(
        limiting_mag_5sigma=21.0,
        predicted_mag=24.0,
        sigma_limit_mag=0.20,
    )
    assert abs(ln_L_faint - 0.0) < 1e-10

    # 4. Flux-space non-detection likelihood:
    # F_lim = 500 nJy (5 sigma), sigma_F = 100 nJy
    # Source predicts faint 50 nJy -> z_F = (500 - 50) / 100 = 4.5 -> Phi(4.5) ~ 1.0
    ln_L_flux_faint = BayesianRecalibrationEngine.evaluate_censored_flux_upper_limit_log_likelihood(
        limiting_flux_njy=500.0,
        predicted_flux_njy=50.0,
        flux_uncertainty_njy=100.0,
    )
    assert abs(ln_L_flux_faint - stats.norm.logcdf(4.5)) < 1e-5
    assert ln_L_flux_faint > -1e-4

    # Source predicts bright 1500 nJy -> z_F = (500 - 1500) / 100 = -10.0 -> ln L = logcdf(-10.0)
    ln_L_flux_bright = BayesianRecalibrationEngine.evaluate_censored_flux_upper_limit_log_likelihood(
        limiting_flux_njy=500.0,
        predicted_flux_njy=1500.0,
        flux_uncertainty_njy=100.0,
    )
    assert abs(ln_L_flux_bright - stats.norm.logcdf(-10.0)) < 1e-5
    assert ln_L_flux_bright < -50.0


def test_genuine_monte_carlo_posterior_predictive_simulation():
    """
    Deliverable 3: Distinguishes classical goodness-of-fit (chi2.sf) from genuine
    Monte Carlo Bayesian posterior predictive checking (PPC).
    """
    # 3 observations well-matched to forward model
    preds = [18.2, 18.8, 19.4]
    tot_errs = [0.08, 0.08, 0.08]

    # Run genuine Monte Carlo replication
    replicates = PosteriorPredictiveChecker.simulate_posterior_predictive_replicates(
        predicted_magnitudes=preds,
        total_uncertainties=tot_errs,
        num_replicates=1000,
        random_seed=42,
    )
    assert len(replicates) == 1000
    # Mean discrepancy for chi2 with dof=3 should be approximately dof = 3.0
    mean_rep = float(np.mean(replicates))
    assert 2.7 < mean_rep < 3.3

    obs_list = [
        CalibratedObservation(
            observation_id="p1", target_id="t1", target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60000.5,
            apparent_mag=18.53, apparent_mag_err=0.06, exposure_time_sec=300.0, raw_data_hash_sha256="h1",
        ),
        CalibratedObservation(
            observation_id="p2", target_id="t1", target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.0,
            apparent_mag=18.90, apparent_mag_err=0.06, exposure_time_sec=300.0, raw_data_hash_sha256="h2",
        ),
        CalibratedObservation(
            observation_id="p3", target_id="t1", target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.5,
            apparent_mag=19.49, apparent_mag_err=0.06, exposure_time_sec=300.0, raw_data_hash_sha256="h3",
        ),
    ]

    assessment = PosteriorPredictiveChecker.evaluate_model_fit(
        target_id="t1",
        hypothesis_posteriors={"KILONOVA": 0.90, "CCSN": 0.10},
        accumulated_observations=obs_list,
        trigger_mjd=60000.0,
        fitted_parameter_count=0,
    )

    kn_critique = assessment.hypotheses_critiques["KILONOVA"]
    assert kn_critique.degrees_of_freedom == 3
    # Classical chi2.sf p-value
    assert 0.10 < kn_critique.goodness_of_fit_p_value < 0.95
    # Bayesian Monte Carlo PPC p-value
    assert 0.10 < kn_critique.bayesian_ppc_p_value < 0.95
    assert kn_critique.is_statistically_consistent is True
    assert kn_critique.evaluation_status == HypothesisEvaluationStatus.STRONGER_EVIDENCE


def test_unmodeled_transient_hypothesis_prevents_posterior_collapse():
    """
    Deliverable 4: Verifies that adding UNMODELED_TRANSIENT prevents artificial
    posterior collapse to 1.000 when evidence is ambiguous or non-detections occur.
    """
    initial_priors = {
        "KILONOVA": 0.45,
        "CCSN": 0.45,
        "UNMODELED_TRANSIENT": 0.10,
    }

    # Ingest Swift UV non-detection
    swift_obs = CalibratedObservation(
        observation_id="swift_uv_test",
        target_id="t_ambiguous",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="SWIFT",
        instrument_id="UVOT",
        filter_id="uvw2",
        measurement_type=MeasurementType.UPPER_LIMIT_CENSORED,
        epoch_mjd=60001.0,
        limiting_mag_5sigma=20.5,
        exposure_time_sec=900.0,
        raw_data_hash_sha256="hash_swift_test",
    )

    result = BayesianRecalibrationEngine.recalibrate_posteriors(
        current_posteriors=initial_priors,
        observation=swift_obs,
        effective_wavelength_angstrom=2600.0,
        regime=WavelengthRegime.UV,
        time_since_trigger_days=1.0,
    )

    posts = result.updated_posteriors
    # Kilonova should increase because it is UV-suppressed
    assert posts["KILONOVA"] > initial_priors["KILONOVA"]
    # But Kilonova MUST NOT collapse to 1.000 because UNMODELED_TRANSIENT absorbs open-world probability
    assert posts["KILONOVA"] < 0.999
    assert posts["UNMODELED_TRANSIENT"] > 0.01


def test_calibrated_spectrum_and_synthetic_passband_projection():
    """
    Deliverable 5: Validates medium-resolution spectral representation (CalibratedSpectrum)
    and passband synthesis matching X-shooter spectroscopic workflows.
    """
    # Create synthetic VLT X-shooter NIR arm spectrum (10000 to 20000 A)
    wls = np.linspace(10000.0, 20000.0, 1000)
    # Peak flux around 15000 A (typical red kilonova spectrum)
    # Flux ~ 1.5e-17 erg/s/cm^2/A -> m_AB ~ 18.2 mag at 15000 A
    flam = 1.5e-17 * np.exp(-0.5 * ((wls - 15000.0) / 3000.0) ** 2)
    flam_err = 0.05 * flam
    mask = [True] * len(wls)
    # Simulate telluric absorption mask between 13500 and 14200 A
    for i, w in enumerate(wls):
        if 13500.0 <= w <= 14200.0:
            mask[i] = False

    raw_hash = CalibratedObservation.compute_payload_hash("RAW_XSHOOTER_FITS_20261124")

    spectrum = CalibratedSpectrum(
        spectrum_id="spec_xshooter_nir_001",
        target_id="tgt_spec_test",
        target_ra_deg=180.0,
        target_dec_deg=10.0,
        facility_id="VLT",
        instrument_id="XSHOOTER",
        arm="NIR",
        spectral_resolution_R=5400.0,
        epoch_mjd=60002.0,
        wavelengths_angstrom=list(wls),
        flux_density_cgs=list(flam),
        flux_density_err_cgs=list(flam_err),
        mask=mask,
        telluric_corrected=True,
        raw_data_hash_sha256=raw_hash,
    )

    # Synthesize H-band broadband photometric measurement (center: 16300 A, FWHM: 2800 A)
    synthetic_obs = spectrum.synthesize_broadband_magnitude(
        filter_id="H_BAND",
        central_wl_angstrom=16300.0,
        fwhm_angstrom=2800.0,
    )

    assert isinstance(synthetic_obs, CalibratedObservation)
    assert synthetic_obs.facility_id == "VLT"
    assert synthetic_obs.instrument_id == "XSHOOTER"
    assert synthetic_obs.filter_id == "H_BAND"
    assert 17.5 < synthetic_obs.apparent_mag < 19.0
    assert synthetic_obs.apparent_mag_err < 0.10
    assert synthetic_obs.provenance_metadata["projected_from_spectrum"] is True
    assert synthetic_obs.provenance_metadata["spectral_arm"] == "NIR"


def test_evidence_ledger_recomputation_and_n_dim_correlated_evidence():
    """
    Deliverable 6: Validates ledger recomputation rollback and N-dimensional
    correlated covariance matrix handling.
    """
    ledger = ObservationEvidenceLedger()
    prior = {"KILONOVA": 0.50, "CCSN": 0.50}

    # Ingest original observation
    obs_v1 = CalibratedObservation(
        observation_id="obs_red_v1", target_id="t_recon", target_ra_deg=180.0, target_dec_deg=10.0,
        facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.0,
        apparent_mag=18.50, apparent_mag_err=0.05, exposure_time_sec=300.0, raw_data_hash_sha256="hash_v1",
    )
    ledger.ingest_observation(obs_v1)
    posts_v1 = ledger.recompute_target_posteriors("t_recon", prior)

    # Ingest reprocessed version (improved PSF zero-point photometry: m = 18.95)
    obs_v2 = CalibratedObservation(
        observation_id="obs_red_v2", target_id="t_recon", target_ra_deg=180.0, target_dec_deg=10.0,
        facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.0,
        apparent_mag=18.95, apparent_mag_err=0.03, exposure_time_sec=300.0,
        raw_data_hash_sha256="hash_v2", is_reprocessed_version_of="obs_red_v1",
    )
    ledger.ingest_observation(obs_v2)

    # Verify v1 is superseded
    active = ledger.get_active_observations_for_target("t_recon")
    assert len(active) == 1
    assert active[0].observation_id == "obs_red_v2"

    # Recompute posteriors from active ledger
    posts_v2 = ledger.recompute_target_posteriors("t_recon", prior)
    # v2 m=18.95 matches Kilonova decay much better than v1 m=18.50
    assert posts_v2["KILONOVA"] > posts_v1["KILONOVA"]

    # Test N-dimensional (N=3) correlated covariance handling
    residuals = np.array([0.05, -0.04, 0.06])
    # 3x3 covariance with shared systematic calibration error of 0.04 mag
    cov = np.array([
        [0.05**2 + 0.04**2, 0.04**2, 0.04**2],
        [0.04**2, 0.05**2 + 0.04**2, 0.04**2],
        [0.04**2, 0.04**2, 0.05**2 + 0.04**2],
    ])
    ln_L_corr = BayesianRecalibrationEngine.evaluate_correlated_group_log_likelihood(residuals, cov)
    assert not math.isnan(ln_L_corr)
    assert not math.isinf(ln_L_corr)

    # Test near-singular regularized inversion
    near_sing_cov = cov.copy()
    near_sing_cov[2, :] = near_sing_cov[1, :] * 0.9999999  # Highly collinear
    ln_L_near_sing = BayesianRecalibrationEngine.evaluate_correlated_group_log_likelihood(
        residuals, near_sing_cov, regularization_epsilon=1e-5
    )
    assert not math.isnan(ln_L_near_sing)


def test_model_misspecification_detection_on_deliberately_different_generating_process():
    """
    Deliverable 7: Verifies that when the true generating process is completely different
    from all candidate models, the system flags MODEL MISSPECIFICATION DETECTED rather
    than erroneously claiming confident discovery.
    """
    # Generate observations from an exotic power-law shock breakout (fading as t^-3.5)
    # Extremely bright at day 0.5 (m = 15.0), fading to m = 21.0 by day 1.5
    obs_exotic = [
        CalibratedObservation(
            observation_id="ex1", target_id="t_exotic", target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60000.5,
            apparent_mag=15.0, apparent_mag_err=0.03, exposure_time_sec=300.0, raw_data_hash_sha256="hex1",
        ),
        CalibratedObservation(
            observation_id="ex2", target_id="t_exotic", target_ra_deg=180.0, target_dec_deg=10.0,
            facility_id="ZTF", instrument_id="ZTF_CAM", filter_id="r", epoch_mjd=60001.5,
            apparent_mag=21.0, apparent_mag_err=0.08, exposure_time_sec=300.0, raw_data_hash_sha256="hex2",
        ),
    ]

    candidate_posteriors = {
        "KILONOVA": 0.50,
        "CCSN": 0.35,
        "TDE_AGN": 0.15,
    }

    assessment = PosteriorPredictiveChecker.evaluate_model_fit(
        target_id="t_exotic",
        hypothesis_posteriors=candidate_posteriors,
        accumulated_observations=obs_exotic,
        trigger_mjd=60000.0,
    )

    # MUST detect model misspecification
    assert assessment.all_models_misspecified is True
    assert assessment.preferred_hypothesis is None
    assert "MODEL MISSPECIFICATION DETECTED" in assessment.audit_verdict

    # Check that each individual model is rejected under assumptions
    for h, critique in assessment.hypotheses_critiques.items():
        assert critique.is_statistically_consistent is False
        assert critique.evaluation_status == HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS
        # p-values must be tiny due to massive discrepancy
        assert critique.goodness_of_fit_p_value < 1e-4
        assert critique.bayesian_ppc_p_value < 1e-3


def test_recommendation_bound_single_use_expiring_tokens():
    """
    Deliverable 8: Validates recommendation-bound, single-use, expiring authorization tokens.
    """
    store = DurableScientificEventStore(db_path=":memory:")
    gate = HumanApprovalGate(event_store=store)

    target = TargetState(
        target_id="tgt_gov",
        ra_deg=180.0,
        dec_deg=10.0,
        trigger_mjd=60000.0,
        base_magnitude=18.0,
        hypothesis_posteriors={"KILONOVA": 0.7, "CCSN": 0.3},
    )

    obs1 = ScheduledObservation(
        target_id="tgt_gov", facility_id="VLT", instrument_id="XSHOOTER", filter_id="NIR_SPEC",
        start_mjd=60001.0, end_mjd=60001.02, exposure_time_sec=600.0, slew_time_sec=30.0,
        readout_time_sec=15.0, airmass=1.1, expected_eig_nats=0.85, weather_score=0.9,
    )
    rec1 = gate.create_recommendation(obs1, target, "Primary observation proposal")

    obs2 = ScheduledObservation(
        target_id="tgt_gov", facility_id="SWIFT", instrument_id="UVOT", filter_id="uvw2",
        start_mjd=60001.5, end_mjd=60001.52, exposure_time_sec=900.0, slew_time_sec=45.0,
        readout_time_sec=10.0, airmass=1.0, expected_eig_nats=0.45, weather_score=1.0,
    )
    rec2 = gate.create_recommendation(obs2, target, "Secondary observation proposal")

    # Issue token for recommendation 1 with TTL = 10.0 seconds
    token1 = gate.issue_authorization_token(
        recommendation_id=rec1.recommendation_id,
        investigator_email="devmehta060505@gmail.com",
        ttl_seconds=10.0,
    )
    assert token1.is_consumed is False
    assert token1.recommendation_id == rec1.recommendation_id

    # 1. Successful dispatch using valid token
    receipt = gate.dispatch_with_token(token1, rec1.recommendation_id)
    assert receipt["dispatch_status"] == "DISPATCHED_TO_FACILITY_QUEUE"
    assert token1.is_consumed is True

    # 2. Second dispatch with SAME token must fail (single-use enforcement!)
    with pytest.raises(TokenValidationError) as exc_single_use:
        gate.dispatch_with_token(token1, rec1.recommendation_id)
    assert "ALREADY BEEN CONSUMED" in str(exc_single_use.value)

    # 3. Token issued for rec1 cannot authorize rec2 (target/recommendation binding!)
    token_rec2_attempt = gate.issue_authorization_token(
        recommendation_id=rec1.recommendation_id,
        investigator_email="devmehta060505@gmail.com",
        ttl_seconds=10.0,
    )
    with pytest.raises(TokenValidationError) as exc_mismatch:
        gate.dispatch_with_token(token_rec2_attempt, rec2.recommendation_id)
    assert "TOKEN TARGET MISMATCH" in str(exc_mismatch.value)

    # 4. Expired token check
    token_expired = gate.issue_authorization_token(
        recommendation_id=rec2.recommendation_id,
        investigator_email="devmehta060505@gmail.com",
        ttl_seconds=0.001,  # immediate expiration
    )
    time.sleep(0.01)
    with pytest.raises(TokenValidationError) as exc_expired:
        gate.dispatch_with_token(token_expired, rec2.recommendation_id)
    assert "EXPIRED" in str(exc_expired.value)
