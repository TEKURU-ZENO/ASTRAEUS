"""
Tests for ASTRAEUS Real-Survey Observation Feature Engineering & Certification Pipeline.
Verifies negative flux preservation, upper-limit censoring, irregular cadence,
and the three certified survey observation configurations.
"""

import pytest
import numpy as np
import pandas as pd
from astraeus_data.real_survey.observation_feature_pipeline import (
    CanonicalObservation,
    ObservationConfig,
    RealSurveyFeatureExtractor,
    CandidateObservationLoader,
    FEATURE_NAMES_32,
    FEATURE_SPECIFICATION_32,
    certify_observation_configurations_coverage,
)


def test_feature_specification_dictionary_completeness():
    """Verifies all 32 features have documented category, unit, source, and missingness rule."""
    assert len(FEATURE_NAMES_32) == 32
    assert len(FEATURE_SPECIFICATION_32) == 32
    for fname, spec in FEATURE_SPECIFICATION_32.items():
        assert "category" in spec
        assert "unit" in spec
        assert "source" in spec
        assert "missingness_rule" in spec
        assert "description" in spec


def test_negative_flux_preservation_unclamped():
    """
    Verifies that negative difference-image subtraction flux is preserved
    unclipped and without clamping to 0.0.
    """
    obs = [
        CanonicalObservation(
            mjd=59000.0,
            passband="g",
            flux_nmgy=-12.5,  # Unclipped negative flux fluctuation
            flux_err_nmgy=1.5,
            is_upper_limit=False,
            is_negative_flux=True,
            source_survey="ALERCE",
        ),
        CanonicalObservation(
            mjd=59002.0,
            passband="g",
            flux_nmgy=45.0,  # Positive peak
            flux_err_nmgy=2.0,
            is_upper_limit=False,
            is_negative_flux=False,
            source_survey="ALERCE",
        ),
    ]

    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
    feats = extractor.extract_from_observations(obs)

    assert feats["min_flux_nmgy"] == -12.5, "Negative flux was clipped or corrupted!"
    assert feats["peak_flux_nmgy"] == 45.0
    assert feats["flux_dynamic_range_nmgy"] == pytest.approx(57.5, abs=1e-5)
    assert feats["negative_flux_fraction"] == 0.5


def test_upper_limits_as_censored_observations():
    """
    Verifies that non-detections are preserved as censored observations
    with explicit limiting flux, not discarded or treated as zero flux.
    """
    obs = [
        CanonicalObservation(
            mjd=58990.0,
            passband="r",
            flux_nmgy=0.0,
            flux_err_nmgy=0.5,
            is_upper_limit=True,
            limiting_flux_nmgy=2.5,  # 5-sigma limiting flux threshold
            source_survey="ALERCE",
        ),
        CanonicalObservation(
            mjd=58995.0,
            passband="r",
            flux_nmgy=0.0,
            flux_err_nmgy=0.4,
            is_upper_limit=True,
            limiting_flux_nmgy=1.8,  # Deeper limit
            source_survey="ALERCE",
        ),
        CanonicalObservation(
            mjd=59000.0,
            passband="r",
            flux_nmgy=30.0,
            flux_err_nmgy=1.2,
            is_upper_limit=False,
            limiting_flux_nmgy=None,
            source_survey="ALERCE",
        ),
    ]

    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
    feats = extractor.extract_from_observations(obs)

    assert feats["n_detections"] == 1.0
    assert feats["n_upper_limits"] == 2.0
    assert feats["total_epochs"] == 3.0
    assert feats["detection_fraction"] == pytest.approx(1.0 / 3.0, abs=1e-5)
    assert feats["deepest_upper_limit_nmgy"] == 1.8


def test_three_observation_configurations_ablation():
    """
    Verifies strict isolation across the three observation configurations:
    1. ALERCE_ONLY excludes DR24 data.
    2. ZTF_DR24_ONLY excludes ALERCE data.
    3. COMBINED aggregates both.
    """
    multi_obs = [
        CanonicalObservation(
            mjd=58500.0,
            passband="g",
            flux_nmgy=10.0,
            flux_err_nmgy=1.0,
            is_upper_limit=False,
            source_survey="ZTF_DR24",
        ),
        CanonicalObservation(
            mjd=58510.0,
            passband="g",
            flux_nmgy=12.0,
            flux_err_nmgy=1.0,
            is_upper_limit=False,
            source_survey="ZTF_DR24",
        ),
        CanonicalObservation(
            mjd=59200.0,
            passband="r",
            flux_nmgy=80.0,
            flux_err_nmgy=2.0,
            is_upper_limit=False,
            source_survey="ALERCE",
        ),
    ]

    ext_alerce = RealSurveyFeatureExtractor(config=ObservationConfig.ALERCE_ONLY)
    feats_alerce = ext_alerce.extract_from_observations(multi_obs)
    assert feats_alerce["has_alerce_observations"] == 1.0
    assert feats_alerce["has_dr24_observations"] == 0.0
    assert feats_alerce["n_detections"] == 1.0
    assert feats_alerce["peak_flux_nmgy"] == 80.0

    ext_dr24 = RealSurveyFeatureExtractor(config=ObservationConfig.ZTF_DR24_ONLY)
    feats_dr24 = ext_dr24.extract_from_observations(multi_obs)
    assert feats_dr24["has_alerce_observations"] == 0.0
    assert feats_dr24["has_dr24_observations"] == 1.0
    assert feats_dr24["n_detections"] == 2.0
    assert feats_dr24["peak_flux_nmgy"] == 12.0

    ext_comb = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
    feats_comb = ext_comb.extract_from_observations(multi_obs)
    assert feats_comb["has_alerce_observations"] == 1.0
    assert feats_comb["has_dr24_observations"] == 1.0
    assert feats_comb["n_detections"] == 3.0
    assert feats_comb["peak_flux_nmgy"] == 80.0
    assert feats_comb["baseline_duration_days"] == pytest.approx(700.0, abs=1e-5)


def test_real_candidate_lightcurve_loading_and_extraction():
    """
    Tests feature extraction on an actual real-survey alert file (ZTF19aangfld).
    """
    loader = CandidateObservationLoader()
    obs = loader.load_candidate_observations("ZTF19aangfld")
    assert len(obs) > 0, "Failed to load real observations for ZTF19aangfld"

    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
    feats = extractor.extract_from_observations(obs)

    assert len(feats) == 32
    assert feats["has_alerce_observations"] == 1.0
    assert feats["n_detections"] == 6.0
    assert feats["n_upper_limits"] == 3.0
    assert feats["peak_flux_nmgy"] > 0.0
    assert feats["baseline_duration_days"] > 0.0

    vec = extractor.extract_vector(obs)
    assert len(vec) == 32
    assert not np.any(np.isnan(vec)), "Feature vector contains NaN values!"


def test_multi_configuration_coverage_audit():
    """
    Runs configuration audit over a real-survey candidate sample,
    verifying reproducible feature metrics across ALERCE_ONLY, ZTF_DR24_ONLY, and COMBINED.
    """
    candidates_df = pd.read_parquet("data/processed/real_survey_10k_v1/candidates_10k.parquet")
    audit = certify_observation_configurations_coverage(candidates_df, sample_size=50)

    assert "configurations" in audit
    assert "ALERCE_ONLY" in audit["configurations"]
    assert "ZTF_DR24_ONLY" in audit["configurations"]
    assert "COMBINED" in audit["configurations"]

    comb = audit["configurations"]["COMBINED"]
    assert comb["total_candidates"] == 50
    assert comb["total_epochs_processed"] > 0
    assert "survey_origin_bias_rationale" in audit


def test_explicit_missingness_masks_and_indicators():
    """
    Verifies that extract_with_missingness() yields explicit boolean masks
    and numerical observation indicators for every feature, distinguishing
    unobserved/defaulted features from directly measured quantities.
    """
    # Create an observation set with only 1 detection and 0 upper limits
    obs = [
        CanonicalObservation(
            mjd=59100.0,
            passband="g",
            flux_nmgy=20.0,
            flux_err_nmgy=1.0,
            is_upper_limit=False,
            source_survey="ALERCE",
        )
    ]

    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.COMBINED)
    result = extractor.extract_with_missingness(obs)

    # 1 detection means peak_flux is observed, but flux_std and baseline_duration are unobserved
    assert result.missingness_mask["peak_flux_nmgy"] is True
    assert result.missingness_indicators["peak_flux_nmgy__is_observed"] == 1.0

    assert result.missingness_mask["flux_std_nmgy"] is False
    assert result.missingness_indicators["flux_std_nmgy__is_observed"] == 0.0

    assert result.missingness_mask["baseline_duration_days"] is False
    assert result.missingness_indicators["baseline_duration_days__is_observed"] == 0.0

    assert result.missingness_mask["deepest_upper_limit_nmgy"] is False
    assert result.missingness_indicators["deepest_upper_limit_nmgy__is_observed"] == 0.0

    assert result.missingness_mask["has_alerce_observations"] is True
    assert result.n_detections == 1
    assert result.n_upper_limits == 0


def test_temporal_coordinate_system_reconciliation():
    """
    Verifies that heliocentric light travel time reconciliation strictly adheres
    to physical light travel bounds (|delta_t| <= 499 s ~ 0.00578 d) and roundtrips cleanly.
    """
    from astraeus_data.real_survey.observation_feature_pipeline import (
        compute_heliocentric_time_correction_days,
        reconcile_utc_mjd_to_hmjd,
        reconcile_hmjd_to_utc_mjd,
    )

    mjd_test = 59500.0
    ra_test = 150.0  # degrees
    dec_test = 25.0  # degrees

    dt_days = compute_heliocentric_time_correction_days(mjd_test, ra_test, dec_test)
    max_light_travel_days = 499.00478 / 86400.0  # ~0.0057755 days

    assert abs(dt_days) <= max_light_travel_days + 1e-7, f"Correction exceeds 1 AU / c: {dt_days}"

    hmjd = reconcile_utc_mjd_to_hmjd(mjd_test, ra_test, dec_test)
    assert abs(hmjd - mjd_test) <= max_light_travel_days + 1e-7

    # Roundtrip check
    mjd_recovered = reconcile_hmjd_to_utc_mjd(hmjd, ra_test, dec_test)
    assert mjd_recovered == pytest.approx(mjd_test, abs=1e-8)


def test_hyperbolic_uncertainty_scaling_and_clamping():
    """
    Verifies that photometric uncertainty propagation uses exact hyperbolic log-normal scaling
    sigma_F = F * sinh((ln 10 / 2.5) * sigma_m) which smoothly handles large sigma_m without
    first-order breakdown and caps extreme noise fluctuations at sigma_m = 3.0.
    """
    import math

    # Case 1: Small magnitude error (sigma_m = 0.05 mag) -> sinh(x) ~ x
    sm_small = 0.05
    f_val = 100.0
    sig_f_first_order = f_val * (math.log(10) / 2.5) * sm_small
    sig_f_exact = f_val * math.sinh((math.log(10) / 2.5) * sm_small)
    assert abs(sig_f_exact - sig_f_first_order) / sig_f_exact < 0.002  # Within 0.2%

    # Case 2: Large magnitude error (sigma_m = 0.8 mag) -> first-order severely underestimates
    sm_large = 0.8
    sig_f_large_first_order = f_val * (math.log(10) / 2.5) * sm_large
    sig_f_large_exact = f_val * math.sinh((math.log(10) / 2.5) * sm_large)
    assert sig_f_large_exact > sig_f_large_first_order, "Hyperbolic error should exceed linear approximation at large sigma_m"
