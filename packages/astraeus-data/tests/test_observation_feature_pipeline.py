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
