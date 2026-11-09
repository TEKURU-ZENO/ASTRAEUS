"""
ASTRAEUS Phase 2 Stage 2 — Temporal Sequence & Data Contract Unit Tests.
Verifies v1.1.2 mathematical conversions, time transformations, causal visibility,
science-preserving hierarchical FPS, and censoring likelihoods.
"""

import pytest
import math
import numpy as np
from datetime import datetime, timezone

from astraeus_data.corpus.time_conversion import (
    compute_tt_from_utc_mjd,
    compute_tdb_from_utc_mjd,
    compute_bmjd_tdb,
    compute_roemer_delay_sec,
    get_leap_seconds_utc_to_tai,
    is_observation_visible_at_cutoff,
    TT_TAI_OFFSET_SEC,
)
from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    QualitySeverity,
    NoiseScaleSource,
    SequenceViewType,
    TemporalSequence,
    AkimaPredictiveModel,
    CANONICAL_AB_ZERO_POINT,
    CORPUS_V10_ZERO_POINT,
    CORPUS_V10_TO_NMGY_SCALE,
    SI_FLUX_DENSITY_PER_NANOMAGGY,
    MICROJANSKYS_PER_NANOMAGGY,
)


class TestPhotometricCalibrationAndSIConversion:
    """Verifies P0.1 SI flux conversion and reference corpus unit migration."""

    def test_si_flux_density_identity(self):
        """Verifies 1 nanomaggy == 3.631e-32 W m^-2 Hz^-1 == 3.631 uJy."""
        one_nmgy = 1.0
        si_flux = ObservationRecord.nanomaggies_to_si_flux_density(one_nmgy)
        ujy_flux = ObservationRecord.nanomaggies_to_microjanskys(one_nmgy)

        assert math.isclose(si_flux, 3.631e-32, rel_tol=1e-6)
        assert math.isclose(ujy_flux, 3.631, rel_tol=1e-6)

    def test_ab_magnitude_zero_point(self):
        """Verifies m_AB = 22.500 corresponds to exactly 1.0 nanomaggy."""
        flux_22_5, _ = ObservationRecord.ab_magnitude_to_nanomaggies(22.500, 0.05)
        assert math.isclose(flux_22_5, 1.000, rel_tol=1e-6)

        flux_15_0, _ = ObservationRecord.ab_magnitude_to_nanomaggies(15.000, 0.01)
        assert math.isclose(flux_15_0, 1000.000, rel_tol=1e-5)

        flux_25_0, _ = ObservationRecord.ab_magnitude_to_nanomaggies(25.000, 0.10)
        assert math.isclose(flux_25_0, 0.100, rel_tol=1e-5)

    def test_corpus_v10_to_nanomaggy_migration(self):
        """Verifies exact F_nmgy = F_27.5 / 100.0 migration from Corpus v1.0."""
        # Standard positive flux: 100 counts in Z0=27.5 -> 1 nmgy in Z0=22.5
        flux_nmgy, sigma_nmgy = ObservationRecord.corpus_v10_to_nanomaggies(100.0, 5.0)
        assert math.isclose(flux_nmgy, 1.0, rel_tol=1e-7)
        assert math.isclose(sigma_nmgy, 0.05, rel_tol=1e-7)

        # Signed negative flux preservation
        neg_flux_nmgy, neg_sigma_nmgy = ObservationRecord.corpus_v10_to_nanomaggies(-50.0, 10.0)
        assert math.isclose(neg_flux_nmgy, -0.500, rel_tol=1e-7)
        assert math.isclose(neg_sigma_nmgy, 0.100, rel_tol=1e-7)


class TestCensoringAndTrainingLikelihood:
    """Verifies P0.2 separation of forced-photometry and catalog upper limits with explicit noise scales."""

    def test_forced_photometry_negative_flux_retention(self):
        """Verifies negative difference flux executes exact Gaussian likelihood, not censored survival."""
        obs = ObservationRecord(
            object_id="TEST-TRANSIENT-001",
            observation_id="OBS-001",
            t_obs_bmjd=60000.0,
            t_pub_utc="2023-03-01T02:00:00Z",
            passband="r",
            survey_id="ZTF",
            obs_type=ObservationType.FORCED_PHOTOMETRY,
            measured_flux_nmgy=-0.800,
            measurement_sigma_nmgy=1.000,
            effective_measurement_sigma_nmgy=1.000,
            provenance_hash="a1b2c3d4"
        )
        assert obs.measured_flux_nmgy == -0.800
        # Loss around prediction mu = 0.0, aleatoric = 0.5
        loss = obs.compute_training_loss(mu_pred=0.0, sigma_aleatoric=0.5)
        # Expected total_var = 0.5^2 + 1.0^2 = 1.25
        # Expected Gaussian NLL = 0.5*ln(2*pi*1.25) + (-0.8)^2 / (2 * 1.25)
        expected_nll = 0.5 * math.log(2 * math.pi * 1.25) + (0.64 / 2.50)
        assert math.isclose(loss, expected_nll, rel_tol=1e-5)

    def test_catalog_upper_limit_survival_likelihood(self):
        """Verifies catalog upper limits evaluate survival likelihood using explicit noise scale."""
        obs = ObservationRecord(
            object_id="TEST-TRANSIENT-002",
            observation_id="OBS-002",
            t_obs_bmjd=60000.5,
            t_pub_utc="2023-03-01T12:00:00Z",
            passband="g",
            survey_id="ZTF",
            obs_type=ObservationType.CATALOG_UPPER_LIMIT,
            upper_limit_flux_nmgy=1.250,
            upper_limit_sigma_threshold=5.0,
            noise_scale_source=NoiseScaleSource.INFERRED_FROM_THRESHOLD,
            effective_measurement_sigma_nmgy=0.250,  # 1.25 / 5.0 = 0.25 nmgy
            provenance_hash="e5f6g7h8"
        )
        assert obs.measured_flux_nmgy is None
        # Model predicts quiescent flux mu = 0.1, aleatoric = 0.2
        loss = obs.compute_training_loss(mu_pred=0.1, sigma_aleatoric=0.2)
        # Total sigma = sqrt(0.2^2 + 0.25^2) = sqrt(0.04 + 0.0625) = sqrt(0.1025) approx 0.320156
        # z = (1.25 - 0.1) / 0.320156 approx 3.59199
        # phi(z) approx 0.999836 -> loss = -ln(phi(z)) should be very small
        assert loss < 0.01


class TestAstronomicalTimeConversion:
    """Verifies P0.3 separation between BMJD(TDB) and operational UTC timeline."""

    def test_utc_to_tai_and_tt_offsets(self):
        """Verifies leap second (+37s) and TT (+32.184s) offsets explicitly."""
        mjd_2024 = 60310.0  # 2024-01-01
        leap_sec = get_leap_seconds_utc_to_tai(mjd_2024)
        assert leap_sec == 37.0

        mjd_tt = compute_tt_from_utc_mjd(mjd_2024)
        expected_tt_mjd = mjd_2024 + (37.0 + TT_TAI_OFFSET_SEC) / 86400.0
        assert math.isclose(mjd_tt, expected_tt_mjd, rel_tol=1e-9)

    def test_roemer_delay_bounds(self):
        """Verifies geometric Roemer delay lies within [-500s, +500s] across observatories and sky coordinates."""
        mjd = 60000.0
        # Test all major optical survey observatories
        observatories = ["ZTF", "PALOMAR", "ATLAS_HKO", "PS1", "ATLAS_MLO", "ATLAS_SAO", "ATLAS_CHL", "DECAM"]
        # Test diverse sky directions: ecliptic plane, north celestial pole, celestial equator
        sky_coords = [(0.0, 0.0), (180.0, 0.0), (90.0, 66.5), (0.0, 90.0), (270.0, -45.0)]

        for obs in observatories:
            for ra, dec in sky_coords:
                roemer_delay = compute_roemer_delay_sec(mjd, ra_deg=ra, dec_deg=dec, observatory=obs)
                assert -500.0 <= roemer_delay <= 500.0, f"Roemer delay {roemer_delay} out of bounds for {obs} at ({ra}, {dec})"

    def test_leap_seconds_historical_transitions(self):
        """Verifies historical leap second insertions per IERS Bulletin C."""
        assert get_leap_seconds_utc_to_tai(57754.0) == 37.0  # 2017-01-01
        assert get_leap_seconds_utc_to_tai(57204.0) == 36.0  # 2015-07-01
        assert get_leap_seconds_utc_to_tai(56109.0) == 35.0  # 2012-07-01
        assert get_leap_seconds_utc_to_tai(54832.0) == 34.0  # 2009-01-01
        assert get_leap_seconds_utc_to_tai(50083.0) == 30.0  # 1996-01-01

    def test_causal_visibility_pipeline_and_ingestion_latency(self):
        """
        Verifies operational visibility invariant:
        Exposure taken at t_obs = 10:00, published at t_pub = 10:30, ingested at t_ingest = 10:35.
        At T_cutoff = 10:32, publication happened but ingestion has not occurred: observation is invisible.
        At T_cutoff = 10:40, both publication and ingestion have occurred: observation is visible.
        """
        t_pub = "2023-05-10T10:30:00Z"
        t_ingest = "2023-05-10T10:35:00Z"
        cutoff_pre_ingest = "2023-05-10T10:32:00Z"
        cutoff_post_ingest = "2023-05-10T10:40:00Z"

        assert not is_observation_visible_at_cutoff(t_pub, cutoff_pre_ingest, t_ingest_utc=t_ingest)
        assert is_observation_visible_at_cutoff(t_pub, cutoff_post_ingest, t_ingest_utc=t_ingest)


class TestCausalSequenceAndLandmarkProtection:
    """Verifies P0.2 and P1.1 causal sequence construction and science-preserving FPS."""

    def _create_mock_sequence(self, n_obs: int = 300) -> List[ObservationRecord]:
        records = []
        base_bmjd = 60000.0
        passbands = ["g", "r", "i"]

        for i in range(n_obs):
            t_bmjd = base_bmjd + i * 0.25  # 6-hour spacing
            t_pub = f"2023-03-01T{(i % 24):02d}:00:00Z"
            pb = passbands[i % len(passbands)]

            # Create transient-like flux peaking at index 150
            phase = i - 150
            flux = 20.0 * math.exp(-(phase ** 2) / (2 * (20 ** 2)))
            noise = 0.5

            if i < 30:
                # Pre-explosion non-detections / upper limits
                rec = ObservationRecord(
                    object_id="SN-MOCK-001",
                    observation_id=f"OBS-{i:04d}",
                    t_obs_bmjd=t_bmjd,
                    t_pub_utc=t_pub,
                    passband=pb,
                    survey_id="ZTF",
                    obs_type=ObservationType.CATALOG_UPPER_LIMIT,
                    upper_limit_flux_nmgy=1.5,
                    upper_limit_sigma_threshold=5.0,
                    effective_measurement_sigma_nmgy=0.3,
                    provenance_hash=f"hash_{i}"
                )
            else:
                rec = ObservationRecord(
                    object_id="SN-MOCK-001",
                    observation_id=f"OBS-{i:04d}",
                    t_obs_bmjd=t_bmjd,
                    t_pub_utc=t_pub,
                    passband=pb,
                    survey_id="ZTF",
                    obs_type=ObservationType.FORCED_PHOTOMETRY,
                    measured_flux_nmgy=flux + 0.1 * (i % 3 - 1),
                    measurement_sigma_nmgy=noise,
                    effective_measurement_sigma_nmgy=noise,
                    provenance_hash=f"hash_{i}"
                )
            records.append(rec)
        return records

    def test_landmark_extraction_causal_vs_retrospective(self):
        """Verifies that causal view only sees peak-so-far, not future peak."""
        records = self._create_mock_sequence(300)
        # Peak occurs at index 150.
        # Construct causal sequence truncated at index 100
        early_records = records[:100]
        causal_seq = TemporalSequence.from_records(
            object_id="SN-MOCK-001",
            records=early_records,
            view_type=SequenceViewType.CAUSAL_FORECAST
        )
        causal_landmarks = causal_seq.extract_landmark_indices(is_causal=True)

        # In causal sequence, max flux in r-band occurs at index ~99, not 150
        max_causal_flux = max(
            causal_seq.observations[idx].measured_flux_nmgy or 0.0
            for idx in causal_landmarks
        )
        # The true global peak flux is ~20.0 (at index 150)
        # In early causal view (up to index 100), flux is ~0.87
        assert max_causal_flux < 5.0  # Future peak of 20.0 is strictly invisible!

    def test_science_preserving_fps_decimation(self):
        """Verifies hierarchical FPS compresses 300 observations to <= 256 while preserving peaks and limits."""
        records = self._create_mock_sequence(300)
        seq = TemporalSequence.from_records("SN-MOCK-001", records, view_type=SequenceViewType.FULL_HISTORY)
        assert len(seq.observations) == 300

        compressed_seq = seq.science_preserving_hierarchical_fps(max_tokens=256)
        assert len(compressed_seq.observations) <= 256
        assert compressed_seq.is_decimated is True
        assert len(compressed_seq.unmerged_records_audit) == 300 - len(compressed_seq.observations)

        # Assert that the global peak flux in each passband is preserved
        for pb in ["g", "r", "i"]:
            raw_peak_flux = max(
                r.measured_flux_nmgy for r in records
                if r.passband == pb and r.measured_flux_nmgy is not None
            )
            compressed_peak_flux = max(
                r.measured_flux_nmgy for r in compressed_seq.observations
                if r.passband == pb and r.measured_flux_nmgy is not None
            )
            assert math.isclose(raw_peak_flux, compressed_peak_flux, abs_tol=1e-5)


class TestAkimaPredictiveModel:
    """Verifies Baseline 1 Akima predictive distribution and Leave-One-Out (LOO) uncertainty."""

    def test_akima_loo_uncertainty_and_extrapolation(self):
        """Verifies Akima produces non-zero LOO error on knots and expanding extrapolation uncertainty."""
        t = np.linspace(0.0, 30.0, 20)
        # Sine-wave light curve
        y = 5.0 * np.sin(t / 5.0) + 10.0
        sigmas = np.full_like(t, 0.2)

        model = AkimaPredictiveModel(tau_decay=15.0, s_extrap_prior=3.0)
        model.fit(t, y, sigmas)

        # Test interpolation at knot
        mu_interp, sigma_interp = model.predict(np.array([15.0]))
        assert math.isclose(mu_interp[0], 5.0 * math.sin(15.0 / 5.0) + 10.0, abs_tol=0.5)
        assert sigma_interp[0] > 0.0  # LOO residual variance must be non-zero!

        # Test future extrapolation at t = 50.0 (past t_max = 30.0)
        mu_extrap, sigma_extrap = model.predict(np.array([50.0]))
        # Uncertainty in future must be strictly larger than local knot uncertainty
        assert sigma_extrap[0] > sigma_interp[0]

        # Analytical CRPS evaluation
        crps = AkimaPredictiveModel.evaluate_crps(y_true=10.0, mu_pred=mu_interp[0], sigma_pred=sigma_interp[0])
        assert crps >= 0.0
