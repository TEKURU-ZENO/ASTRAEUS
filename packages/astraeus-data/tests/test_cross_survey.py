"""
Unit tests for ASTRAEUS Cross-Survey Data Contracts (Pan-STARRS1 & DECam) and Photometric Harmonization.
"""

import numpy as np
import pytest

from astraeus_data.schemas.panstarrs import (
    PanstarrsFilter,
    PanstarrsMeasurement,
    PanstarrsLightCurve,
    PANSTARRS_EFFECTIVE_WAVELENGTHS,
)
from astraeus_data.schemas.decam import (
    DecamFilter,
    DecamMeasurement,
    DecamLightCurve,
    DECAM_EFFECTIVE_WAVELENGTHS,
)
from astraeus_data.photometry.harmonization import (
    PhotometricHarmonizer,
    HarmonizedPhotometricEpoch,
    SURVEY_FILTER_METADATA,
)


def test_panstarrs_schema_and_upper_limits():
    # 1. Secure Detection
    det = PanstarrsMeasurement(
        mjd=56000.123,
        filter_band=PanstarrsFilter.G_P1,
        is_detection=True,
        psf_mag=18.50,
        psf_mag_err=0.03,
        psf_flux_ujy=144.54,
        psf_flux_err_ujy=4.0,
        limiting_mag=21.5,
        psf_qf=0.98,
        info_flags=0,
    )
    assert det.is_clean is True
    assert det.psf_mag == 18.50

    # 2. Upper Limit (Non-detection)
    uplim = PanstarrsMeasurement(
        mjd=56002.456,
        filter_band=PanstarrsFilter.R_P1,
        is_detection=False,
        psf_mag=None,
        psf_mag_err=None,
        psf_flux_ujy=1.2,
        psf_flux_err_ujy=3.5,
        limiting_mag=21.2,
        psf_qf=0.95,
        info_flags=0,
    )
    assert uplim.is_clean is True
    assert uplim.is_detection is False
    assert uplim.psf_mag is None

    # 3. Light Curve Packet
    lc = PanstarrsLightCurve(
        obj_id="PSO J180.2500+25.5000",
        ra_deg=180.25,
        dec_deg=25.50,
        epochs=[det, uplim],
    )
    assert lc.num_detections == 1
    assert lc.num_upper_limits == 1
    assert len(lc.raw_packet_hash) == 64


def test_decam_schema_and_quality_flags():
    # Clean observation
    meas = DecamMeasurement(
        mjd=58000.5,
        filter_band=DecamFilter.I_DECAM,
        is_detection=True,
        mag=19.20,
        mag_err=0.04,
        flux_ujy=75.86,
        flux_err_ujy=2.8,
        limiting_mag=22.5,
        fwhm_arcsec=1.1,
        airmass=1.15,
        exposure_time_sec=90.0,
        cp_flags=0,
    )
    assert meas.is_clean is True

    # Bad seeing / corrupted flag
    bad_meas = DecamMeasurement(
        mjd=58001.5,
        filter_band=DecamFilter.Z_DECAM,
        is_detection=True,
        mag=17.5,
        mag_err=0.1,
        flux_ujy=363.0,
        flux_err_ujy=33.0,
        limiting_mag=20.0,
        fwhm_arcsec=2.8, # Bad seeing > 2.2
        airmass=2.3,     # High airmass > 2.0
        exposure_time_sec=30.0,
        cp_flags=16,     # Non-zero CP flag
    )
    assert bad_meas.is_clean is False

    lc = DecamLightCurve(
        entity_id="DECam-DES-SN18abc",
        ra_deg=45.1234,
        dec_deg=-30.5678,
        epochs=[meas, bad_meas],
    )
    assert lc.num_detections == 2
    assert len(lc.raw_packet_hash) == 64


def test_photometric_harmonizer_flux_magnitude_roundtrip():
    mag_true = 19.50
    mag_err_true = 0.05

    # 1. Mag to Flux
    flux_ujy, flux_err_ujy = PhotometricHarmonizer.mag_to_flux_ujy(mag_true, mag_err_true)
    assert flux_ujy > 0.0
    assert flux_err_ujy > 0.0

    # 2. Flux to Mag round-trip
    mag_round, mag_err_round = PhotometricHarmonizer.flux_ujy_to_mag(flux_ujy, flux_err_ujy)
    assert mag_round is not None
    assert mag_err_round is not None
    assert np.isclose(mag_round, mag_true, atol=1e-5)
    assert np.isclose(mag_err_round, mag_err_true, atol=1e-5)

    # 3. Non-positive flux returns None for magnitude
    neg_mag, neg_err = PhotometricHarmonizer.flux_ujy_to_mag(-5.0, 2.0)
    assert neg_mag is None
    assert neg_err is None


def test_extinction_and_passband_mapping():
    # Extinction on g-band
    f_raw = 100.0  # uJy
    ebv = 0.05
    f_dered = PhotometricHarmonizer.apply_galactic_extinction(f_raw, "ZTF_g", ebv)
    assert f_dered > f_raw

    # Passband matching: ZTF_g (4770 A) -> Pan-STARRS g_P1 (4866 A) delta ~ 96 A
    best_ps1, delta_ps1 = PhotometricHarmonizer.find_nearest_passband("ZTF_g", "P1")
    assert best_ps1 == "g_P1"
    assert delta_ps1 == pytest.approx(96.0, abs=5.0)

    # ZTF_r (6390 A) -> DECam r_DECam (6415 A) delta ~ 25 A
    best_decam, delta_decam = PhotometricHarmonizer.find_nearest_passband("ZTF_r", "DECam")
    assert best_decam == "r_DECam"
    assert delta_decam == pytest.approx(25.0, abs=5.0)


def test_cross_survey_bayesian_matcher():
    from astraeus_data.crossmatch.cross_survey_matcher import (
        CrossSurveyMatcher,
        CrossSurveyMatchStatus,
        SurveyCandidate,
    )

    matcher = CrossSurveyMatcher(completeness_prior_q=0.90)

    # 1. Single Close Counterpart -> Secure Match
    cand_close = SurveyCandidate(
        catalog_name="Pan-STARRS1",
        source_id="PSO_123",
        ra_deg=100.00005,  # ~ 0.18 arcsec away
        dec_deg=20.00000,
        mag=18.5,
        filter_band="g_P1",
    )
    res_secure = matcher.match_source(
        query_id="ztf_src_01",
        query_ra=100.0,
        query_dec=20.0,
        query_pos_err_arcsec=0.1,
        candidates=[cand_close],
    )
    assert res_secure.match_status == CrossSurveyMatchStatus.SECURE_MATCH
    assert res_secure.posterior_match_probability > 0.85
    assert res_secure.separation_arcsec < 0.5

    # 2. Competing Counterparts -> Ambiguous Match
    cand_comp1 = SurveyCandidate(
        catalog_name="DECam",
        source_id="DECam_01",
        ra_deg=100.0002,  # ~ 0.7 arcsec
        dec_deg=20.0,
        mag=19.0,
        filter_band="r_DECam",
    )
    cand_comp2 = SurveyCandidate(
        catalog_name="DECam",
        source_id="DECam_02",
        ra_deg=100.00025, # ~ 0.9 arcsec
        dec_deg=20.0,
        mag=19.2,
        filter_band="r_DECam",
    )
    res_ambig = matcher.match_source(
        query_id="ztf_src_02",
        query_ra=100.0,
        query_dec=20.0,
        query_pos_err_arcsec=0.2,
        candidates=[cand_comp1, cand_comp2],
    )
    assert res_ambig.match_status == CrossSurveyMatchStatus.AMBIGUOUS_MATCH

    # 3. Training/Test Spatial Overlap Quarantine Audit
    train_entities = [{"entity_id": "ztf_train_001", "ra_deg": 150.0, "dec_deg": 10.0}]
    eval_entities = [
        {"entity_id": "ps1_eval_overlap", "ra_deg": 150.0002, "dec_deg": 10.0},  # ~ 0.7 arcsec overlap (< 3 arcsec)
        {"entity_id": "ps1_eval_clean", "ra_deg": 150.0100, "dec_deg": 10.0},    # ~ 35 arcsec away
    ]
    quarantined = matcher.audit_training_test_overlap(train_entities, eval_entities)
    assert "ps1_eval_overlap" in quarantined
    assert "ps1_eval_clean" not in quarantined
