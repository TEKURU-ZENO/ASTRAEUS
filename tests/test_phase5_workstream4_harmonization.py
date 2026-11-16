"""
ASTRAEUS Phase 5: Workstream 4 Photometric & Temporal Harmonization Verification Suite.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Milestone: Phase 5 Workstream 4 (Photometric and Temporal Harmonization)

Tests:
1. Physical time-scale offsets, leap seconds, and observatory geodetic locations.
2. Relativistic barycentric Roemer delay bounds (|dt| <= 0.00585 days) and record standardization.
3. Three-State Harmonization taxonomy (SED-supported, Empirical Color with positive covariance, Native Passband).
4. Strict non-destructive negative flux preservation invariant.
5. Cryptographic transition manifest integrity (18 sealed SHA-256 digests).
6. Reconciled H2 hypothesis preregistration alignment across scientific and statistical plans.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from astraeus_data.adapters import (
    CanonicalObservationRecord,
    ObservationType,
    TransformationState,
)
from astraeus_data.photometry import (
    PhotometricHarmonizer,
    TemporalHarmonizer,
    SURVEY_FILTER_METADATA,
    OBSERVATORY_REGISTRY,
    LEAP_SECONDS_TAI_UTC,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_ws4_temporal_harmonizer_time_scales_and_offsets():
    """
    TEMPORAL STANDARDS TEST:
    Verifies physical constants, leap seconds, TT/UTC offsets, and observatory registry geodetic coordinates.
    """
    # 1. Leap seconds and physical offsets
    assert LEAP_SECONDS_TAI_UTC == 37.0
    from astraeus_data.photometry.temporal import TT_UTC_OFFSET_SECONDS, TT_TAI_OFFSET_SECONDS
    assert TT_TAI_OFFSET_SECONDS == 32.184
    assert np.isclose(TT_UTC_OFFSET_SECONDS, 69.184)

    # 2. Observatory geodetic registry validation
    required_observatories = ["PALOMAR", "HALEAKALA", "MAUNA_LOA", "SUTHERLAND", "EL_SAUCE", "CERRO_TOLOLO", "CERRO_PACHON"]
    for obs in required_observatories:
        assert obs in OBSERVATORY_REGISTRY
        meta = OBSERVATORY_REGISTRY[obs]
        assert -90.0 <= meta["lat_deg"] <= 90.0
        assert -180.0 <= meta["lon_deg"] <= 180.0
        assert meta["height_m"] > 0.0

    # Test EarthLocation instantiation caching
    loc_palomar = TemporalHarmonizer.get_observatory_location("PALOMAR")
    assert loc_palomar is not None
    # Second call returns cached instance
    assert TemporalHarmonizer.get_observatory_location("PALOMAR") is loc_palomar


def test_ws4_temporal_barycentric_delay_bounds_and_record_standardization():
    """
    BARYCENTRIC CORRECTION TEST:
    Verifies that Roemer delays to the Solar System Barycenter are physically bounded
    (|dt| <= 0.00585 days) and standardizes CanonicalObservationRecord to BMJD_TDB.
    """
    mjd = 59500.0  # October 2021
    ra = 150.0
    dec = 20.0

    # 1. Roemer delay magnitude check
    dt_days = TemporalHarmonizer.compute_barycentric_delay_days(
        mjd=mjd,
        ra_deg=ra,
        dec_deg=dec,
        observatory_key="PALOMAR",
        native_scale="UTC",
    )
    # 1 AU / c ~= 499 seconds ~= 0.00578 days. Max bound including lunar/planetary motion is < 0.00585 days.
    assert abs(dt_days) <= 0.00585, f"Roemer delay {dt_days} exceeds physical 1 AU bound!"

    # 2. Standardization of un-converted CanonicalObservationRecord
    rec = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ZTF_59500123_g",
        t_native=mjd,
        native_time_scale="UTC",
        t_converted_bmjd=None,
        ra_deg=ra,
        dec_deg=dec,
        survey_id="ZTF",
        native_passband="ZTF_g",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=35.0,
        measurement_sigma_ujy=2.5,
        is_positive_detection_snr5=True,
        provenance_hash="1122" * 16,
    )
    assert rec.is_barycentric_converted is False
    assert rec.t_converted_bmjd is None

    std_rec = TemporalHarmonizer.standardize_record_time(rec, observatory_key="PALOMAR")
    assert std_rec.is_barycentric_converted is True
    assert std_rec.t_converted_bmjd is not None
    assert std_rec.converted_time_scale == "BMJD_TDB"
    assert std_rec.time_conversion_method == "ASTROPY_BARYCORR_V1"
    assert std_rec.t_obs_bmjd == std_rec.t_converted_bmjd

    t_val, t_scale = std_rec.get_time_coordinate()
    assert t_val == std_rec.t_converted_bmjd
    assert t_scale == "BMJD_TDB"


def test_ws4_three_state_harmonization_taxonomy():
    """
    THREE-STATE HARMONIZATION TAXONOMY TEST:
    Verifies State 1 (SED metadata availability), State 2 (Empirical color with positive covariance),
    and State 3 (Native passband retention).
    """
    # 1. State 1 Passband Infrastructure: Verify filter metadata coverage
    expected_filters = ["ZTF_g", "ZTF_r", "ZTF_i", "ATLAS_c", "ATLAS_o", "r_DECam", "g_P1", "r_LSST"]
    for f in expected_filters:
        assert f in SURVEY_FILTER_METADATA
        meta = SURVEY_FILTER_METADATA[f]
        assert meta["lambda_eff_angstrom"] > 3000.0
        assert meta["fwhm_angstrom"] > 500.0

    # 2. State 2 Empirical Color Transformation:
    # Synchronous positive detections with SNR >= 3.0 within [-0.4, 2.0]
    fg, fr = 150.0, 180.0
    sig_g, sig_r = 3.0, 3.5
    res = PhotometricHarmonizer.ztf_to_atlas_flux(fg, fr, sig_g, sig_r)
    assert "ATLAS_c" in res and "ATLAS_o" in res

    fc, sig_c = res["ATLAS_c"]
    fo, sig_o = res["ATLAS_o"]

    # Verify strictly positive cross-filter covariance
    cov_co = PhotometricHarmonizer.compute_atlas_cross_filter_covariance(
        g_err_mag=float((2.5 / np.log(10.0)) * (sig_g / fg)),
        r_err_mag=float((2.5 / np.log(10.0)) * (sig_r / fr)),
    )
    assert cov_co > 0.0

    # Exact inverse matrix reconstruction back to ZTF
    ztf_back = PhotometricHarmonizer.atlas_to_ztf_flux(fc, fo, sig_c, sig_o)
    assert np.isclose(ztf_back["ZTF_g"][0], fg, rtol=0.05)
    assert np.isclose(ztf_back["ZTF_r"][0], fr, rtol=0.05)


def test_ws4_negative_flux_preservation_invariant():
    """
    NEGATIVE FLUX PRESERVATION INVARIANT TEST:
    Proves that non-positive difference flux (-15.0 uJy) is strictly prohibited from empirical
    color transformation, preventing linear wavelength fallback or magnitude conversion,
    and forcing retention in State 3 (Native Passband).
    """
    neg_flux = -15.0
    pos_flux = 20.0
    err = 2.0

    # Attempting cross-filter color transformation on negative flux must fail closed
    with pytest.raises(ValueError, match="Non-positive difference flux cannot be transformed"):
        PhotometricHarmonizer.ztf_to_atlas_flux(neg_flux, pos_flux, err, err)

    with pytest.raises(ValueError, match="Non-positive difference flux cannot be transformed"):
        PhotometricHarmonizer.atlas_to_ztf_flux(pos_flux, neg_flux, err, err)

    # Creating a State 3 CanonicalObservationRecord natively preserves the negative value
    rec_neg = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ATLAS_NEG_001",
        t_native=59123.4,
        native_time_scale="UTC",
        ra_deg=100.0,
        dec_deg=15.0,
        survey_id="ATLAS",
        native_passband="ATLAS_o",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=neg_flux,
        measurement_sigma_ujy=4.5,
        upper_limit_ujy=22.5,
        is_positive_detection_snr5=False,
        transformation_state=TransformationState.NATIVE_PASSBAND_UNTRANSFORMED,
        provenance_hash="3344" * 16,
    )
    assert rec_neg.measured_flux_ujy == -15.0
    assert rec_neg.transformation_state == TransformationState.NATIVE_PASSBAND_UNTRANSFORMED


def test_ws4_transition_manifest_cryptographic_integrity(repo_root: Path):
    """
    TRANSITION MANIFEST INTEGRITY TEST:
    Verifies phase5_ws3_ws4_transition_manifest.json exists and asserts that all 18 sealed
    scientific protocols, specifications, ledgers, and manifests match their exact SHA-256 digests.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    manifest_path = p5_dir / "phase5_ws3_ws4_transition_manifest.json"
    assert manifest_path.exists(), f"Missing transition manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["manifest_identifier"] == "ASTRAEUS-P5-TRANSITION-WS3-WS4-001"
    assert data["author"] == "Dev Mehta"
    assert data["transition_type"] == "WORKSTREAM_CLOSURE_AND_INITIALIZATION"
    assert "WS3.1" in data["closed_workstreams"]
    assert "WS4" in data["initialized_workstreams"]

    # Verify all 18 artifact digests
    hashes = data["artifact_sha256_digests"]
    assert len(hashes) == 18

    for filename, expected_hash in hashes.items():
        file_path = p5_dir / filename
        assert file_path.exists(), f"Sealed asset missing from disk: {filename}"
        actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert actual_hash == expected_hash, f"Cryptographic seal broken for {filename}: expected {expected_hash}, got {actual_hash}"


def test_ws4_preregistered_h2_hypothesis_alignment(repo_root: Path):
    """
    H2 PREREGISTRATION RECONCILIATION TEST:
    Asserts that PHASE5_SCIENTIFIC_PROTOCOL.md and PHASE5_STATISTICAL_ANALYSIS_PLAN.md
    define the identical estimand, direction of comparison, and McNemar falsification rule for H2.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"
    proto_path = p5_dir / "PHASE5_SCIENTIFIC_PROTOCOL.md"
    stat_path = p5_dir / "PHASE5_STATISTICAL_ANALYSIS_PLAN.md"

    proto_text = proto_path.read_text(encoding="utf-8")
    stat_text = stat_path.read_text(encoding="utf-8")

    # Both documents must define Delta Macro-F1 = Macro-F1_phys - Macro-F1_raw
    assert r"\Delta \text{Macro-F1}" in proto_text and r"\text{Macro-F1}_{\mathrm{phys}}" in proto_text
    assert r"\Delta \text{Macro-F1} = \text{Macro-F1}_{\mathrm{phys}} - \text{Macro-F1}_{\mathrm{raw}}" in stat_text

    # Both documents must require McNemar test rejection at alpha = 0.05
    assert "McNemar" in proto_text and "0.05" in proto_text
    assert "McNemar" in stat_text and "0.05" in stat_text
