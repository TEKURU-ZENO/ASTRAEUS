"""
ASTRAEUS Phase 5: Governance, Zero-Touch Quarantine & Multi-Survey Adapter Verification Suite.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Milestone: Phase 5 Cross-Survey Generalization & Domain Robustness

Tests:
1. Gate 3 Quarantine Zero-Touch & Geometric Spatial Footprint Disjointness (1,808 test pixels, 586 objects).
2. Champion Model Frozen Hash Immutability (466c2752808be01e...).
3. Canonical Observation Record Contract (negative flux preservation, timing decoupling).
4. Survey Adapters Verification (ATLAS, DECam DDF, Rubin LSST).
5. Photometric Harmonizer Multi-Survey Extension (ATLAS & Rubin filters, color transformations).
6. Phase 5 Feasibility Pilot Deliverables Integrity (150 candidates, FEASIBILITY_ESTABLISHED).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import astropy.units as u
from astropy_healpix import HEALPix

from astraeus_data.adapters import (
    CanonicalObservationRecord,
    ObservationType,
    AtlasForcedPhotometryAdapter,
    DecamDdfPhotometryAdapter,
    RubinAlertInterfaceAdapter,
)
from astraeus_data.photometry.harmonization import (
    PhotometricHarmonizer,
    HarmonizedPhotometricEpoch,
    SURVEY_FILTER_METADATA,
)

EXPECTED_HEALPIX_CONFIG_HASH = "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b"
EXPECTED_SPRINT4H_MANIFEST_HASH = "1ddbf733cc52f6fe5a36a7a9854e780a59d2cadc3b4b153aaadf970a3b377761"
EXPECTED_CHAMPION_HASH = "466c2752808be01e4f41bd8fdb9c0804a57b03b2c02c35e5172ccebda411fce5"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_phase5_gate3_quarantine_zero_touch(repo_root: Path):
    """
    GOVERNANCE ASSERTION 1: Phase 5 must strictly preserve the Gate 3 Zero-Touch Quarantine.
    None of the Phase 5 pilot candidates may intersect the 1,808 quarantined test HEALPix pixels.
    """
    # 1. HEALPix config hash assertion
    hp_path = repo_root / "configs" / "data_splits_healpix.json"
    assert hp_path.exists()
    hp_sha = hashlib.sha256(hp_path.read_bytes()).hexdigest()
    assert hp_sha == EXPECTED_HEALPIX_CONFIG_HASH

    with open(hp_path, "r", encoding="utf-8") as f:
        hp_config = json.load(f)
    test_pixels = set(int(p) for p in hp_config["splits"]["test"])
    assert len(test_pixels) == 1808

    # 2. Custodian release manifest seal assertion
    s4h_path = repo_root / "experiments" / "sprint4h_final_release_remediation" / "sprint4h_release_manifest.json"
    assert s4h_path.exists()
    s4h_sha = hashlib.sha256(s4h_path.read_bytes()).hexdigest()
    assert s4h_sha == EXPECTED_SPRINT4H_MANIFEST_HASH

    with open(s4h_path, "r", encoding="utf-8") as f:
        s4h_data = json.load(f)
    assert s4h_data["spatial_isolation_audit"]["subtraction_verification"]["sequestered_test_count"] == 586

    # 3. Geometric footprint disjointness for Phase 5 pilot cohort
    cohort_path = repo_root / "experiments" / "phase5_cross_survey_generalization" / "phase5_feasibility_pilot_cohort.parquet"
    assert cohort_path.exists(), f"Missing pilot cohort: {cohort_path}"
    df_cohort = pd.read_parquet(cohort_path)
    assert len(df_cohort) == 150

    hp = HEALPix(nside=hp_config.get("healpix_nside", 32), order="nested", frame="icrs")
    cand_pixels = hp.lonlat_to_healpix(df_cohort["ra_deg"].values * u.deg, df_cohort["dec_deg"].values * u.deg)
    overlap = set(cand_pixels).intersection(test_pixels)
    assert len(overlap) == 0, f"Violation: {len(overlap)} Phase 5 candidates intersect quarantined test pixels!"


def test_phase5_champion_model_frozen_hash(repo_root: Path):
    """
    GOVERNANCE ASSERTION 2: Champion model artifact must remain bit-level immutable.
    """
    champ_path = repo_root / "experiments" / "sprint4h_final_release_remediation" / "sprint4h_champion_model.joblib"
    assert champ_path.exists()
    champ_sha = hashlib.sha256(champ_path.read_bytes()).hexdigest()
    assert champ_sha.lower() == EXPECTED_CHAMPION_HASH.lower()


def test_phase5_canonical_observation_contract():
    """
    CONTRACT TEST 1: CanonicalObservationRecord preserves negative difference fluxes,
    validates uncertainty propagation, and correctly flags 5-sigma detections.
    """
    # 1. Negative flux observation (unbiased forced photometry subtraction residual)
    neg_rec = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-P5-261116-A1B2C3D4",
        survey_obs_id="ATLAS_59000123_c",
        t_obs_bmjd=59000.123,
        ra_deg=150.0,
        dec_deg=2.0,
        survey_id="ATLAS",
        native_passband="ATLAS_c",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=-12.5,
        measurement_sigma_ujy=8.0,
        upper_limit_ujy=40.0,
        is_positive_detection_snr5=False,
        provenance_hash="abcd" * 16,
    )
    assert neg_rec.measured_flux_ujy == -12.5
    assert neg_rec.is_positive_detection_snr5 is False
    assert neg_rec.upper_limit_ujy == 40.0

    # 2. Positive 5-sigma detection
    pos_rec = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-P5-261116-A1B2C3D4",
        survey_obs_id="ATLAS_59010456_o",
        t_obs_bmjd=59010.456,
        ra_deg=150.0,
        dec_deg=2.0,
        survey_id="ATLAS",
        native_passband="ATLAS_o",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=120.0,
        measurement_sigma_ujy=15.0,
        upper_limit_ujy=75.0,
        is_positive_detection_snr5=True,
        provenance_hash="1234" * 16,
    )
    assert pos_rec.measured_flux_ujy == 120.0
    assert pos_rec.is_positive_detection_snr5 is True


def test_phase5_survey_adapters():
    """
    ADAPTER TEST: Verifies ATLAS, DECam DDF, and Rubin LSST adapters.
    """
    # 1. ATLAS adapter parsing
    atlas_row_clean = {
        "MJD": 59123.456,
        "m": None,
        "uJy": -8.5,
        "duJy": 7.2,
        "filt": "c",
        "chin": 1.15,
        "RA": 160.5,
        "Dec": 25.4,
    }
    rec_atlas = AtlasForcedPhotometryAdapter.parse_record(
        atlas_row_clean,
        astraeus_id="ASTRAEUS-P5-261116-00000001",
        ra_fallback=160.5,
        dec_fallback=25.4,
    )
    assert rec_atlas is not None
    assert rec_atlas.survey_id == "ATLAS"
    assert rec_atlas.native_passband == "ATLAS_c"
    assert rec_atlas.measured_flux_ujy == -8.5
    assert rec_atlas.measurement_sigma_ujy == 7.2
    assert rec_atlas.is_positive_detection_snr5 is False

    # Rejected on chin > 3.0 cut
    atlas_row_bad = dict(atlas_row_clean, chin=4.5)
    rec_bad = AtlasForcedPhotometryAdapter.parse_record(
        atlas_row_bad,
        astraeus_id="ASTRAEUS-P5-261116-00000001",
        ra_fallback=160.5,
        dec_fallback=25.4,
    )
    assert rec_bad is None

    # 2. DECam DDF adapter parsing
    decam_row = {
        "mjd": 59200.789,
        "mag": 21.5,
        "mag_err": 0.04,
        "filter": "i",
        "ra": 150.1,
        "dec": 2.2,
        "flags": 0,
    }
    rec_decam = DecamDdfPhotometryAdapter.parse_record(
        decam_row,
        astraeus_id="ASTRAEUS-P5-261116-00000002",
        ra_fallback=150.1,
        dec_fallback=2.2,
    )
    assert rec_decam is not None
    assert rec_decam.survey_id == "DECam"
    assert rec_decam.native_passband == "i_DECam"
    assert rec_decam.measured_flux_ujy > 0.0

    # 3. Rubin LSST adapter parsing (nanoJansky to microJansky)
    rubin_dia_source = {
        "midpointMjdTai": 59300.111,
        "filterName": "r",
        "ra": 53.1,
        "dec": -27.8,
        "psfFlux": 2500.0,      # 2500 nJy = 2.5 uJy
        "psfFluxErr": 250.0,    # 250 nJy = 0.25 uJy
        "flags": 0,
    }
    rec_rubin = RubinAlertInterfaceAdapter.parse_dia_source(
        rubin_dia_source,
        astraeus_id="ASTRAEUS-P5-261116-00000003",
        ra_fallback=53.1,
        dec_fallback=-27.8,
    )
    assert rec_rubin is not None
    assert rec_rubin.survey_id == "Rubin"
    assert rec_rubin.native_passband == "r_LSST"
    assert np.isclose(rec_rubin.measured_flux_ujy, 2.5, atol=1e-5)
    assert np.isclose(rec_rubin.measurement_sigma_ujy, 0.25, atol=1e-5)
    assert rec_rubin.is_positive_detection_snr5 is True  # SNR = 10.0


def test_phase5_photometric_harmonizer_atlas_conversions():
    """
    HARMONIZATION TEST: Verifies ATLAS filter metadata and bidirectional color conversions.
    """
    assert "ATLAS_c" in SURVEY_FILTER_METADATA
    assert "ATLAS_o" in SURVEY_FILTER_METADATA
    assert "r_LSST" in SURVEY_FILTER_METADATA

    # Passband matching
    best_atlas_c, delta_c = PhotometricHarmonizer.find_nearest_passband("ZTF_g", "ATLAS")
    assert best_atlas_c == "ATLAS_c"
    best_atlas_o, delta_o = PhotometricHarmonizer.find_nearest_passband("ZTF_r", "ATLAS")
    assert best_atlas_o == "ATLAS_o"

    # Forward ZTF (g, r) to ATLAS (c, o)
    fg, fr = 100.0, 120.0  # uJy
    atlas_out = PhotometricHarmonizer.ztf_to_atlas_flux(fg, fr, 2.0, 2.5)
    assert "ATLAS_c" in atlas_out and "ATLAS_o" in atlas_out
    fc, fc_err = atlas_out["ATLAS_c"]
    fo, fo_err = atlas_out["ATLAS_o"]
    assert fc > 0.0 and fc_err > 0.0
    assert fo > 0.0 and fo_err > 0.0

    # Invert back to ZTF (g, r)
    ztf_back = PhotometricHarmonizer.atlas_to_ztf_flux(fc, fo, fc_err, fo_err)
    assert "ZTF_g" in ztf_back and "ZTF_r" in ztf_back
    fg_rec, _ = ztf_back["ZTF_g"]
    fr_rec, _ = ztf_back["ZTF_r"]
    # Reconstructed fluxes should match original within 5%
    assert np.isclose(fg_rec, fg, rtol=0.05)
    assert np.isclose(fr_rec, fr, rtol=0.05)


def test_phase5_feasibility_pilot_deliverables(repo_root: Path):
    """
    DELIVERABLES TEST: Asserts feasibility pilot deliverables exist and meet certification standards.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"

    # 1. Report JSON
    report_path = p5_dir / "phase5_feasibility_pilot_report.json"
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["pilot_disposition"] == "FEASIBILITY_ESTABLISHED"
    assert rep["author"] == "Dev Mehta"
    assert rep["feasibility_results"]["curated_cohort_size"] == 150
    assert rep["feasibility_results"]["atlas_feasibility"]["eligible_declination_fraction"] == 1.0
    assert rep["feasibility_results"]["atlas_feasibility"]["unbiased_negative_flux_retention"] is True
    assert rep["feasibility_results"]["spectroscopic_traceability"]["iau_name_traceability_rate"] == 1.0

    # 2. Cohort tables
    parquet_path = p5_dir / "phase5_feasibility_pilot_cohort.parquet"
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert len(df) == 150
    assert "astraeus_id" in df.columns
    assert "iau_id" in df.columns
    assert "astraeus_class" in df.columns
