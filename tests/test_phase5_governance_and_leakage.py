"""
ASTRAEUS Phase 5: Governance, Zero-Touch Quarantine, Adapter Fail-Closed & Bayesian Matcher Verification Suite.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Milestone: Phase 5 Cross-Survey Generalization & Domain Robustness

Tests:
1. Gate 3 Quarantine Zero-Touch & Geometric Footprint Disjointness (1,808 test pixels, 586 objects).
2. Champion Model Frozen Hash Immutability (466c2752808be01e...).
3. Canonical Observation Record Contract v2.0 (timestamp semantics, fail-closed uncertainties).
4. Survey Adapters Fail-Closed Verification (ATLAS, DECam DDF, Rubin LSST).
5. Photometric Harmonizer Negative Flux Rejection (strict ban on unphysical linear fallback).
6. Competing-Hypothesis Bayesian Matcher (H0 vs Hj normalization, candidate competition).
7. Stable Entity Registry (permanent IDs, decoupling entity from coordinates).
8. Workstream 3 Identity Pilot Deliverables & Four-Tier Yield Accounting.
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
    TransformationState,
    AtlasForcedPhotometryAdapter,
    DecamDdfPhotometryAdapter,
    RubinAlertInterfaceAdapter,
)
from astraeus_data.photometry.harmonization import (
    PhotometricHarmonizer,
    HarmonizedPhotometricEpoch,
    SURVEY_FILTER_METADATA,
)
from astraeus_data.crossmatch.cross_survey_matcher import (
    CrossSurveyMatcher,
    SurveyCandidate,
    CrossSurveyMatchStatus,
)
from astraeus_data.registry import StableEntityRegistry, StableEntityRecord

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


def test_phase5_canonical_observation_contract_v2():
    """
    CONTRACT TEST v2.0: CanonicalObservationRecord preserves physical time scales,
    natively preserves negative difference flux, and fails closed on non-positive uncertainty.
    """
    # 1. Negative flux observation (unbiased forced photometry subtraction residual)
    rec = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ATLAS_59000123_c",
        t_native=59000.123,
        native_time_scale="UTC",
        exposure_midpoint_convention="MIDPOINT",
        t_converted_bmjd=None,
        converted_time_scale=None,
        time_conversion_method="NATIVE_UNCONVERTED",
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
        zero_point_ab=23.900,
        transformation_state=TransformationState.NATIVE_PASSBAND_UNTRANSFORMED,
        provenance_hash="abcd" * 16,
    )
    assert rec.measured_flux_ujy == -12.5
    assert rec.is_positive_detection_snr5 is False
    assert rec.native_time_scale == "UTC"
    assert rec.transformation_state == TransformationState.NATIVE_PASSBAND_UNTRANSFORMED

    # 2. Assert fail-closed validation on non-positive uncertainty
    with pytest.raises(ValueError):
        CanonicalObservationRecord(
            astraeus_id="ASTRAEUS-EVT-00000001",
            survey_obs_id="BAD_OBS",
            t_native=59000.0,
            t_obs_bmjd=59000.0,
            ra_deg=150.0,
            dec_deg=2.0,
            survey_id="ATLAS",
            native_passband="ATLAS_c",
            obs_type=ObservationType.DIFFERENCE_FORCED,
            measured_flux_ujy=10.0,
            measurement_sigma_ujy=-1.0,  # Invalid negative sigma!
            is_positive_detection_snr5=False,
            provenance_hash="1234" * 16,
        )


def test_phase5_survey_adapters_fail_closed():
    """
    ADAPTER FAIL-CLOSED TEST: Verifies ATLAS, DECam DDF, and Rubin adapters
    reject unknown filters and non-positive uncertainties without silent defaults.
    """
    # 1. ATLAS: Rejects unknown filter
    bad_atlas_filter = {
        "MJD": 59123.456,
        "uJy": 10.0,
        "duJy": 2.0,
        "filt": "unknown_filter",
        "chin": 1.2,
    }
    assert AtlasForcedPhotometryAdapter.parse_record(bad_atlas_filter, "ASTRAEUS-EVT-00000001", 150.0, 2.0) is None

    # ATLAS: Rejects non-positive uncertainty
    bad_atlas_err = {
        "MJD": 59123.456,
        "uJy": 10.0,
        "duJy": -0.5,
        "filt": "c",
        "chin": 1.2,
    }
    assert AtlasForcedPhotometryAdapter.parse_record(bad_atlas_err, "ASTRAEUS-EVT-00000001", 150.0, 2.0) is None

    # ATLAS: Valid record parses cleanly with UTC scale
    good_atlas = {
        "MJD": 59123.456,
        "uJy": -5.0,
        "duJy": 2.5,
        "filt": "c",
        "chin": 1.1,
    }
    rec_atlas = AtlasForcedPhotometryAdapter.parse_record(good_atlas, "ASTRAEUS-EVT-00000001", 150.0, 2.0)
    assert rec_atlas is not None
    assert rec_atlas.native_passband == "ATLAS_c"
    assert rec_atlas.native_time_scale == "UTC"
    assert rec_atlas.measured_flux_ujy == -5.0

    # 2. DECam: Rejects unknown filter (no default to r_DECam!)
    bad_decam_filter = {
        "mjd": 59200.0,
        "flux_ujy": 15.0,
        "flux_err_ujy": 2.0,
        "filter": "nonexistent_band",
    }
    assert DecamDdfPhotometryAdapter.parse_record(bad_decam_filter, "ASTRAEUS-EVT-00000002", 150.0, 2.0) is None

    # 3. Rubin: Preserves TAI time scale and converts nJy to uJy
    rubin_alert = {
        "midpointMjdTai": 59300.5,
        "filterName": "r",
        "ra": 53.1,
        "dec": -27.8,
        "psfFlux": 3500.0,      # 3500 nJy = 3.5 uJy
        "psfFluxErr": 350.0,    # 350 nJy = 0.35 uJy
    }
    rec_rubin = RubinAlertInterfaceAdapter.parse_dia_source(rubin_alert, "ASTRAEUS-EVT-00000003", 53.1, -27.8)
    assert rec_rubin is not None
    assert rec_rubin.native_time_scale == "TAI"
    assert np.isclose(rec_rubin.measured_flux_ujy, 3.5, atol=1e-5)
    assert np.isclose(rec_rubin.measurement_sigma_ujy, 0.35, atol=1e-5)


def test_phase5_photometric_harmonizer_rejection_of_negative_flux_interpolation():
    """
    HARMONIZATION TEST: Proves strict ban on linear interpolation of negative difference flux.
    """
    # 1. Non-positive flux must raise ValueError per ASTRAEUS-P5-HARMON-002
    with pytest.raises(ValueError, match="Non-positive difference flux cannot be transformed"):
        PhotometricHarmonizer.ztf_to_atlas_flux(-10.0, 15.0, 2.0, 2.0)

    with pytest.raises(ValueError, match="Non-positive difference flux cannot be transformed"):
        PhotometricHarmonizer.atlas_to_ztf_flux(15.0, -5.0, 2.0, 2.0)

    # 2. Positive fluxes in valid color range transform accurately
    fg, fr = 100.0, 120.0
    out = PhotometricHarmonizer.ztf_to_atlas_flux(fg, fr, 2.0, 2.5)
    assert "ATLAS_c" in out and "ATLAS_o" in out
    fc, fo = out["ATLAS_c"][0], out["ATLAS_o"][0]

    # Invert back to ZTF
    ztf_back = PhotometricHarmonizer.atlas_to_ztf_flux(fc, fo, out["ATLAS_c"][1], out["ATLAS_o"][1])
    assert np.isclose(ztf_back["ZTF_g"][0], fg, rtol=0.05)
    assert np.isclose(ztf_back["ZTF_r"][0], fr, rtol=0.05)


def test_phase5_competing_hypothesis_bayesian_matcher():
    """
    BAYESIAN MATCHER TEST: Verifies competing hypothesis mathematics (H0 vs Hj) and normalization.
    """
    matcher = CrossSurveyMatcher(completeness_prior_q=0.90, max_search_radius_arcsec=5.0)

    # 1. Single close candidate (0.05 arcsec offset)
    cand_close = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="CAND_CLOSE",
        ra_deg=100.00001,
        dec_deg=10.00001,
        ra_err_arcsec=0.05,
        dec_err_arcsec=0.05,
        mag=18.0,
        filter_band="ATLAS_o",
    )
    res_single = matcher.match_source("QUERY_1", 100.0, 10.0, 0.05, [cand_close])
    assert res_single.match_status == CrossSurveyMatchStatus.SECURE_MATCH
    assert res_single.posterior_match_probability >= 0.95
    # Normalization: P(H1) + P(H0) == 1.0
    assert np.isclose(res_single.posterior_match_probability + res_single.posterior_no_counterpart_probability, 1.0, atol=1e-5)

    # 2. Candidate competition: two candidates within 0.12 arcsec
    cand_comp1 = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="COMP_1",
        ra_deg=100.00002,
        dec_deg=10.0,
        ra_err_arcsec=0.05,
        dec_err_arcsec=0.05,
        mag=18.0,
        filter_band="ATLAS_o",
    )
    cand_comp2 = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="COMP_2",
        ra_deg=100.0,
        dec_deg=10.00002,
        ra_err_arcsec=0.05,
        dec_err_arcsec=0.05,
        mag=18.1,
        filter_band="ATLAS_o",
    )
    res_comp = matcher.match_source("QUERY_2", 100.0, 10.0, 0.05, [cand_comp1, cand_comp2])
    # When two candidates compete closely, status must be AMBIGUOUS_MATCH
    assert res_comp.match_status == CrossSurveyMatchStatus.AMBIGUOUS_MATCH
    assert res_comp.competing_candidates_count == 2
    # Neither candidate can achieve >= 0.90
    assert res_comp.posterior_match_probability < 0.90


def test_phase5_stable_entity_registry():
    """
    REGISTRY TEST: Verifies permanent entity registration and decoupling from coordinates.
    """
    registry = StableEntityRegistry()
    ent1 = registry.register_entity(
        reference_ra_deg=120.5,
        reference_dec_deg=35.2,
        iau_name="SN 2020abc",
        ztf_id="ZTF20aaaxyz",
        spectral_class="SN_Ia",
    )
    assert ent1.astraeus_id == "ASTRAEUS-EVT-00000001"
    assert ent1.iau_name == "SN 2020abc"

    # Registering same ZTF ID returns identical permanent entity
    ent2 = registry.register_entity(
        reference_ra_deg=120.50005,  # Slightly refined coordinate
        reference_dec_deg=35.20002,
        ztf_id="ZTF20aaaxyz",
    )
    assert ent2.astraeus_id == ent1.astraeus_id


def test_phase5_identity_pilot_deliverables_and_four_tier_yield(repo_root: Path):
    """
    DELIVERABLES & FOUR-TIER YIELD TEST:
    Asserts Workstream 3 deliverables exist, adhere to the 4-tier yield accounting,
    and verify calibration monotonicity.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"

    # 1. Report JSON
    rep_path = p5_dir / "phase5_identity_matching_pilot_report.json"
    assert rep_path.exists()
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Asserts 4-tier yield separation
    yields = rep["four_tier_yield_accounting"]
    assert yields["tier1_geometric_footprint_eligible"] == 150
    assert yields["tier2_modeled_visibility_yield"] == 150
    assert yields["tier3_observed_retrieval_secure_matches"] == 142
    assert yields["tier4_feature_evaluable_yield"] == 142
    assert yields["ambiguous_conflict_count"] == 8

    # 2. Ledgers
    id_parquet = p5_dir / "phase5_cross_survey_identity_ledger.parquet"
    ev_parquet = p5_dir / "phase5_spectroscopic_evidence_ledger.parquet"
    cf_parquet = p5_dir / "phase5_identity_conflict_ledger.parquet"

    assert id_parquet.exists()
    assert ev_parquet.exists()
    assert cf_parquet.exists()

    df_id = pd.read_parquet(id_parquet)
    df_ev = pd.read_parquet(ev_parquet)
    df_cf = pd.read_parquet(cf_parquet)

    assert len(df_id) == 150
    assert len(df_ev) == 150
    assert len(df_cf) == 8

    # 3. Calibration report
    calib_path = p5_dir / "phase5_match_calibration_report.json"
    assert calib_path.exists()
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    assert calib["offset_monotonicity_verified"] is True
    assert calib["negative_control_rejection_rate"] == 1.0
