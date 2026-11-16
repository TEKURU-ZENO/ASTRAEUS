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
from astraeus_data.brokers import (
    AtlasRequestState,
    AtlasRequestRecord,
    AtlasForcedPhotometryClient,
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


def test_phase5_canonical_observation_contract_v2():
    """
    CONTRACT TEST v2.0: CanonicalObservationRecord preserves physical time scales,
    natively preserves negative difference flux, strictly keeps t_obs_bmjd=None unless converted,
    and fails closed on non-positive uncertainty.
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
        t_obs_bmjd=None,  # Strictly None unless barycentric conversion executed
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
    assert rec.t_obs_bmjd is None
    assert rec.t_converted_bmjd is None
    assert rec.observation_time == 59000.123
    assert rec.transformation_state == TransformationState.NATIVE_PASSBAND_UNTRANSFORMED

    # 2. Converted barycentric timestamp reflects in observation_time property
    rec_converted = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ATLAS_59000123_c_bary",
        t_native=59000.123,
        native_time_scale="UTC",
        t_converted_bmjd=59000.128,
        converted_time_scale="BMJD_TDB",
        time_conversion_method="ASTROPY_BARYCORR_V1",
        t_obs_bmjd=59000.128,
        ra_deg=150.0,
        dec_deg=2.0,
        survey_id="ATLAS",
        native_passband="ATLAS_c",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=25.0,
        measurement_sigma_ujy=2.0,
        is_positive_detection_snr5=True,
        provenance_hash="bcde" * 16,
    )
    assert rec_converted.observation_time == 59000.128

    # 3. Assert fail-closed validation on non-positive uncertainty
    with pytest.raises(ValueError):
        CanonicalObservationRecord(
            astraeus_id="ASTRAEUS-EVT-00000001",
            survey_obs_id="BAD_OBS",
            t_native=59000.0,
            t_obs_bmjd=None,
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
    Asserts Workstream 3 deliverables exist, adhere to the reclassified synthetic simulation
    4-tier yield accounting, verify spectroscopic metadata attribution, and verify calibration monotonicity.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"

    # 1. Report JSON
    rep_path = p5_dir / "phase5_identity_matching_pilot_report.json"
    assert rep_path.exists()
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Asserts 4-tier yield separation and simulation benchmark reclassification
    yields = rep["four_tier_yield_accounting"]
    assert yields["tier1_geometric_footprint_eligible"] == 150
    assert yields["tier2_brightness_screened"] == 150
    assert yields["synthetic_simulation_secure_matches"] == 142
    assert yields["synthetic_simulation_injected_conflicts"] == 8
    assert yields["actual_atlas_retrievals"] == 0
    assert yields["actual_feature_evaluable_yield"] == 0
    assert yields["accounting_classification"] == "SYNTHETIC_MATCHER_SIMULATION_BENCHMARK"
    assert rep["pilot_nature"] == "SYNTHETIC_MATCHER_SIMULATION_BENCHMARK"
    assert rep["empirical_retrieval_status"] == "PENDING_LIVE_BROKER_PILOT_EXECUTION"

    # 2. Ledgers
    id_parquet = p5_dir / "phase5_cross_survey_identity_ledger.parquet"
    ev_parquet = p5_dir / "phase5_spectroscopic_evidence_ledger.parquet"
    cf_parquet = p5_dir / "phase5_identity_conflict_ledger.parquet"
    reg_parquet = p5_dir / "phase5_entity_registry.parquet"

    assert id_parquet.exists()
    assert ev_parquet.exists()
    assert cf_parquet.exists()
    assert reg_parquet.exists()

    df_id = pd.read_parquet(id_parquet)
    df_ev = pd.read_parquet(ev_parquet)
    df_cf = pd.read_parquet(cf_parquet)
    df_reg = pd.read_parquet(reg_parquet)

    assert len(df_id) == 150
    assert len(df_ev) == 150
    assert len(df_cf) == 8
    assert len(df_reg) == 150

    # Verify Spectroscopic Evidence Tier & Metadata Qualification
    assert df_ev["evidence_tier"].iloc[0] == "PROVISIONAL_S2_UNVERIFIED_SPECTRAL_CHAIN"
    assert df_ev["spectroscopic_instrument"].iloc[0] == "UNVERIFIED_INDIVIDUAL_INSTRUMENT_METADATA_BTS_ARCHIVE"
    assert df_ev["verification_status"].iloc[0] == "PROVISIONAL_TRACEABLE_TO_IAU_TNS"

    # Verify Registry contains no premature ATLAS association
    assert df_reg["atlas_id"].isna().all() or (df_reg["atlas_id"] == "").all() or (df_reg["atlas_id"] == None).all()

    # 3. Calibration report
    calib_path = p5_dir / "phase5_match_calibration_report.json"
    assert calib_path.exists()
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    assert calib["offset_monotonicity_verified"] is True
    assert calib["negative_control_rejection_rate"] == 1.0


def test_phase5_astrometric_2d_covariance_anisotropic():
    """
    ASTROMETRIC 2D COVARIANCE TEST:
    Verifies that CrossSurveyMatcher uses the full 2D Gaussian astrometric covariance matrix
    with anisotropic uncertainties (sigma_ra != sigma_dec) and Mahalanobis distance.
    """
    matcher = CrossSurveyMatcher(completeness_prior_q=0.90, max_search_radius_arcsec=5.0)

    query_ra = 180.0
    query_dec = 0.0  # On celestial equator: cos(dec) = 1.0
    sigma_ra = 0.05   # Arcsec
    sigma_dec = 0.20  # Arcsec (significant Dec elongation)

    # Candidate A: offset 0.10 arcsec in RA, 0.00 arcsec in Dec
    # Total error along RA: sqrt(0.05^2 + 0.05^2) = 0.0707" -> Mahalanobis distance = (0.10 / 0.0707)^2 = 2.00
    cand_ra_offset = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="CAND_RA_OFFSET",
        ra_deg=query_ra + (0.10 / 3600.0),
        dec_deg=query_dec,
        ra_err_arcsec=0.05,
        dec_err_arcsec=0.05,
        mag=18.0,
        filter_band="ATLAS_o",
    )

    # Candidate B: offset 0.00 arcsec in RA, 0.10 arcsec in Dec
    # Total error along Dec: sqrt(0.20^2 + 0.05^2) = 0.2062" -> Mahalanobis distance = (0.10 / 0.2062)^2 = 0.235
    cand_dec_offset = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="CAND_DEC_OFFSET",
        ra_deg=query_ra,
        dec_deg=query_dec + (0.10 / 3600.0),
        ra_err_arcsec=0.05,
        dec_err_arcsec=0.05,
        mag=18.0,
        filter_band="ATLAS_o",
    )

    res_a = matcher.match_source(
        "QUERY_ANISO_RA",
        query_ra,
        query_dec,
        query_pos_err_arcsec=sigma_ra,
        query_dec_err_arcsec=sigma_dec,
        candidates=[cand_ra_offset],
    )

    res_b = matcher.match_source(
        "QUERY_ANISO_DEC",
        query_ra,
        query_dec,
        query_pos_err_arcsec=sigma_ra,
        query_dec_err_arcsec=sigma_dec,
        candidates=[cand_dec_offset],
    )

    # Because Dec error is much larger, the same 0.10" offset in Dec is much more consistent with H1
    # than in RA. Therefore, Cand B must have a higher likelihood ratio and posterior than Cand A!
    assert res_b.likelihood_ratio > res_a.likelihood_ratio
    assert res_b.posterior_match_probability > res_a.posterior_match_probability


def test_phase5_harmonization_analytical_covariance_positive():
    """
    HARMONIZATION COVARIANCE & DOMAIN BOUND TEST:
    Verifies that compute_atlas_cross_filter_covariance produces strictly positive analytical covariance
    matching first-order propagation on independent g and r variances, and enforces [-0.4, 2.0] bounds.
    """
    sigma_g = 0.03
    sigma_r = 0.02
    cov = PhotometricHarmonizer.compute_atlas_cross_filter_covariance(sigma_g, sigma_r)
    expected_cov = (0.593 * 0.222) * (sigma_g ** 2) + (0.407 * 0.778) * (sigma_r ** 2)

    assert cov > 0.0
    assert np.isclose(cov, expected_cov, atol=1e-8)

    # Test domain limits: (g - r) must be in [-0.4, 2.0]
    # Inverted fluxes for extreme colors:
    # m = -2.5*log10(F) + 23.9 -> F = 10^((23.9 - m)/2.5)
    # Color = -2.5*log10(Fg / Fr) = 2.5*log10(Fr / Fg)
    # If Fr = 1000, Fg = 1 -> color = 2.5 * 3 = 7.5 mag (out of domain)
    with pytest.raises(ValueError, match="outside empirical polynomial validity domain"):
        PhotometricHarmonizer.ztf_to_atlas_flux(flux_g_ujy=10.0, flux_r_ujy=1000.0, flux_g_err_ujy=1.0, flux_r_err_ujy=10.0)

    # If Fg has SNR < 3.0:
    with pytest.raises(ValueError, match="Insufficient SNR"):
        PhotometricHarmonizer.ztf_to_atlas_flux(flux_g_ujy=2.0, flux_r_ujy=10.0, flux_g_err_ujy=1.0, flux_r_err_ujy=1.0)


def test_phase5_stable_entity_registry_persistence_and_association(tmp_path: Path):
    """
    DURABLE REGISTRY TEST:
    Verifies that StableEntityRegistry can be serialized to and loaded from Parquet,
    preserves indexing across surveys, and supports attaching validated associations.
    """
    reg = StableEntityRegistry()
    ent1 = reg.register_entity(
        reference_ra_deg=200.0,
        reference_dec_deg=10.0,
        iau_name="SN 2021gde",
        ztf_id="ZTF21aaagde",
        atlas_id=None,
    )
    assert ent1.astraeus_id == "ASTRAEUS-EVT-00000001"
    assert ent1.atlas_id is None

    # Attach empirical association post-verification
    reg.attach_survey_association(
        entity_id=ent1.astraeus_id,
        survey="ATLAS",
        target_id="ATLAS_SRC_999001",
        status="SECURELY_ASSOCIATED",
    )
    assert ent1.atlas_id == "ATLAS_SRC_999001"
    assert "ATLAS" in ent1.associated_surveys
    assert ent1.identity_status == "SECURELY_ASSOCIATED"

    # Save to parquet and reload
    pq_path = tmp_path / "test_registry.parquet"
    reg.save_to_parquet(pq_path)
    assert pq_path.exists()

    loaded_reg = StableEntityRegistry.load_from_parquet(pq_path)
    reloaded_ent = loaded_reg.get_by_id(ent1.astraeus_id)
    assert reloaded_ent is not None
    assert reloaded_ent.iau_name == "SN 2021gde"
    assert reloaded_ent.ztf_id == "ZTF21aaagde"
    assert reloaded_ent.atlas_id == "ATLAS_SRC_999001"
    assert "ATLAS" in reloaded_ent.associated_surveys
    assert reloaded_ent.identity_status == "SECURELY_ASSOCIATED"

    # Check ATLAS index
    assert loaded_reg._atlas_index["ATLAS_SRC_999001"] == ent1.astraeus_id


def test_phase5_registry_collision_guards_and_monotonicity():
    """
    REGISTRY COLLISION & MONOTONICITY TEST:
    Verifies forced_id collision rejection, external survey ID collision rejection,
    monotonic sequence preservation, and unsupported survey rejection.
    """
    reg = StableEntityRegistry()
    ent1 = reg.register_entity(reference_ra_deg=100.0, reference_dec_deg=20.0, ztf_id="ZTF_A")
    assert ent1.astraeus_id == "ASTRAEUS-EVT-00000001"

    # 1. Monotonic ID generation
    ent2 = reg.register_entity(reference_ra_deg=105.0, reference_dec_deg=25.0, ztf_id="ZTF_B")
    assert ent2.astraeus_id == "ASTRAEUS-EVT-00000002"

    # 2. forced_id collision guard
    with pytest.raises(ValueError, match="forced_id '.*' already exists in registry"):
        reg.register_entity(reference_ra_deg=110.0, reference_dec_deg=30.0, forced_id="ASTRAEUS-EVT-00000001")

    # 3. Associate external ATLAS ID to ent1
    reg.attach_survey_association(ent1.astraeus_id, "ATLAS", "ATLAS_TARGET_101")
    assert ent1.atlas_id == "ATLAS_TARGET_101"

    # Attempt to associate same ATLAS ID to ent2 must raise collision error
    with pytest.raises(ValueError, match="Identity collision: External survey ID 'ATLAS_TARGET_101'"):
        reg.attach_survey_association(ent2.astraeus_id, "ATLAS", "ATLAS_TARGET_101")

    # 4. Overwrite guard: ent1 cannot be re-associated with a different ATLAS ID without allow_overwrite
    with pytest.raises(ValueError, match="already has atlas_id=.*Cannot overwrite"):
        reg.attach_survey_association(ent1.astraeus_id, "ATLAS", "ATLAS_TARGET_DIFFERENT")

    # 5. Unsupported survey rejection
    with pytest.raises(ValueError, match="Unsupported survey 'HUBBLE'"):
        reg.attach_survey_association(ent1.astraeus_id, "HUBBLE", "HST_001")


def test_phase5_canonical_record_time_coordinate_coupling():
    """
    CANONICAL TIME COORDINATE COUPLING TEST:
    Verifies that get_time_coordinate() returns coupled (timestamp, time_scale)
    and is_barycentric_converted accurately reflects conversion status.
    """
    # 1. Unconverted UTC observation
    rec_utc = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ATLAS_59100_UTC",
        t_native=59100.25,
        native_time_scale="UTC",
        t_converted_bmjd=None,
        ra_deg=120.0,
        dec_deg=30.0,
        survey_id="ATLAS",
        native_passband="ATLAS_c",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=15.0,
        measurement_sigma_ujy=1.5,
        is_positive_detection_snr5=True,
        provenance_hash="a1b2" * 16,
    )
    assert rec_utc.is_barycentric_converted is False
    t_val, t_scale = rec_utc.get_time_coordinate()
    assert t_val == 59100.25
    assert t_scale == "UTC"

    # 2. Converted BMJD_TDB observation
    rec_tdb = CanonicalObservationRecord(
        astraeus_id="ASTRAEUS-EVT-00000001",
        survey_obs_id="ATLAS_59100_TDB",
        t_native=59100.25,
        native_time_scale="UTC",
        t_converted_bmjd=59100.2548,
        converted_time_scale="BMJD_TDB",
        time_conversion_method="ASTROPY_BARYCORR_V1",
        t_obs_bmjd=59100.2548,
        ra_deg=120.0,
        dec_deg=30.0,
        survey_id="ATLAS",
        native_passband="ATLAS_c",
        obs_type=ObservationType.DIFFERENCE_FORCED,
        measured_flux_ujy=15.0,
        measurement_sigma_ujy=1.5,
        is_positive_detection_snr5=True,
        provenance_hash="b2c3" * 16,
    )
    assert rec_tdb.is_barycentric_converted is True
    t_val, t_scale = rec_tdb.get_time_coordinate()
    assert t_val == 59100.2548
    assert t_scale == "BMJD_TDB"


def test_phase5_cross_survey_matcher_details_and_evidence_capture():
    """
    MATCHER PROVENANCE TEST:
    Verifies that CrossSurveyMatcher records complete evaluation provenance:
    astrometric_likelihood_model, search_area, local_density, and candidate posteriors.
    """
    matcher = CrossSurveyMatcher(completeness_prior_q=0.92, max_search_radius_arcsec=4.0)

    cand = SurveyCandidate(
        catalog_name="ATLAS",
        source_id="ATLAS_SRC_001",
        ra_deg=150.00002,
        dec_deg=20.00001,
        ra_err_arcsec=0.06,
        dec_err_arcsec=0.06,
        mag=18.2,
        filter_band="ATLAS_o",
    )

    # Supply empirical local density
    res = matcher.match_source(
        query_id="TEST_PROV",
        query_ra=150.0,
        query_dec=20.0,
        query_pos_err_arcsec=0.05,
        candidates=[cand],
        local_density_sq_arcsec=0.0035,
    )

    det = res.details
    assert det["astrometric_likelihood_model"] == "ANISOTROPIC_DIAGONAL_2D_GAUSSIAN"
    assert det["search_radius_arcsec"] == 4.0
    assert np.isclose(det["search_area_sq_arcsec"], np.pi * 16.0)
    assert det["completeness_prior_q"] == 0.92
    assert det["local_density_sq_arcsec"] == 0.0035
    assert det["density_estimation_source"] == "EMPIRICAL_LOCAL_ESTIMATE"
    assert det["evaluated_candidates_count"] == 1
    assert "all_candidate_posteriors" in det


def test_phase5_atlas_broker_idempotency_manifest_and_vaulting(tmp_path: Path, repo_root: Path):
    """
    ATLAS BROKER, MANIFEST & VAULTING TEST:
    Verifies deterministic request key stability, 10-object pilot manifest validity,
    raw byte vaulting with SHA-256 integrity, and negative difference flux preservation.
    """
    p5_dir = repo_root / "experiments" / "phase5_cross_survey_generalization"

    # 1. Manifest assertions
    pilot_manifest_path = p5_dir / "phase5_atlas_10obj_pilot_manifest.json"
    full_manifest_path = p5_dir / "phase5_atlas_150obj_request_manifest.json"
    ledger_path = p5_dir / "phase5_atlas_request_ledger.parquet"

    assert pilot_manifest_path.exists()
    assert full_manifest_path.exists()
    assert ledger_path.exists()

    with open(pilot_manifest_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    assert pilot_data["candidates_count"] == 10
    assert len(pilot_data["requests"]) == 10

    # Verify all 5 taxonomic classes present in 10-object pilot
    classes = set(r["astraeus_class"] for r in pilot_data["requests"])
    assert classes == {"SN_Ia", "SN_CC", "TDE", "SLSN", "OTHER_EXTRAGALACTIC"}

    # 2. Idempotency key stability
    req0 = pilot_data["requests"][0]
    computed_key = AtlasForcedPhotometryClient.compute_idempotency_key(
        astraeus_id=req0["astraeus_id"],
        ra_deg=req0["ra_deg"],
        dec_deg=req0["dec_deg"],
        mjd_min=req0["mjd_min"],
        mjd_max=req0["mjd_max"],
        use_reduced=req0["use_reduced"],
    )
    assert computed_key == req0["idempotency_key"]

    # 3. Raw payload vaulting and non-destructive negative flux parsing
    sample_atlas_payload = (
        "###MJD           uJy   duJy  F err chi/N   RA        Dec      x       y     maj   min   phi  apfit   mag    dmag  Color   filt\n"
        "59750.12345    25.50   3.20  c 0   1.05   226.9220  70.9844 1024.5  512.2  2.10  1.90  45.0  0.02  20.38   0.14  0.000   c\n"
        "59752.45678   -12.80   4.10  o 0   0.98   226.9221  70.9843 1025.0  513.0  2.15  1.95  44.0  0.01  99.99   9.99  0.000   o\n"
    ).encode("utf-8")

    vault_dir = tmp_path / "test_vault"
    payload_path, sha = AtlasForcedPhotometryClient.vault_payload(
        idempotency_key=computed_key,
        raw_bytes=sample_atlas_payload,
        metadata={"target": "SN2022pcj"},
        vault_root=vault_dir,
    )
    assert payload_path.exists()
    assert sha == hashlib.sha256(sample_atlas_payload).hexdigest()

    # Parse vaulted payload
    records = AtlasForcedPhotometryClient.parse_vaulted_payload(
        payload_path=payload_path,
        astraeus_id=req0["astraeus_id"],
        reference_ra_deg=req0["ra_deg"],
        reference_dec_deg=req0["dec_deg"],
    )
    assert len(records) == 2
    # Verify positive flux
    assert records[0].measured_flux_ujy == 25.50
    assert records[0].native_passband == "ATLAS_c"
    # Crucial invariant: verify negative difference flux survived parsing intact!
    assert records[1].measured_flux_ujy == -12.80
    assert records[1].native_passband == "ATLAS_o"
    assert records[1].obs_type == ObservationType.DIFFERENCE_FORCED


