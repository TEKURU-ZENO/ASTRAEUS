"""
ASTRAEUS Gate 4 Phase 4B: Protocol Architecture, Authoritative Geometry & Photometric Physics Test Suite.

Verifies:
1. Authoritative HEALPix split geometry: 1,808 frozen test pixels at NSIDE=32, NESTED, ICRS.
2. Complete spatial isolation (>60'') of all 35 pre-registered pilot cohort candidates.
3. Physically consistent signed-flux SNR calculation and positive detection discrimination.
4. Stratification integrity, coverage, and disjointness of the pilot cohort specification.
5. End-to-end payload parsing and non-detection preservation.
"""

import hashlib
import json
import math
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import numpy as np
import pandas as pd
import pytest

from experiments.gate4b_broker_acquisition_and_cohort_expansion.astrometry_isolation_audit import (
    SpatialIsolationAuditor,
    EXPECTED_HEALPIX_CONFIG_HASH,
    HEALPIX_CONFIG_PATH,
)
from experiments.gate4b_broker_acquisition_and_cohort_expansion.broker_client import (
    AlerceBrokerClient,
    compute_signed_flux_and_snr,
)

PILOT_SPEC_PATH = Path("experiments/gate4b_broker_acquisition_and_cohort_expansion/pilot_cohort_specification.json")


def test_authoritative_healpix_geometry():
    """Verify authoritative HEALPix configuration and test partition size."""
    assert HEALPIX_CONFIG_PATH.exists(), f"Missing {HEALPIX_CONFIG_PATH}"
    config_hash = hashlib.sha256(HEALPIX_CONFIG_PATH.read_bytes()).hexdigest()
    assert config_hash == EXPECTED_HEALPIX_CONFIG_HASH

    with open(HEALPIX_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["healpix_nside"] == 32
    test_pixels = data["splits"]["test"]
    assert len(test_pixels) == 1808
    assert len(set(test_pixels)) == 1808

    # Verify native astropy_healpix cone search functionality
    hp = HEALPix(nside=32, order="nested", frame="icrs")
    first_test_sc = hp.healpix_to_skycoord(test_pixels[0])
    cone_pixels = hp.cone_search_skycoord(first_test_sc, radius=60.0 * u.arcsec)
    assert any(int(p) in set(test_pixels) for p in cone_pixels)


def test_pilot_cohort_spatial_isolation():
    """Verify that every candidate in the 35-object pilot cohort is strictly >60'' spatially isolated."""
    assert PILOT_SPEC_PATH.exists(), f"Missing {PILOT_SPEC_PATH}"
    with open(PILOT_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    auditor = SpatialIsolationAuditor()
    candidates = spec["candidates"]
    assert len(candidates) == 35

    for cand in candidates:
        res = auditor.audit_single_coordinate(
            ra_deg=cand["ra_deg"], dec_deg=cand["dec_deg"], buffer_arcsec=60.0
        )
        assert res["is_spatially_isolated"] is True, (
            f"Candidate {cand['ztf_id']} violates spatial isolation! Status: {res['spatial_isolation_status']}"
        )
        assert res["is_exact_test_pixel_overlap"] is False
        assert res["is_buffer_test_pixel_overlap"] is False
        assert len(res["intersecting_test_pixels_in_buffer"]) == 0


def test_signed_flux_and_snr_physics():
    """Verify physically consistent signed difference flux and SNR calculation."""
    # 1. High-significance positive difference detection
    res_pos = compute_signed_flux_and_snr(magpsf=18.0, sigmapsf=0.1, isdiffpos=1)
    assert res_pos["flux_nmgy"] > 0
    assert math.isclose(res_pos["signed_snr"], 10.85736, rel_tol=1e-4)
    assert res_pos["is_positive_detection_snr5"] is True

    # 2. Negative difference flux (transient fainter than reference template)
    # Even with very small magnitude uncertainty (sigma_m = 0.05 mag), SNR must be negative!
    res_neg = compute_signed_flux_and_snr(magpsf=18.0, sigmapsf=0.05, isdiffpos=0)
    assert res_neg["flux_nmgy"] < 0
    assert res_neg["signed_snr"] < 0
    assert math.isclose(res_neg["signed_snr"], -21.7147, rel_tol=1e-4)
    # MUST NOT be classified as a positive detection
    assert res_neg["is_positive_detection_snr5"] is False

    # 3. Alternative representation of negative difference flag (-1 or 'f')
    res_neg_alt = compute_signed_flux_and_snr(magpsf=18.0, sigmapsf=0.1, isdiffpos=-1)
    assert res_neg_alt["flux_nmgy"] < 0
    assert res_neg_alt["signed_snr"] < 0
    assert res_neg_alt["is_positive_detection_snr5"] is False

    # 4. Low-significance positive measurement (sigma_m = 0.5 mag -> SNR ~ 2.17)
    res_low = compute_signed_flux_and_snr(magpsf=18.0, sigmapsf=0.5, isdiffpos=1)
    assert res_low["flux_nmgy"] > 0
    assert math.isclose(res_low["signed_snr"], 2.17147, rel_tol=1e-4)
    # Fails SNR >= 5.0 criterion
    assert res_low["is_positive_detection_snr5"] is False

    # 5. Boundary condition: sigma_m = 0.217147 mag -> SNR == 5.0
    res_bound = compute_signed_flux_and_snr(magpsf=18.0, sigmapsf=0.2171472, isdiffpos=1)
    assert math.isclose(res_bound["signed_snr"], 5.0, rel_tol=1e-4)
    assert res_bound["is_positive_detection_snr5"] is True


def test_pilot_strata_breakdown_and_disjointness():
    """Verify pilot cohort composition, stratification, and metadata integrity."""
    with open(PILOT_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    assert spec["total_candidates"] == 35
    strata = spec["strata_breakdown"]
    assert strata["REPRODUCIBILITY_FEATURE_EVALUABLE"] == 5
    assert strata["KNOWN_SINGLE_BAND_DISCRIMINATION"] == 3
    assert strata["BTS_DISJOINT_RARE_CLASS"] == 4
    assert strata["REPRESENTATIVE_SN_IA_PENDING"] == 11
    assert strata["REPRESENTATIVE_SN_CC_PENDING"] == 12

    candidates = spec["candidates"]
    ztf_ids = [c["ztf_id"] for c in candidates]
    assert len(ztf_ids) == 35
    assert len(set(ztf_ids)) == 35

    # Check rare class representations
    rare_classes = {c["astraeus_class"] for c in candidates if c["pilot_stratum"] == "BTS_DISJOINT_RARE_CLASS"}
    assert "TDE" in rare_classes
    assert "SLSN" in rare_classes
    assert "OTHER_EXTRAGALACTIC" in rare_classes

    # Check single-band candidate identities
    single_ids = [c["ztf_id"] for c in candidates if c["pilot_stratum"] == "KNOWN_SINGLE_BAND_DISCRIMINATION"]
    assert sorted(single_ids) == ["ZTF18abzpbpi", "ZTF19aalfsqm", "ZTF23aaihsvm"]


def test_alerce_payload_parser():
    """Verify raw ALeRCE payload parsing, non-detection preservation, and evaluability categorization."""
    client = AlerceBrokerClient()

    # Mock payload with 3 positive g detections (SNR >= 5), 3 positive r detections (SNR >= 5),
    # 2 negative difference detections, and 4 non-detections (upper limits)
    mock_payload = {
        "detections": [
            {"mjd": 58300.1, "fid": 1, "magpsf": 18.0, "sigmapsf": 0.1, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            {"mjd": 58302.1, "fid": 1, "magpsf": 18.2, "sigmapsf": 0.1, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            {"mjd": 58304.1, "fid": 1, "magpsf": 18.5, "sigmapsf": 0.1, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            {"mjd": 58301.1, "fid": 2, "magpsf": 17.5, "sigmapsf": 0.08, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            {"mjd": 58303.1, "fid": 2, "magpsf": 17.8, "sigmapsf": 0.09, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            {"mjd": 58305.1, "fid": 2, "magpsf": 18.0, "sigmapsf": 0.12, "isdiffpos": 1, "ra": 10.0, "dec": 20.0},
            # Negative difference fluxes:
            {"mjd": 58290.0, "fid": 1, "magpsf": 19.5, "sigmapsf": 0.15, "isdiffpos": 0, "ra": 10.0, "dec": 20.0},
            {"mjd": 58291.0, "fid": 2, "magpsf": 19.8, "sigmapsf": 0.18, "isdiffpos": -1, "ra": 10.0, "dec": 20.0},
        ],
        "non_detections": [
            {"mjd": 58280.0, "fid": 1, "diffmaglim": 20.5},
            {"mjd": 58282.0, "fid": 2, "diffmaglim": 20.3},
            {"mjd": 58320.0, "fid": 1, "diffmaglim": 20.2},
            {"mjd": 58322.0, "fid": 2, "diffmaglim": 20.0},
        ],
    }

    parsed = client.parse_raw_lightcurve_payload(mock_payload)
    assert parsed["total_raw_detections"] == 8
    assert parsed["total_raw_nondetections"] == 4
    assert parsed["negative_detections_count"] == 2
    assert parsed["n_det_snr5_g"] == 3
    assert parsed["n_det_snr5_r"] == 3
    assert parsed["n_det_snr5_total"] == 6
    assert parsed["has_dual_band_snr5"] is True
    assert parsed["derived_feature_status"] == "FEATURE_EVALUABLE"

    # Verify that negative detections are preserved with negative flux and negative SNR
    neg_dets = [d for d in parsed["detections"] if d["isdiffpos"] in (0, -1)]
    assert len(neg_dets) == 2
    for nd in neg_dets:
        assert nd["flux_nmgy"] < 0
        assert nd["signed_snr"] < 0
        assert nd["is_positive_detection_snr5"] is False
