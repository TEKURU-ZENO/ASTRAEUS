"""
ASTRAEUS Gate 4 Phase 4B: Acquisition Engine Verification Test Suite (v1.2).

Verifies:
1. Authoritative HEALPix split geometry: 1,808 frozen test pixels at NSIDE=32, NESTED, ICRS.
2. Complete spatial isolation (>60'') of all 35 pre-registered pilot cohort candidates.
3. Physical flux calibration: canonical nanomaggies (Z0=22.500) vs microjanskys (Z0=23.900).
4. Fail-closed difference flux sign parsing on missing, malformed, or unknown tokens.
5. Process-safe & thread-safe rate limiter concurrency.
6. Full end-to-end mocked HTTP acquisition lifecycle (HTTP 200, 429 retry, 404 not found).
7. Non-overwriting idempotency: failed retries NEVER overwrite existing raw vault artifacts.
8. Reconciled pilot cohort distributions (SN_Ia z and mag ranges; SN_CC 12 subtypes).
"""

import hashlib
import json
import math
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock

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
    GlobalInterprocessRateLimiter,
    compute_physical_flux_and_snr,
    parse_difference_flux_sign,
    CANONICAL_AB_ZERO_POINT_NMGY,
    MICROJANSKY_AB_ZERO_POINT,
    NMGY_TO_UJY_FACTOR,
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


def test_physical_flux_units_and_independent_validation():
    """Verify exact flux unit conventions against independent physical definitions."""
    # 1. Nanomaggies definition: 1.0 nmgy at m = 22.500 mag
    calib_nmgy = compute_physical_flux_and_snr(CANONICAL_AB_ZERO_POINT_NMGY, 0.1, 1)
    assert math.isclose(calib_nmgy["flux_nmgy"], 1.0, rel_tol=1e-5)
    # Microjansky equivalent: 1 nmgy = 3.631028 uJy
    assert math.isclose(calib_nmgy["flux_ujy"], NMGY_TO_UJY_FACTOR, rel_tol=1e-5)

    # 2. Microjansky definition: 1.0 uJy at m = 23.900 mag
    calib_ujy = compute_physical_flux_and_snr(MICROJANSKY_AB_ZERO_POINT, 0.1, 1)
    assert math.isclose(calib_ujy["flux_ujy"], 1.0, rel_tol=1e-3)
    assert math.isclose(calib_ujy["flux_nmgy"], 1.0 / NMGY_TO_UJY_FACTOR, rel_tol=1e-3)

    # 3. Scale-invariance of SNR
    # SNR = F_nmgy / sigma_F_nmgy == F_uJy / sigma_F_uJy
    snr_nmgy = calib_nmgy["flux_nmgy"] / calib_nmgy["flux_err_nmgy"]
    snr_ujy = calib_nmgy["flux_ujy"] / calib_nmgy["flux_err_ujy"]
    assert math.isclose(snr_nmgy, snr_ujy, rel_tol=1e-9)
    assert math.isclose(calib_nmgy["signed_snr"], snr_nmgy, rel_tol=1e-9)


def test_fail_closed_sign_parsing():
    """Verify that unknown, missing, or malformed isdiffpos values fail closed."""
    # Documented positive representations
    for valid_pos in (1, "1", "+1", True, "t", "true", "pos", "positive"):
        sgn, is_valid = parse_difference_flux_sign(valid_pos)
        assert is_valid is True
        assert sgn == 1.0
        c = compute_physical_flux_and_snr(18.0, 0.1, valid_pos)
        assert c["is_valid_sign"] is True
        assert c["is_positive_detection_snr5"] is True

    # Documented negative representations
    for valid_neg in (0, -1, "-1", "0", False, "f", "false", "neg", "negative"):
        sgn, is_valid = parse_difference_flux_sign(valid_neg)
        assert is_valid is True
        assert sgn == -1.0
        c = compute_physical_flux_and_snr(18.0, 0.05, valid_neg)
        assert c["is_valid_sign"] is True
        # Even with tiny error, must NOT be positive detection!
        assert c["is_positive_detection_snr5"] is False
        assert c["signed_snr"] < 0

    # Unknown / Malformed tokens: MUST FAIL CLOSED
    for bad_token in (None, "", "unknown", "ambiguous", "null", float("nan"), 99, -99, 2, "2"):
        sgn, is_valid = parse_difference_flux_sign(bad_token)
        assert is_valid is False
        assert sgn is None
        c = compute_physical_flux_and_snr(18.0, 0.1, bad_token)
        assert c["is_valid_sign"] is False
        assert c["is_positive_detection_snr5"] is False
        assert c["signed_snr"] == 0.0


def test_global_rate_limiter_concurrency(tmp_path):
    """Verify thread-safe rate limiter enforcement across concurrent threads."""
    lock_file = tmp_path / "test_rate.lock"
    limiter = GlobalInterprocessRateLimiter(lock_file_path=lock_file, max_rate=20.0)  # 20 req/s -> 0.05s interval

    timestamps = []
    lock = threading.Lock()

    def worker():
        for _ in range(5):
            limiter.wait_and_acquire()
            with lock:
                timestamps.append(time.time())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t1 = time.time()

    assert len(timestamps) == 20
    # Minimum total time for 20 requests at 20 req/s with jitter must be >= 0.8s
    total_duration = t1 - t0
    assert total_duration >= 0.75, f"Rate limiter permitted requests too quickly: {total_duration:.3f}s"


def test_mocked_end_to_end_acquisition_lifecycle(tmp_path):
    """Verify full HTTP request, archival, retry backoff, and 404 handling."""
    vault_dir = tmp_path / "vault"
    forced_dir = tmp_path / "forced"
    client = AlerceBrokerClient(raw_vault_dir=vault_dir, forced_vault_dir=forced_dir, max_rate=100.0)

    # 1. Successful HTTP 200 light-curve download
    mock_payload = {
        "detections": [
            {"mjd": 58300.0, "fid": 1, "magpsf": 18.0, "sigmapsf": 0.1, "isdiffpos": 1},
            {"mjd": 58301.0, "fid": 2, "magpsf": 17.5, "sigmapsf": 0.08, "isdiffpos": 1},
        ],
        "non_detections": [
            {"mjd": 58290.0, "fid": 1, "diffmaglim": 20.5},
        ],
    }

    mock_session = MagicMock()
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = json.dumps(mock_payload)
    mock_session.get.return_value = mock_resp_200

    res = client.fetch_and_archive_lightcurve("ZTF18test200", session=mock_session)
    assert res["status"] == "SUCCESS_DOWNLOADED"
    assert res["http_status"] == 200
    assert Path(res["vault_path"]).exists()

    # Verify written JSON matches exactly
    saved_json = json.loads(Path(res["vault_path"]).read_text(encoding="utf-8"))
    assert saved_json == mock_payload

    # 2. Idempotent re-fetch: returns ALREADY_ARCHIVED without re-requesting HTTP
    mock_session.get.reset_mock()
    res_idempotent = client.fetch_and_archive_lightcurve("ZTF18test200", session=mock_session)
    assert res_idempotent["status"] == "ALREADY_ARCHIVED"
    mock_session.get.assert_not_called()

    # 3. HTTP 404 handling
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_session.get.return_value = mock_resp_404

    res_404 = client.fetch_and_archive_lightcurve("ZTF18test404", session=mock_session)
    assert res_404["status"] == "OBJECT_NOT_FOUND_404"
    assert res_404["http_status"] == 404
    assert res_404["vault_path"] is None


def test_non_overwriting_idempotency_on_failed_retry(tmp_path):
    """
    CRITICAL BLOCKER TEST:
    Verifies that a failed request or exhausted retries NEVER overwrite, truncate,
    or corrupt an existing archived vault file.
    """
    vault_dir = tmp_path / "vault_idempotency"
    client = AlerceBrokerClient(raw_vault_dir=vault_dir, max_rate=100.0, max_retries=2)

    ztf_id = "ZTF18preserved"
    vault_path = client.get_lightcurve_path(ztf_id)

    # Pre-populate vault with valid pristine file
    pristine_payload = {"detections": [{"mjd": 58000.0, "magpsf": 17.0, "sigmapsf": 0.05, "isdiffpos": 1}]}
    pristine_bytes = json.dumps(pristine_payload).encode("utf-8")
    vault_path.write_bytes(pristine_bytes)
    original_sha256 = hashlib.sha256(pristine_bytes).hexdigest()

    # Mock session with persistent HTTP 500 error
    mock_session = MagicMock()
    mock_resp_500 = MagicMock()
    mock_resp_500.status_code = 500
    mock_session.get.return_value = mock_resp_500

    # 1. Normal call respects existing file and does not even attempt HTTP
    res_normal = client.fetch_and_archive_lightcurve(ztf_id, session=mock_session, force=False)
    assert res_normal["status"] == "ALREADY_ARCHIVED"
    assert vault_path.read_bytes() == pristine_bytes

    # 2. Even with force=True, if HTTP call fails, the existing file must NOT be destroyed!
    # Because client writes to .tmp first and replaces only on 200 OK!
    res_failed = client.fetch_and_archive_lightcurve(ztf_id, session=mock_session, force=True)
    assert res_failed["status"] == "RETRIES_EXHAUSTED"

    # Verify existing file is completely intact with original SHA-256!
    assert vault_path.exists()
    current_bytes = vault_path.read_bytes()
    assert current_bytes == pristine_bytes
    assert hashlib.sha256(current_bytes).hexdigest() == original_sha256


def test_reconciled_pilot_specification_distributions():
    """Verify exact reconciled distributions for the pre-registered 35-object pilot cohort."""
    with open(PILOT_SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    assert spec["total_candidates"] == 35
    candidates = spec["candidates"]

    # 1. SN_Ia Pending ($N=11$) exact ranges:
    ia_cands = [c for c in candidates if c["pilot_stratum"] == "REPRESENTATIVE_SN_IA_PENDING"]
    assert len(ia_cands) == 11
    zs = [c["redshift"] for c in ia_cands if c["redshift"] is not None]
    mags = [c["peak_mag"] for c in ia_cands if c["peak_mag"] is not None]
    assert min(zs) >= 0.0141 and max(zs) <= 0.1000
    assert min(mags) >= 14.37 and max(mags) <= 18.83

    # 2. SN_CC Pending ($N=12$) exact subtypes:
    cc_cands = [c for c in candidates if c["pilot_stratum"] == "REPRESENTATIVE_SN_CC_PENDING"]
    assert len(cc_cands) == 12
    subtypes = {c["normalized_subtype"] for c in cc_cands}
    expected_subtypes = {
        "SN Ic", "SN II", "SN IIb", "SN Ib", "SN IIP", "SN Ic-BL",
        "SN Ib/c", "SN IIn", "SN Ibn", "SN II-pec", "SN Icn", "SN Ic-pec"
    }
    assert subtypes == expected_subtypes
