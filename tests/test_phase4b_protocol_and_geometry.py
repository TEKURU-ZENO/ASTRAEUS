"""
ASTRAEUS Gate 4 Phase 4B: Acquisition Engine Verification Test Suite (v1.3).

Verifies:
1. Authoritative HEALPix split geometry: 1,808 frozen test pixels at NSIDE=32, NESTED, ICRS.
2. Complete spatial isolation (>60'') of all 35 pre-registered pilot cohort candidates.
3. Physical flux calibration: canonical nanomaggies (Z0=22.500) vs microjanskys (Z0=23.900).
4. Fail-closed difference flux sign parsing on missing, malformed, or unknown tokens (Blocker C).
5. Process-safe & thread-safe rate limiter concurrency.
6. Full end-to-end mocked HTTP acquisition lifecycle (HTTP 200, 404 not found).
7. Non-overwriting idempotency: failed retries NEVER overwrite existing raw vault artifacts.
8. Reconciled pilot cohort distributions (SN_Ia z and mag ranges; SN_CC 12 subtypes).
9. Windows multi-process rate ceiling obedience without contention deadlocks (Blocker A).
10. Pre-commit JSON validation preventing malformed or invalid schema archival (Blocker B).
11. Explicit HTTP 429 and 5xx bounded exponential backoff and retry recovery.
12. Downstream feature extractor fail-closed filtering of NaN/invalid photometry (Blocker C).
13. Forced photometry dedicated lifecycle and strict vault separation.
14. Mathematical provenance and exact specification of conformal quantile index k=507.
"""

import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time
from unittest.mock import MagicMock

# Ensure repository root is on sys.path for clean multi-platform test execution
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import numpy as np
import pandas as pd
import pytest

from astraeus_data.real_survey.observation_feature_pipeline import (
    CanonicalObservation,
    ObservationConfig,
    RealSurveyFeatureExtractor,
)
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
    _multiprocess_rate_limiter_worker,
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

    # Unknown / Malformed tokens: MUST FAIL CLOSED (Blocker C)
    for bad_token in (None, "", "unknown", "ambiguous", "null", float("nan"), 99, -99, 2, "2"):
        sgn, is_valid = parse_difference_flux_sign(bad_token)
        assert is_valid is False
        assert sgn is None
        c = compute_physical_flux_and_snr(18.0, 0.1, bad_token)
        assert c["is_valid_sign"] is False
        assert c["is_valid_measurement"] is False
        assert c["is_positive_detection_snr5"] is False
        assert math.isnan(c["signed_snr"])
        assert math.isnan(c["flux_nmgy"])
        assert math.isnan(c["flux_ujy"])


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
    """Verify full HTTP request, atomic archival, idempotency, and 404 handling."""
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


def test_windows_multiprocess_rate_limiter(tmp_path):
    """
    CRITICAL BLOCKER A TEST:
    Verify cross-process rate ceiling obedience on Windows using independent OS processes.
    Asserts that 3 separate child processes contending on the same byte 0 lock file
    strictly obey the shared rate limit (10 req/s -> min 0.10s interval) without deadlocks.
    """
    lock_file = tmp_path / "multiprocess_rate.lock"
    max_rate = 10.0  # 10 req/s -> 0.10s interval
    reqs_per_proc = 2
    n_procs = 3
    total_reqs = reqs_per_proc * n_procs

    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    procs = [
        ctx.Process(
            target=_multiprocess_rate_limiter_worker,
            args=(str(lock_file), max_rate, reqs_per_proc, queue),
        )
        for _ in range(n_procs)
    ]

    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=15.0)
    t1 = time.time()

    results = []
    while not queue.empty():
        results.append(queue.get())

    assert len(results) == total_reqs, f"Expected {total_reqs} results, got {len(results)}"
    total_duration = t1 - t0
    # Minimum total time for 6 requests at 10 req/s with jitter must be >= 0.45s
    assert total_duration >= 0.45, f"Multi-process rate limiter permitted requests too quickly: {total_duration:.3f}s"


def test_malformed_json_validation_before_commit(tmp_path):
    """
    CRITICAL BLOCKER B TEST:
    Verify that malformed or non-dictionary JSON returned with HTTP 200 is validated
    BEFORE committing to the permanent vault, preventing corrupted vault artifacts.
    """
    vault_dir = tmp_path / "vault_malformed"
    client = AlerceBrokerClient(raw_vault_dir=vault_dir, max_rate=100.0, max_retries=2)

    mock_session = MagicMock()

    # 1. Broken / Truncated JSON payload with HTTP 200
    mock_resp_broken = MagicMock()
    mock_resp_broken.status_code = 200
    mock_resp_broken.text = '{"detections": [{"mjd": 58000.0, "magpsf": 18.0,'  # Syntax error!
    mock_session.get.return_value = mock_resp_broken

    bad_id = "ZTF18brokenJson"
    bad_vault_path = client.get_lightcurve_path(bad_id)

    res_broken = client.fetch_and_archive_lightcurve(bad_id, session=mock_session)
    assert res_broken["status"] == "MALFORMED_JSON_PAYLOAD"
    assert res_broken["vault_path"] is None
    assert not bad_vault_path.exists(), f"Corrupted file was written to vault: {bad_vault_path}"

    # 2. Valid JSON syntax but invalid schema (non-dict root e.g. list)
    mock_resp_nondict = MagicMock()
    mock_resp_nondict.status_code = 200
    mock_resp_nondict.text = json.dumps(["unexpected", "list", "root"])
    mock_session.get.return_value = mock_resp_nondict

    bad_id_2 = "ZTF18nonDictJson"
    bad_vault_path_2 = client.get_lightcurve_path(bad_id_2)

    res_nondict = client.fetch_and_archive_lightcurve(bad_id_2, session=mock_session)
    assert res_nondict["status"] == "INVALID_JSON_SCHEMA"
    assert res_nondict["vault_path"] is None
    assert not bad_vault_path_2.exists(), f"Invalid schema file was written to vault: {bad_vault_path_2}"


def test_explicit_http_429_and_5xx_retries(tmp_path):
    """
    Verify bounded exponential backoff and retry recovery across HTTP 429 and 5xx errors:
    1st call -> HTTP 429 (Rate Limit Exceeded)
    2nd call -> HTTP 503 (Service Unavailable)
    3rd call -> HTTP 200 (OK with valid light curve)
    """
    vault_dir = tmp_path / "vault_retries"
    client = AlerceBrokerClient(raw_vault_dir=vault_dir, max_rate=50.0, max_retries=5)

    mock_session = MagicMock()
    mock_resp_429 = MagicMock()
    mock_resp_429.status_code = 429

    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503

    valid_payload = {
        "detections": [{"mjd": 58100.0, "fid": 1, "magpsf": 18.2, "sigmapsf": 0.05, "isdiffpos": 1}],
        "non_detections": [],
    }
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = json.dumps(valid_payload)

    mock_session.get.side_effect = [mock_resp_429, mock_resp_503, mock_resp_200]

    res = client.fetch_and_archive_lightcurve("ZTF18retrySuccess", session=mock_session)
    assert res["status"] == "SUCCESS_DOWNLOADED"
    assert res["http_status"] == 200
    assert res["retries_performed"] == 2
    assert Path(res["vault_path"]).exists()
    assert json.loads(Path(res["vault_path"]).read_text(encoding="utf-8")) == valid_payload
    assert mock_session.get.call_count == 3


def test_invalid_photometry_downstream_handling():
    """
    CRITICAL BLOCKER C TEST:
    Verifies that downstream feature extraction cleanly filters out invalid/NaN photometry measurements
    and does NOT silently consume invalid values as physical measurements or cause numerical corruption.
    """
    extractor = RealSurveyFeatureExtractor(config=ObservationConfig.ALERCE_ONLY)

    # 1. Mix of valid detections, non-detections, and invalid (NaN) measurements
    obs = [
        # Valid detection 1 (g-band, positive)
        CanonicalObservation(
            mjd=58300.0, passband="g", flux_nmgy=25.0, flux_err_nmgy=1.2,
            is_upper_limit=False, source_survey="ALERCE"
        ),
        # Invalid detection (Blocker C: NaN flux produced by compute_physical_flux_and_snr on bad sign)
        CanonicalObservation(
            mjd=58301.0, passband="g", flux_nmgy=float("nan"), flux_err_nmgy=0.1,
            is_upper_limit=False, source_survey="ALERCE"
        ),
        # Valid detection 2 (r-band, positive)
        CanonicalObservation(
            mjd=58302.0, passband="r", flux_nmgy=30.0, flux_err_nmgy=1.5,
            is_upper_limit=False, source_survey="ALERCE"
        ),
        # Valid upper limit / non-detection
        CanonicalObservation(
            mjd=58303.0, passband="r", flux_nmgy=0.0, flux_err_nmgy=0.5,
            is_upper_limit=True, limiting_flux_nmgy=5.0, source_survey="ALERCE"
        ),
        # Valid negative difference detection (fading transient)
        CanonicalObservation(
            mjd=58304.0, passband="r", flux_nmgy=-2.5, flux_err_nmgy=0.8,
            is_upper_limit=False, is_negative_flux=True, source_survey="ALERCE"
        ),
    ]

    feats = extractor.extract_from_observations(obs)

    # The 3 valid detections are preserved (NaN detection is cleanly filtered out)
    assert feats["n_detections"] == 3.0
    assert feats["n_upper_limits"] == 1.0
    assert feats["peak_flux_nmgy"] == 30.0
    assert feats["min_flux_nmgy"] == -2.5  # Negative flux preserved!
    assert math.isclose(feats["mean_flux_nmgy"], (25.0 + 30.0 - 2.5) / 3.0, rel_tol=1e-5)

    # All 32 extracted features MUST be strictly finite floats
    for k, v in feats.items():
        assert math.isfinite(v), f"Feature {k} was corrupted with non-finite value: {v}"

    # 2. Degenerate case: ALL detections have NaN flux
    all_bad_obs = [
        CanonicalObservation(
            mjd=58300.0, passband="g", flux_nmgy=float("nan"), flux_err_nmgy=0.1,
            is_upper_limit=False, source_survey="ALERCE"
        ),
    ]
    degenerate_feats = extractor.extract_from_observations(all_bad_obs)
    assert degenerate_feats["n_detections"] == 0.0
    assert degenerate_feats["peak_flux_nmgy"] == 0.0
    for k, v in degenerate_feats.items():
        assert math.isfinite(v), f"Degenerate feature {k} is non-finite: {v}"


def test_forced_photometry_archival_and_separation(tmp_path):
    """
    Verify forced photometry lifecycle and strict isolation from alert light curves:
    1. Archives strictly to forced_vault_dir ({ztf_id}_forced.json).
    2. Leaves difference light curve vault path untouched.
    3. Validates JSON before commit.
    4. Handles 404 cleanly.
    5. Re-requesting returns FORCED_ALREADY_ARCHIVED without HTTP call.
    """
    raw_dir = tmp_path / "raw_vault"
    forced_dir = tmp_path / "forced_vault"
    client = AlerceBrokerClient(raw_vault_dir=raw_dir, forced_vault_dir=forced_dir, max_rate=100.0)

    ztf_id = "ZTF18forcedObj"
    lc_path = client.get_lightcurve_path(ztf_id)
    forced_path = client.get_forced_path(ztf_id)

    mock_payload = {"forced_photometry": [{"mjd": 58500.0, "forced_flux": 12.5, "forced_err": 1.2}]}
    mock_session = MagicMock()
    mock_resp_200 = MagicMock()
    mock_resp_200.status_code = 200
    mock_resp_200.text = json.dumps(mock_payload)
    mock_session.get.return_value = mock_resp_200

    # 1. Successful download and archival
    res = client.fetch_and_archive_forced_photometry(ztf_id, session=mock_session)
    assert res["status"] == "FORCED_SUCCESS_DOWNLOADED"
    assert res["vault_path"] == str(forced_path)
    assert forced_path.exists()
    assert not lc_path.exists(), "Forced photometry must NEVER write to difference light curve vault!"
    assert json.loads(forced_path.read_text(encoding="utf-8")) == mock_payload

    # 2. Idempotent retrieval
    mock_session.get.reset_mock()
    res_idem = client.fetch_and_archive_forced_photometry(ztf_id, session=mock_session)
    assert res_idem["status"] == "FORCED_ALREADY_ARCHIVED"
    mock_session.get.assert_not_called()

    # 3. Malformed JSON validation before commit
    mock_resp_bad = MagicMock()
    mock_resp_bad.status_code = 200
    mock_resp_bad.text = "{ malformed forced json ... "
    mock_session.get.return_value = mock_resp_bad

    bad_id = "ZTF18badForced"
    bad_forced_path = client.get_forced_path(bad_id)
    res_bad = client.fetch_and_archive_forced_photometry(bad_id, session=mock_session)
    assert res_bad["status"] == "FORCED_MALFORMED_JSON"
    assert res_bad["vault_path"] is None
    assert not bad_forced_path.exists()

    # 4. HTTP 404 handling
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_session.get.return_value = mock_resp_404

    notfound_id = "ZTF18notfoundForced"
    res_404 = client.fetch_and_archive_forced_photometry(notfound_id, session=mock_session)
    assert res_404["status"] == "FORCED_NOT_FOUND_404"
    assert res_404["vault_path"] is None


def test_conformal_k507_specification_and_provenance():
    """
    Verifies the exact statistical specification and mathematical provenance of the
    conformal quantile index k=507 required for Gate 4C sequence modeling calibration:
    1. Calibration population size: N_calib = 562 (from the 80/20 train/calibration split of the N=2,808 development cohort).
    2. Target coverage: 1 - alpha = 0.90 (significance alpha = 0.10).
    3. Exact finite-sample quantile formula:
       k = ceil((N_calib + 1) * (1 - alpha)) = ceil(563 * 0.90) = ceil(506.7) = 507.
    4. 0-indexed order statistic array index: k - 1 = 506.
    5. Conservative tie-handling rule: np.sort(scores)[k - 1], which guarantees
       empirical coverage P(Y in C(X)) >= 1 - alpha under exchangeability.
    """
    n_dev_total = 2808
    n_train_proper = 2246
    n_calib = 562
    assert n_train_proper + n_calib == n_dev_total

    alpha = 0.10
    target_coverage = 1.0 - alpha

    # Exact finite-sample quantile formula
    k_float = (n_calib + 1) * target_coverage  # 563 * 0.90 = 506.7
    k_ceil = math.ceil(k_float)
    assert k_ceil == 507

    # 0-indexed rank index
    rank_0_indexed = k_ceil - 1
    assert rank_0_indexed == 506

    # Verify order statistic evaluation on synthetic conformity scores
    np.random.seed(42)
    synthetic_scores = np.random.uniform(0.0, 1.0, size=n_calib)
    sorted_scores = np.sort(synthetic_scores)
    q_hat = sorted_scores[rank_0_indexed]

    # Verify coverage guarantee on calibration set: fraction of scores <= q_hat
    empirical_coverage = np.mean(synthetic_scores <= q_hat)
    assert empirical_coverage >= target_coverage
    assert math.isclose(empirical_coverage, 507 / 562, rel_tol=1e-5)


def test_unknown_passband_and_malformed_photometry_quarantine():
    """
    INGESTION-LEVEL SAFEGUARD TEST (Reviewer Qualification 3):
    Verifies that unexpected malformed values (string mag/sig) and unknown filter IDs
    are safely quarantined at ingestion without raising unhandled exceptions or misclassifying.
    """
    client = AlerceBrokerClient()
    raw_payload = {
        "detections": [
            {"mjd": 58000.0, "fid": 99, "magpsf": 18.0, "sigmapsf": 0.1, "isdiffpos": 1},  # Unknown filter 99
            {"mjd": 58001.0, "fid": 1, "magpsf": "corrupt_str", "sigmapsf": 0.1, "isdiffpos": 1},  # Malformed mag string
            {"mjd": 58002.0, "fid": 2, "magpsf": 17.5, "sigmapsf": "bad_sig", "isdiffpos": 1},  # Malformed sig string
            {"mjd": 58003.0, "fid": 2, "magpsf": 17.5, "sigmapsf": 0.08, "isdiffpos": 1},  # Valid detection
        ],
        "non_detections": [
            {"mjd": 58290.0, "fid": "invalid_fid", "diffmaglim": 20.5},
            {"mjd": 58291.0, "fid": 1, "diffmaglim": "corrupt_lim"},
        ],
    }
    parsed = client.parse_raw_lightcurve_payload(raw_payload)
    assert parsed["unknown_passband_count"] == 1
    assert parsed["malformed_photometry_count"] == 2
    assert parsed["detections"][0]["is_valid_passband"] is False
    assert parsed["detections"][0]["is_valid_measurement"] is False
    assert parsed["detections"][0]["is_positive_detection_snr5"] is False
    assert parsed["detections"][1]["is_valid_measurement"] is False
    assert parsed["detections"][2]["is_valid_measurement"] is False
    assert parsed["detections"][3]["is_valid_measurement"] is True
    assert parsed["non_detections"][0]["is_valid_passband"] is False
    assert math.isnan(parsed["non_detections"][1]["diffmaglim"])


def test_phase4b1_pilot_acquisition_deliverables_and_integrity():
    """
    Verifies the complete deliverable suite for Phase 4B.1 Pilot Acquisition:
    1. Acquisition ledger exists (CSV and Parquet) with exactly 35 candidates, all HTTP 200.
    2. Integrity manifest confirms valid JSON and zero duplicates.
    3. Photometric audit captures detections, non-detections, and passbands.
    4. 5/5 designated local reference objects achieve identical feature reproducibility.
    5. 4/4 rare transients are certified as ingestion-only in Population 3 with zero inference.
    6. Operational report demonstrates 0 rate ceiling violations across all requests.
    """
    base_dir = Path("experiments/gate4b_broker_acquisition_and_cohort_expansion")
    ledger_csv = base_dir / "phase4b1_pilot_acquisition_ledger.csv"
    ledger_parquet = base_dir / "phase4b1_pilot_acquisition_ledger.parquet"
    integrity_manifest = base_dir / "phase4b1_pilot_integrity_manifest.json"
    photo_csv = base_dir / "phase4b1_photometric_audit.csv"
    feat_json = base_dir / "phase4b1_feature_reproducibility_audit.json"
    rare_json = base_dir / "phase4b1_rare_event_isolation_audit.json"
    op_json = base_dir / "phase4b1_operational_report.json"

    assert ledger_csv.exists()
    assert ledger_parquet.exists()
    assert integrity_manifest.exists()
    assert photo_csv.exists()
    assert feat_json.exists()
    assert rare_json.exists()
    assert op_json.exists()

    df_ledger = pd.read_csv(ledger_csv)
    assert len(df_ledger) == 35
    assert set(df_ledger["http_status_code"]) == {200}
    assert all(df_ledger["file_size_bytes"] > 0)
    assert all(df_ledger["sha256_checksum"].str.len() == 64)

    with open(integrity_manifest, "r", encoding="utf-8") as f:
        iman = json.load(f)
    assert iman["total_candidates_requested"] == 35
    assert iman["successful_retrievals"] == 35
    assert iman["failed_retrievals"] == 0
    assert iman["duplicate_candidate_count"] == 0

    with open(feat_json, "r", encoding="utf-8") as f:
        fdat = json.load(f)
    repro = fdat["reproducibility_audit_5_objects"]
    assert len(repro) == 5
    for zid, comp in repro.items():
        assert comp["is_exact_feature_match"] is True

    with open(rare_json, "r", encoding="utf-8") as f:
        rdat = json.load(f)
    assert len(rdat) == 4
    for zid, r in rdat.items():
        assert r["model_inference_executed"] is False
        assert r["prediction_scored"] is False
        assert r["threshold_adjusted"] is False
        assert r["governance_status"] == "STRICTLY_ISOLATED_INGESTION_ONLY"

    with open(op_json, "r", encoding="utf-8") as f:
        odat = json.load(f)
    assert odat["rate_ceiling_violations"] == 0
    assert odat["total_requests"] == 35
