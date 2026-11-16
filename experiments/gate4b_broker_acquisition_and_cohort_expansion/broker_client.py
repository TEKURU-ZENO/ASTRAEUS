"""
ASTRAEUS Gate 4 Phase 4B: Production ALeRCE Broker API Client & Photometric Calibration Engine.

Milestone: Phase 4B.0 — Acquisition Engine Verification
Author: Antigravity (Google DeepMind) in coordination with Scientific Review
Date: November 16, 2026
Version: 1.2 (Addressing Blocker Findings: Flux Units, Fail-Closed Sign Parsing,
             Global Inter-Process Rate Limiting, and End-to-End Vault Lifecycle)

Physical Specifications:
1. Canonical Flux Units: Nanomaggies (nmgy), Z0 = 22.500 mag.
   F_nmgy = sgn(isdiffpos) * 10^(-0.4 * (m_AB - 22.500))
   sigma_F_nmgy = |F_nmgy| * sinh((ln 10 / 2.5) * sigma_m)
2. Microjansky Scaling: Microjanskys (uJy), Z0 = 23.900 mag.
   1 nmgy = 3.631028 uJy
   F_uJy = sgn(isdiffpos) * 10^(-0.4 * (m_AB - 23.900)) = 3.631028 * F_nmgy
   sigma_F_uJy = 3.631028 * sigma_F_nmgy
3. Scale-Invariant Signed SNR:
   SNR = F_nmgy / sigma_F_nmgy = F_uJy / sigma_F_uJy
4. Fail-Closed Sign Security:
   Unknown, missing, or malformed isdiffpos values fail closed (is_positive_detection_snr5 = False)
   and are recorded in the invalid-sign audit ledger.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Cross-platform file locking imports
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

# Photometric calibration constants
CANONICAL_AB_ZERO_POINT_NMGY = 22.500  # 1 nmgy = 3.631e-6 Jy (m_AB = 22.500)
MICROJANSKY_AB_ZERO_POINT = 23.900    # 1 uJy = 1e-6 Jy (m_AB = 23.90005 ~ 23.900)
NMGY_TO_UJY_FACTOR = 3.631028198428519  # Exact conversion factor: 10^(0.4 * 1.4)
LN10_OVER_2P5 = math.log(10.0) / 2.5   # ~ 0.9210340371976184

# Documented valid sign tokens
POSITIVE_DIFF_TOKENS = {1, "1", "+1", True, "t", "true", "pos", "positive"}
NEGATIVE_DIFF_TOKENS = {0, -1, "-1", "0", False, "f", "false", "neg", "negative"}


def parse_difference_flux_sign(raw_val: Any) -> Tuple[Optional[float], bool]:
    """
    Parse difference flux sign with strict fail-closed security.

    Returns:
    --------
    (sgn, is_valid):
        If valid positive: (+1.0, True)
        If valid negative: (-1.0, True)
        If missing, null, or unrecognized: (None, False)  <-- FAILS CLOSED
    """
    if raw_val is None:
        return None, False

    # Check boolean explicitly before numeric checks
    if isinstance(raw_val, bool):
        return (1.0, True) if raw_val else (-1.0, True)

    if isinstance(raw_val, (int, float)):
        if math.isnan(raw_val):
            return None, False
        if raw_val in POSITIVE_DIFF_TOKENS:
            return 1.0, True
        if raw_val in NEGATIVE_DIFF_TOKENS:
            return -1.0, True
        return None, False

    if isinstance(raw_val, str):
        cleaned = raw_val.strip().lower()
        if cleaned in POSITIVE_DIFF_TOKENS:
            return 1.0, True
        if cleaned in NEGATIVE_DIFF_TOKENS:
            return -1.0, True
        return None, False

    return None, False


def compute_physical_flux_and_snr(
    magpsf: float, sigmapsf: float, isdiffpos: Any
) -> Dict[str, Union[float, bool, str]]:
    """
    Computes physical difference flux in canonical nanomaggies (nmgy) and microjanskys (uJy),
    along with physically consistent signed signal-to-noise ratio.

    Fails closed: Any missing, null, or malformed sign token yields is_positive_detection_snr5 = False.
    """
    sgn, is_valid_sign = parse_difference_flux_sign(isdiffpos)

    # Validate numerical inputs
    is_valid_mag = math.isfinite(magpsf)
    is_valid_err = math.isfinite(sigmapsf) and (sigmapsf > 0.0)

    if (not is_valid_sign) or (not is_valid_mag) or (not is_valid_err):
        return {
            "flux_nmgy": 0.0,
            "flux_err_nmgy": float("inf"),
            "flux_ujy": 0.0,
            "flux_err_ujy": float("inf"),
            "signed_snr": 0.0,
            "is_valid_sign": bool(is_valid_sign),
            "is_positive_detection_snr5": False,
            "validation_note": "FAIL_CLOSED_INVALID_INPUT" if not (is_valid_mag and is_valid_err) else "FAIL_CLOSED_UNKNOWN_SIGN",
        }

    # 1. Canonical Nanomaggies (Z0 = 22.500)
    # F_nmgy = 10^(-0.4 * (m - 22.5))
    abs_flux_nmgy = 10.0 ** (-0.4 * (magpsf - CANONICAL_AB_ZERO_POINT_NMGY))
    flux_nmgy = sgn * abs_flux_nmgy

    # Hyperbolic log-normal error propagation: sigma_F = F * sinh((ln 10 / 2.5) * sigma_m)
    # Clamping extreme sigma_m at 3.0 mag to prevent numerical overflow in sinh
    clamped_sig = min(max(float(sigmapsf), 1e-4), 3.0)
    flux_err_nmgy = abs_flux_nmgy * math.sinh(LN10_OVER_2P5 * clamped_sig)

    # 2. Microjanskys (Z0 = 23.900)
    # 1 nmgy = 3.631028 uJy
    abs_flux_ujy = abs_flux_nmgy * NMGY_TO_UJY_FACTOR
    flux_ujy = sgn * abs_flux_ujy
    flux_err_ujy = flux_err_nmgy * NMGY_TO_UJY_FACTOR

    # 3. Signed SNR: Scale invariant ratio F / sigma_F
    # For small errors, SNR ~ sgn * 1.085736 / sigma_m
    signed_snr = flux_nmgy / flux_err_nmgy

    # Strict positive detection rule:
    # Requires documented positive flux (sgn > 0) AND signed SNR >= +5.0
    is_pos_det = (sgn > 0.0) and (signed_snr >= 5.0)

    return {
        "flux_nmgy": float(flux_nmgy),
        "flux_err_nmgy": float(flux_err_nmgy),
        "flux_ujy": float(flux_ujy),
        "flux_err_ujy": float(flux_err_ujy),
        "signed_snr": float(signed_snr),
        "is_valid_sign": True,
        "is_positive_detection_snr5": bool(is_pos_det),
        "validation_note": "VALID_CALIBRATED",
    }


class GlobalInterprocessRateLimiter:
    """
    Genuinely thread-safe and inter-process rate limiter with uniform random jitter.
    Uses OS-level file locking (msvcrt on Windows, fcntl on POSIX) and threading.Lock.
    """

    def __init__(
        self,
        lock_file_path: Path = Path("data/external_validity_v1/raw/alerce_external/.rate_limiter.lock"),
        max_rate: float = 5.0,
    ):
        self.lock_file_path = Path(lock_file_path)
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate  # 0.20s for 5.0 req/s
        self._thread_lock = threading.Lock()

    def _acquire_file_lock(self, file_obj) -> None:
        if sys.platform == "win32":
            # Windows non-blocking lock polling
            while True:
                try:
                    msvcrt.locking(file_obj.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except (IOError, OSError):
                    time.sleep(0.005)
        else:
            # POSIX blocking lock
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX)

    def _release_file_lock(self, file_obj) -> None:
        if sys.platform == "win32":
            try:
                msvcrt.locking(file_obj.fileno(), msvcrt.LK_UNLCK, 1)
            except (IOError, OSError):
                pass
        else:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def wait_and_acquire(self) -> float:
        """
        Coordinates across threads and processes to enforce the global rate ceiling.
        Returns the duration waited in seconds.
        """
        with self._thread_lock:
            total_waited = 0.0
            with open(self.lock_file_path, "a+") as f:
                self._acquire_file_lock(f)
                try:
                    f.seek(0)
                    content = f.read().strip()
                    now = time.time()
                    last_time = float(content) if content else 0.0
                    elapsed = now - last_time

                    if elapsed < self.min_interval:
                        jitter = random.uniform(0.02, 0.06)
                        sleep_time = (self.min_interval - elapsed) + jitter
                        time.sleep(sleep_time)
                        total_waited = sleep_time
                        now = time.time()

                    # Write new timestamp
                    f.seek(0)
                    f.truncate(0)
                    f.write(f"{now:.6f}")
                    f.flush()
                finally:
                    self._release_file_lock(f)
            return total_waited


class AlerceBrokerClient:
    """
    Production-grade ALeRCE API client with complete request-to-vault lifecycle,
    bounded exponential backoff, rate limiting, and immutable artifact preservation.
    """

    BASE_URL = "https://api.alerce.online/ztf/v1"
    CLIENT_VERSION = "ASTRAEUS-AlerceBrokerClient-v1.2"

    def __init__(
        self,
        raw_vault_dir: Path = Path("data/external_validity_v1/raw/alerce_external"),
        forced_vault_dir: Path = Path("data/external_validity_v1/raw/alerce_forced"),
        max_rate: float = 5.0,
        request_timeout: float = 15.0,
        max_retries: int = 5,
    ):
        self.raw_vault_dir = Path(raw_vault_dir)
        self.forced_vault_dir = Path(forced_vault_dir)
        self.raw_vault_dir.mkdir(parents=True, exist_ok=True)
        self.forced_vault_dir.mkdir(parents=True, exist_ok=True)

        self.max_rate = max_rate
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        lock_path = self.raw_vault_dir / ".rate_limiter.lock"
        self.rate_limiter = GlobalInterprocessRateLimiter(lock_file_path=lock_path, max_rate=max_rate)

    def get_lightcurve_path(self, ztf_id: str) -> Path:
        return self.raw_vault_dir / f"{ztf_id}_lightcurve.json"

    def get_forced_path(self, ztf_id: str) -> Path:
        return self.forced_vault_dir / f"{ztf_id}_forced.json"

    def fetch_and_archive_lightcurve(
        self, ztf_id: str, session: Optional[Any] = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete lifecycle for difference light curve:
        1. Checks existing artifact (idempotent / non-overwriting).
        2. Acquires global inter-process rate limiter token.
        3. Queries ALeRCE API with bounded exponential backoff on 429/5xx.
        4. Writes exact raw JSON response atomically to vault.
        5. Computes SHA-256 and parses photometric records.
        """
        vault_path = self.get_lightcurve_path(ztf_id)

        # Idempotency Protection: never overwrite existing raw response
        if vault_path.exists() and not force:
            raw_bytes = vault_path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            try:
                raw_data = json.loads(raw_bytes.decode("utf-8"))
                parsed = self.parse_raw_lightcurve_payload(raw_data)
                return {
                    "ztf_id": ztf_id,
                    "status": "ALREADY_ARCHIVED",
                    "http_status": 200,
                    "vault_path": str(vault_path),
                    "sha256": sha256,
                    "file_size_bytes": len(raw_bytes),
                    "parsed_metrics": parsed,
                }
            except Exception as e:
                pass  # Fall through to re-fetch if corrupted and force=True

        url = f"{self.BASE_URL}/objects/{ztf_id}/lightcurve"
        params = {"survey": "ztf"}
        headers = {
            "User-Agent": f"{self.CLIENT_VERSION} (DeepMind-ASTRAEUS)",
            "Accept": "application/json",
        }

        # Lazy import requests to allow isolated testing without global dependency
        import requests
        http_session = session if session is not None else requests.Session()

        attempt = 0
        last_error = None
        last_status = None

        while attempt < self.max_retries:
            attempt += 1
            self.rate_limiter.wait_and_acquire()

            try:
                resp = http_session.get(url, params=params, headers=headers, timeout=self.request_timeout)
                last_status = resp.status_code

                if resp.status_code == 200:
                    raw_text = resp.text
                    raw_bytes = raw_text.encode("utf-8")
                    sha256 = hashlib.sha256(raw_bytes).hexdigest()

                    # Atomic write to temporary file
                    tmp_path = vault_path.with_suffix(".tmp")
                    tmp_path.write_bytes(raw_bytes)
                    tmp_path.replace(vault_path)

                    # Parse verified payload
                    raw_data = json.loads(raw_text)
                    parsed = self.parse_raw_lightcurve_payload(raw_data)

                    return {
                        "ztf_id": ztf_id,
                        "status": "SUCCESS_DOWNLOADED",
                        "http_status": 200,
                        "vault_path": str(vault_path),
                        "sha256": sha256,
                        "file_size_bytes": len(raw_bytes),
                        "parsed_metrics": parsed,
                        "retries_performed": attempt - 1,
                    }

                elif resp.status_code == 404:
                    return {
                        "ztf_id": ztf_id,
                        "status": "OBJECT_NOT_FOUND_404",
                        "http_status": 404,
                        "vault_path": None,
                        "sha256": None,
                        "error_message": f"Object {ztf_id} not found on ALeRCE broker.",
                        "retries_performed": attempt - 1,
                    }

                elif resp.status_code in (429, 500, 502, 503, 504):
                    # Bounded exponential backoff with jitter
                    backoff = min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.1, 0.5)
                    time.sleep(backoff)
                    continue

                else:
                    return {
                        "ztf_id": ztf_id,
                        "status": f"HTTP_ERROR_{resp.status_code}",
                        "http_status": resp.status_code,
                        "vault_path": None,
                        "sha256": None,
                        "error_message": resp.text[:200],
                        "retries_performed": attempt - 1,
                    }

            except Exception as exc:
                last_error = str(exc)
                backoff = min(8.0, 0.5 * (2 ** attempt)) + random.uniform(0.1, 0.5)
                time.sleep(backoff)

        # Retries exhausted: ensure existing files are NEVER touched or corrupted
        return {
            "ztf_id": ztf_id,
            "status": "RETRIES_EXHAUSTED",
            "http_status": last_status,
            "vault_path": None,
            "sha256": None,
            "error_message": last_error or "Maximum retry attempts exceeded.",
            "retries_performed": attempt,
        }

    def fetch_and_archive_forced_photometry(
        self, ztf_id: str, session: Optional[Any] = None, force: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieves and archives forced photometry as a separate independent product.
        Never merged into alert difference light curves.
        """
        vault_path = self.get_forced_path(ztf_id)

        if vault_path.exists() and not force:
            raw_bytes = vault_path.read_bytes()
            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            return {
                "ztf_id": ztf_id,
                "status": "FORCED_ALREADY_ARCHIVED",
                "http_status": 200,
                "vault_path": str(vault_path),
                "sha256": sha256,
            }

        url = f"{self.BASE_URL}/objects/{ztf_id}/forced_photometry"
        headers = {
            "User-Agent": f"{self.CLIENT_VERSION} (DeepMind-ASTRAEUS)",
            "Accept": "application/json",
        }

        import requests
        http_session = session if session is not None else requests.Session()

        self.rate_limiter.wait_and_acquire()
        try:
            resp = http_session.get(url, headers=headers, timeout=self.request_timeout)
            if resp.status_code == 200:
                raw_text = resp.text
                raw_bytes = raw_text.encode("utf-8")
                sha256 = hashlib.sha256(raw_bytes).hexdigest()

                tmp_path = vault_path.with_suffix(".tmp")
                tmp_path.write_bytes(raw_bytes)
                tmp_path.replace(vault_path)

                return {
                    "ztf_id": ztf_id,
                    "status": "FORCED_SUCCESS_DOWNLOADED",
                    "http_status": 200,
                    "vault_path": str(vault_path),
                    "sha256": sha256,
                }
            else:
                return {
                    "ztf_id": ztf_id,
                    "status": f"FORCED_HTTP_{resp.status_code}",
                    "http_status": resp.status_code,
                    "vault_path": None,
                    "sha256": None,
                }
        except Exception as exc:
            return {
                "ztf_id": ztf_id,
                "status": "FORCED_REQUEST_FAILED",
                "http_status": None,
                "vault_path": None,
                "sha256": None,
                "error_message": str(exc),
            }

    def parse_raw_lightcurve_payload(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses raw ALeRCE JSON payload with physical calibration, fail-closed sign validation,
        and multi-dimensional evaluability metrics.
        """
        raw_detections = raw_data.get("detections", [])
        raw_nondetections = raw_data.get("non_detections", [])

        parsed_detections = []
        pos_det_snr5_g = 0
        pos_det_snr5_r = 0
        neg_detections_count = 0
        invalid_sign_count = 0

        for det in raw_detections:
            mag = float(det.get("magpsf", float("nan")))
            sig = float(det.get("sigmapsf", float("nan")))
            diffpos = det.get("isdiffpos")
            fid = int(det.get("fid", 0))

            calib = compute_physical_flux_and_snr(mag, sig, diffpos)

            if not calib["is_valid_sign"]:
                invalid_sign_count += 1
            elif calib["flux_nmgy"] < 0:
                neg_detections_count += 1

            if calib["is_positive_detection_snr5"]:
                if fid == 1:
                    pos_det_snr5_g += 1
                elif fid == 2:
                    pos_det_snr5_r += 1

            parsed_detections.append({
                "mjd": float(det.get("mjd", 0.0)),
                "fid": fid,
                "passband": "g" if fid == 1 else ("r" if fid == 2 else "i"),
                "magpsf": mag,
                "sigmapsf": sig,
                "isdiffpos_raw": diffpos,
                "ra": float(det.get("ra", float("nan"))),
                "dec": float(det.get("dec", float("nan"))),
                **calib,
            })

        parsed_nondetections = []
        for nd in raw_nondetections:
            fid = int(nd.get("fid", 0))
            diffmaglim = float(nd.get("diffmaglim", float("nan")))

            # 5-sigma limiting flux in nanomaggies: F_lim = 10^(-0.4 * (diffmaglim - 22.500))
            if math.isfinite(diffmaglim):
                ul_nmgy = 10.0 ** (-0.4 * (diffmaglim - CANONICAL_AB_ZERO_POINT_NMGY))
                ul_ujy = ul_nmgy * NMGY_TO_UJY_FACTOR
            else:
                ul_nmgy = float("nan")
                ul_ujy = float("nan")

            parsed_nondetections.append({
                "mjd": float(nd.get("mjd", 0.0)),
                "fid": fid,
                "passband": "g" if fid == 1 else ("r" if fid == 2 else "i"),
                "diffmaglim": diffmaglim,
                "limiting_flux_nmgy": ul_nmgy,
                "limiting_flux_ujy": ul_ujy,
            })

        total_snr5 = pos_det_snr5_g + pos_det_snr5_r
        has_dual_band_snr5 = (pos_det_snr5_g > 0) and (pos_det_snr5_r > 0)

        if total_snr5 >= 5 and has_dual_band_snr5:
            eval_status = "FEATURE_EVALUABLE"
        elif total_snr5 >= 5 and not has_dual_band_snr5:
            eval_status = "SINGLE_BAND_ONLY"
        elif total_snr5 > 0:
            eval_status = "INSUFFICIENT_DETECTIONS"
        elif len(raw_nondetections) > 0:
            eval_status = "NON_DETECTIONS_ONLY"
        else:
            eval_status = "EMPTY_RECORD"

        return {
            "total_raw_detections": len(raw_detections),
            "total_raw_nondetections": len(raw_nondetections),
            "invalid_sign_detections_count": invalid_sign_count,
            "negative_detections_count": neg_detections_count,
            "n_det_snr5_g": pos_det_snr5_g,
            "n_det_snr5_r": pos_det_snr5_r,
            "n_det_snr5_total": total_snr5,
            "has_dual_band_snr5": has_dual_band_snr5,
            "derived_feature_status": eval_status,
            "detections": parsed_detections,
            "non_detections": parsed_nondetections,
        }

    def generate_acquisition_manifest(
        self, records: List[Dict[str, Any]], output_path: Path
    ) -> Dict[str, Any]:
        """Generates cryptographic acquisition manifest for all processed requests."""
        manifest = {
            "manifest_type": "ASTRAEUS Phase 4B Broker Acquisition Manifest",
            "client_version": self.CLIENT_VERSION,
            "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_requests": len(records),
            "status_counts": {},
            "records": records,
        }
        for r in records:
            st = r.get("status", "UNKNOWN")
            manifest["status_counts"][st] = manifest["status_counts"].get(st, 0) + 1

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


def self_test():
    """Verify units, sign parsing, and error propagation."""
    # 1. Known reference check: m = 22.500 must equal exactly 1.0 nanomaggie
    calib_1nmgy = compute_physical_flux_and_snr(22.500, 0.1, 1)
    assert math.isclose(calib_1nmgy["flux_nmgy"], 1.0, rel_tol=1e-5), "1 nmgy definition violated!"
    assert math.isclose(calib_1nmgy["flux_ujy"], 3.631028, rel_tol=1e-5), "Microjansky scaling violated!"

    # 2. Known reference check: m = 23.900 corresponds to ~1.0 uJy (within astronomical rounding of Z0=23.900 vs exact 23.90007)
    calib_1ujy = compute_physical_flux_and_snr(23.900, 0.1, 1)
    assert math.isclose(calib_1ujy["flux_ujy"], 1.0, rel_tol=1e-3), "1 uJy definition violated!"
    assert math.isclose(calib_1ujy["flux_nmgy"], 1.0 / 3.631028, rel_tol=1e-3), "Nanomaggie scaling violated!"

    # 3. Fail-closed security on invalid / missing sign
    for bad_sign in (None, "unknown", "foo", float("nan"), 99, ""):
        bad_calib = compute_physical_flux_and_snr(18.0, 0.1, bad_sign)
        assert bad_calib["is_positive_detection_snr5"] is False, f"Failed to fail closed on sign {bad_sign}"
        assert bad_calib["is_valid_sign"] is False

    # 4. Negative flux check: must fail positive detection
    neg_calib = compute_physical_flux_and_snr(18.0, 0.05, 0)
    assert neg_calib["flux_nmgy"] < 0
    assert neg_calib["signed_snr"] < 0
    assert neg_calib["is_positive_detection_snr5"] is False

    print("self_test passed: Calibration and fail-closed security verified.")


if __name__ == "__main__":
    self_test()
