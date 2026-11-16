"""
ASTRAEUS Phase 5: DECam DDF Photometry Adapter (Fail-Closed v2.0).

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ARCH-002 / ASTRAEUS-P5-FEAS-001
"""

import hashlib
import json
import math
from typing import List, Dict, Any, Optional
import numpy as np

from astraeus_data.adapters.canonical import (
    CanonicalObservationRecord,
    ObservationType,
    TransformationState,
)


class DecamDdfPhotometryAdapter:
    """
    Adapter for Dark Energy Camera (DECam) Deep Drilling Fields (DDF) Photometry.
    Enforces strict fail-closed validation on unknown passbands and non-positive uncertainties.
    """

    DECAM_FILTER_MAP = {
        "u": "u_DECam",
        "u_decam": "u_DECam",
        "g": "g_DECam",
        "g_decam": "g_DECam",
        "r": "r_DECam",
        "r_decam": "r_DECam",
        "i": "i_DECam",
        "i_decam": "i_DECam",
        "z": "z_DECam",
        "z_decam": "z_DECam",
        "y": "Y_DECam",
        "y_decam": "Y_DECam",
        "Y": "Y_DECam",
    }

    @staticmethod
    def _compute_hash(row: Dict[str, Any]) -> str:
        s = json.dumps(row, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @classmethod
    def parse_record(
        cls,
        row: Dict[str, Any],
        astraeus_id: str,
        ra_fallback: float,
        dec_fallback: float,
    ) -> Optional[CanonicalObservationRecord]:
        # 1. Filter Validation (Fail-closed: no default)
        filt_raw = row.get("filter") or row.get("band")
        if not filt_raw or str(filt_raw).strip().lower() not in cls.DECAM_FILTER_MAP:
            return None
        native_passband = cls.DECAM_FILTER_MAP[str(filt_raw).strip().lower()]

        # 2. Timestamp
        mjd_val = row.get("mjd") or row.get("MJD")
        if mjd_val is None:
            return None
        mjd = float(mjd_val)
        if math.isnan(mjd) or mjd <= 0:
            return None

        ra = float(row.get("ra") or row.get("RA", ra_fallback))
        dec = float(row.get("dec") or row.get("Dec", dec_fallback))

        # Check native flags: 0x0008 (cosmic ray), 0x0010 (edge defect)
        flags = int(row.get("flags", 0))
        if flags != 0 and (flags & 0x0008 or flags & 0x0010):
            return None

        # 3. Flux and Uncertainty Validation (Fail-closed)
        if "flux_ujy" in row and "flux_err_ujy" in row and row["flux_err_ujy"] is not None:
            flux_ujy = float(row["flux_ujy"])
            sigma_ujy = float(row["flux_err_ujy"])
        elif "mag" in row and "mag_err" in row and row["mag"] is not None and row["mag_err"] is not None:
            mag = float(row["mag"])
            mag_err = float(row["mag_err"])
            if mag <= 0 or mag_err <= 0 or math.isnan(mag_err):
                return None
            flux_ujy = float(10.0 ** (-(mag - 23.90) / 2.5))
            sigma_ujy = float((np.log(10.0) / 2.5) * flux_ujy * mag_err)
        elif "flux" in row and "flux_err" in row and row["flux_err"] is not None:
            zpt = float(row.get("zpt", 25.0))
            flux_cnt = float(row["flux"])
            err_cnt = float(row["flux_err"])
            scale = 10.0 ** ((23.90 - zpt) / 2.5)
            flux_ujy = float(flux_cnt * scale)
            sigma_ujy = float(err_cnt * scale)
        else:
            return None

        # Strict uncertainty validation
        if math.isnan(sigma_ujy) or math.isinf(sigma_ujy) or sigma_ujy <= 0.0:
            return None

        if math.isnan(flux_ujy) or math.isinf(flux_ujy):
            return None

        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)
        obs_id = str(row.get("obs_id") or f"DECam_{int(mjd * 1000)}_{filt_raw}")
        prov_hash = cls._compute_hash(row)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=obs_id,
            survey_id="DECam",
            provenance_hash=prov_hash,
            t_native=mjd,
            native_time_scale="UTC",
            exposure_midpoint_convention="MIDPOINT",
            t_converted_bmjd=None,
            converted_time_scale=None,
            time_conversion_method="NATIVE_UNCONVERTED",
            t_obs_bmjd=mjd,
            t_err_days=0.0010,  # ~90s exposure
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.04,
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_FORCED,
            measured_flux_ujy=flux_ujy,
            measurement_sigma_ujy=sigma_ujy,
            upper_limit_ujy=upper_limit_ujy,
            is_positive_detection_snr5=is_detection,
            zero_point_ab=23.900,
            transformation_state=TransformationState.NATIVE_PASSBAND_UNTRANSFORMED,
            native_flags=flags,
            quality_bitmask=flags,
            is_clean=True,
            quarantine_reason=None,
        )

    @classmethod
    def parse_lightcurve(
        cls,
        rows: List[Dict[str, Any]],
        astraeus_id: str,
        ra_fallback: float,
        dec_fallback: float,
    ) -> List[CanonicalObservationRecord]:
        records = []
        for r in rows:
            rec = cls.parse_record(r, astraeus_id, ra_fallback, dec_fallback)
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda x: x.t_native)
        return records
