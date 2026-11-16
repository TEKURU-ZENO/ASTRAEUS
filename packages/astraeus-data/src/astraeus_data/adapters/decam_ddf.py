"""
ASTRAEUS Phase 5: DECam DDF Photometry Adapter.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ARCH-001 / ASTRAEUS-P5-FEAS-001
"""

import hashlib
import json
from typing import List, Dict, Any, Optional
import numpy as np

from astraeus_data.adapters.canonical import CanonicalObservationRecord, ObservationType


class DecamDdfPhotometryAdapter:
    """
    Adapter for Dark Energy Camera (DECam) Deep Drilling Fields (DDF) Photometry.
    Standardizes calibrated time series across Blanco 4m telescope deep fields (COSMOS, ECDFS, XMM-LSS).
    """

    DECAM_FILTER_MAP = {
        "u": "u_DECam",
        "g": "g_DECam",
        "r": "r_DECam",
        "i": "i_DECam",
        "z": "z_DECam",
        "y": "Y_DECam",
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
        """
        Parses a single row from a DECam DDF candidate light curve.
        """
        mjd = float(row.get("mjd") or row.get("MJD"))
        filt_raw = str(row.get("filter") or row.get("band", "r")).strip().lower()
        native_passband = cls.DECAM_FILTER_MAP.get(filt_raw, "r_DECam")

        ra = float(row.get("ra") or row.get("RA", ra_fallback))
        dec = float(row.get("dec") or row.get("Dec", dec_fallback))

        # Check flags: 0 = clean detection
        flags = int(row.get("flags", 0))
        if flags != 0 and (flags & 0x0008 or flags & 0x0010): # cosmic ray or edge defect
            return None

        # Flux conversion: if flux_ujy present directly
        if "flux_ujy" in row and "flux_err_ujy" in row:
            flux_ujy = float(row["flux_ujy"])
            sigma_ujy = max(float(row["flux_err_ujy"]), 1e-3)
        elif "mag" in row and "mag_err" in row and row["mag"] is not None:
            mag = float(row["mag"])
            mag_err = float(row["mag_err"])
            flux_ujy = float(10.0 ** (-(mag - 23.90) / 2.5))
            sigma_ujy = float((np.log(10.0) / 2.5) * flux_ujy * mag_err)
        elif "flux" in row and "flux_err" in row:
            # Native counts with zero-point
            zpt = float(row.get("zpt", 25.0))
            flux_cnt = float(row["flux"])
            err_cnt = float(row["flux_err"])
            # flux_ujy = flux_cnt * 10^((23.90 - zpt) / 2.5)
            scale = 10.0 ** ((23.90 - zpt) / 2.5)
            flux_ujy = float(flux_cnt * scale)
            sigma_ujy = max(float(err_cnt * scale), 1e-3)
        else:
            return None

        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)
        obs_id = str(row.get("obs_id") or f"DECam_{int(mjd * 1000)}_{filt_raw}")
        prov_hash = cls._compute_hash(row)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=obs_id,
            t_obs_bmjd=mjd,
            t_err_days=0.0010,  # ~90-second exposure ~ 0.0010 days
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.04,  # DECam pixel scale 0.263 arcsec/pixel
            survey_id="DECam",
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_FORCED,
            measured_flux_ujy=flux_ujy,
            measurement_sigma_ujy=sigma_ujy,
            upper_limit_ujy=upper_limit_ujy,
            is_positive_detection_snr5=is_detection,
            quality_bitmask=flags,
            provenance_hash=prov_hash,
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
        records.sort(key=lambda x: x.t_obs_bmjd)
        return records
