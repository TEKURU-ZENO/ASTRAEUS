"""
ASTRAEUS Phase 5: ATLAS Forced Photometry Adapter.

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


class AtlasForcedPhotometryAdapter:
    """
    Adapter for Asteroid Terrestrial-impact Last Alert System (ATLAS) Forced Photometry.
    Converts raw forced difference photometry (ASCII/JSON/dicts) into CanonicalObservationRecord models.
    Guarantees non-destructive retention of negative difference flux and zero-point consistency.
    """

    CHIN_QUALITY_CUT = 3.0  # Quality cut: chi/N <= 3.0 avoids bad subtractions and cosmic rays

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
        Parses a single row from an ATLAS forced photometry result.
        Returns None if quality flags fail (e.g. chi/N > CHIN_QUALITY_CUT).
        """
        # Extract MJD
        mjd = float(row.get("MJD") or row.get("mjd"))
        
        # Filter band: 'c' (cyan) or 'o' (orange)
        filt_raw = str(row.get("filt") or row.get("filter", "o")).strip().lower()
        native_passband = "ATLAS_c" if filt_raw.startswith("c") else "ATLAS_o"

        # Quality check: chi/N
        chin = float(row.get("chin") or row.get("chi/N", 1.0))
        if chin > cls.CHIN_QUALITY_CUT:
            return None

        # Coordinates
        ra = float(row.get("ra") or row.get("RA", ra_fallback))
        dec = float(row.get("dec") or row.get("Dec", dec_fallback))

        # Flux and uncertainty in microJanskys
        # ATLAS forced photometry provides 'uJy' and 'duJy' directly
        if "uJy" in row and "duJy" in row:
            flux_ujy = float(row["uJy"])
            sigma_ujy = max(float(row["duJy"]), 1e-3)
        elif "flux" in row and "flux_err" in row:
            flux_ujy = float(row["flux"])
            sigma_ujy = max(float(row["flux_err"]), 1e-3)
        else:
            # If magnitudes provided
            m = row.get("m")
            dm = row.get("dm", 0.1)
            if m is not None and float(m) > 0:
                m_val = float(m)
                dm_val = float(dm)
                flux_ujy = float(10.0 ** (-(m_val - 23.90) / 2.5))
                sigma_ujy = float((np.log(10.0) / 2.5) * flux_ujy * dm_val)
            else:
                return None

        # 5-sigma detection check and limiting flux
        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)

        obs_id = str(row.get("obs_id") or f"ATLAS_{int(mjd * 1000)}_{filt_raw}")
        prov_hash = cls._compute_hash(row)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=obs_id,
            t_obs_bmjd=mjd,
            t_err_days=0.00035,  # 30-second exposure ~ 0.00035 days
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.10,  # ATLAS PSF FWHM ~ 1.5 - 2.5 arcsec
            survey_id="ATLAS",
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_FORCED,
            measured_flux_ujy=flux_ujy,
            measurement_sigma_ujy=sigma_ujy,
            upper_limit_ujy=upper_limit_ujy,
            is_positive_detection_snr5=is_detection,
            quality_bitmask=0,
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
        """
        Parses full light curve array, sorting chronologically.
        """
        records = []
        for r in rows:
            rec = cls.parse_record(r, astraeus_id, ra_fallback, dec_fallback)
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda x: x.t_obs_bmjd)
        return records
