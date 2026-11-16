"""
ASTRAEUS Phase 5: ATLAS Forced Photometry Adapter (Fail-Closed v2.0).

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ARCH-002 / ASTRAEUS-P5-FEAS-001
"""

import hashlib
import json
import math
from enum import Enum
from typing import List, Dict, Any, Optional
import numpy as np

from astraeus_data.adapters.canonical import (
    CanonicalObservationRecord,
    ObservationType,
    TransformationState,
)


class AtlasRetrievalStatus(str, Enum):
    VALID_PHOTOMETRY = "VALID_PHOTOMETRY"
    NO_RESULT = "NO_RESULT"
    FAILED_RETRIEVAL = "FAILED_RETRIEVAL"
    INVALID_RESPONSE = "INVALID_RESPONSE"


class AtlasForcedPhotometryAdapter:
    """
    Adapter for Asteroid Terrestrial-impact Last Alert System (ATLAS) Forced Photometry.
    Converts raw forced difference photometry into CanonicalObservationRecord models.
    Enforces strict fail-closed handling on unknown filters and invalid uncertainties.
    """

    CHIN_QUALITY_CUT = 3.0  # Quality cut: chi/N <= 3.0 avoids bad subtractions and cosmic rays

    VALID_ATLAS_FILTERS = {
        "c": "ATLAS_c",
        "cyan": "ATLAS_c",
        "atlas_c": "ATLAS_c",
        "o": "ATLAS_o",
        "orange": "ATLAS_o",
        "atlas_o": "ATLAS_o",
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
        Parses a single row from an ATLAS forced photometry result.
        Returns None if quality flags fail (e.g. chi/N > CHIN_QUALITY_CUT) or if data is invalid.
        Enforces strict fail-closed validation: no default filter, no uncertainty flooring.
        """
        # 1. Filter Validation (Fail-closed)
        filt_raw = row.get("filt") or row.get("filter")
        if not filt_raw or str(filt_raw).strip().lower() not in cls.VALID_ATLAS_FILTERS:
            # Strictly reject unknown or missing filters
            return None
        native_passband = cls.VALID_ATLAS_FILTERS[str(filt_raw).strip().lower()]

        # 2. MJD Timestamp
        mjd_val = row.get("MJD") or row.get("mjd")
        if mjd_val is None:
            return None
        mjd = float(mjd_val)
        if math.isnan(mjd) or math.isinf(mjd) or mjd <= 0:
            return None

        # 3. Quality check: chi/N
        chin_val = row.get("chin") or row.get("chi/N", 1.0)
        try:
            chin = float(chin_val)
        except (ValueError, TypeError):
            return None
        if math.isnan(chin) or chin > cls.CHIN_QUALITY_CUT:
            return None

        # 4. Coordinates
        ra = float(row.get("ra") or row.get("RA", ra_fallback))
        dec = float(row.get("dec") or row.get("Dec", dec_fallback))

        # 5. Flux and Uncertainty Validation (Fail-closed)
        if "uJy" in row and "duJy" in row and row["duJy"] is not None:
            flux_ujy = float(row["uJy"])
            sigma_ujy = float(row["duJy"])
        elif "flux" in row and "flux_err" in row and row["flux_err"] is not None:
            flux_ujy = float(row["flux"])
            sigma_ujy = float(row["flux_err"])
        elif "m" in row and "dm" in row and row["m"] is not None and row["dm"] is not None:
            m_val = float(row["m"])
            dm_val = float(row["dm"])
            if m_val <= 0 or dm_val <= 0 or math.isnan(dm_val):
                return None
            flux_ujy = float(10.0 ** (-(m_val - 23.90) / 2.5))
            sigma_ujy = float((np.log(10.0) / 2.5) * flux_ujy * dm_val)
        else:
            return None

        # Strict uncertainty assertion: must be positive, non-zero, and finite
        if math.isnan(sigma_ujy) or math.isinf(sigma_ujy) or sigma_ujy <= 0.0:
            return None

        if math.isnan(flux_ujy) or math.isinf(flux_ujy):
            return None

        # 5-sigma detection check and limiting flux
        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)

        obs_id = str(row.get("obs_id") or f"ATLAS_{int(mjd * 1000)}_{filt_raw}")
        prov_hash = cls._compute_hash(row)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=obs_id,
            survey_id="ATLAS",
            provenance_hash=prov_hash,
            t_native=mjd,
            native_time_scale="UTC",
            exposure_midpoint_convention="MIDPOINT",
            t_converted_bmjd=None,  # Set when high-accuracy barycentric ephemeris available
            converted_time_scale=None,
            time_conversion_method="NATIVE_UNCONVERTED",
            t_obs_bmjd=None,  # Strictly None when uncomputed
            t_err_days=0.00035,  # 30-second exposure ~ 0.00035 days
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.10,
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_FORCED,
            measured_flux_ujy=flux_ujy,
            measurement_sigma_ujy=sigma_ujy,
            upper_limit_ujy=upper_limit_ujy,
            is_positive_detection_snr5=is_detection,
            zero_point_ab=23.900,
            transformation_state=TransformationState.NATIVE_PASSBAND_UNTRANSFORMED,
            native_flags=int(row.get("flags", 0)),
            quality_bitmask=0,
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
