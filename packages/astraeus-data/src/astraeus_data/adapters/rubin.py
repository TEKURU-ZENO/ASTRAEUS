"""
ASTRAEUS Phase 5: Rubin LSST Alert Interface Adapter (Fail-Closed v2.0).

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


class RubinAlertInterfaceAdapter:
    """
    Forward-compatible adapter for Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) alert packets.
    Converts simulated or public alert schema DIASource and DIAForcedPhot records to CanonicalObservationRecord.
    Preserves TAI time-scale semantics and handles nanoJansky (Z0 = 31.400) to microJansky (Z0 = 23.900) scaling.
    """

    LSST_FILTER_MAP = {
        "u": "u_LSST",
        "u_lsst": "u_LSST",
        "g": "g_LSST",
        "g_lsst": "g_LSST",
        "r": "r_LSST",
        "r_lsst": "r_LSST",
        "i": "i_LSST",
        "i_lsst": "i_LSST",
        "z": "z_LSST",
        "z_lsst": "z_LSST",
        "y": "y_LSST",
        "y_lsst": "y_LSST",
    }

    @staticmethod
    def _compute_hash(data: Dict[str, Any]) -> str:
        s = json.dumps(data, sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @classmethod
    def parse_dia_source(
        cls,
        dia_source: Dict[str, Any],
        astraeus_id: str,
        ra_fallback: float,
        dec_fallback: float,
    ) -> Optional[CanonicalObservationRecord]:
        """
        Parses a single DIASource record from an LSST alert packet.
        Rubin DIA fluxes are natively reported in nanoJanskys (nJy).
        1 microJansky = 1000 nanoJanskys.
        Native time scale is International Atomic Time (TAI).
        """
        # 1. Filter Validation (Fail-closed: no default)
        band = dia_source.get("filterName") or dia_source.get("band")
        if not band or str(band).strip().lower() not in cls.LSST_FILTER_MAP:
            return None
        native_passband = cls.LSST_FILTER_MAP[str(band).strip().lower()]

        # 2. Timestamp: Reported in TAI
        mjd_val = dia_source.get("midpointMjdTai") or dia_source.get("mjd")
        if mjd_val is None:
            return None
        midpoint_mjd_tai = float(mjd_val)
        if math.isnan(midpoint_mjd_tai) or midpoint_mjd_tai <= 0:
            return None

        ra = float(dia_source.get("ra", ra_fallback))
        dec = float(dia_source.get("dec", dec_fallback))
        flags = int(dia_source.get("flags", 0))

        # 3. Flux and Uncertainty Conversion (Fail-closed)
        if "psfFlux" in dia_source and "psfFluxErr" in dia_source and dia_source["psfFluxErr"] is not None:
            flux_njy = float(dia_source["psfFlux"])
            flux_err_njy = float(dia_source["psfFluxErr"])
            flux_ujy = float(flux_njy / 1000.0)
            sigma_ujy = float(flux_err_njy / 1000.0)
        elif "flux_ujy" in dia_source and "flux_err_ujy" in dia_source and dia_source["flux_err_ujy"] is not None:
            flux_ujy = float(dia_source["flux_ujy"])
            sigma_ujy = float(dia_source["flux_err_ujy"])
        else:
            return None

        # Strict uncertainty validation: strictly positive and finite
        if math.isnan(sigma_ujy) or math.isinf(sigma_ujy) or sigma_ujy <= 0.0:
            return None

        if math.isnan(flux_ujy) or math.isinf(flux_ujy):
            return None

        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)
        dia_source_id = str(dia_source.get("diaSourceId") or f"LSST_{int(midpoint_mjd_tai * 1000)}_{band}")
        prov_hash = cls._compute_hash(dia_source)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=dia_source_id,
            survey_id="Rubin",
            provenance_hash=prov_hash,
            t_native=midpoint_mjd_tai,
            native_time_scale="TAI",
            exposure_midpoint_convention="MIDPOINT",
            t_converted_bmjd=None,  # Explicitly None until barycentric ephemeris applied
            converted_time_scale=None,
            time_conversion_method="NATIVE_UNCONVERTED",
            t_obs_bmjd=None,  # Strictly None when uncomputed
            t_err_days=0.00035,  # 30-second visit
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.015,
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_ALERT,
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
