"""
ASTRAEUS Phase 5: Rubin LSST Alert Interface Adapter.

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


class RubinAlertInterfaceAdapter:
    """
    Forward-compatible adapter for Vera C. Rubin Observatory Legacy Survey of Space and Time (LSST) alert packets.
    Converts simulated or public alert schema DIASource and DIAForcedPhot records to CanonicalObservationRecord.
    Handles nanoJansky (Z0 = 31.400) to microJansky (Z0 = 23.900) scaling.
    """

    LSST_FILTER_MAP = {
        "u": "u_LSST",
        "g": "g_LSST",
        "r": "r_LSST",
        "i": "i_LSST",
        "z": "z_LSST",
        "y": "y_LSST",
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
        """
        midpoint_mjd = float(dia_source.get("midpointMjdTai") or dia_source.get("mjd"))
        band = str(dia_source.get("filterName") or dia_source.get("band", "r")).strip().lower()
        native_passband = cls.LSST_FILTER_MAP.get(band, f"{band}_LSST")

        ra = float(dia_source.get("ra", ra_fallback))
        dec = float(dia_source.get("dec", dec_fallback))

        # Check flags
        flags = int(dia_source.get("flags", 0))

        # Flux conversion: native flux is psfFlux in nJy
        if "psfFlux" in dia_source and "psfFluxErr" in dia_source:
            flux_njy = float(dia_source["psfFlux"])
            flux_err_njy = max(float(dia_source["psfFluxErr"]), 1.0)
            flux_ujy = float(flux_njy / 1000.0)
            sigma_ujy = max(float(flux_err_njy / 1000.0), 1e-4)
        elif "flux_ujy" in dia_source and "flux_err_ujy" in dia_source:
            flux_ujy = float(dia_source["flux_ujy"])
            sigma_ujy = max(float(dia_source["flux_err_ujy"]), 1e-4)
        else:
            return None

        is_detection = bool(flux_ujy > 0.0 and (flux_ujy / sigma_ujy) >= 5.0)
        upper_limit_ujy = float(5.0 * sigma_ujy)
        dia_source_id = str(dia_source.get("diaSourceId") or f"LSST_{int(midpoint_mjd * 1000)}_{band}")
        prov_hash = cls._compute_hash(dia_source)

        return CanonicalObservationRecord(
            astraeus_id=astraeus_id,
            survey_obs_id=dia_source_id,
            t_obs_bmjd=midpoint_mjd,
            t_err_days=0.00035,  # 30-second visit
            ra_deg=ra,
            dec_deg=dec,
            astrometric_sigma_arcsec=0.015,  # LSST astrometric precision ~ 10-20 mas
            survey_id="Rubin",
            native_passband=native_passband,
            obs_type=ObservationType.DIFFERENCE_ALERT,
            measured_flux_ujy=flux_ujy,
            measurement_sigma_ujy=sigma_ujy,
            upper_limit_ujy=upper_limit_ujy,
            is_positive_detection_snr5=is_detection,
            quality_bitmask=flags,
            provenance_hash=prov_hash,
        )
