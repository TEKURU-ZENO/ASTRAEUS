"""
ASTRAEUS Phase 5: Canonical Observation Record Model.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ARCH-001
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ObservationType(str, Enum):
    DIFFERENCE_FORCED = "DIFFERENCE_FORCED"
    DIFFERENCE_ALERT = "DIFFERENCE_ALERT"
    TOTAL_STACK = "TOTAL_STACK"


class CanonicalObservationRecord(BaseModel):
    """
    Standardized, survey-agnostic astrophysical observation event.
    Preserves negative difference fluxes, decouples astrophysical observation
    time from operational publication time, and guarantees full cryptographic provenance.
    """
    # Unique Internal Identifiers
    astraeus_id: str = Field(..., description="Canonical transient object identifier (ASTRAEUS-P5-YYMMDD-XXXXXXXX).")
    survey_obs_id: str = Field(..., description="Native provider observation identifier.")

    # Spatio-Temporal Coordinates
    t_obs_bmjd: float = Field(..., description="Barycentric Modified Julian Date (TDB scale) at exposure midpoint.")
    t_pub_utc: Optional[datetime] = Field(default=None, description="ISO 8601 UTC timestamp when packet was published.")
    t_err_days: float = Field(default=0.0001, description="Exposure duration / timing uncertainty.")

    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="ICRS Right Ascension in degrees.")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="ICRS Declination in degrees.")
    astrometric_sigma_arcsec: float = Field(default=0.05, gt=0.0, description="1-sigma astrometric uncertainty.")

    # Photometric Measurements
    survey_id: str = Field(..., description="'ZTF', 'ATLAS', 'DECam', 'PS1', or 'Rubin'.")
    native_passband: str = Field(..., description="Native survey passband (e.g. 'ZTF_g', 'ATLAS_c', 'r_DECam').")
    obs_type: ObservationType = Field(default=ObservationType.DIFFERENCE_FORCED, description="Observation type.")
    
    # Crucial invariant: flux can be negative in forced/difference photometry
    measured_flux_ujy: float = Field(..., description="Standardized flux density in microJanskys (AB Z0 = 23.900).")
    measurement_sigma_ujy: float = Field(..., gt=0.0, description="1-sigma heteroscedastic uncertainty in microJanskys.")
    upper_limit_ujy: Optional[float] = Field(default=None, description="5-sigma limiting flux for non-detections.")
    is_positive_detection_snr5: bool = Field(..., description="True if flux > 0 and flux / sigma >= 5.0.")
    
    quality_bitmask: int = Field(default=0, description="Survey-specific image subtraction quality flags (0 = clean).")
    provenance_hash: str = Field(..., description="SHA-256 digest of original raw provider payload or row.")
