"""
ASTRAEUS Phase 5: Canonical Observation Record Model (Contract v2.0).

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ARCH-002
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ObservationType(str, Enum):
    DIFFERENCE_FORCED = "DIFFERENCE_FORCED"
    DIFFERENCE_ALERT = "DIFFERENCE_ALERT"
    TOTAL_STACK = "TOTAL_STACK"
    CATALOG_APER = "CATALOG_APER"
    UPPER_LIMIT_THRESHOLD = "UPPER_LIMIT_THRESHOLD"
    MISSING_MEASUREMENT = "MISSING_MEASUREMENT"


class TransformationState(str, Enum):
    NATIVE_PASSBAND_UNTRANSFORMED = "NATIVE_PASSBAND_UNTRANSFORMED"
    EMPIRICAL_COLOR_TRANSFORMED = "EMPIRICAL_COLOR_TRANSFORMED"
    SED_SUPPORTED_TRANSFORMED = "SED_SUPPORTED_TRANSFORMED"
    NON_TRANSFORMABLE = "NON_TRANSFORMABLE"


class CanonicalObservationRecord(BaseModel):
    """
    Standardized, survey-agnostic astrophysical observation event (Contract v2.0).
    Preserves negative difference fluxes, enforces complete physical timestamp semantics,
    guarantees fail-closed data validation, and maintains cryptographic provenance.
    """
    # 1. Identity & Provenance
    astraeus_id: str = Field(..., description="Canonical persistent entity ID (ASTRAEUS-EVT-XXXXX or lookup key).")
    survey_obs_id: str = Field(..., description="Native provider observation identifier.")
    survey_id: str = Field(..., description="'ZTF', 'ATLAS', 'DECam', 'PS1', or 'Rubin'.")
    provenance_hash: str = Field(..., description="SHA-256 digest of original raw provider payload or row.")

    # 2. Spatio-Temporal Coordinates & Physical Time Scales
    t_native: float = Field(..., description="Native exposure timestamp as reported by the provider archive.")
    native_time_scale: str = Field(default="UTC", description="'UTC', 'TAI', 'TDB', or 'TT'.")
    exposure_midpoint_convention: str = Field(default="MIDPOINT", description="'MIDPOINT', 'START_PLUS_HALF_EXPTIME', 'SHUTTER_OPEN'.")
    t_converted_bmjd: Optional[float] = Field(
        default=None,
        description="Barycentric Modified Julian Date (TDB scale), strictly None unless barycentric correction was computed."
    )
    converted_time_scale: Optional[str] = Field(
        default=None,
        description="'BMJD_TDB' if barycentric conversion succeeded, else None."
    )
    time_conversion_method: str = Field(
        default="NATIVE_UNCONVERTED",
        description="e.g. 'ASTROPY_BARYCORR_V1' or 'NATIVE_UNCONVERTED'."
    )
    t_obs_bmjd: Optional[float] = Field(
        default=None,
        description="Deprecated alias for t_converted_bmjd. Strictly None unless barycentric correction is applied."
    )
    t_pub_utc: Optional[datetime] = Field(default=None, description="ISO 8601 UTC timestamp when packet was published.")
    t_err_days: float = Field(default=0.0001, description="Exposure duration / timing uncertainty in days.")

    @property
    def observation_time(self) -> float:
        """
        Returns numeric timestamp: t_converted_bmjd if computed, else t_native.
        WARNING: This numeric value MUST NOT be assumed to be BMJD(TDB) without inspecting
        `is_barycentric_converted`. For precision timing, call `get_time_coordinate()`.
        """
        return self.t_converted_bmjd if self.t_converted_bmjd is not None else self.t_native

    @property
    def is_barycentric_converted(self) -> bool:
        """Returns True strictly when verified barycentric conversion to BMJD(TDB) was executed."""
        return self.t_converted_bmjd is not None

    def get_time_coordinate(self) -> Tuple[float, str]:
        """
        Returns the coupled (timestamp, time_scale) tuple.
        Prevents scientific analysis code from consuming a naked numeric timestamp without its time scale.
        """
        if self.t_converted_bmjd is not None:
            return self.t_converted_bmjd, self.converted_time_scale or "BMJD_TDB"
        return self.t_native, self.native_time_scale

    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="ICRS Right Ascension in degrees.")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="ICRS Declination in degrees.")
    astrometric_sigma_arcsec: float = Field(default=0.05, gt=0.0, description="1-sigma astrometric uncertainty.")

    # 3. Photometric Measurement & Semantics
    native_passband: str = Field(..., description="Native survey passband (e.g. 'ZTF_g', 'ATLAS_c', 'r_DECam').")
    obs_type: ObservationType = Field(default=ObservationType.DIFFERENCE_FORCED, description="Observation construction type.")
    
    # Crucial invariant: flux can be negative in forced/difference photometry
    measured_flux_ujy: float = Field(..., description="Standardized flux density in microJanskys (AB Z0 = 23.900).")
    measurement_sigma_ujy: float = Field(..., gt=0.0, description="1-sigma heteroscedastic uncertainty in microJanskys (must be strictly > 0).")
    upper_limit_ujy: Optional[float] = Field(default=None, description="5-sigma limiting flux for non-detections.")
    is_positive_detection_snr5: bool = Field(..., description="True if flux > 0 and flux / sigma >= 5.0.")

    # 4. Calibration & Transformation
    zero_point_ab: Optional[float] = Field(default=23.900, description="Zero-point magnitude on AB system.")
    zero_point_err: Optional[float] = Field(default=None, description="Zero-point calibration uncertainty.")
    photometric_group_id: Optional[str] = Field(default=None, description="Group ID for shared calibration covariance.")
    transformation_state: TransformationState = Field(
        default=TransformationState.NATIVE_PASSBAND_UNTRANSFORMED,
        description="Harmonization transformation state."
    )
    
    # 5. Quality & Fail-Closed Status
    native_flags: int = Field(default=0, description="Native survey reduction bitmask.")
    quality_bitmask: int = Field(default=0, description="Standardized quality bitmask (0 = clean).")
    is_clean: bool = Field(default=True, description="True if observation passed all provider quality cuts.")
    quarantine_reason: Optional[str] = Field(default=None, description="Reason if quarantined; None if clean.")
