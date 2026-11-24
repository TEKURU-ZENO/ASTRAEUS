"""
ASTRAEUS Sequential Scientific Feedback - Calibrated Observation Models (WS7.6-A).
Defines schemas for calibrated astronomical measurements, censored non-detection upper limits,
measurement provenance, and calibration metadata across multi-facility instruments.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class MeasurementType(str, Enum):
    """Observational measurement category."""
    POINT_DETECTION = "POINT_DETECTION"              # Significant detection (SNR >= 5.0)
    UPPER_LIMIT_CENSORED = "UPPER_LIMIT_CENSORED"    # Non-detection / 3-sigma or 5-sigma upper limit
    SPECTRUM_SUMMARY = "SPECTRUM_SUMMARY"            # Extracted continuum / emission line feature
    POLARIMETRY = "POLARIMETRY"                      # Linear/circular polarization measurement


class CalibrationStatus(str, Enum):
    """Verification grade of photometric and astrometric calibration."""
    CALIBRATED_NOMINAL = "CALIBRATED_NOMINAL"        # Fully validated against reference catalogs
    CALIBRATION_SUSPECT = "CALIBRATION_SUSPECT"      # Discrepant zero point or unmodeled seeing loss
    CALIBRATION_FAILED = "CALIBRATION_FAILED"        # Corrupted flatfield, bad tracking, saturated
    UNCALIBRATED = "UNCALIBRATED"                    # Raw counts without absolute flux scale


class CalibratedObservation(BaseModel):
    """
    Canonical representation of a calibrated multi-facility observation payload.
    Supports optical, NIR, space UV/X-ray, and radio regimes.
    """
    observation_id: str = Field(..., description="Unique immutable observation identifier")
    target_id: str
    target_ra_deg: float = Field(..., ge=0.0, lt=360.0)
    target_dec_deg: float = Field(..., ge=-90.0, le=90.0)
    facility_id: str
    instrument_id: str
    filter_id: str
    measurement_type: MeasurementType = MeasurementType.POINT_DETECTION
    epoch_mjd: float = Field(..., description="Mid-point observation epoch in MJD")
    
    # Photometric measurements
    apparent_mag: Optional[float] = Field(default=None, description="Apparent magnitude in AB system")
    apparent_mag_err: Optional[float] = Field(default=None, description="1-sigma photometric uncertainty")
    measured_flux_njy: Optional[float] = None
    measured_flux_err_njy: Optional[float] = None

    # Censored upper limits (critical for non-detections!)
    limiting_mag_5sigma: Optional[float] = Field(default=None, description="5-sigma limiting magnitude")
    limiting_flux_njy: Optional[float] = None

    # Observation environment & telemetry
    exposure_time_sec: float = Field(..., gt=0.0)
    airmass: float = Field(default=1.0, ge=1.0, le=5.0)
    seeing_fwhm_arcsec: float = Field(default=0.8, ge=0.0)
    
    # Calibration & Systematic Error Lineage
    calibration_status: CalibrationStatus = CalibrationStatus.CALIBRATED_NOMINAL
    calibration_reference_catalog: Optional[str] = "Gaia_DR3"
    calibration_zero_point: Optional[float] = None
    nominal_zero_point: Optional[float] = None
    pipeline_version: str = "astraeus_photometry_v1.0"
    instrument_flags: List[str] = Field(default_factory=list)
    
    # Correlation & Revision Tracking
    raw_data_hash_sha256: str = Field(..., description="SHA-256 fingerprint of raw telemetry/FITS payload")
    is_reprocessed_version_of: Optional[str] = Field(default=None, description="Parent observation ID if reprocessed")
    shared_calibration_group_id: Optional[str] = Field(default=None, description="ID identifying shared calibration reference")
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_detection(self) -> bool:
        return self.measurement_type == MeasurementType.POINT_DETECTION and self.apparent_mag is not None

    def is_upper_limit(self) -> bool:
        return self.measurement_type == MeasurementType.UPPER_LIMIT_CENSORED and self.limiting_mag_5sigma is not None

    @classmethod
    def compute_payload_hash(cls, raw_content: str | bytes) -> str:
        """Derives SHA-256 hash from raw observation bytes or text."""
        if isinstance(raw_content, str):
            raw_content = raw_content.encode("utf-8")
        return hashlib.sha256(raw_content).hexdigest()
