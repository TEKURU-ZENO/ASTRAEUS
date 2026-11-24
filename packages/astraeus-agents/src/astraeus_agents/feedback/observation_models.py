"""
ASTRAEUS Sequential Scientific Feedback - Calibrated Observation Models (WS7.6-A).
Defines schemas for calibrated astronomical measurements, censored non-detection upper limits,
measurement provenance, and calibration metadata across multi-facility instruments.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import hashlib
import json
import numpy as np
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


class CalibratedSpectrum(BaseModel):
    """
    Medium/high-resolution 1D calibrated astronomical spectrum (e.g. VLT X-shooter).
    Preserves exact spectral wavelength grid, flux density, error array, telluric masking,
    and enables synthetic broad-band passband projection.
    """
    spectrum_id: str
    target_id: str
    target_ra_deg: float = Field(..., ge=0.0, lt=360.0)
    target_dec_deg: float = Field(..., ge=-90.0, le=90.0)
    facility_id: str = "VLT"
    instrument_id: str = "XSHOOTER"
    arm: str = "NIR"  # "UVB", "VIS", "NIR"
    spectral_resolution_R: float = 5000.0
    epoch_mjd: float
    
    # 1D spectral arrays
    wavelengths_angstrom: List[float]
    flux_density_cgs: List[float]       # F_lambda in erg / s / cm^2 / Angstrom
    flux_density_err_cgs: List[float]   # 1-sigma uncertainty
    mask: List[bool]                    # True = valid physical pixel; False = masked sky/telluric absorption
    
    # Metadata
    telluric_corrected: bool = True
    flux_calibrated: bool = True
    pipeline_version: str = "xshooter_reduction_v3.4"
    raw_data_hash_sha256: str
    provenance_metadata: Dict[str, Any] = Field(default_factory=dict)

    def synthesize_broadband_magnitude(
        self,
        filter_id: str,
        central_wl_angstrom: float,
        fwhm_angstrom: float,
    ) -> CalibratedObservation:
        r"""
        Projects high-resolution 1D calibrated spectrum onto an integrated passband filter.
        Computes synthetic AB magnitude via transmission curve integration:
            f_nu = \int F_lambda lambda T(lambda) dlambda / (c \int (T(lambda)/lambda) dlambda)
            m_AB = -2.5 * log10(f_nu) - 48.60
        """
        wls = np.array(self.wavelengths_angstrom)
        flam = np.array(self.flux_density_cgs)
        flam_err = np.array(self.flux_density_err_cgs)
        valid = np.array(self.mask, dtype=bool)

        # Gaussian transmission profile for filter
        sigma_filter = fwhm_angstrom / 2.355
        transmission = np.exp(-0.5 * ((wls - central_wl_angstrom) / max(1.0, sigma_filter)) ** 2)
        
        # Apply mask
        weight = transmission * valid
        weight_sum = np.sum(weight)
        if weight_sum <= 1e-12:
            mean_flam = 1e-18
            mean_err = 1e-19
        else:
            mean_flam = float(np.sum(flam * weight) / weight_sum)
            mean_err = float(np.sqrt(np.sum((weight * flam_err) ** 2)) / weight_sum)

        # Convert F_lambda to F_nu at central wavelength:
        # F_nu [erg/s/cm^2/Hz] = F_lambda * lambda^2 / c
        c_cgs = 2.99792458e18  # Angstrom / s
        f_nu = max(1e-35, mean_flam * (central_wl_angstrom ** 2) / c_cgs)
        f_nu_err = max(1e-36, mean_err * (central_wl_angstrom ** 2) / c_cgs)
        
        # Convert to AB magnitude: m_AB = -2.5 * log10(f_nu) - 48.60
        m_ab = float(-2.5 * math.log10(f_nu) - 48.60)
        m_ab_err = float(1.0857 * (f_nu_err / f_nu))
        f_njy = f_nu / 1e-32

        return CalibratedObservation(
            observation_id=f"syn_{self.spectrum_id}_{filter_id.lower()}",
            target_id=self.target_id,
            target_ra_deg=self.target_ra_deg,
            target_dec_deg=self.target_dec_deg,
            facility_id=self.facility_id,
            instrument_id=self.instrument_id,
            filter_id=filter_id,
            measurement_type=MeasurementType.POINT_DETECTION,
            epoch_mjd=self.epoch_mjd,
            apparent_mag=round(m_ab, 3),
            apparent_mag_err=round(max(0.01, min(0.50, m_ab_err)), 4),
            measured_flux_njy=round(f_njy, 2),
            exposure_time_sec=600.0,
            calibration_status=CalibrationStatus.CALIBRATED_NOMINAL,
            raw_data_hash_sha256=self.raw_data_hash_sha256,
            provenance_metadata={
                "source_spectrum_id": self.spectrum_id,
                "spectral_arm": self.arm,
                "central_wavelength_angstrom": central_wl_angstrom,
                "filter_fwhm_angstrom": fwhm_angstrom,
                "spectral_resolution_R": self.spectral_resolution_R,
                "projected_from_spectrum": True,
            },
        )
