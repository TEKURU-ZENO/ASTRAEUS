"""
ASTRAEUS Pan-STARRS1 (PS1) Data Contract Module.
Preserves raw PS1 3pi Survey measurements across g_P1, r_P1, i_P1, z_P1, y_P1 filters,
photometric zero points, instrumental bitmasks, and upper limits.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib
from pydantic import BaseModel, Field, field_validator


class PanstarrsFilter(str, Enum):
    G_P1 = "g_P1"
    R_P1 = "r_P1"
    I_P1 = "i_P1"
    Z_P1 = "z_P1"
    Y_P1 = "y_P1"


# Effective wavelengths (Angstroms) from Tonry et al. 2012
PANSTARRS_EFFECTIVE_WAVELENGTHS = {
    PanstarrsFilter.G_P1: 4866.0,
    PanstarrsFilter.R_P1: 6215.0,
    PanstarrsFilter.I_P1: 7545.0,
    PanstarrsFilter.Z_P1: 8679.0,
    PanstarrsFilter.Y_P1: 9633.0,
}


class PanstarrsMeasurement(BaseModel):
    """
    Individual photometric epoch from Pan-STARRS1 Medium Deep or 3pi survey.
    Preserves detection status, calibrated PSF magnitude, instrumental flux, and quality flags.
    """
    mjd: float = Field(..., ge=45000.0, le=75000.0, description="Modified Julian Date of exposure mid-point.")
    filter_band: PanstarrsFilter
    is_detection: bool = Field(..., description="True if source detected >= 5 sigma; False for upper limit.")
    psf_mag: Optional[float] = Field(default=None, description="Calibrated AB magnitude; None if non-detection.")
    psf_mag_err: Optional[float] = Field(default=None, ge=0.0, description="1-sigma uncertainty on psf_mag.")
    psf_flux_ujy: float = Field(..., description="Calibrated flux in microJanskys (uJy). Negative or near-zero for upper limits.")
    psf_flux_err_ujy: float = Field(..., gt=0.0, description="1-sigma flux uncertainty in microJanskys (uJy).")
    limiting_mag: float = Field(..., description="5-sigma limiting AB magnitude of the exposure.")
    zero_point_ab: float = Field(default=25.0, description="AB zero point used for flux calibration.")
    sky_background: float = Field(default=0.0, description="Local sky background counts.")
    psf_qf: float = Field(default=1.0, ge=0.0, le=1.0, description="PSF quality factor: fraction of unmasked pixels in PSF core.")
    info_flags: int = Field(default=0, ge=0, description="PS1 bitmask of processing and defect flags.")

    @property
    def is_clean(self) -> bool:
        """Determines if measurement is free from defects, blending, and diffraction spikes."""
        return self.psf_qf >= 0.85 and (self.info_flags & 0x10080000) == 0


class PanstarrsLightCurve(BaseModel):
    """
    Complete multi-epoch Pan-STARRS1 time series for a transient or variable source.
    """
    obj_id: str = Field(..., description="Canonical Pan-STARRS object identifier (e.g., 'PSO J123.4567+45.6789').")
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    epochs: List[PanstarrsMeasurement] = Field(default_factory=list)
    raw_packet_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.raw_packet_hash:
            payload = f"{self.obj_id}:{self.ra_deg:.6f}:{self.dec_deg:.6f}:{len(self.epochs)}"
            self.raw_packet_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def num_detections(self) -> int:
        return sum(1 for ep in self.epochs if ep.is_detection)

    @property
    def num_upper_limits(self) -> int:
        return sum(1 for ep in self.epochs if not ep.is_detection)
