"""
ASTRAEUS Dark Energy Camera (DECam) Data Contract Module.
Preserves raw DECam measurements across u_DECam, g_DECam, r_DECam, i_DECam, z_DECam, Y_DECam,
NOIRLab Community Pipeline (CP) flags, seeing FWHM, zero points, and upper limits.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import hashlib
from pydantic import BaseModel, Field


class DecamFilter(str, Enum):
    U_DECAM = "u_DECam"
    G_DECAM = "g_DECam"
    R_DECAM = "r_DECam"
    I_DECAM = "i_DECam"
    Z_DECAM = "z_DECam"
    Y_DECAM = "Y_DECam"


# Effective wavelengths (Angstroms) from Flaugher et al. 2015 / Abbott et al. 2018
DECAM_EFFECTIVE_WAVELENGTHS = {
    DecamFilter.U_DECAM: 3560.0,
    DecamFilter.G_DECAM: 4720.0,
    DecamFilter.R_DECAM: 6415.0,
    DecamFilter.I_DECAM: 7835.0,
    DecamFilter.Z_DECAM: 9260.0,
    DecamFilter.Y_DECAM: 10095.0,
}


class DecamMeasurement(BaseModel):
    """
    Individual photometric epoch from Dark Energy Camera (DECam / DES / DELVE).
    Preserves exposure conditions, seeing FWHM, CP bitmasks, and upper limits.
    """
    mjd: float = Field(..., ge=45000.0, le=75000.0, description="Modified Julian Date mid-exposure.")
    filter_band: DecamFilter
    is_detection: bool = Field(..., description="True if source >= 5 sigma; False for upper limit.")
    mag: Optional[float] = Field(default=None, description="Calibrated AB magnitude; None for non-detections.")
    mag_err: Optional[float] = Field(default=None, ge=0.0, description="1-sigma uncertainty on mag.")
    flux_ujy: float = Field(..., description="Calibrated flux in microJanskys (uJy). Negative or near-zero for non-detections.")
    flux_err_ujy: float = Field(..., gt=0.0, description="1-sigma flux uncertainty in microJanskys (uJy).")
    limiting_mag: float = Field(..., description="5-sigma limiting AB magnitude of the exposure.")
    zero_point_ab: float = Field(default=30.0, description="Community pipeline photometric zero point.")
    fwhm_arcsec: float = Field(..., ge=0.1, le=10.0, description="Delivered seeing FWHM in arcseconds.")
    airmass: float = Field(default=1.2, ge=1.0, le=3.5, description="Airmass of observation.")
    exposure_time_sec: float = Field(..., gt=0.0, description="Exposure duration in seconds.")
    cp_flags: int = Field(default=0, ge=0, description="Community Pipeline quality bitmask (0 = uncorrupted).")

    @property
    def is_clean(self) -> bool:
        """Determines if measurement meets science-quality thresholds without saturated or bad pixels."""
        return self.cp_flags == 0 and self.fwhm_arcsec <= 2.2 and self.airmass <= 2.0


class DecamLightCurve(BaseModel):
    """
    Complete multi-epoch DECam time series for a transient or variable entity.
    """
    entity_id: str = Field(..., description="Canonical DECam target identifier.")
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    epochs: List[DecamMeasurement] = Field(default_factory=list)
    raw_packet_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.raw_packet_hash:
            payload = f"{self.entity_id}:{self.ra_deg:.6f}:{self.dec_deg:.6f}:{len(self.epochs)}"
            self.raw_packet_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def num_detections(self) -> int:
        return sum(1 for ep in self.epochs if ep.is_detection)

    @property
    def num_upper_limits(self) -> int:
        return sum(1 for ep in self.epochs if not ep.is_detection)
