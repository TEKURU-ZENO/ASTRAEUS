"""
ASTRAEUS Rubin Observatory / LSST Alert Packet Schema (WS7.1).
Defines Pydantic models for Rubin diaSource, diaObject, and diaForcedSource records.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class RubinDiaSource(BaseModel):
    """Calibrated difference image source detection from Rubin LSST."""
    diaSourceId: int = Field(..., description="Unique 64-bit integer ID of the detection")
    diaObjectId: Optional[int] = Field(None, description="Associated stationary or persistent object ID")
    midPointTai: float = Field(..., description="Mid-exposure International Atomic Time in MJD")
    filterName: Literal["u", "g", "r", "i", "z", "y"] = Field(..., description="Rubin optical filter band")
    ra: float = Field(..., ge=0.0, lt=360.0, description="Right Ascension (ICRS, deg)")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination (ICRS, deg)")
    raErr: float = Field(..., ge=0.0, description="Astrometric 1-sigma uncertainty in RA (deg)")
    decErr: float = Field(..., ge=0.0, description="Astrometric 1-sigma uncertainty in Dec (deg)")
    psfFlux: float = Field(..., description="Calibrated point-spread-function flux (nJy)")
    psfFluxErr: float = Field(..., gt=0.0, description="1-sigma uncertainty on psfFlux (nJy)")
    snr: float = Field(..., description="Signal-to-noise ratio of detection")
    isDiffPos: bool = Field(True, description="True if positive detection on difference image")

    @property
    def apparent_magnitude_ab(self) -> float:
        """Converts nanoJansky flux to AB magnitude: m_AB = -2.5 * log10(F / 3631 Jy)."""
        import math
        if self.psfFlux <= 0.0:
            return 99.0
        # 1 nJy = 1e-9 Jy; m_AB = -2.5 * log10(psfFlux * 1e-9 / 3631) = 31.4 - 2.5 * log10(psfFlux)
        return float(31.4 - 2.5 * math.log10(self.psfFlux))

    @property
    def magnitude_err(self) -> float:
        """Computes symmetric 1-sigma magnitude error from flux SNR."""
        import math
        if self.snr <= 0.0:
            return 9.9
        return float(1.0857 / self.snr)


class RubinDiaObject(BaseModel):
    """Aggregate multi-epoch astrophysical transient/variable object from Rubin LSST."""
    diaObjectId: int = Field(..., description="Unique 64-bit integer object identifier")
    ra: float = Field(..., ge=0.0, lt=360.0, description="Weighted mean RA (ICRS, deg)")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Weighted mean Dec (ICRS, deg)")
    nDiaSources: int = Field(..., ge=1, description="Number of associated difference detections")
    hostGalaxyId: Optional[int] = Field(None, description="Nearby associated host galaxy identifier")
    hostDistanceArcsec: Optional[float] = Field(None, ge=0.0, description="Angular separation from host center")


class RubinAlertPacket(BaseModel):
    """Complete Rubin LSST alert notification packet."""
    alertId: int = Field(..., description="Unique alert broadcast identifier")
    diaSource: RubinDiaSource = Field(..., description="Current triggering difference detection")
    diaObject: Optional[RubinDiaObject] = Field(None, description="Historical multi-epoch object metadata")
    prvDiaSources: List[RubinDiaSource] = Field(default_factory=list, description="Historical light-curve detections")

    def compute_payload_hash(self) -> str:
        """Calculates deterministic SHA-256 hash of the alert payload."""
        canonical_dict = {
            "alertId": self.alertId,
            "sourceId": self.diaSource.diaSourceId,
            "mjd": self.diaSource.midPointTai,
            "filter": self.diaSource.filterName,
            "flux": self.diaSource.psfFlux,
            "ra": self.diaSource.ra,
            "dec": self.diaSource.dec,
        }
        serialized = json.dumps(canonical_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
