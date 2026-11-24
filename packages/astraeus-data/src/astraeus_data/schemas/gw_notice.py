"""
ASTRAEUS Gravitational-Wave Notice Schema (WS7.1).
Defines Pydantic models for LIGO/Virgo/KAGRA (LVK) GCN Circular and VOEvent notices.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class GwNoticePacket(BaseModel):
    """Calibrated LVK multi-messenger gravitational-wave alert packet."""
    superevent_id: str = Field(..., description="Unique LVK candidate identifier (e.g., S261124a)")
    notice_type: Literal["EARLY_WARNING", "PRELIMINARY", "INITIAL", "UPDATE", "RETRACTION"] = Field(
        ..., description="Alert pipeline latency classification"
    )
    time_utc: str = Field(..., description="Merger GPS time converted to ISO UTC string")
    event_time_mjd: float = Field(..., description="Merger event time in Modified Julian Date")
    false_alarm_rate_hz: float = Field(..., gt=0.0, description="Estimated False Alarm Rate in Hertz")
    far_yr: float = Field(..., gt=0.0, description="Estimated FAR inverse frequency (events per year)")
    skymap_url: Optional[str] = Field(None, description="FITS skymap download location (HEALPix FITS)")
    area_90_deg2: float = Field(..., ge=0.0, description="90% credible sky localization area in square degrees")
    distance_mean_mpc: float = Field(..., ge=0.0, description="Luminosity distance posterior mean in Megaparsecs")
    distance_std_mpc: float = Field(..., ge=0.0, description="Luminosity distance 1-sigma uncertainty (Mpc)")
    
    # Astrophysical classification probabilities (sum to ~1.0)
    prob_bns: float = Field(..., ge=0.0, le=1.0, description="Binary Neutron Star probability")
    prob_nsbh: float = Field(..., ge=0.0, le=1.0, description="Neutron Star - Black Hole probability")
    prob_bbh: float = Field(..., ge=0.0, le=1.0, description="Binary Black Hole probability")
    prob_terrestrial: float = Field(..., ge=0.0, le=1.0, description="Terrestrial / instrumental noise probability")
    prob_has_ns: float = Field(..., ge=0.0, le=1.0, description="Probability that at least one component is a neutron star (< 3 M_sun)")
    prob_has_remnant: float = Field(..., ge=0.0, le=1.0, description="Probability of non-zero ejecta remnant outside horizon")

    # Spatial centroid (approximate peak of probability density)
    central_ra_deg: Optional[float] = Field(None, ge=0.0, lt=360.0, description="Approximate RA centroid (deg)")
    central_dec_deg: Optional[float] = Field(None, ge=-90.0, le=90.0, description="Approximate Dec centroid (deg)")

    @property
    def is_electromagnetic_counterpart_promising(self) -> bool:
        """Determines if notice warrants immediate optical / ToO follow-up."""
        return (
            self.notice_type != "RETRACTION"
            and self.prob_terrestrial < 0.10
            and self.prob_has_ns > 0.50
            and self.prob_has_remnant > 0.20
            and self.area_90_deg2 < 1000.0
        )

    def compute_payload_hash(self) -> str:
        """Calculates deterministic SHA-256 digest of the GW notice."""
        canonical = {
            "superevent_id": self.superevent_id,
            "notice_type": self.notice_type,
            "mjd": self.event_time_mjd,
            "area_90": self.area_90_deg2,
            "dist": self.distance_mean_mpc,
            "prob_bns": self.prob_bns,
            "prob_nsbh": self.prob_nsbh,
        }
        serialized = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
