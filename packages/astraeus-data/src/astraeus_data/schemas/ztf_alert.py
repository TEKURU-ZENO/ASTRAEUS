"""
ASTRAEUS ZTF Alert Schema Contract.
Preserves raw alert packets, survey quality flags, detection vs upper-limit status,
and timescale standards (MJD / BJD_TDB).
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
import hashlib
import json


class ZtfCandidate(BaseModel):
    """
    Direct representation of a ZTF alert candidate table.
    Retains all survey-specific quality flags and telemetry.
    """
    jd: float = Field(..., description="Julian Date of observation.")
    fid: int = Field(..., description="Filter ID: 1=g, 2=r, 3=i.")
    pid: int = Field(..., description="Processing ID.")
    diffmaglim: float = Field(..., description="5-sigma limiting magnitude in difference image.")
    isdiffpos: Union[str, int, bool] = Field(..., description="'t' or 1 for positive difference flux, 'f' or 0 for negative.")
    ra: float = Field(..., ge=0.0, lt=360.0, description="Right Ascension in degrees (J2000).")
    dec: float = Field(..., ge=-90.0, le=90.0, description="Declination in degrees (J2000).")
    magpsf: Optional[float] = Field(default=None, description="PSF fit magnitude (None if upper limit).")
    sigmapsf: Optional[float] = Field(default=None, ge=0.0, description="1-sigma uncertainty of magpsf.")
    chipsf: Optional[float] = Field(default=None, description="Reduced chi-square of PSF fit.")
    fwhm: Optional[float] = Field(default=None, ge=0.0, description="Full Width at Half Maximum in pixels.")
    rb: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Real-Bogus score from random forest.")
    drb: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Deep Learning Real-Bogus score (Braai).")
    scorr: Optional[float] = Field(default=None, description="Peak correlation score in difference image.")

    @property
    def mjd(self) -> float:
        """Converts Julian Date (JD) to Modified Julian Date (MJD)."""
        return self.jd - 2400000.5

    @property
    def filter_name(self) -> str:
        """Maps numerical filter ID to canonical passband string."""
        mapping = {1: "ZTF_g", 2: "ZTF_r", 3: "ZTF_i"}
        return mapping.get(self.fid, f"ZTF_unknown_{self.fid}")

    @property
    def is_detection(self) -> bool:
        """Determines whether epoch is a genuine detection or non-detection upper limit."""
        return self.magpsf is not None and self.sigmapsf is not None and self.magpsf > 0.0


class ZtfAlertPacket(BaseModel):
    """
    ZTF Alert Packet contract preserving raw payload provenance.
    """
    schema_version: str = "astraeus:schema:ztf_alert:v1.0"
    candid: int = Field(..., description="Unique alert candidate ID (64-bit integer).")
    objectId: str = Field(..., description="Canonical ZTF object name (e.g. 'ZTF20acvhrgm').")
    candidate: ZtfCandidate = Field(..., description="Current trigger detection.")
    prv_candidates: List[ZtfCandidate] = Field(
        default_factory=list,
        description="Historical detections and upper limits within past 30 days."
    )
    raw_payload_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 fingerprint of the original raw JSON/AVRO packet."
    )
    raw_packet_json: Optional[str] = Field(
        default=None,
        description="Verbatim serialized raw packet for archival preservation."
    )

    def compute_payload_hash(self) -> str:
        """Computes deterministic SHA-256 of candid and objectId for idempotent replay."""
        content = f"{self.candid}:{self.objectId}:{self.candidate.jd}:{self.candidate.ra}:{self.candidate.dec}"
        h = hashlib.sha256(content.encode()).hexdigest()
        self.raw_payload_hash = h
        return h
