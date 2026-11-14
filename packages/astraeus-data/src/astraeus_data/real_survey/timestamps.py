"""
ASTRAEUS Real-Survey 5-Timestamp & Operational Availability Semantics (P0.3).

Enforces strict separation among the five distinct astronomical / pipeline timestamps:
1. `t_observation` (`t_obs_mjd_utc`, `t_obs_bmjd_tdb`): physical telescope exposure midpoint.
2. `t_publication` (`t_pub_utc`): external broker/alert stream publication timestamp (ONLY when
   independently observed; NEVER fabricated from `t_observation` or `t_ingestion`).
3. `t_ingestion` (`t_ingest_utc`): local pipeline / ASTRAEUS ingestion timestamp.
4. `t_classification` (`t_class_utc`): external classification report timestamp (when available).
5. `t_retrieval` (`t_retrieval_utc`): local raw-snapshot HTTP retrieval / download timestamp.

Epochs lacking an independently known `t_pub_utc` carry explicit
`AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN` and `has_operational_pub_time=False`, isolating
archival retrospective evaluation from real-time operational causal-cutoff evaluation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Iterable, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from astraeus_data.corpus.time_conversion import compute_bmjd_tdb


class AvailabilityStatus(str, Enum):
    """Explicit operational alert publication-time knowledge status."""

    KNOWN_OPERATIONAL = "KNOWN_OPERATIONAL"
    AVAILABILITY_TIME_UNKNOWN = "AVAILABILITY_TIME_UNKNOWN"


class TimestampBundle(BaseModel):
    """Immutable 5-timestamp record preserving physical vs operational vs archival semantics."""

    model_config = ConfigDict(frozen=True)

    t_obs_mjd_utc: Optional[float] = None
    t_obs_bmjd_tdb: Optional[float] = None
    t_pub_utc: Optional[str] = None
    t_ingest_utc: Optional[str] = None
    t_class_utc: Optional[str] = None
    t_retrieval_utc: Optional[str] = None
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN

    @model_validator(mode="after")
    def _validate_availability_invariant(self) -> "TimestampBundle":
        pub = self.t_pub_utc.strip() if isinstance(self.t_pub_utc, str) else None
        if pub == "":
            object.__setattr__(self, "t_pub_utc", None)
            pub = None

        if pub is None and self.availability_status == AvailabilityStatus.KNOWN_OPERATIONAL:
            raise ValueError(
                "KNOWN_OPERATIONAL requires an independently observed `t_pub_utc`; "
                "use AVAILABILITY_TIME_UNKNOWN when alert publication time is unknown."
            )
        if pub is not None and self.availability_status == AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN:
            raise ValueError(
                "AVAILABILITY_TIME_UNKNOWN must not carry a fabricated `t_pub_utc`."
            )
        return self

    @property
    def is_operationally_causal_eligible(self) -> bool:
        """True iff this record has a verified operational alert publication timestamp."""
        return (
            self.availability_status == AvailabilityStatus.KNOWN_OPERATIONAL
            and self.t_pub_utc is not None
        )

    @property
    def is_archival_retrospective_eligible(self) -> bool:
        """True iff this record has a valid physical exposure epoch for retrospective use."""
        return self.t_obs_bmjd_tdb is not None or self.t_obs_mjd_utc is not None


def build_observation_timestamps(
    mjd_utc: Optional[float],
    ra_deg: float,
    dec_deg: float,
    t_pub_utc: Optional[str] = None,
    t_ingest_utc: Optional[str] = None,
    t_class_utc: Optional[str] = None,
    t_retrieval_utc: Optional[str] = None,
    observatory: str = "ZTF",
) -> TimestampBundle:
    """
    Constructs a verified `TimestampBundle` for an observation without ever equating
    `t_observation`, `t_publication`, or `t_ingestion`.
    """
    bmjd = (
        compute_bmjd_tdb(float(mjd_utc), float(ra_deg), float(dec_deg), observatory=observatory)
        if mjd_utc is not None
        else None
    )
    cleaned_pub = t_pub_utc.strip() if isinstance(t_pub_utc, str) and t_pub_utc.strip() else None
    status = (
        AvailabilityStatus.KNOWN_OPERATIONAL
        if cleaned_pub is not None
        else AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN
    )
    return TimestampBundle(
        t_obs_mjd_utc=float(mjd_utc) if mjd_utc is not None else None,
        t_obs_bmjd_tdb=bmjd,
        t_pub_utc=cleaned_pub,
        t_ingest_utc=t_ingest_utc,
        t_class_utc=t_class_utc,
        t_retrieval_utc=t_retrieval_utc,
        availability_status=status,
    )


def summarize_availability(records: Iterable[Any]) -> Dict[str, int]:
    """Counts known-operational vs AVAILABILITY_TIME_UNKNOWN records in an observation sequence."""
    known = 0
    unknown = 0
    for rec in records:
        has_pub = bool(getattr(rec, "has_operational_pub_time", False)) and bool(
            getattr(rec, "t_pub_utc", None)
        )
        if has_pub:
            known += 1
        else:
            unknown += 1
    return {
        "total_records": known + unknown,
        "known_operational_pub_time_count": known,
        "availability_time_unknown_count": unknown,
    }
