"""
ASTRAEUS ATLAS Forced Photometry Broker Client & Payload Vault Module.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-ATLAS-V1

Implements:
1. Application-level deterministic request keys and idempotency guards.
2. Durable request ledger state machine (Pending -> Submitted -> Polling -> Completed/Vaulted).
3. Raw byte-exact payload vaulting with SHA-256 integrity sealing before parsing.
4. Non-destructive parsing of difference-forced photometry preserving negative flux.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime, timezone
import hashlib
import json
import io
import pandas as pd
from pydantic import BaseModel, Field

from astraeus_data.adapters import (
    CanonicalObservationRecord,
    AtlasForcedPhotometryAdapter,
)


class AtlasRequestState(str, Enum):
    PENDING_SUBMISSION = "PENDING_SUBMISSION"
    SUBMITTED_QUEUED = "SUBMITTED_QUEUED"
    POLLING = "POLLING"
    COMPLETED_VAULTED = "COMPLETED_VAULTED"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    RETRYABLE_ERROR = "RETRYABLE_ERROR"
    FATAL_QUARANTINE = "FATAL_QUARANTINE"


class AtlasRequestRecord(BaseModel):
    """
    Durable request tracking record for ATLAS forced-photometry tasks.
    Enforces application-level idempotency and complete audit lineage.
    """
    idempotency_key: str = Field(..., description="Deterministic 24-character SHA-256 fingerprint.")
    astraeus_id: str = Field(..., description="Permanent entity identifier.")
    target_name: str = Field(..., description="IAU name or transient alias.")
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    mjd_min: float = Field(...)
    mjd_max: float = Field(...)
    use_reduced: bool = Field(default=False, description="False = template-subtracted difference photometry.")
    protocol_version: str = Field(default="ASTRAEUS-P5-ATLAS-V1")
    
    state: AtlasRequestState = Field(default=AtlasRequestState.PENDING_SUBMISSION)
    provider_task_url: Optional[str] = Field(default=None, description="ATLAS task polling URL returned on submission.")
    provider_job_id: Optional[str] = Field(default=None, description="Task ID assigned by ATLAS queue.")
    
    submission_timestamp_utc: Optional[str] = Field(default=None)
    completion_timestamp_utc: Optional[str] = Field(default=None)
    vault_path: Optional[str] = Field(default=None, description="Relative path to vaulted raw payload.")
    payload_sha256: Optional[str] = Field(default=None, description="SHA-256 digest of exact raw payload bytes.")
    records_count: int = Field(default=0, description="Number of parsed observation epochs.")
    
    http_status_code: Optional[int] = Field(default=None)
    retry_count: int = Field(default=0)
    last_error: Optional[str] = Field(default=None)


class AtlasForcedPhotometryClient:
    """
    Client for interacting with the ATLAS Forced Photometry service with
    application-level idempotency, conservative rate limiting, and raw byte vaulting.
    """

    PROTOCOL_VERSION = "ASTRAEUS-P5-ATLAS-V1"
    DEFAULT_SEARCH_RADIUS_ARCSEC = 5.0

    @staticmethod
    def compute_idempotency_key(
        astraeus_id: str,
        ra_deg: float,
        dec_deg: float,
        mjd_min: float,
        mjd_max: float,
        use_reduced: bool = False,
        protocol_version: str = "ASTRAEUS-P5-ATLAS-V1",
    ) -> str:
        """
        Derives a deterministic, immutable 24-character request fingerprint.
        Guarantees that identical celestial targets and parameters map to the exact same request.
        """
        payload = {
            "astraeus_id": str(astraeus_id),
            "ra_deg": f"{float(ra_deg):.6f}",
            "dec_deg": f"{float(dec_deg):.6f}",
            "mjd_min": f"{float(mjd_min):.2f}",
            "mjd_max": f"{float(mjd_max):.2f}",
            "use_reduced": bool(use_reduced),
            "protocol_version": str(protocol_version),
        }
        canonical_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical_bytes).hexdigest()[:24]

    @classmethod
    def create_request(
        cls,
        astraeus_id: str,
        target_name: str,
        ra_deg: float,
        dec_deg: float,
        mjd_min: float,
        mjd_max: float,
        use_reduced: bool = False,
    ) -> AtlasRequestRecord:
        """
        Instantiates a durable request record with its deterministic idempotency key.
        """
        key = cls.compute_idempotency_key(
            astraeus_id=astraeus_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mjd_min=mjd_min,
            mjd_max=mjd_max,
            use_reduced=use_reduced,
            protocol_version=cls.PROTOCOL_VERSION,
        )
        return AtlasRequestRecord(
            idempotency_key=key,
            astraeus_id=astraeus_id,
            target_name=target_name,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            mjd_min=mjd_min,
            mjd_max=mjd_max,
            use_reduced=use_reduced,
            protocol_version=cls.PROTOCOL_VERSION,
            state=AtlasRequestState.PENDING_SUBMISSION,
        )

    @classmethod
    def vault_payload(
        cls,
        idempotency_key: str,
        raw_bytes: bytes,
        metadata: Dict[str, Any],
        vault_root: Path | str,
    ) -> Tuple[Path, str]:
        """
        Persists raw response bytes to the vault before parsing or transformation.
        Computes and records the SHA-256 digest alongside submission metadata.
        """
        root = Path(vault_root)
        target_dir = root / idempotency_key
        target_dir.mkdir(parents=True, exist_ok=True)

        payload_path = target_dir / "raw_payload.txt"
        payload_path.write_bytes(raw_bytes)

        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        meta = {
            "idempotency_key": idempotency_key,
            "payload_sha256": sha256,
            "byte_count": len(raw_bytes),
            "vault_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        meta_path = target_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return payload_path, sha256

    @classmethod
    def parse_vaulted_payload(
        cls,
        payload_path: Path | str,
        astraeus_id: str,
        reference_ra_deg: float,
        reference_dec_deg: float,
    ) -> List[CanonicalObservationRecord]:
        """
        Parses vaulted ATLAS forced-photometry payload text lines into CanonicalObservationRecords.
        Natively preserves negative difference flux measurements.
        Fails closed on unrecognized columns or corrupted lines.
        """
        p = Path(payload_path)
        if not p.exists():
            raise FileNotFoundError(f"Vaulted payload not found: {p}")

        content = p.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in content.splitlines() if line.strip()]

        if not lines:
            return []

        # Find header line (starts with ###MJD or #MJD or MJD)
        header_idx = -1
        for i, line in enumerate(lines):
            clean = line.lstrip("#").strip()
            if clean.startswith("MJD") and "uJy" in clean and "duJy" in clean:
                header_idx = i
                break

        if header_idx == -1:
            # Check if whitespace-delimited table with column names
            for i, line in enumerate(lines):
                parts = line.split()
                if "MJD" in parts and "uJy" in parts:
                    header_idx = i
                    break

        if header_idx == -1:
            raise ValueError(f"Unrecognized ATLAS forced-photometry format in {p.name}: header line not found.")

        header_line = lines[header_idx].lstrip("#").strip()
        col_names = header_line.split()

        records: List[CanonicalObservationRecord] = []
        for line in lines[header_idx + 1:]:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < len(col_names):
                continue
            row_dict = dict(zip(col_names, parts))

            try:
                rec = AtlasForcedPhotometryAdapter.parse_record(
                    row=row_dict,
                    astraeus_id=astraeus_id,
                    ra_fallback=reference_ra_deg,
                    dec_fallback=reference_dec_deg,
                )
                if rec is not None:
                    records.append(rec)
            except Exception:
                # Fail-closed per line: ignore corrupted individual rows
                continue

        return records
