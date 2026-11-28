"""
ASTRAEUS Unified Scientific Event Ingestion Engine (WS7.1).
Multi-survey ingestion, normalization, and quarantine enforcement for:
1. Zakatman Transient Facility (ZTF) alerts
2. Vera C. Rubin Observatory (LSST) difference alerts
3. LIGO/Virgo/KAGRA (LVK) gravitational-wave GCN notices

Features:
- Cryptographic idempotency via raw SHA-256 payload hashing
- Live Gate 3 Quarantine Firewall enforcement (1,808 frozen HEALPix pixels)
- Normalized multi-messenger UnifiedScientificEvent container
- Dead-letter routing for malformed or quarantined packets

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import json
import hashlib
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Union, Literal

from astraeus_data.schemas.ztf_alert import ZtfAlertPacket
from astraeus_data.schemas.rubin_alert import RubinAlertPacket
from astraeus_data.schemas.gw_notice import GwNoticePacket
from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class IngestionStatus(str, Enum):
    INGESTED = "INGESTED"
    DUPLICATE_SKIPPED = "DUPLICATE_SKIPPED"
    QUARANTINED_COVENANT = "QUARANTINED_COVENANT"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"


@dataclass
class UnifiedScientificEvent:
    """Canonical multi-messenger event schema for Phase 7 autonomous triage and investigation."""
    event_id: str
    survey_origin: Literal["ZTF", "RUBIN", "LVK_GW"]
    ra_deg: float
    dec_deg: float
    healpix_pixel_nside32: int
    event_time_mjd: float
    photometry_points: List[Dict[str, Any]]
    event_metadata: Dict[str, Any]
    raw_payload_hash: str
    quarantine_cleared: bool = True
    is_too_promising: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventIngestionResult:
    status: IngestionStatus
    event_id: str
    raw_hash: str
    unified_event: Optional[UnifiedScientificEvent] = None
    error_message: Optional[str] = None


class ScientificEventIngestionEngine:
    """Production multi-survey ingestion engine with zero-touch quarantine protection."""

    def __init__(self, storage_root: Union[str, Path] = "data/events"):
        self.storage_root = Path(storage_root)
        self.raw_dir = self.storage_root / "raw"
        self.unified_dir = self.storage_root / "unified"
        self.quarantine_dir = self.storage_root / "quarantine"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.seen_hashes: set[str] = set()

    def _check_idempotency(self, payload_hash: str, event_id: str) -> Optional[EventIngestionResult]:
        if payload_hash in self.seen_hashes:
            return EventIngestionResult(
                status=IngestionStatus.DUPLICATE_SKIPPED,
                event_id=event_id,
                raw_hash=payload_hash,
            )
        return None

    def ingest_rubin_packet(self, packet: RubinAlertPacket) -> EventIngestionResult:
        """Ingests and normalizes a Rubin LSST difference alert."""
        payload_hash = packet.compute_payload_hash()
        event_id = f"evt:rubin:{packet.diaSource.diaSourceId}"

        if dup := self._check_idempotency(payload_hash, event_id):
            return dup

        ra = packet.diaSource.ra
        dec = packet.diaSource.dec

        # Quarantine verification
        try:
            hp_pix = QuarantineFirewall.assert_not_quarantined(ra, dec, identifier=event_id, context="rubin_ingestion")
        except QuarantineViolationError as e:
            # Route to dead-letter quarantine
            q_path = self.quarantine_dir / f"{packet.diaSource.diaSourceId}_rubin_quarantine.json"
            with open(q_path, "w", encoding="utf-8") as f:
                f.write(packet.model_dump_json(indent=2))
            return EventIngestionResult(
                status=IngestionStatus.QUARANTINED_COVENANT,
                event_id=event_id,
                raw_hash=payload_hash,
                error_message=str(e),
            )

        # Assemble light curve photometry points
        phot_points = []
        for src in packet.prvDiaSources:
            phot_points.append({
                "mjd": src.midPointTai,
                "filter": src.filterName,
                "filter_band": src.filterName,
                "flux_njy": src.psfFlux,
                "flux_err_njy": src.psfFluxErr,
                "mag": src.apparent_magnitude_ab,
                "mag_ab": src.apparent_magnitude_ab,
                "magerr": src.magnitude_err,
                "mag_err": src.magnitude_err,
                "snr": src.snr,
            })

        # Add current triggering detection
        phot_points.append({
            "mjd": packet.diaSource.midPointTai,
            "filter": packet.diaSource.filterName,
            "filter_band": packet.diaSource.filterName,
            "flux_njy": packet.diaSource.psfFlux,
            "flux_err_njy": packet.diaSource.psfFluxErr,
            "mag": packet.diaSource.apparent_magnitude_ab,
            "mag_ab": packet.diaSource.apparent_magnitude_ab,
            "magerr": packet.diaSource.magnitude_err,
            "mag_err": packet.diaSource.magnitude_err,
            "snr": packet.diaSource.snr,
        })

        is_too = packet.diaSource.snr > 15.0 and packet.diaSource.apparent_magnitude_ab < 21.0

        unified = UnifiedScientificEvent(
            event_id=event_id,
            survey_origin="RUBIN",
            ra_deg=ra,
            dec_deg=dec,
            healpix_pixel_nside32=hp_pix,
            event_time_mjd=packet.diaSource.midPointTai,
            photometry_points=phot_points,
            event_metadata={
                "alertId": packet.alertId,
                "diaObjectId": packet.diaSource.diaObjectId,
                "filter": packet.diaSource.filterName,
                "snr": packet.diaSource.snr,
            },
            raw_payload_hash=payload_hash,
            quarantine_cleared=True,
            is_too_promising=is_too,
        )

        # Persist raw and unified
        with open(self.raw_dir / f"{payload_hash}.json", "w", encoding="utf-8") as f:
            f.write(packet.model_dump_json(indent=2))
        with open(self.unified_dir / f"{event_id.replace(':', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(unified.to_dict(), f, indent=2)

        self.seen_hashes.add(payload_hash)
        return EventIngestionResult(
            status=IngestionStatus.INGESTED,
            event_id=event_id,
            raw_hash=payload_hash,
            unified_event=unified,
        )

    def ingest_gw_notice(self, notice: GwNoticePacket) -> EventIngestionResult:
        """Ingests and normalizes an LVK Gravitational Wave notice."""
        payload_hash = notice.compute_payload_hash()
        event_id = f"evt:gw:{notice.superevent_id}"

        if dup := self._check_idempotency(payload_hash, event_id):
            return dup

        ra = notice.central_ra_deg if notice.central_ra_deg is not None else 0.0
        dec = notice.central_dec_deg if notice.central_dec_deg is not None else 0.0

        # Check quarantine on centroid
        try:
            hp_pix = QuarantineFirewall.assert_not_quarantined(ra, dec, identifier=event_id, context="gw_notice_ingestion")
        except QuarantineViolationError as e:
            q_path = self.quarantine_dir / f"{notice.superevent_id}_gw_quarantine.json"
            with open(q_path, "w", encoding="utf-8") as f:
                f.write(notice.model_dump_json(indent=2))
            return EventIngestionResult(
                status=IngestionStatus.QUARANTINED_COVENANT,
                event_id=event_id,
                raw_hash=payload_hash,
                error_message=str(e),
            )

        unified = UnifiedScientificEvent(
            event_id=event_id,
            survey_origin="LVK_GW",
            ra_deg=ra,
            dec_deg=dec,
            healpix_pixel_nside32=hp_pix,
            event_time_mjd=notice.event_time_mjd,
            photometry_points=[],  # No direct optical photometry yet
            event_metadata={
                "notice_type": notice.notice_type,
                "area_90_deg2": notice.area_90_deg2,
                "distance_mean_mpc": notice.distance_mean_mpc,
                "distance_std_mpc": notice.distance_std_mpc,
                "prob_bns": notice.prob_bns,
                "prob_nsbh": notice.prob_nsbh,
                "prob_has_remnant": notice.prob_has_remnant,
                "skymap_url": notice.skymap_url,
            },
            raw_payload_hash=payload_hash,
            quarantine_cleared=True,
            is_too_promising=notice.is_electromagnetic_counterpart_promising,
        )

        with open(self.raw_dir / f"{payload_hash}.json", "w", encoding="utf-8") as f:
            f.write(notice.model_dump_json(indent=2))
        with open(self.unified_dir / f"{event_id.replace(':', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(unified.to_dict(), f, indent=2)

        self.seen_hashes.add(payload_hash)
        return EventIngestionResult(
            status=IngestionStatus.INGESTED,
            event_id=event_id,
            raw_hash=payload_hash,
            unified_event=unified,
        )

    def ingest_ztf_packet(self, packet: ZtfAlertPacket) -> EventIngestionResult:
        """Ingests and normalizes a ZTF optical alert packet."""
        payload_hash = packet.compute_payload_hash()
        candid_val = getattr(packet.candidate, "candid", None) or getattr(packet, "candid", 0)
        event_id = f"evt:ztf:{candid_val}"

        if dup := self._check_idempotency(payload_hash, event_id):
            return dup

        ra = packet.candidate.ra
        dec = packet.candidate.dec

        try:
            hp_pix = QuarantineFirewall.assert_not_quarantined(ra, dec, identifier=event_id, context="ztf_ingestion")
        except QuarantineViolationError as e:
            q_path = self.quarantine_dir / f"{candid_val}_ztf_quarantine.json"
            with open(q_path, "w", encoding="utf-8") as f:
                f.write(packet.model_dump_json(indent=2))
            return EventIngestionResult(
                status=IngestionStatus.QUARANTINED_COVENANT,
                event_id=event_id,
                raw_hash=payload_hash,
                error_message=str(e),
            )

        filter_map = {1: "g", 2: "r", 3: "i"}
        phot_points = []
        for prv in packet.prv_candidates:
            if prv.magpsf is not None and prv.sigmapsf is not None:
                phot_points.append({
                    "mjd": prv.jd - 2400000.5,
                    "filter": filter_map.get(prv.fid, f"fid_{prv.fid}"),
                    "mag_ab": prv.magpsf,
                    "mag_err": prv.sigmapsf,
                    "snr": 1.0857 / prv.sigmapsf if prv.sigmapsf > 0 else 0.0,
                })

        # Add triggering point
        phot_points.append({
            "mjd": packet.candidate.jd - 2400000.5,
            "filter": filter_map.get(packet.candidate.fid, f"fid_{packet.candidate.fid}"),
            "mag_ab": packet.candidate.magpsf,
            "mag_err": packet.candidate.sigmapsf,
            "snr": 1.0857 / packet.candidate.sigmapsf if packet.candidate.sigmapsf > 0 else 0.0,
        })

        is_too = packet.candidate.magpsf < 19.5 and packet.candidate.sigmapsf < 0.10

        unified = UnifiedScientificEvent(
            event_id=event_id,
            survey_origin="ZTF",
            ra_deg=ra,
            dec_deg=dec,
            healpix_pixel_nside32=hp_pix,
            event_time_mjd=packet.candidate.jd - 2400000.5,
            photometry_points=phot_points,
            event_metadata={
                "objectId": packet.objectId,
                "candid": candid_val,
                "drb": packet.candidate.drb,
            },
            raw_payload_hash=payload_hash,
            quarantine_cleared=True,
            is_too_promising=is_too,
        )

        with open(self.raw_dir / f"{payload_hash}.json", "w", encoding="utf-8") as f:
            f.write(packet.model_dump_json(indent=2))
        with open(self.unified_dir / f"{event_id.replace(':', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(unified.to_dict(), f, indent=2)

        self.seen_hashes.add(payload_hash)
        return EventIngestionResult(
            status=IngestionStatus.INGESTED,
            event_id=event_id,
            raw_hash=payload_hash,
            unified_event=unified,
        )
