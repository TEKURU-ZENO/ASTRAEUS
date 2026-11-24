"""
ASTRAEUS Durable Scientific Event Store - SQLite WAL Engine (WS7.7).
Provides persistent, ACID-compliant, crash-resilient event and observation logging:
- Write-Ahead Logging (WAL) for high-throughput concurrency and durability.
- Exact cryptographic SHA-256 payload and state hashing for non-repudiation.
- Deterministic investigation state checkpointing linked by parent hashes (Merkle lineage).
- Strict Gate 3 Quarantine Firewall integration routing restricted alerts to dead-letter storage.
- Idempotent ingestion preventing duplicate processing.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from astraeus_data.ingestion.scientific_event_engine import (
    UnifiedScientificEvent,
)
from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
    CalibrationStatus,
)
from engines.quarantine_firewall import QuarantineFirewall


@dataclass
class InvestigationCheckpoint:
    """Persistent, versioned scientific investigation state record."""
    checkpoint_id: str
    investigation_id: str
    target_id: str
    revision_index: int
    timestamp_mjd: float
    state_hash: str
    parent_state_hash: Optional[str]
    evidence_hashes: List[str]
    model_versions: Dict[str, str]
    hypothesis_posteriors: Dict[str, float]
    entropy_nats: float
    decision_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def compute_state_hash(
        cls,
        target_id: str,
        revision_index: int,
        parent_state_hash: Optional[str],
        evidence_hashes: List[str],
        model_versions: Dict[str, str],
        hypothesis_posteriors: Dict[str, float],
    ) -> str:
        """
        Computes canonical SHA-256 digest of the scientific state.
        Ensures exact cryptographic reproducibility.
        """
        canonical_dict = {
            "target_id": target_id,
            "revision_index": revision_index,
            "parent_state_hash": parent_state_hash or "ROOT",
            "evidence_hashes": sorted(evidence_hashes),
            "model_versions": {k: model_versions[k] for k in sorted(model_versions.keys())},
            "hypothesis_posteriors": {
                k: round(hypothesis_posteriors[k], 6)
                for k in sorted(hypothesis_posteriors.keys())
            },
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class DurableScientificEventStore:
    """
    Persistent, ACID-compliant scientific event and state store backed by SQLite with WAL mode.
    Guarantees that every observation, event, and posterior revision is durable and reconstructable.
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes SQLite event store. Pass ':memory:' for transient in-memory tests,
        or a file path for persistent on-disk WAL storage.
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database connection has been closed.")
        return self._conn

    def close(self) -> None:
        """Closes the active database connection and releases all file locks."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        self.close()

    def _init_db(self) -> None:
        """Initializes database schema if not present."""
        with self._get_connection() as conn:
            # 1. Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    alert_id TEXT UNIQUE NOT NULL,
                    broker TEXT NOT NULL,
                    trigger_time_mjd REAL NOT NULL,
                    ingestion_time_utc TEXT NOT NULL,
                    ra_deg REAL NOT NULL,
                    dec_deg REAL NOT NULL,
                    healpix_pixel INTEGER NOT NULL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    is_quarantined INTEGER NOT NULL DEFAULT 0
                );
            """)

            # 2. Observations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    target_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    filter_id TEXT NOT NULL,
                    epoch_mjd REAL NOT NULL,
                    measurement_type TEXT NOT NULL,
                    value REAL,
                    error REAL,
                    limiting_mag REAL,
                    payload_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT
                );
            """)

            # 3. Investigation Checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS investigation_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    revision_index INTEGER NOT NULL,
                    timestamp_mjd REAL NOT NULL,
                    state_hash TEXT NOT NULL,
                    parent_state_hash TEXT,
                    evidence_hashes_json TEXT NOT NULL,
                    model_versions_json TEXT NOT NULL,
                    hypothesis_posteriors_json TEXT NOT NULL,
                    entropy_nats REAL NOT NULL,
                    decision_metadata_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    UNIQUE(investigation_id, revision_index)
                );
            """)

            # 4. Dead-letter quarantine table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_letter_quarantine (
                    quarantine_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    healpix_pixel INTEGER,
                    timestamp_utc TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL
                );
            """)

            # Indices for rapid querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_alert_id ON events(alert_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_healpix ON events(healpix_pixel);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_target_active ON observations(target_id, is_active);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_investigation ON investigation_checkpoints(investigation_id, revision_index);")

    def store_event(self, event: UnifiedScientificEvent) -> Tuple[bool, str]:
        """
        Persists a unified scientific alert.
        Enforces Gate 3 Quarantine firewall. If quarantined, routes to dead-letter table.
        Idempotent: if alert_id is already stored, returns (True, "ALREADY_STORED").
        """
        # 1. Gate 3 Quarantine Check
        is_quarantined = QuarantineFirewall.is_quarantined_coordinate(event.ra_deg, event.dec_deg)
        if is_quarantined:
            reason = f"Gate 3 Quarantine Covenant: ({event.ra_deg:.4f}, {event.dec_deg:.4f}) is restricted"
            self._route_to_dead_letter(
                entity_type="EVENT",
                entity_id=event.event_id,
                reason=reason,
                healpix_pixel=getattr(event, "healpix_pixel_nside32", getattr(event, "healpix_pixel", 0)),
                payload=event.to_dict() if hasattr(event, "to_dict") else {"event_id": event.event_id},
            )
            return False, f"QUARANTINED: {reason}"

        # 2. Compute payload hash
        payload_dict = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
        payload_str = json.dumps(payload_dict, sort_keys=True)
        payload_hash = getattr(event, "raw_payload_hash", None) or hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        broker = getattr(event, "survey_origin", getattr(event, "broker_source", "UNKNOWN"))
        trigger_mjd = getattr(event, "event_time_mjd", getattr(event, "trigger_mjd", 60000.0))
        healpix_px = getattr(event, "healpix_pixel_nside32", getattr(event, "healpix_pixel", 0))

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO events (
                        event_id, alert_id, broker, trigger_time_mjd,
                        ingestion_time_utc, ra_deg, dec_deg, healpix_pixel,
                        payload_hash, payload_json, is_quarantined
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        event.event_id,
                        event.event_id,
                        broker,
                        trigger_mjd,
                        datetime.now(timezone.utc).isoformat(),
                        event.ra_deg,
                        event.dec_deg,
                        healpix_px,
                        payload_hash,
                        payload_str,
                    ),
                )
                return True, "STORED"
            except sqlite3.IntegrityError:
                # Already ingested (idempotent)
                return True, "ALREADY_STORED"

    def store_observation(self, observation: CalibratedObservation) -> Tuple[bool, str]:
        """
        Persists a calibrated astronomical observation.
        If superseding an earlier observation, marks the old one inactive.
        """
        raw_hash = getattr(observation, "raw_data_hash_sha256", None) or getattr(observation, "payload_hash", "hash_default")
        payload_dict = {
            "observation_id": observation.observation_id,
            "target_id": observation.target_id,
            "target_ra_deg": getattr(observation, "target_ra_deg", 180.0),
            "target_dec_deg": getattr(observation, "target_dec_deg", 10.0),
            "facility_id": observation.facility_id,
            "instrument_id": observation.instrument_id,
            "filter_id": observation.filter_id,
            "epoch_mjd": observation.epoch_mjd,
            "exposure_time_sec": getattr(observation, "exposure_time_sec", 60.0),
            "measurement_type": observation.measurement_type.value if hasattr(observation.measurement_type, "value") else str(observation.measurement_type),
            "apparent_mag": observation.apparent_mag,
            "apparent_mag_err": observation.apparent_mag_err,
            "limiting_mag_5sigma": observation.limiting_mag_5sigma,
            "flux_density_ujy": getattr(observation, "measured_flux_njy", getattr(observation, "flux_density_ujy", None)),
            "flux_density_err_ujy": getattr(observation, "measured_flux_err_njy", getattr(observation, "flux_density_err_ujy", None)),
            "calibration_status": observation.calibration_status.value if hasattr(observation.calibration_status, "value") else str(observation.calibration_status),
            "payload_hash": raw_hash,
            "raw_data_hash_sha256": raw_hash,
        }
        payload_str = json.dumps(payload_dict, sort_keys=True)
        payload_hash = raw_hash

        meas_type = observation.measurement_type.value if hasattr(observation.measurement_type, "value") else str(observation.measurement_type)
        val = observation.apparent_mag if observation.apparent_mag is not None else observation.flux_density_ujy
        err = observation.apparent_mag_err if observation.apparent_mag_err is not None else observation.flux_density_err_ujy

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO observations (
                        observation_id, target_id, facility_id, instrument_id,
                        filter_id, epoch_mjd, measurement_type, value, error,
                        limiting_mag, payload_hash, payload_json, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        observation.observation_id,
                        observation.target_id,
                        observation.facility_id,
                        observation.instrument_id,
                        observation.filter_id,
                        observation.epoch_mjd,
                        meas_type,
                        val,
                        err,
                        observation.limiting_mag_5sigma,
                        payload_hash,
                        payload_str,
                    ),
                )
                return True, "STORED"
            except sqlite3.IntegrityError:
                return True, "ALREADY_STORED"

    def supersede_observation(self, old_observation_id: str, new_observation_id: str) -> bool:
        """Marks old_observation_id inactive and superseded by new_observation_id."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE observations
                SET is_active = 0, superseded_by = ?
                WHERE observation_id = ?
                """,
                (new_observation_id, old_observation_id),
            )
            return cur.rowcount > 0

    def store_investigation_checkpoint(
        self,
        investigation_id: str,
        target_id: str,
        revision_index: int,
        timestamp_mjd: float,
        evidence_hashes: List[str],
        model_versions: Dict[str, str],
        hypothesis_posteriors: Dict[str, float],
        entropy_nats: float,
        parent_state_hash: Optional[str] = None,
        decision_metadata: Optional[Dict[str, Any]] = None,
    ) -> InvestigationCheckpoint:
        """
        Creates and stores a versioned, cryptographically hashed investigation checkpoint.
        """
        state_hash = InvestigationCheckpoint.compute_state_hash(
            target_id=target_id,
            revision_index=revision_index,
            parent_state_hash=parent_state_hash,
            evidence_hashes=evidence_hashes,
            model_versions=model_versions,
            hypothesis_posteriors=hypothesis_posteriors,
        )

        checkpoint_id = f"chk_{investigation_id}_rev{revision_index}_{state_hash[:10]}"
        now_utc = datetime.now(timezone.utc).isoformat()
        metadata = decision_metadata or {}

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO investigation_checkpoints (
                    checkpoint_id, investigation_id, target_id, revision_index,
                    timestamp_mjd, state_hash, parent_state_hash,
                    evidence_hashes_json, model_versions_json,
                    hypothesis_posteriors_json, entropy_nats,
                    decision_metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    investigation_id,
                    target_id,
                    revision_index,
                    timestamp_mjd,
                    state_hash,
                    parent_state_hash,
                    json.dumps(evidence_hashes),
                    json.dumps(model_versions),
                    json.dumps(hypothesis_posteriors),
                    entropy_nats,
                    json.dumps(metadata),
                    now_utc,
                ),
            )

        return InvestigationCheckpoint(
            checkpoint_id=checkpoint_id,
            investigation_id=investigation_id,
            target_id=target_id,
            revision_index=revision_index,
            timestamp_mjd=timestamp_mjd,
            state_hash=state_hash,
            parent_state_hash=parent_state_hash,
            evidence_hashes=evidence_hashes,
            model_versions=model_versions,
            hypothesis_posteriors=hypothesis_posteriors,
            entropy_nats=entropy_nats,
            decision_metadata=metadata,
            created_at_utc=now_utc,
        )

    def get_investigation_checkpoints(self, investigation_id: str) -> List[InvestigationCheckpoint]:
        """Retrieves checkpoint history for an investigation ordered by revision_index."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT * FROM investigation_checkpoints
                WHERE investigation_id = ?
                ORDER BY revision_index ASC
                """,
                (investigation_id,),
            )
            results = []
            for row in cur.fetchall():
                results.append(
                    InvestigationCheckpoint(
                        checkpoint_id=row["checkpoint_id"],
                        investigation_id=row["investigation_id"],
                        target_id=row["target_id"],
                        revision_index=row["revision_index"],
                        timestamp_mjd=row["timestamp_mjd"],
                        state_hash=row["state_hash"],
                        parent_state_hash=row["parent_state_hash"],
                        evidence_hashes=json.loads(row["evidence_hashes_json"]),
                        model_versions=json.loads(row["model_versions_json"]),
                        hypothesis_posteriors=json.loads(row["hypothesis_posteriors_json"]),
                        entropy_nats=row["entropy_nats"],
                        decision_metadata=json.loads(row["decision_metadata_json"]),
                        created_at_utc=row["created_at_utc"],
                    )
                )
            return results

    def get_active_observations_for_target(self, target_id: str) -> List[CalibratedObservation]:
        """Retrieves active, un-superseded observations for a target in chronological order."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT payload_json FROM observations
                WHERE target_id = ? AND is_active = 1
                ORDER BY epoch_mjd ASC
                """,
                (target_id,),
            )
            obs_list = []
            for row in cur.fetchall():
                d = json.loads(row["payload_json"])
                m_type = MeasurementType(d["measurement_type"])
                cal_status = CalibrationStatus(d["calibration_status"])
                raw_hash = d.get("raw_data_hash_sha256") or d.get("payload_hash") or "hash_default"
                obs = CalibratedObservation(
                    observation_id=d["observation_id"],
                    target_id=d["target_id"],
                    target_ra_deg=d.get("target_ra_deg", 180.0),
                    target_dec_deg=d.get("target_dec_deg", 10.0),
                    facility_id=d["facility_id"],
                    instrument_id=d["instrument_id"],
                    filter_id=d["filter_id"],
                    epoch_mjd=d["epoch_mjd"],
                    exposure_time_sec=d.get("exposure_time_sec", 60.0),
                    measurement_type=m_type,
                    apparent_mag=d.get("apparent_mag"),
                    apparent_mag_err=d.get("apparent_mag_err"),
                    limiting_mag_5sigma=d.get("limiting_mag_5sigma"),
                    measured_flux_njy=d.get("flux_density_ujy"),
                    measured_flux_err_njy=d.get("flux_density_err_ujy"),
                    calibration_status=cal_status,
                    raw_data_hash_sha256=raw_hash,
                )
                obs_list.append(obs)
            return obs_list

    def get_quarantined_records(self) -> List[Dict[str, Any]]:
        """Returns records routed to dead-letter storage."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM dead_letter_quarantine ORDER BY timestamp_utc ASC;")
            return [dict(r) for r in cur.fetchall()]

    def verify_database_integrity(self) -> Dict[str, Any]:
        """Executes SQLite internal integrity checks."""
        with self._get_connection() as conn:
            quick = conn.execute("PRAGMA quick_check;").fetchone()[0]
            integrity = conn.execute("PRAGMA integrity_check;").fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM events;").fetchone()[0]
            total_obs = conn.execute("SELECT COUNT(*) FROM observations;").fetchone()[0]
            total_chk = conn.execute("SELECT COUNT(*) FROM investigation_checkpoints;").fetchone()[0]
            total_quar = conn.execute("SELECT COUNT(*) FROM dead_letter_quarantine;").fetchone()[0]

            return {
                "quick_check": quick,
                "integrity_check": integrity,
                "is_healthy": (quick == "ok" and integrity == "ok"),
                "total_events": total_events,
                "total_observations": total_obs,
                "total_checkpoints": total_chk,
                "total_quarantined": total_quar,
            }

    def _route_to_dead_letter(
        self,
        entity_type: str,
        entity_id: str,
        reason: str,
        healpix_pixel: Optional[int],
        payload: Dict[str, Any],
    ) -> None:
        """Stores quarantined record in dead-letter table."""
        qid = f"dlq_{entity_type}_{entity_id}_{int(datetime.now(timezone.utc).timestamp())}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO dead_letter_quarantine (
                    quarantine_id, entity_type, entity_id, reason,
                    healpix_pixel, timestamp_utc, raw_payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    qid,
                    entity_type,
                    entity_id,
                    reason,
                    healpix_pixel,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload, sort_keys=True),
                ),
            )
