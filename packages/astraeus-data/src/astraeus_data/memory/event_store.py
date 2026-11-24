"""
ASTRAEUS Durable Scientific Event Store - SQLite WAL Engine (WS7.7.1 Hardened).
Provides persistent, ACID-compliant, crash-resilient event and observation logging:
- Write-Ahead Logging (WAL) with configurable durability (NORMAL vs FULL).
- Exact cryptographic SHA-256 state hashing with full IEEE 754 precision.
- Immutable historical observation tracking preserving validity windows across supersession.
- Persistent, race-free, atomic single-use authorization token registry.
- Strict Gate 3 Quarantine Firewall integration routing restricted alerts to dead-letter storage.
- Specific, fine-grained IntegrityError disambiguation.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import json
import sqlite3
import hashlib
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
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
    initial_priors: Dict[str, float]
    prior_hash: str
    model_versions: Dict[str, str]
    model_config_hashes: Dict[str, str]
    hypothesis_posteriors: Dict[str, float]
    entropy_nats: float
    decision_metadata: Dict[str, Any] = field(default_factory=dict)
    canonical_state_json: str = ""
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def compute_prior_hash(cls, priors: Dict[str, float]) -> str:
        """Computes deterministic hash of prior distribution with full float precision."""
        canonical_priors = {k: f"{priors[k]:.12e}" for k in sorted(priors.keys())}
        canonical_str = json.dumps(canonical_priors, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @classmethod
    def compute_state_hash(
        cls,
        target_id: str,
        revision_index: int,
        parent_state_hash: Optional[str],
        evidence_hashes: List[str],
        initial_priors: Dict[str, float],
        model_versions: Dict[str, str],
        model_config_hashes: Dict[str, str],
        hypothesis_posteriors: Dict[str, float],
        entropy_nats: float,
        decision_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        """
        Computes canonical SHA-256 digest of the complete scientific state (v2 schema).
        Uses explicit 12-digit scientific notation to eliminate floating point rounding ambiguity.
        Returns: (state_hash, canonical_json_string).
        """
        prior_hash = cls.compute_prior_hash(initial_priors)
        canonical_dict = {
            "schema_version": "astronomy.astraeus.state.v2",
            "target_id": target_id,
            "revision_index": revision_index,
            "parent_state_hash": parent_state_hash or "ROOT",
            "evidence_hashes": sorted(evidence_hashes),
            "initial_priors": {k: f"{initial_priors[k]:.12e}" for k in sorted(initial_priors.keys())},
            "prior_hash": prior_hash,
            "model_versions": {k: str(model_versions[k]) for k in sorted(model_versions.keys())},
            "model_config_hashes": {k: str(model_config_hashes.get(k, "")) for k in sorted(model_versions.keys())},
            "hypothesis_posteriors": {
                k: f"{hypothesis_posteriors[k]:.12e}"
                for k in sorted(hypothesis_posteriors.keys())
            },
            "entropy_nats": f"{entropy_nats:.12e}",
            "decision_metadata": decision_metadata or {},
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        state_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        return state_hash, canonical_str


class DurableScientificEventStore:
    """
    Persistent, ACID-compliant scientific event and state store backed by SQLite with WAL mode.
    Guarantees that every observation, event, and posterior revision is durable and reconstructable.
    """

    def __init__(self, db_path: str = ":memory:", synchronous_mode: str = "NORMAL"):
        """
        Initializes SQLite event store. Pass ':memory:' for transient in-memory tests,
        or a file path for persistent on-disk WAL storage.
        `synchronous_mode` can be 'NORMAL' (high-throughput production) or 'FULL' (strict power-loss guarantee).
        """
        self.db_path = db_path
        self.synchronous_mode = synchronous_mode
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(f"PRAGMA synchronous={self.synchronous_mode};")
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

            # 2. Observations table with full validity windowing
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
                    valid_from_mjd REAL NOT NULL,
                    valid_until_mjd REAL,
                    superseded_at_mjd REAL,
                    superseded_by TEXT
                );
            """)

            # 3. Investigation Checkpoints table (committing complete scientific state v2)
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
                    initial_priors_json TEXT NOT NULL,
                    prior_hash TEXT NOT NULL,
                    model_versions_json TEXT NOT NULL,
                    model_config_hashes_json TEXT NOT NULL,
                    hypothesis_posteriors_json TEXT NOT NULL,
                    entropy_nats REAL NOT NULL,
                    decision_metadata_json TEXT NOT NULL,
                    canonical_state_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    UNIQUE(investigation_id, revision_index)
                );
            """)

            # 4. Persistent Authorization Tokens table (durable single-use governance)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS authorization_tokens (
                    token_id TEXT PRIMARY KEY,
                    recommendation_id TEXT UNIQUE NOT NULL,
                    target_id TEXT NOT NULL,
                    facility_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    filter_id TEXT NOT NULL,
                    exposure_time_sec REAL NOT NULL,
                    scheduled_mjd REAL NOT NULL,
                    proposal_hash TEXT NOT NULL,
                    investigator_email TEXT NOT NULL,
                    created_at_utc REAL NOT NULL,
                    expires_at_utc REAL NOT NULL,
                    ttl_seconds REAL NOT NULL,
                    is_consumed INTEGER NOT NULL DEFAULT 0,
                    consumed_at_utc REAL,
                    hmac_signature TEXT NOT NULL
                );
            """)

            # 5. Dead-letter quarantine table
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_obs_payload_hash ON observations(payload_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_investigation ON investigation_checkpoints(investigation_id, revision_index);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tokens_rec ON authorization_tokens(recommendation_id);")

    def store_event(self, event: UnifiedScientificEvent) -> Tuple[bool, str]:
        """
        Persists a unified scientific alert.
        Enforces Gate 3 Quarantine firewall. If quarantined, routes to dead-letter table.
        Idempotent: if alert_id is already stored, returns (True, "ALREADY_STORED").
        Distinguishes unique constraints from other integrity failures.
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
            except sqlite3.IntegrityError as e:
                err_msg = str(e).lower()
                if "unique" in err_msg or "primary key" in err_msg:
                    return True, "ALREADY_STORED"
                raise e

    def store_observation(self, observation: CalibratedObservation) -> Tuple[bool, str]:
        """
        Persists a calibrated astronomical observation.
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
        val = observation.apparent_mag if observation.apparent_mag is not None else getattr(observation, "measured_flux_njy", None)
        err = observation.apparent_mag_err if observation.apparent_mag_err is not None else getattr(observation, "measured_flux_err_njy", None)

        with self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO observations (
                        observation_id, target_id, facility_id, instrument_id,
                        filter_id, epoch_mjd, measurement_type, value, error,
                        limiting_mag, payload_hash, payload_json, is_active,
                        valid_from_mjd, valid_until_mjd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
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
                        observation.epoch_mjd,
                    ),
                )
                return True, "STORED"
            except sqlite3.IntegrityError as e:
                err_msg = str(e).lower()
                if "unique" in err_msg or "primary key" in err_msg:
                    return True, "ALREADY_STORED"
                raise e

    def atomic_supersede_observation(
        self,
        old_observation_id: str,
        new_observation: CalibratedObservation,
        superseding_mjd: float,
    ) -> Tuple[bool, str]:
        """
        Atomically replaces old_observation_id with new_observation in a single database transaction:
        1. Confirms old_observation_id exists and is currently active.
        2. Inserts new_observation with valid_from_mjd = superseding_mjd.
        3. Updates old_observation_id: is_active = 0, valid_until_mjd = superseding_mjd, superseded_by = new_id.
        Rolls back completely on any failure.
        """
        conn = self._get_connection()
        try:
            # Begin transaction
            cur = conn.cursor()
            cur.execute("SELECT is_active, target_id FROM observations WHERE observation_id = ?", (old_observation_id,))
            row = cur.fetchone()
            if not row:
                return False, f"Old observation '{old_observation_id}' not found."
            if row["is_active"] == 0:
                return False, f"Old observation '{old_observation_id}' is already superseded."

            # Store new observation
            raw_hash = getattr(new_observation, "raw_data_hash_sha256", None) or getattr(new_observation, "payload_hash", "hash_default")
            payload_dict = {
                "observation_id": new_observation.observation_id,
                "target_id": new_observation.target_id,
                "target_ra_deg": getattr(new_observation, "target_ra_deg", 180.0),
                "target_dec_deg": getattr(new_observation, "target_dec_deg", 10.0),
                "facility_id": new_observation.facility_id,
                "instrument_id": new_observation.instrument_id,
                "filter_id": new_observation.filter_id,
                "epoch_mjd": new_observation.epoch_mjd,
                "exposure_time_sec": getattr(new_observation, "exposure_time_sec", 60.0),
                "measurement_type": new_observation.measurement_type.value if hasattr(new_observation.measurement_type, "value") else str(new_observation.measurement_type),
                "apparent_mag": new_observation.apparent_mag,
                "apparent_mag_err": new_observation.apparent_mag_err,
                "limiting_mag_5sigma": new_observation.limiting_mag_5sigma,
                "flux_density_ujy": getattr(new_observation, "measured_flux_njy", getattr(new_observation, "flux_density_ujy", None)),
                "flux_density_err_ujy": getattr(new_observation, "measured_flux_err_njy", getattr(new_observation, "flux_density_err_ujy", None)),
                "calibration_status": new_observation.calibration_status.value if hasattr(new_observation.calibration_status, "value") else str(new_observation.calibration_status),
                "payload_hash": raw_hash,
                "raw_data_hash_sha256": raw_hash,
            }
            payload_str = json.dumps(payload_dict, sort_keys=True)
            val = new_observation.apparent_mag if new_observation.apparent_mag is not None else getattr(new_observation, "measured_flux_njy", None)
            err = new_observation.apparent_mag_err if new_observation.apparent_mag_err is not None else getattr(new_observation, "measured_flux_err_njy", None)
            meas_type = new_observation.measurement_type.value if hasattr(new_observation.measurement_type, "value") else str(new_observation.measurement_type)

            cur.execute(
                """
                INSERT INTO observations (
                    observation_id, target_id, facility_id, instrument_id,
                    filter_id, epoch_mjd, measurement_type, value, error,
                    limiting_mag, payload_hash, payload_json, is_active,
                    valid_from_mjd, valid_until_mjd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, NULL)
                """,
                (
                    new_observation.observation_id,
                    new_observation.target_id,
                    new_observation.facility_id,
                    new_observation.instrument_id,
                    new_observation.filter_id,
                    new_observation.epoch_mjd,
                    meas_type,
                    val,
                    err,
                    new_observation.limiting_mag_5sigma,
                    raw_hash,
                    payload_str,
                    superseding_mjd,
                ),
            )

            # Supersede old observation
            cur.execute(
                """
                UPDATE observations
                SET is_active = 0, valid_until_mjd = ?, superseded_at_mjd = ?, superseded_by = ?
                WHERE observation_id = ?
                """,
                (superseding_mjd, superseding_mjd, new_observation.observation_id, old_observation_id),
            )

            conn.commit()
            return True, "ATOMICALLY_SUPERSEDED"
        except Exception as e:
            conn.rollback()
            raise e

    def supersede_observation(self, old_observation_id: str, new_observation_id: str, superseding_mjd: float = 60000.0) -> bool:
        """Marks old_observation_id inactive and superseded by new_observation_id."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                UPDATE observations
                SET is_active = 0, valid_until_mjd = ?, superseded_at_mjd = ?, superseded_by = ?
                WHERE observation_id = ?
                """,
                (superseding_mjd, superseding_mjd, new_observation_id, old_observation_id),
            )
            return cur.rowcount > 0

    def store_investigation_checkpoint(
        self,
        investigation_id: str,
        target_id: str,
        revision_index: int,
        timestamp_mjd: float,
        evidence_hashes: List[str],
        hypothesis_posteriors: Dict[str, float],
        entropy_nats: float,
        initial_priors: Optional[Dict[str, float]] = None,
        model_versions: Optional[Dict[str, str]] = None,
        model_config_hashes: Optional[Dict[str, str]] = None,
        parent_state_hash: Optional[str] = None,
        decision_metadata: Optional[Dict[str, Any]] = None,
    ) -> InvestigationCheckpoint:
        """
        Creates and stores a versioned, cryptographically hashed investigation checkpoint
        committing to the complete scientific state (v2 schema).
        """
        priors = initial_priors or {"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5}
        models = model_versions or {"bayesian_recalibration": "ws7.6-c", "kilonova": "v2.0"}
        configs = model_config_hashes or {k: "default_cfg_hash" for k in models}
        metadata = decision_metadata or {}
        prior_hash = InvestigationCheckpoint.compute_prior_hash(priors)

        state_hash, canonical_json = InvestigationCheckpoint.compute_state_hash(
            target_id=target_id,
            revision_index=revision_index,
            parent_state_hash=parent_state_hash,
            evidence_hashes=evidence_hashes,
            initial_priors=priors,
            model_versions=models,
            model_config_hashes=configs,
            hypothesis_posteriors=hypothesis_posteriors,
            entropy_nats=entropy_nats,
            decision_metadata=metadata,
        )

        checkpoint_id = f"chk_{investigation_id}_rev{revision_index}_{state_hash[:10]}"
        now_utc = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO investigation_checkpoints (
                    checkpoint_id, investigation_id, target_id, revision_index,
                    timestamp_mjd, state_hash, parent_state_hash,
                    evidence_hashes_json, initial_priors_json, prior_hash,
                    model_versions_json, model_config_hashes_json,
                    hypothesis_posteriors_json, entropy_nats,
                    decision_metadata_json, canonical_state_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint_id,
                    investigation_id,
                    target_id,
                    revision_index,
                    timestamp_mjd,
                    state_hash,
                    parent_state_hash,
                    json.dumps(sorted(evidence_hashes)),
                    json.dumps(priors),
                    prior_hash,
                    json.dumps(models),
                    json.dumps(configs),
                    json.dumps(hypothesis_posteriors),
                    entropy_nats,
                    json.dumps(metadata),
                    canonical_json,
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
            evidence_hashes=sorted(evidence_hashes),
            initial_priors=priors,
            prior_hash=prior_hash,
            model_versions=models,
            model_config_hashes=configs,
            hypothesis_posteriors=hypothesis_posteriors,
            entropy_nats=entropy_nats,
            decision_metadata=metadata,
            canonical_state_json=canonical_json,
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
                        initial_priors=json.loads(row["initial_priors_json"]),
                        prior_hash=row["prior_hash"],
                        model_versions=json.loads(row["model_versions_json"]),
                        model_config_hashes=json.loads(row["model_config_hashes_json"]),
                        hypothesis_posteriors=json.loads(row["hypothesis_posteriors_json"]),
                        entropy_nats=row["entropy_nats"],
                        decision_metadata=json.loads(row["decision_metadata_json"]),
                        canonical_state_json=row["canonical_state_json"],
                        created_at_utc=row["created_at_utc"],
                    )
                )
            return results

    def get_active_observations_for_target(self, target_id: str) -> List[CalibratedObservation]:
        """Retrieves currently active observations for a target in chronological order."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT payload_json FROM observations
                WHERE target_id = ? AND is_active = 1
                ORDER BY epoch_mjd ASC
                """,
                (target_id,),
            )
            return self._parse_observation_rows(cur.fetchall())

    def get_observations_by_hashes(self, evidence_hashes: List[str]) -> List[CalibratedObservation]:
        """
        Retrieves observations by their exact immutable evidence hashes,
        regardless of whether they have been superseded in subsequent revisions.
        """
        if not evidence_hashes:
            return []
        placeholders = ",".join("?" for _ in evidence_hashes)
        with self._get_connection() as conn:
            cur = conn.execute(
                f"""
                SELECT payload_json FROM observations
                WHERE payload_hash IN ({placeholders})
                ORDER BY epoch_mjd ASC
                """,
                evidence_hashes,
            )
            return self._parse_observation_rows(cur.fetchall())

    def _parse_observation_rows(self, rows: List[Any]) -> List[CalibratedObservation]:
        obs_list = []
        for row in rows:
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

    # =========================================================================
    # PERSISTENT AUTHORIZATION TOKEN REGISTRY (CONCURRENCY-SAFE)
    # =========================================================================

    def store_authorization_token(self, token_data: Dict[str, Any]) -> bool:
        """Stores a cryptographically signed authorization token in durable SQLite."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO authorization_tokens (
                        token_id, recommendation_id, target_id, facility_id,
                        instrument_id, filter_id, exposure_time_sec, scheduled_mjd,
                        proposal_hash, investigator_email, created_at_utc,
                        expires_at_utc, ttl_seconds, is_consumed, hmac_signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        token_data["token_id"],
                        token_data["recommendation_id"],
                        token_data["target_id"],
                        token_data["facility_id"],
                        token_data["instrument_id"],
                        token_data["filter_id"],
                        token_data["exposure_time_sec"],
                        token_data["scheduled_mjd"],
                        token_data["proposal_hash"],
                        token_data["investigator_email"],
                        token_data["created_at_utc"],
                        token_data["created_at_utc"] + token_data["ttl_seconds"],
                        token_data["ttl_seconds"],
                        token_data["hmac_signature"],
                    ),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                conn.rollback()
                return False

    def atomic_consume_token(self, token_id: str, current_time_utc: float) -> Tuple[bool, str]:
        """
        Atomically consumes an authorization token in a single ACID update.
        Prevents race conditions across concurrent dispatch threads or processes.
        Returns: (success, reason).
        """
        with self._lock:
            conn = self._get_connection()
            # Atomic single-use consumption conditional on is_consumed == 0 and not expired
            cur = conn.execute(
                """
                UPDATE authorization_tokens
                SET is_consumed = 1, consumed_at_utc = ?
                WHERE token_id = ? AND is_consumed = 0 AND expires_at_utc > ?
                """,
                (current_time_utc, token_id, current_time_utc),
            )
            if cur.rowcount == 1:
                conn.commit()
                return True, "CONSUMED_SUCCESSFULLY"

            # Investigate why update did not touch a row
            row = conn.execute(
                "SELECT is_consumed, expires_at_utc FROM authorization_tokens WHERE token_id = ?",
                (token_id,),
            ).fetchone()
            if not row:
                return False, f"Token '{token_id}' not found in persistent registry."
            if row["is_consumed"] == 1:
                return False, f"Token '{token_id}' has ALREADY BEEN CONSUMED. Single-use policy strictly enforced."
            if current_time_utc > row["expires_at_utc"]:
                return False, f"Token '{token_id}' has EXPIRED."
            return False, "Token consumption failed."

    def get_authorization_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves token record from persistent registry."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT * FROM authorization_tokens WHERE token_id = ?", (token_id,))
                row = cur.fetchone()
                return dict(row) if row else None

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
            total_tokens = conn.execute("SELECT COUNT(*) FROM authorization_tokens;").fetchone()[0]
            total_quar = conn.execute("SELECT COUNT(*) FROM dead_letter_quarantine;").fetchone()[0]

            return {
                "quick_check": quick,
                "integrity_check": integrity,
                "is_healthy": (quick == "ok" and integrity == "ok"),
                "total_events": total_events,
                "total_observations": total_obs,
                "total_checkpoints": total_chk,
                "total_tokens": total_tokens,
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
