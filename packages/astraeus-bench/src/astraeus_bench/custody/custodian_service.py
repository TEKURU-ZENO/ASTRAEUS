"""
ASTRAEUS Standalone Custodian Authority & Independent KMS/HSM Service (WS10.2).

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class RollbackViolationError(RuntimeError):
    """Raised when ledger rollback, sequence regression, or history deletion is detected."""
    pass


class ExternalKeyAuthority(abc.ABC):
    """
    Abstract interface for cryptographic signing keys held strictly outside
    the ASTRAEUS inference and deliberation agent process memory.
    """

    @property
    @abc.abstractmethod
    def key_id(self) -> str:
        """Unique identifier of the signing key."""
        pass

    @abc.abstractmethod
    def sign(self, payload: bytes) -> str:
        """Produces a cryptographic HMAC-SHA256 signature string."""
        pass

    @abc.abstractmethod
    def verify(self, payload: bytes, signature_hex: str) -> bool:
        """Verifies a signature using constant-time comparison."""
        pass


class EnvironmentExternalKeyProvider(ExternalKeyAuthority):
    """
    Key authority that loads private keys strictly from secure environment
    variables or restricted filesystem key-files outside the repository workspace.
    """

    def __init__(
        self,
        key_id: str = "ENV_KMS_KEY_V1",
        env_var_name: str = "ASTRAEUS_CUSTODIAN_HMAC_KEY",
        key_file_path: Optional[str] = None,
    ) -> None:
        self._key_id = key_id
        raw_key = os.environ.get(env_var_name)
        if not raw_key and key_file_path and Path(key_file_path).exists():
            with open(key_file_path, "rb") as f:
                self._key_bytes = f.read().strip()
        elif raw_key:
            self._key_bytes = raw_key.encode("utf-8")
        else:
            raise ValueError(
                f"EXTERNAL KEY ACQUISITION FAILED: No external key available in environment variable "
                f"'{env_var_name}' or key file '{key_file_path}'."
            )

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._key_bytes, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature_hex: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature_hex)


class SimulatedKMSAuthority(ExternalKeyAuthority):
    """
    Simulates a secure Hardware Security Module (HSM) / Cloud KMS enclave.
    Private key material is encapsulated and never leaked outside the authority.
    """

    def __init__(self, key_id: str = "KMS_HSM_ENCLAVE_ROOT", private_key_bytes: Optional[bytes] = None) -> None:
        self._key_id = key_id
        self._secret = private_key_bytes or secrets.token_bytes(32)

    @property
    def key_id(self) -> str:
        return self._key_id

    def sign(self, payload: bytes) -> str:
        return hmac.new(self._secret, payload, hashlib.sha256).hexdigest()

    def verify(self, payload: bytes, signature_hex: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature_hex)


class SignedStateCheckpoint(BaseModel):
    """Cryptographically signed ledger state checkpoint for rollback detection."""
    checkpoint_id: str
    sequence_number: int
    root_ledger_hash: str
    total_record_count: int
    timestamp_epoch_sec: float
    timestamp_iso: str
    key_id: str
    custodian_signature_hex: str


class StandaloneCustodianService:
    """
    Independent Observatory Custodian Service (WS10.2).
    Operates with an isolated key authority, enforces persistent SQLite storage,
    prohibits UPDATE/DELETE with database engine triggers, cryptographically binds
    signatures into hash-chain records, verifies HMAC signatures during audit,
    and issues monotonic signed state checkpoints.
    """

    def __init__(
        self,
        custodian_id: str = "INDEPENDENT_OBSERVATORY_CUSTODIAN_SERVICE",
        key_authority: Optional[ExternalKeyAuthority] = None,
        ledger_db_path: Optional[str] = None,
        allow_ephemeral_for_testing: bool = False,
        require_external_key: bool = True,
    ) -> None:
        self.custodian_id = custodian_id

        if key_authority is None:
            if require_external_key:
                # Attempt to load from environment
                env_key = os.environ.get("ASTRAEUS_CUSTODIAN_HMAC_KEY")
                if env_key:
                    self.key_authority = EnvironmentExternalKeyProvider()
                else:
                    raise ValueError(
                        f"EXTERNAL KEY REQUIRED: Custodian service '{custodian_id}' requires an independent external key authority. "
                        "Set ASTRAEUS_CUSTODIAN_HMAC_KEY or supply an ExternalKeyAuthority instance."
                    )
            else:
                self.key_authority = SimulatedKMSAuthority(key_id=f"{custodian_id}_INTERNAL")
        else:
            self.key_authority = key_authority

        if ledger_db_path is None or ledger_db_path == ":memory:":
            if not allow_ephemeral_for_testing:
                raise ValueError(
                    "MANDATORY PERSISTENCE VIOLATION: ledger_db_path must point to a durable file on disk. "
                    "In-memory databases (':memory:') are prohibited unless allow_ephemeral_for_testing=True."
                )
            self.ledger_db_path = ":memory:"
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.ledger_db_path = ledger_db_path
            self._mem_conn = None

        self._sequence_counter = 0
        self._init_sqlite_ledger()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        return sqlite3.connect(self.ledger_db_path)

    def _init_sqlite_ledger(self) -> None:
        conn = self._get_connection()
        if self._mem_conn is None:
            conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS custodian_ledger (
                record_index INTEGER PRIMARY KEY AUTOINCREMENT,
                prev_record_hash TEXT NOT NULL,
                record_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                signature_hex TEXT NOT NULL,
                recorded_at_epoch_sec REAL NOT NULL,
                recorded_at_iso TEXT NOT NULL,
                record_hash TEXT NOT NULL
            );
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_custodian_ledger_update
            BEFORE UPDATE ON custodian_ledger
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY VIOLATION: UPDATE operations are strictly prohibited on custodian_ledger');
            END;
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS prevent_custodian_ledger_delete
            BEFORE DELETE ON custodian_ledger
            BEGIN
                SELECT RAISE(ABORT, 'IMMUTABILITY VIOLATION: DELETE operations are strictly prohibited on custodian_ledger');
            END;
        """)
        conn.commit()

        # Initialize sequence counter from existing DB
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM custodian_ledger;")
        self._sequence_counter = cur.fetchone()[0]

        if self._mem_conn is None:
            conn.close()

    def register_commitment(
        self,
        target_id: str,
        hypothesis_id: str,
        hypothesis_hash: str,
        predictions: List[Dict[str, Any]],
        falsification_criteria: List[str],
    ) -> Dict[str, Any]:
        """
        Authoritatively records pre-registered scientific prediction commitment.
        Binds custodian signature into record hash.
        """
        now_dt = datetime.now(timezone.utc)
        epoch_sec = now_dt.timestamp()
        iso_str = now_dt.isoformat()
        cid = f"commit_{target_id}_{hypothesis_id}"

        sig = self.key_authority.sign(f"{cid}:{target_id}:{hypothesis_hash}:{iso_str}".encode("utf-8"))

        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT record_hash FROM custodian_ledger ORDER BY record_index DESC LIMIT 1;")
            row = cur.fetchone()
            prev_hash = row[0] if row else "0" * 64

            payload = {
                "commitment_id": cid,
                "target_id": target_id,
                "hypothesis_hash": hypothesis_hash,
                "predictions": predictions,
                "falsification_criteria": falsification_criteria,
            }
            payload_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

            # Cryptographically bind signature into record hash
            rec_bytes = f"{prev_hash}:{cid}:{target_id}:{payload_hash}:{sig}:{epoch_sec}".encode("utf-8")
            rec_hash = hashlib.sha256(rec_bytes).hexdigest()

            cur.execute("""
                INSERT INTO custodian_ledger (
                    prev_record_hash, record_type, target_id, payload_hash, payload_json,
                    signature_hex, recorded_at_epoch_sec, recorded_at_iso, record_hash
                ) VALUES (?, 'COMMITMENT', ?, ?, ?, ?, ?, ?, ?);
            """, (prev_hash, target_id, payload_hash, payload_str, sig, epoch_sec, iso_str, rec_hash))
            conn.commit()
            self._sequence_counter += 1
        finally:
            if self._mem_conn is None:
                conn.close()

        return {
            "commitment_id": cid,
            "target_id": target_id,
            "frozen_hypothesis_hash": hypothesis_hash,
            "pre_registered_predictions": predictions,
            "falsification_criteria": falsification_criteria,
            "custodian_id": self.custodian_id,
            "key_id": self.key_authority.key_id,
            "custodian_signature_hex": sig,
            "committed_at_epoch_sec": epoch_sec,
            "committed_at_iso": iso_str,
            "record_hash": rec_hash,
        }

    def grant_unblinded_access(self, target_id: str) -> Tuple[float, str]:
        """
        Grants access to withheld observations strictly if prior commitment exists.
        Records access event with cryptographic signature bound into the hash chain.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT record_index, recorded_at_epoch_sec FROM custodian_ledger WHERE target_id = ? AND record_type = 'COMMITMENT';",
                (target_id,)
            )
            commit_row = cur.fetchone()
            if not commit_row:
                raise ValueError(
                    f"ACCESS DENIED: No pre-registration commitment found for target '{target_id}'. "
                    "Withheld data cannot be accessed without prior immutable commitment."
                )

            now_dt = datetime.now(timezone.utc)
            unblind_sec = now_dt.timestamp()
            unblind_iso = now_dt.isoformat()

            cur.execute("SELECT record_hash FROM custodian_ledger ORDER BY record_index DESC LIMIT 1;")
            row = cur.fetchone()
            prev_hash = row[0] if row else "0" * 64

            token_sig = self.key_authority.sign(f"UNBLIND:{target_id}:{unblind_sec}".encode("utf-8"))
            rec_bytes = f"{prev_hash}:UNBLIND:{target_id}:{token_sig}:{unblind_sec}".encode("utf-8")
            rec_hash = hashlib.sha256(rec_bytes).hexdigest()

            cur.execute("""
                INSERT INTO custodian_ledger (
                    prev_record_hash, record_type, target_id, payload_hash, payload_json,
                    signature_hex, recorded_at_epoch_sec, recorded_at_iso, record_hash
                ) VALUES (?, 'UNBLIND_ACCESS', ?, '', '', ?, ?, ?, ?);
            """, (prev_hash, target_id, token_sig, unblind_sec, unblind_iso, rec_hash))
            conn.commit()
            self._sequence_counter += 1
        finally:
            if self._mem_conn is None:
                conn.close()

        return (unblind_sec, token_sig)

    def create_signed_state_checkpoint(self) -> SignedStateCheckpoint:
        """
        Generates a cryptographically signed state checkpoint representing the
        exact state of the ledger, sequence counter, and latest root hash.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*), MAX(record_hash) FROM custodian_ledger;")
            cnt, latest_hash = cur.fetchone()
            latest_hash = latest_hash or ("0" * 64)
        finally:
            if self._mem_conn is None:
                conn.close()

        now_dt = datetime.now(timezone.utc)
        epoch_sec = now_dt.timestamp()
        iso_str = now_dt.isoformat()
        chk_id = f"chk_{cnt}_{epoch_sec:.2f}"

        sig_payload = f"{chk_id}:{cnt}:{latest_hash}:{epoch_sec}".encode("utf-8")
        sig = self.key_authority.sign(sig_payload)

        return SignedStateCheckpoint(
            checkpoint_id=chk_id,
            sequence_number=cnt,
            root_ledger_hash=latest_hash,
            total_record_count=cnt,
            timestamp_epoch_sec=epoch_sec,
            timestamp_iso=iso_str,
            key_id=self.key_authority.key_id,
            custodian_signature_hex=sig,
        )

    def verify_ledger_integrity(self) -> Tuple[bool, List[str]]:
        """
        Performs exhaustive mathematical verification of the custodian ledger:
        1. Checks active existence of SQLite immutability triggers.
        2. Checks contiguous monotonic sequence index (detects row deletion).
        3. Verifies continuous hash-chaining (prev_record_hash matches preceding record_hash).
        4. Recomputes payload hashes and verifies equality.
        5. Recomputes record hashes (supporting signature-bound and legacy formats).
        6. Cryptographically validates HMAC-SHA256 signatures for every record against the key authority.
        """
        notes: List[str] = []
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            # 1. Trigger verification
            cur.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
            active_triggers = {row[0] for row in cur.fetchall()}
            if not {"prevent_custodian_ledger_update", "prevent_custodian_ledger_delete"}.issubset(active_triggers):
                notes.append("SCHEMA TAMPERING: Required SQLite immutability triggers have been dropped or modified.")

            cur.execute("""
                SELECT record_index, prev_record_hash, record_type, target_id,
                       payload_hash, payload_json, signature_hex, recorded_at_epoch_sec,
                       recorded_at_iso, record_hash
                FROM custodian_ledger
                ORDER BY record_index ASC;
            """)
            rows = cur.fetchall()
        finally:
            if self._mem_conn is None:
                conn.close()

        if not rows:
            return (True, ["Ledger is empty."])

        expected_prev = "0" * 64
        expected_idx = 1
        is_valid = True

        for r in rows:
            idx, p_hash, r_type, tid, pl_hash, pl_json, sig, epoch, iso_str, r_hash = r

            # Check sequence continuity
            if idx != expected_idx:
                is_valid = False
                notes.append(f"SEQUENCE BREAK: Record index discontinuous. Expected {expected_idx}, got {idx} (possible row deletion).")

            # Check hash-chain pointer continuity
            if p_hash != expected_prev:
                is_valid = False
                notes.append(f"HASH CHAIN BREAK at index {idx}: stored prev_hash {p_hash} does not match expected {expected_prev}.")

            # Recompute payload and record hashes
            if r_type == "COMMITMENT":
                recomputed_pl_hash = hashlib.sha256(pl_json.encode("utf-8")).hexdigest()
                if recomputed_pl_hash != pl_hash:
                    is_valid = False
                    notes.append(f"PAYLOAD TAMPERING at index {idx}: payload_json hash {recomputed_pl_hash} != stored {pl_hash}.")

                try:
                    pld = json.loads(pl_json)
                    cid = pld.get("commitment_id", f"commit_{tid}")
                    hyp_hash = pld.get("hypothesis_hash", "")
                except Exception:
                    cid = f"commit_{tid}"
                    hyp_hash = ""

                # Verify HMAC signature
                expected_sig = self.key_authority.sign(f"{cid}:{tid}:{hyp_hash}:{iso_str}".encode("utf-8"))
                if not hmac.compare_digest(expected_sig, sig):
                    is_valid = False
                    notes.append(f"SIGNATURE TAMPERING at index {idx}: HMAC signature invalid for commitment {cid}.")

                # Check record hash (signature-bound or legacy)
                rec_bytes_bound = f"{p_hash}:{cid}:{tid}:{pl_hash}:{sig}:{epoch}".encode("utf-8")
                rec_bytes_legacy = f"{p_hash}:{cid}:{tid}:{pl_hash}:{epoch}".encode("utf-8")
                hash_bound = hashlib.sha256(rec_bytes_bound).hexdigest()
                hash_legacy = hashlib.sha256(rec_bytes_legacy).hexdigest()

                if r_hash not in (hash_bound, hash_legacy):
                    is_valid = False
                    notes.append(f"RECORD HASH MISMATCH at index {idx}: recomputed {hash_bound} != stored {r_hash}.")

            elif r_type == "UNBLIND_ACCESS":
                # Verify HMAC signature
                expected_sig = self.key_authority.sign(f"UNBLIND:{tid}:{epoch}".encode("utf-8"))
                if not hmac.compare_digest(expected_sig, sig):
                    is_valid = False
                    notes.append(f"SIGNATURE TAMPERING at index {idx}: HMAC signature invalid for unblind access {tid}.")

                rec_bytes = f"{p_hash}:UNBLIND:{tid}:{epoch}".encode("utf-8")
                rec_bytes_sig = f"{p_hash}:UNBLIND:{tid}:{sig}:{epoch}".encode("utf-8")
                hash_base = hashlib.sha256(rec_bytes).hexdigest()
                hash_sig = hashlib.sha256(rec_bytes_sig).hexdigest()

                if r_hash not in (hash_base, hash_sig):
                    is_valid = False
                    notes.append(f"RECORD HASH MISMATCH at index {idx}: recomputed {hash_sig} != stored {r_hash}.")

            expected_prev = r_hash
            expected_idx = idx + 1

        if any("SCHEMA TAMPERING" in n for n in notes):
            is_valid = False

        return (is_valid, notes)


class CustodianServiceClient:
    """
    Client-side boundary wrapper for ASTRAEUS inference agents.
    Communicates with the standalone custodian service without direct database
    or private key access, maintaining local checkpoint history to detect rollback.
    """

    def __init__(self, service: StandaloneCustodianService) -> None:
        self._service = service
        self._latest_known_checkpoint: Optional[SignedStateCheckpoint] = None

    def register_commitment(
        self,
        target_id: str,
        hypothesis_id: str,
        hypothesis_hash: str,
        predictions: List[Dict[str, Any]],
        falsification_criteria: List[str],
    ) -> Dict[str, Any]:
        result = self._service.register_commitment(
            target_id=target_id,
            hypothesis_id=hypothesis_id,
            hypothesis_hash=hypothesis_hash,
            predictions=predictions,
            falsification_criteria=falsification_criteria,
        )
        self.refresh_and_verify_checkpoint()
        return result

    def grant_unblinded_access(self, target_id: str) -> Tuple[float, str]:
        res = self._service.grant_unblinded_access(target_id)
        self.refresh_and_verify_checkpoint()
        return res

    def refresh_and_verify_checkpoint(self) -> SignedStateCheckpoint:
        chk = self._service.create_signed_state_checkpoint()
        # Verify checkpoint signature
        sig_payload = f"{chk.checkpoint_id}:{chk.sequence_number}:{chk.root_ledger_hash}:{chk.timestamp_epoch_sec}".encode("utf-8")
        if not self._service.key_authority.verify(sig_payload, chk.custodian_signature_hex):
            raise RollbackViolationError("CHECKPOINT SIGNATURE INVALID: State checkpoint was not signed by custodian key authority.")

        # Check monotonic sequence progression
        if self._latest_known_checkpoint is not None:
            if chk.sequence_number < self._latest_known_checkpoint.sequence_number:
                raise RollbackViolationError(
                    f"LEDGER ROLLBACK DETECTED: Sequence regressed from {self._latest_known_checkpoint.sequence_number} to {chk.sequence_number}."
                )
        self._latest_known_checkpoint = chk
        return chk

    def verify_ledger_integrity(self) -> Tuple[bool, List[str]]:
        return self._service.verify_ledger_integrity()
