"""
ASTRAEUS Real-Survey Immutable Snapshot & Retrieval Provenance Manifest.

Records cryptographic digests, source URIs, HTTP metadata, schema columns,
and git code state for every external real-sky download so downstream cohorts
and benchmarks remain 100% reproducible and auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


def compute_file_sha256(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Computes streaming SHA-256 hex digest of a local snapshot file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def current_git_commit() -> str:
    """Returns short HEAD commit SHA if inside a git worktree, else 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


class RetrievalManifestEntry(BaseModel):
    """Immutable provenance record for a single fetched raw survey artifact."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    source_system: str          # e.g. "TNS_PUBLIC_OBJECTS", "ALERCE_ZTF_V1", "ZTF_DR24_HATS"
    source_uri: str
    local_relative_path: str
    retrieved_at_utc: str
    sha256: str
    size_bytes: int
    row_count: Optional[int] = None
    schema_columns: List[str] = Field(default_factory=list)
    http_status: Optional[int] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    query_params: Dict[str, Any] = Field(default_factory=dict)
    git_commit: str = "unknown"


class SnapshotManifest(BaseModel):
    """Collection of raw real-survey snapshot entries with integrity verification."""

    schema_version: str = "real_survey_manifest_v1"
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    entries: List[RetrievalManifestEntry] = Field(default_factory=list)

    def add_or_update(self, entry: RetrievalManifestEntry) -> None:
        self.entries = [e for e in self.entries if e.artifact_id != entry.artifact_id]
        self.entries.append(entry)

    def save(self, path: Union[str, Path]) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SnapshotManifest":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    def verify_all(self, root_dir: Union[str, Path]) -> Dict[str, Any]:
        """Verifies every entry's local file exists, matches size_bytes, and matches SHA-256."""
        root = Path(root_dir)
        failures: List[Dict[str, str]] = []
        verified_count = 0

        for entry in self.entries:
            target = root / entry.local_relative_path
            if not target.exists():
                failures.append(
                    {"artifact_id": entry.artifact_id, "reason": f"missing_file:{target}"}
                )
                continue
            actual_size = target.stat().st_size
            if actual_size != entry.size_bytes:
                failures.append(
                    {
                        "artifact_id": entry.artifact_id,
                        "reason": f"size_mismatch:expected={entry.size_bytes},actual={actual_size}",
                    }
                )
                continue
            actual_sha = compute_file_sha256(target)
            if actual_sha != entry.sha256:
                failures.append(
                    {
                        "artifact_id": entry.artifact_id,
                        "reason": f"sha256_mismatch:expected={entry.sha256},actual={actual_sha}",
                    }
                )
                continue
            verified_count += 1

        return {
            "passed": len(failures) == 0 and len(self.entries) > 0,
            "total_entries": len(self.entries),
            "verified_count": verified_count,
            "failures": failures,
        }
