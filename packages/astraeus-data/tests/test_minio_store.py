"""
Tests for ASTRAEUS MinIO / S3 Object Store and Cryptographic Integrity.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from astraeus_data.storage.minio_store import AstraeusObjectStore


@pytest.fixture
def object_store():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = AstraeusObjectStore(local_root=tmp_dir)
        yield store


def test_object_persistence_and_checksum_verification(object_store):
    """Verifies write, SHA-256 checksum computation, and bit-level retrieval."""
    test_data = b'{"survey": "ZTF", "candidate_id": "CAND_0001", "flux": 12.5}'
    bucket = "astraeus-alerts"
    key = "2026/01/cand_0001.json"

    put_res = object_store.put_object(bucket, key, test_data)
    assert put_res["size_bytes"] == len(test_data)
    assert "sha256" in put_res

    # Retrieve with correct hash
    retrieved = object_store.get_object(bucket, key, expected_sha256=put_res["sha256"])
    assert retrieved == test_data

    # Retrieve with invalid hash -> raises ValueError
    with pytest.raises(ValueError, match="Cryptographic hash mismatch"):
        object_store.get_object(bucket, key, expected_sha256="0000000000000000000000000000000000000000000000000000000000000000")


def test_object_store_manifest_snapshots_persistence(object_store):
    """
    Ingests all 33 raw snapshot files from the manifest into object storage
    and cryptographically validates their SHA-256 signatures upon read.
    """
    manifest_path = "data/raw/real_survey_v1/snapshot_manifest.json"
    if not os.path.exists(manifest_path):
        pytest.skip("Snapshot manifest not present")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    assert len(entries) == 33

    persisted_count = 0
    bucket = "astraeus-snapshots"

    for entry in entries:
        rel_path = entry["local_relative_path"]
        full_path = os.path.join("data/raw/real_survey_v1", rel_path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                content = f.read()

            key = rel_path.replace("\\", "/")
            put_res = object_store.put_object(bucket, key, content)
            assert put_res["sha256"] == entry["sha256"], f"Hash mismatch for {rel_path}"

            # Read back and re-verify
            verified_bytes = object_store.get_object(bucket, key, expected_sha256=entry["sha256"])
            assert len(verified_bytes) == entry["size_bytes"]
            persisted_count += 1

    assert persisted_count == 33
    all_keys = object_store.list_objects(bucket)
    assert len(all_keys) == 33


def test_object_deletion(object_store):
    """Verifies clean deletion of objects."""
    bucket = "temp-bucket"
    key = "test_file.txt"
    object_store.put_object(bucket, key, b"delete_me")
    assert key in object_store.list_objects(bucket)

    deleted = object_store.delete_object(bucket, key)
    assert deleted is True
    assert key not in object_store.list_objects(bucket)
