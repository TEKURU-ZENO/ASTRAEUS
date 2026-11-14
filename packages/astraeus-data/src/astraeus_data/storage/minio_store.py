"""
ASTRAEUS Object Store (MinIO / S3 Integration)
Provides secure object persistence for raw alerts, lightcurve JSONs, and catalog snapshots
with end-to-end SHA-256 cryptographic integrity verification.
"""

from __future__ import annotations

import os
import hashlib
import io
import json
from typing import Optional, Dict, Any, List
from pathlib import Path


class AstraeusObjectStore:
    """
    Object Storage Manager supporting MinIO S3 API with local filesystem fallback.
    Guarantees bit-level reproducibility and SHA-256 validation.
    """
    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        secure: bool = False,
        local_root: str = "data/storage/minio_objects",
    ):
        self.endpoint = endpoint
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self._minio_client = None
        self.mode = "local"

        if endpoint:
            try:
                from minio import Minio
                self._minio_client = Minio(
                    endpoint=endpoint,
                    access_key=access_key or "minioadmin",
                    secret_key=secret_key or "minioadmin",
                    secure=secure,
                )
                # Check connection
                self._minio_client.list_buckets()
                self.mode = "minio_live"
            except Exception:
                # Fall back to local object storage mode
                self._minio_client = None
                self.mode = "local"

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/json",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Stores an object, computing and returning its SHA-256 digest.
        """
        computed_sha256 = hashlib.sha256(data).hexdigest()
        byte_size = len(data)

        if self.mode == "minio_live" and self._minio_client:
            if not self._minio_client.bucket_exists(bucket_name):
                self._minio_client.make_bucket(bucket_name)
            meta = metadata or {}
            meta["sha256"] = computed_sha256
            self._minio_client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=io.BytesIO(data),
                length=byte_size,
                content_type=content_type,
                metadata=meta,
            )
        else:
            # Local storage backend
            bucket_dir = self.local_root / bucket_name
            target_path = bucket_dir / object_name
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(data)
            
            # Store metadata
            meta_path = target_path.with_suffix(target_path.suffix + ".meta.json")
            meta = metadata or {}
            meta.update({
                "sha256": computed_sha256,
                "size_bytes": byte_size,
                "content_type": content_type,
            })
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        return {
            "bucket_name": bucket_name,
            "object_name": object_name,
            "sha256": computed_sha256,
            "size_bytes": byte_size,
            "storage_mode": self.mode,
        }

    def get_object(
        self,
        bucket_name: str,
        object_name: str,
        expected_sha256: Optional[str] = None,
    ) -> bytes:
        """
        Retrieves an object and cryptographically verifies its SHA-256 checksum.
        Raises ValueError if hash mismatch occurs.
        """
        if self.mode == "minio_live" and self._minio_client:
            response = self._minio_client.get_object(bucket_name, object_name)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
        else:
            target_path = self.local_root / bucket_name / object_name
            if not target_path.exists():
                raise FileNotFoundError(f"Object {bucket_name}/{object_name} does not exist in store.")
            with open(target_path, "rb") as f:
                data = f.read()

        computed_sha256 = hashlib.sha256(data).hexdigest()
        if expected_sha256 and computed_sha256 != expected_sha256:
            raise ValueError(
                f"Cryptographic hash mismatch for {bucket_name}/{object_name}: "
                f"expected {expected_sha256}, got {computed_sha256}"
            )

        return data

    def list_objects(self, bucket_name: str) -> List[str]:
        """Lists all object keys in the specified bucket."""
        if self.mode == "minio_live" and self._minio_client:
            return [obj.object_name for obj in self._minio_client.list_objects(bucket_name)]
        else:
            bucket_dir = self.local_root / bucket_name
            if not bucket_dir.exists():
                return []
            return [
                str(p.relative_to(bucket_dir)).replace("\\", "/")
                for p in bucket_dir.glob("**/*")
                if p.is_file() and not p.name.endswith(".meta.json")
            ]

    def delete_object(self, bucket_name: str, object_name: str) -> bool:
        """Deletes an object and its metadata."""
        if self.mode == "minio_live" and self._minio_client:
            self._minio_client.remove_object(bucket_name, object_name)
            return True
        else:
            target_path = self.local_root / bucket_name / object_name
            meta_path = target_path.with_suffix(target_path.suffix + ".meta.json")
            deleted = False
            if target_path.exists():
                target_path.unlink()
                deleted = True
            if meta_path.exists():
                meta_path.unlink()
            return deleted
