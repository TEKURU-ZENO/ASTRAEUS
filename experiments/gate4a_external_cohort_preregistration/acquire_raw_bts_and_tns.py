"""
ASTRAEUS Gate 4 Phase 4A: Raw BTS & TNS Catalog Acquisition.

GOVERNANCE CONTRACT:
- Authorized under Phase 4A Conditional Authorization.
- Acquires immutable raw BTS catalog from Caltech BTS Sample Explorer.
- Verifies and anchors supplementary TNS public objects snapshot.
- Computes SHA-256 cryptographic checksums, exact row counts, and column schemas.
- ZERO feature extraction, model training, or inference performed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil

import httpx
import pandas as pd

BTS_URL = "https://sites.astro.caltech.edu/ztf/bts/explorer.php?f=s&subsample=all&format=csv"
LOCAL_TNS_ZIP = Path("data/raw/real_survey_v1/tns/tns_public_objects.csv.zip")

OUT_BTS_DIR = Path("data/external_validity_v1/raw/bts")
OUT_TNS_DIR = Path("data/external_validity_v1/raw/tns")
OUT_PROV_DIR = Path("data/external_validity_v1/provenance")


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print("=" * 80)
    print("ASTRAEUS GATE 4 PHASE 4A: RAW DATA ACQUISITION & PROVENANCE RECORDING")
    print(f"Timestamp: {now_utc}")
    print("=" * 80)

    OUT_BTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PROV_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Acquire Official Caltech BTS Catalog
    bts_file = OUT_BTS_DIR / "bts_raw_snapshot.csv"
    print(f"\n[BTS Acquisition] Fetching raw catalog from {BTS_URL}...")
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        resp = client.get(BTS_URL)
        resp.raise_for_status()
        content = resp.content

    with open(bts_file, "wb") as f:
        f.write(content)

    bts_sha256 = compute_sha256(bts_file)
    bts_bytes = bts_file.stat().st_size
    print(f"  -> BTS snapshot saved: {bts_file}")
    print(f"  -> File size: {bts_bytes} bytes")
    print(f"  -> SHA-256: {bts_sha256}")

    # Inspect BTS CSV Structure
    df_bts = pd.read_csv(bts_file)
    bts_row_count = len(df_bts)
    bts_columns = list(df_bts.columns)
    print(f"  -> Exact Row Count: {bts_row_count}")
    print(f"  -> Column Count   : {len(bts_columns)}")
    print(f"  -> Columns        : {bts_columns}")

    # 2. Link & Verify Supplementary TNS Snapshot
    tns_dest = OUT_TNS_DIR / "tns_public_objects.csv.zip"
    print(f"\n[TNS Snapshot] Anchoring verified local TNS snapshot from {LOCAL_TNS_ZIP}...")
    assert LOCAL_TNS_ZIP.exists(), f"Local TNS snapshot not found at {LOCAL_TNS_ZIP}"
    shutil.copy2(LOCAL_TNS_ZIP, tns_dest)

    tns_sha256 = compute_sha256(tns_dest)
    tns_bytes = tns_dest.stat().st_size
    print(f"  -> TNS snapshot saved: {tns_dest}")
    print(f"  -> File size: {tns_bytes} bytes")
    print(f"  -> SHA-256: {tns_sha256}")

    expected_tns_sha256 = "5d3fa52497de6f24491ebf25715fa9b617a220f662b2fc5102f9baa02e094ad8"
    assert tns_sha256 == expected_tns_sha256, (
        f"TNS SHA mismatch: {tns_sha256} vs {expected_tns_sha256}"
    )
    print("  -> TNS Snapshot Cryptographic Integrity Verified: EXACT MATCH.")

    # 3. Write Provenance Manifest
    manifest = {
        "milestone": "ASTRAEUS Gate 4 Phase 4A Raw Data Acquisition",
        "acquisition_timestamp_utc": now_utc,
        "governance_status": "RAW_DATA_ACQUIRED_MODEL_FROZEN",
        "sources": {
            "bts_primary_catalog": {
                "source_url": BTS_URL,
                "local_path": str(bts_file),
                "sha256": bts_sha256,
                "size_bytes": bts_bytes,
                "row_count": bts_row_count,
                "columns": bts_columns,
                "source_organization": "Caltech / Zwicky Transient Facility (ZTF)",
                "survey_programme": "Bright Transient Survey (BTS)",
            },
            "tns_supplementary_snapshot": {
                "source_system": "Transient Name Server (TNS)",
                "local_path": str(tns_dest),
                "sha256": tns_sha256,
                "size_bytes": tns_bytes,
                "row_count": 212833,
                "format": "csv.zip",
                "source_organization": "IAU / Weizmann Institute of Science",
            },
        },
    }

    manifest_file = OUT_PROV_DIR / "raw_acquisition_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Manifest] Raw acquisition manifest saved to {manifest_file}")
    print(f"[Manifest] Manifest SHA-256: {compute_sha256(manifest_file)}")
    print("=" * 80)
    print("PHASE 4A RAW DATA ACQUISITION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
