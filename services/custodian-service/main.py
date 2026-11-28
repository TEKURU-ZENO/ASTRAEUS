"""
ASTRAEUS Custodian Service Standalone Server Process (WS10.2).
Runs in an isolated process boundary, managing cryptographic signing keys
and the persistent SQLite WAL ledger outside the agent address space.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from astraeus_bench.custody.custodian_service import (
    StandaloneCustodianService,
    EnvironmentExternalKeyProvider,
    SimulatedKMSAuthority,
)


def run_service(db_path: str, port: int = 8443, allow_ephemeral: bool = False):
    print(f"[CUSTODIAN-SERVICE] Starting Standalone Custodian Authority Service...")
    print(f"[CUSTODIAN-SERVICE] Ledger database: {db_path}")

    env_key = os.environ.get("ASTRAEUS_CUSTODIAN_HMAC_KEY")
    if env_key:
        key_auth = EnvironmentExternalKeyProvider()
        print(f"[CUSTODIAN-SERVICE] Key Authority initialized from environment KMS key: {key_auth.key_id}")
    else:
        key_auth = SimulatedKMSAuthority(key_id="KMS_STANDALONE_DEFAULT_V1")
        print(f"[CUSTODIAN-SERVICE] Key Authority initialized with isolated enclave key: {key_auth.key_id}")

    service = StandaloneCustodianService(
        custodian_id="INDEPENDENT_OBSERVATORY_CUSTODIAN_SERVICE",
        key_authority=key_auth,
        ledger_db_path=db_path,
        allow_ephemeral_for_testing=allow_ephemeral,
        require_external_key=False,
    )

    chk = service.create_signed_state_checkpoint()
    print(f"[CUSTODIAN-SERVICE] Service online. Initial checkpoint: {chk.checkpoint_id}, sequence: {chk.sequence_number}")
    return service


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRAEUS Custodian Service")
    parser.add_argument("--db-path", type=str, default="data/custodian_ledger.db", help="Path to ledger sqlite database")
    parser.add_argument("--port", type=int, default=8443, help="Service IPC port")
    parser.add_argument("--ephemeral", action="store_true", help="Allow ephemeral in-memory ledger for tests")
    args = parser.parse_args()

    run_service(db_path=args.db_path, port=args.port, allow_ephemeral=args.ephemeral)
