"""
ASTRAEUS Protocol Hash Independent Verification Tool.
Reproducibly loads the frozen evaluation protocol artifact,
applies strict RFC 8785 Canonical JSON serialization,
and verifies the authoritative SHA-256 integrity digest.
"""

import sys
import json
import hashlib
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "packages" / "astraeus-core" / "src"))
sys.path.insert(0, str(repo_root / "packages" / "astraeus-data" / "src"))

from astraeus_data.corpus.builder import ReferenceCorpusBuilder
from astraeus_data.corpus.protocol import FrozenEvaluationProtocol

# Authoritative SHA-256 protocol digest for Reference Corpus v1.0.4
AUTHORITATIVE_PROTOCOL_SHA256 = "7df44b73fb80148c14718104da07393699e1ad002a4116401fba79a65b221583"


def verify_authoritative_protocol() -> bool:
    print("=" * 80)
    print("ASTRAEUS Protocol Hash Verification (RFC 8785 Canonical JSON)")
    print("=" * 80)

    # 1. Build authoritative corpus
    builder = ReferenceCorpusBuilder()
    records = builder.build_reference_corpus_v1()

    # 2. Freeze evaluation protocol v1.0.4
    protocol = FrozenEvaluationProtocol.freeze(records, version="1.0.4")

    print(f"Protocol Version:              {protocol.protocol_version}")
    print(f"Frozen Val Objects:            {len(protocol.frozen_val_ids)}")
    print(f"Frozen Test Objects:           {len(protocol.frozen_test_ids)}")
    print(f"Frozen Quarantine Objects:     {len(protocol.frozen_quarantine_ids)}")
    print(f"Bound Tasks:                   {len(protocol.tasks)}")
    print(f"Bound Metrics:                 {len(protocol.metric_definitions)}")
    print(f"Exclusion Invariant:           {protocol.dataset_b_exclusion_contract[:75]}...")

    # 3. Independent RFC 8785 Canonical Serialization
    payload = {
        "version": protocol.protocol_version,
        "tasks": [t.model_dump() for t in protocol.tasks],
        "val_ids": protocol.frozen_val_ids,
        "test_ids": protocol.frozen_test_ids,
        "quarantine_ids": protocol.frozen_quarantine_ids,
        "metric_definitions": protocol.metric_definitions,
        "dataset_b_exclusion_contract": protocol.dataset_b_exclusion_contract,
    }
    canon_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recomputed_sha256 = hashlib.sha256(canon_bytes).hexdigest()

    print(f"\nAuthoritative Target Hash:     {AUTHORITATIVE_PROTOCOL_SHA256}")
    print(f"Recomputed Canonical Digest:   {recomputed_sha256}")
    print(f"Protocol Internal Digest:      {protocol.protocol_sha256}")

    matches_authority = (recomputed_sha256 == AUTHORITATIVE_PROTOCOL_SHA256)
    matches_internal = (recomputed_sha256 == protocol.protocol_sha256)

    if matches_authority and matches_internal:
        print("\n[VERIFICATION RESULT] SUCCESS: Authoritative protocol digest cryptographically verified.")
        return True
    else:
        print("\n[VERIFICATION RESULT] FAILURE: Hash mismatch detected!")
        return False


if __name__ == "__main__":
    is_valid = verify_authoritative_protocol()
    sys.exit(0 if is_valid else 1)
