"""
ASTRAEUS Custody Service & External Cryptographic Authority Package (WS10.2).
Provides standalone custodian process boundaries, external KMS/HSM key interfaces,
tamper-evident hash-chain verification with HMAC signature enforcement,
and signed state checkpoint rollback detection.
"""

from astraeus_bench.custody.custodian_service import (
    ExternalKeyAuthority,
    EnvironmentExternalKeyProvider,
    SimulatedKMSAuthority,
    StandaloneCustodianService,
    CustodianServiceClient,
    SignedStateCheckpoint,
    RollbackViolationError,
)

__all__ = [
    "ExternalKeyAuthority",
    "EnvironmentExternalKeyProvider",
    "SimulatedKMSAuthority",
    "StandaloneCustodianService",
    "CustodianServiceClient",
    "SignedStateCheckpoint",
    "RollbackViolationError",
]
