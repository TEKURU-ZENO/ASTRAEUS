"""
ASTRAEUS Scientific Replication Network (WS9.7).
Provides independent clean-room re-execution, exact numerical replay,
and cryptographic provenance verification for published scientific results.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
)


class ReplicationStatusOutcome(str, Enum):
    """Outcome of an independent scientific replication audit."""
    REPRODUCED_EXACT = "REPRODUCED_EXACT"
    REPRODUCED_WITHIN_NUMERICAL_TOLERANCE = "REPRODUCED_WITHIN_NUMERICAL_TOLERANCE"
    FAILED_NUMERICAL_DIVERGENCE = "FAILED_NUMERICAL_DIVERGENCE"
    FAILED_PROVENANCE_MISMATCH = "FAILED_PROVENANCE_MISMATCH"


class ReplicationAuditReport(BaseModel):
    """Formal audit report produced by the Scientific Replication Network."""
    investigation_id: str
    target_claim_id: str
    replication_status: ReplicationStatusOutcome
    max_relative_discrepancy: float
    discrepancy_table: Dict[str, float]
    provenance_hash_verified: bool
    is_fully_replicated: bool
    auditor_signature: str
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScientificReplicationNetwork:
    """
    Independent scientific verification auditor.
    Takes research packages and runs clean-room replay to verify exact reproducibility.
    """

    def __init__(self, numerical_tolerance_epsilon: float = 1e-4) -> None:
        self.epsilon = numerical_tolerance_epsilon

    def audit_research_package(
        self,
        investigation_id: str,
        target_claim_id: str,
        claimed_results: Dict[str, float],
        input_parameters: Dict[str, float],
        random_seed: int,
        claimed_evidence_hash: str,
        actual_evidence_hash: str,
    ) -> ReplicationAuditReport:
        """
        Executes an independent clean-room replay of the scientific pipeline
        and compares newly computed observables against claimed values.
        """
        # 1. Verify cryptographic evidence provenance
        provenance_ok = (claimed_evidence_hash == actual_evidence_hash)
        if not provenance_ok:
            return ReplicationAuditReport(
                investigation_id=investigation_id,
                target_claim_id=target_claim_id,
                replication_status=ReplicationStatusOutcome.FAILED_PROVENANCE_MISMATCH,
                max_relative_discrepancy=1.0,
                discrepancy_table={"evidence_hash": 1.0},
                provenance_hash_verified=False,
                is_fully_replicated=False,
                auditor_signature="REPLICATION_NETWORK_VETO",
            )

        # 2. Clean-room independent re-execution
        clean_engine = ComputationalExperimentEngine(random_seed=random_seed)
        replayed_sim = clean_engine.forward_simulate_kilonova(
            ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
            expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
            lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
            epochs_days=[1.0, 3.0],
            filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
            provenance=DataProvenanceTag.SIMULATED_TESTBED,
            inject_noise=False,
        )

        replayed_map = {f"mag_{p.filter_id}_t{p.epoch_days:.0f}": p.simulated_mag for p in replayed_sim}

        # 3. Discrepancy comparison
        discrepancies: Dict[str, float] = {}
        max_rel_diff = 0.0

        for key, claimed_val in claimed_results.items():
            if key in replayed_map:
                actual_val = replayed_map[key]
                abs_diff = abs(claimed_val - actual_val)
                rel_diff = abs_diff / max(1e-6, abs(claimed_val))
                discrepancies[key] = round(rel_diff, 6)
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff

        is_reproduced = max_rel_diff <= self.epsilon
        if is_reproduced:
            status = (
                ReplicationStatusOutcome.REPRODUCED_EXACT
                if max_rel_diff == 0.0
                else ReplicationStatusOutcome.REPRODUCED_WITHIN_NUMERICAL_TOLERANCE
            )
        else:
            status = ReplicationStatusOutcome.FAILED_NUMERICAL_DIVERGENCE

        audit_sig_payload = f"{investigation_id}:{target_claim_id}:{status.value}:{max_rel_diff:.6f}"
        sig = hashlib.sha256(audit_sig_payload.encode()).hexdigest()

        return ReplicationAuditReport(
            investigation_id=investigation_id,
            target_claim_id=target_claim_id,
            replication_status=status,
            max_relative_discrepancy=round(max_rel_diff, 6),
            discrepancy_table=discrepancies,
            provenance_hash_verified=True,
            is_fully_replicated=is_reproduced,
            auditor_signature=sig,
        )
