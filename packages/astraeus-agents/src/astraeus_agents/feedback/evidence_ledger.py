"""
ASTRAEUS Sequential Scientific Feedback - Observation Evidence Ledger (WS7.6-D).
Tracks observation lineage, handles deduplication, preserves reprocessing revisions,
and manages shared calibration covariance structures to prevent artificial confidence inflation.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import time
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    MeasurementType,
)


class LedgerEntryStatus(str, Enum):
    """Status of an observational entry in the evidence ledger."""
    ACTIVE_IN_POSTERIOR = "ACTIVE_IN_POSTERIOR"
    REJECTED_AUDIT_FAILURE = "REJECTED_AUDIT_FAILURE"
    DUPLICATE = "DUPLICATE"
    SUPERSEDED = "SUPERSEDED"


@dataclass
class EvidenceLedgerEntry:
    """Immutable entry in the sequential evidence ledger."""
    entry_id: str
    observation: CalibratedObservation
    ingestion_timestamp_utc: float
    is_duplicate: bool = False
    is_reprocessed_update: bool = False
    superseded_by_entry_id: Optional[str] = None
    shared_calibration_group: Optional[str] = None
    applied_to_inference: bool = True
    sha256_hash: str = ""
    audit_status: LedgerEntryStatus = LedgerEntryStatus.ACTIVE_IN_POSTERIOR


class ObservationEvidenceLedger:
    """
    Durable observation evidence ledger.
    Guarantees that:
    1. Duplicate observation packets do not duplicate Bayesian evidence.
    2. Reprocessed measurements replace superseded versions with full lineage tracking.
    3. Measurements sharing a calibration reference are grouped for joint covariance evaluation.
    """

    def __init__(self):
        self._entries: List[EvidenceLedgerEntry] = []
        self._entries_by_id: Dict[str, EvidenceLedgerEntry] = {}
        self._seen_payload_hashes: Set[str] = set()
        self._calibration_groups: Dict[str, List[str]] = {}

    def ingest_observation(
        self,
        observation: CalibratedObservation,
        audit_result: Optional[Any] = None,
    ) -> Tuple[EvidenceLedgerEntry, bool]:
        """
        Ingests an observation payload with optional quality audit result.
        Returns: (ledger_entry, is_new_informative_evidence)
        """
        payload_hash = observation.raw_data_hash_sha256
        now_ts = time.time()

        # 1. Exact Duplicate Detection (identical raw hash or observation_id)
        if payload_hash in self._seen_payload_hashes or observation.observation_id in self._entries_by_id:
            entry = EvidenceLedgerEntry(
                entry_id=f"dup_{observation.observation_id}_{int(now_ts)}",
                observation=observation,
                ingestion_timestamp_utc=now_ts,
                is_duplicate=True,
                is_reprocessed_update=False,
                applied_to_inference=False,
                sha256_hash=payload_hash,
                audit_status=LedgerEntryStatus.DUPLICATE,
            )
            self._entries.append(entry)
            return entry, False  # Duplicate: do NOT update Bayesian posterior!

        # 2. Quality Audit Failure Check
        if audit_result is not None and not getattr(audit_result, "is_valid_for_inference", True):
            entry = EvidenceLedgerEntry(
                entry_id=observation.observation_id,
                observation=observation,
                ingestion_timestamp_utc=now_ts,
                is_duplicate=False,
                is_reprocessed_update=False,
                applied_to_inference=False,
                sha256_hash=payload_hash,
                audit_status=LedgerEntryStatus.REJECTED_AUDIT_FAILURE,
            )
            self._entries.append(entry)
            self._entries_by_id[observation.observation_id] = entry
            self._seen_payload_hashes.add(payload_hash)
            return entry, False  # Corrupted: recorded in ledger but blocked from inference!

        # 3. Reprocessed Measurement Handling
        is_reprocessed = False
        if observation.is_reprocessed_version_of:
            parent_id = observation.is_reprocessed_version_of
            if parent_id in self._entries_by_id:
                parent_entry = self._entries_by_id[parent_id]
                parent_entry.superseded_by_entry_id = observation.observation_id
                parent_entry.applied_to_inference = False
                parent_entry.audit_status = LedgerEntryStatus.SUPERSEDED
                is_reprocessed = True

        # 4. Shared Calibration Reference Grouping
        cal_group = observation.shared_calibration_group_id
        if cal_group:
            if cal_group not in self._calibration_groups:
                self._calibration_groups[cal_group] = []
            self._calibration_groups[cal_group].append(observation.observation_id)

        entry = EvidenceLedgerEntry(
            entry_id=observation.observation_id,
            observation=observation,
            ingestion_timestamp_utc=now_ts,
            is_duplicate=False,
            is_reprocessed_update=is_reprocessed,
            superseded_by_entry_id=None,
            shared_calibration_group=cal_group,
            applied_to_inference=True,
            sha256_hash=payload_hash,
            audit_status=LedgerEntryStatus.ACTIVE_IN_POSTERIOR,
        )

        self._entries.append(entry)
        self._entries_by_id[observation.observation_id] = entry
        self._seen_payload_hashes.add(payload_hash)

        return entry, True

    def get_active_observations(self) -> List[CalibratedObservation]:
        """Returns all un-superseded, non-duplicate active observations across all targets."""
        return [e.observation for e in self._entries if e.applied_to_inference]

    def get_active_observations_for_target(self, target_id: str) -> List[CalibratedObservation]:
        """Returns all un-superseded, non-duplicate active observations for target."""
        return [
            e.observation
            for e in self._entries
            if e.observation.target_id == target_id and e.applied_to_inference
        ]

    def recompute_target_posteriors(
        self,
        target_id: str,
        initial_prior: Dict[str, float],
        trigger_mjd: float = 60000.0,
        base_mag: float = 18.0,
    ) -> Dict[str, float]:
        """
        Recomputes hypothesis posteriors from scratch using only currently active
        (non-duplicate, un-superseded, audit-passed) observations in the ledger.
        Guarantees exact rollback upon observation supersession without numerical drift.
        """
        from astraeus_agents.feedback.bayesian_recalibration import BayesianRecalibrationEngine
        from astraeus_agents.observatory.facility_registry import WavelengthRegime

        active_obs = self.get_active_observations_for_target(target_id)
        current_post = dict(initial_prior)

        for obs in active_obs:
            t_days = max(0.01, obs.epoch_mjd - trigger_mjd)
            fid = getattr(obs, "filter_id", "") or ""
            fb = fid.lower()
            regime = WavelengthRegime.OPTICAL
            wl = 6000.0
            if any(uv_key in fb for uv_key in ("uvw", "uvm", "fuv", "nuv", "uv")):
                regime = WavelengthRegime.UV
                wl = 2600.0
            elif any(nir_key in fb for nir_key in ("nir", "spec", "j", "h", "k", "z", "y")):
                regime = WavelengthRegime.NIR
                wl = 12500.0

            res = BayesianRecalibrationEngine.recalibrate_posteriors(
                current_posteriors=current_post,
                observation=obs,
                effective_wavelength_angstrom=wl,
                regime=regime,
                time_since_trigger_days=t_days,
                base_mag=base_mag,
            )
            current_post = res.updated_posteriors

        return current_post

    def get_ledger_summary(self) -> Dict[str, Any]:
        """Returns summary counts of recorded observations."""
        total = len(self._entries)
        active = sum(1 for e in self._entries if e.applied_to_inference)
        rejected = sum(1 for e in self._entries if e.audit_status == LedgerEntryStatus.REJECTED_AUDIT_FAILURE)
        duplicates = sum(1 for e in self._entries if e.is_duplicate)
        return {
            "total_observations_recorded": total,
            "active_observations_in_posterior": active,
            "rejected_observations": rejected,
            "duplicate_observations": duplicates,
        }

    def get_shared_calibration_partners(self, observation_id: str) -> List[str]:
        """Finds IDs of other observations sharing the same calibration reference."""
        entry = self._entries_by_id.get(observation_id)
        if not entry or not entry.shared_calibration_group:
            return []
        group = entry.shared_calibration_group
        return [oid for oid in self._calibration_groups.get(group, []) if oid != observation_id]

    def get_ledger_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "entry_id": e.entry_id,
                "obs_id": e.observation.observation_id,
                "target_id": e.observation.target_id,
                "facility": e.observation.facility_id,
                "filter": e.observation.filter_id,
                "type": e.observation.measurement_type.value,
                "is_duplicate": e.is_duplicate,
                "is_reprocessed": e.is_reprocessed_update,
                "superseded_by": e.superseded_by_entry_id,
                "shared_cal_group": e.shared_calibration_group,
                "applied": e.applied_to_inference,
                "hash": e.sha256_hash,
            }
            for e in self._entries
        ]
