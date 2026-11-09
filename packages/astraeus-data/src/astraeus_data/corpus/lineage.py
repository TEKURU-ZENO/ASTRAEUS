"""
ASTRAEUS Dataset B Lineage & Provenance Graph Audit Module.
Enforces explicit parentage, observation provenance, and host identity tracking
to ensure pretraining datasets (Dataset B) contain zero derivatives, cutouts,
or augmentations derived from the frozen evaluation test cohort (Dataset A).
"""

from typing import List, Optional, Set, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from astraeus_data.corpus.record import ReferenceObjectRecord


class LineageStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TEST_OVERLAP_REJECTED = "TEST_OVERLAP_REJECTED"
    LINEAGE_UNAVAILABLE_QUARANTINED = "LINEAGE_UNAVAILABLE_QUARANTINED"


class DatasetBSampleLineage(BaseModel):
    """
    Explicit provenance graph lineage for a pretraining (Dataset B) sample.
    Tracks parent physical object, source observations, and host galaxy references.
    """
    model_config = ConfigDict(frozen=True)

    sample_id: str
    parent_physical_object_id: Optional[str] = None
    source_observation_ids: List[str] = Field(default_factory=list)
    host_galaxy_id: Optional[str] = None
    transformation_history: List[str] = Field(default_factory=list)
    lineage_status: LineageStatus = LineageStatus.VERIFIED
    rejection_reason: Optional[str] = None


class ProvenanceGraphAuditor:
    """
    Audits Dataset B samples against Dataset A test objects to prevent
    information leakage through derived light curves, cutouts, or counterparts.
    """

    def __init__(self, test_records: List[ReferenceObjectRecord]) -> None:
        self.forbidden_object_ids: Set[str] = set()
        self.forbidden_observation_ids: Set[str] = set()
        self.forbidden_host_ids: Set[str] = set()

        for rec in test_records:
            self.forbidden_object_ids.add(rec.canonical_id)
            for alias in rec.aliases:
                self.forbidden_object_ids.add(alias)
            for obs_id in rec.observation_leakage_ids:
                self.forbidden_observation_ids.add(obs_id)
            if rec.host_leakage_cluster_id:
                self.forbidden_host_ids.add(rec.host_leakage_cluster_id)
            if rec.host_association and rec.host_association.host_name:
                self.forbidden_host_ids.add(rec.host_association.host_name)

    def audit_sample(self, sample: DatasetBSampleLineage) -> DatasetBSampleLineage:
        """
        Audits a single Dataset B sample against the forbidden test sets.
        Returns a new DatasetBSampleLineage instance with updated status.
        """
        # Rule 1: Missing lineage must be quarantined
        if not sample.parent_physical_object_id and not sample.source_observation_ids:
            return DatasetBSampleLineage(
                sample_id=sample.sample_id,
                parent_physical_object_id=sample.parent_physical_object_id,
                source_observation_ids=sample.source_observation_ids,
                host_galaxy_id=sample.host_galaxy_id,
                transformation_history=sample.transformation_history,
                lineage_status=LineageStatus.LINEAGE_UNAVAILABLE_QUARANTINED,
                rejection_reason="MISSING_PARENT_AND_OBSERVATION_LINEAGE",
            )

        # Rule 2: Physical object matching test set
        if sample.parent_physical_object_id and sample.parent_physical_object_id in self.forbidden_object_ids:
            return DatasetBSampleLineage(
                sample_id=sample.sample_id,
                parent_physical_object_id=sample.parent_physical_object_id,
                source_observation_ids=sample.source_observation_ids,
                host_galaxy_id=sample.host_galaxy_id,
                transformation_history=sample.transformation_history,
                lineage_status=LineageStatus.TEST_OVERLAP_REJECTED,
                rejection_reason=f"PARENT_OBJECT_IN_TEST_SET:{sample.parent_physical_object_id}",
            )

        # Rule 3: Observation matching test set
        for obs_id in sample.source_observation_ids:
            if obs_id in self.forbidden_observation_ids:
                return DatasetBSampleLineage(
                    sample_id=sample.sample_id,
                    parent_physical_object_id=sample.parent_physical_object_id,
                    source_observation_ids=sample.source_observation_ids,
                    host_galaxy_id=sample.host_galaxy_id,
                    transformation_history=sample.transformation_history,
                    lineage_status=LineageStatus.TEST_OVERLAP_REJECTED,
                    rejection_reason=f"SOURCE_OBSERVATION_IN_TEST_SET:{obs_id}",
                )

        # Rule 4: Host galaxy matching test set
        if sample.host_galaxy_id and sample.host_galaxy_id in self.forbidden_host_ids:
            return DatasetBSampleLineage(
                sample_id=sample.sample_id,
                parent_physical_object_id=sample.parent_physical_object_id,
                source_observation_ids=sample.source_observation_ids,
                host_galaxy_id=sample.host_galaxy_id,
                transformation_history=sample.transformation_history,
                lineage_status=LineageStatus.TEST_OVERLAP_REJECTED,
                rejection_reason=f"HOST_GALAXY_IN_TEST_SET:{sample.host_galaxy_id}",
            )

        return DatasetBSampleLineage(
            sample_id=sample.sample_id,
            parent_physical_object_id=sample.parent_physical_object_id,
            source_observation_ids=sample.source_observation_ids,
            host_galaxy_id=sample.host_galaxy_id,
            transformation_history=sample.transformation_history,
            lineage_status=LineageStatus.VERIFIED,
            rejection_reason=None,
        )

    def audit_batch(self, samples: List[DatasetBSampleLineage]) -> Dict[str, Any]:
        """
        Audits a collection of Dataset B samples.
        """
        verified = []
        rejected = []
        quarantined = []

        for s in samples:
            audited = self.audit_sample(s)
            if audited.lineage_status == LineageStatus.VERIFIED:
                verified.append(audited)
            elif audited.lineage_status == LineageStatus.TEST_OVERLAP_REJECTED:
                rejected.append(audited)
            else:
                quarantined.append(audited)

        return {
            "total_samples": len(samples),
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "quarantined_count": len(quarantined),
            "is_clean": len(rejected) == 0 and len(quarantined) == 0,
            "rejections": [
                {"sample_id": r.sample_id, "reason": r.rejection_reason}
                for r in rejected
            ],
            "quarantined": [
                {"sample_id": q.sample_id, "reason": q.rejection_reason}
                for q in quarantined
            ],
        }
