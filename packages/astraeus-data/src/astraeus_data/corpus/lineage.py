"""
ASTRAEUS Dataset B Lineage & Provenance Graph Audit Module.
Enforces explicit parentage, observation provenance, and host identity tracking
to ensure pretraining datasets (Dataset B) contain zero derivatives, cutouts,
or augmentations derived from the frozen evaluation test cohort (Dataset A).
Includes cycle detection, recursive transitive ancestry checks, and pilot cohort auditing.
"""

from typing import List, Optional, Set, Dict, Any, Tuple
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from astraeus_data.corpus.record import ReferenceObjectRecord


class LineageStatus(str, Enum):
    VERIFIED = "VERIFIED"
    TEST_OVERLAP_REJECTED = "TEST_OVERLAP_REJECTED"
    LINEAGE_UNAVAILABLE_QUARANTINED = "LINEAGE_UNAVAILABLE_QUARANTINED"
    CYCLE_DETECTED_REJECTED = "CYCLE_DETECTED_REJECTED"


class DatasetBSampleLineage(BaseModel):
    """
    Explicit provenance graph lineage for a pretraining (Dataset B) sample.
    Tracks direct parent physical objects, full ancestor chains, source observations,
    and host galaxy references across all transformations and augmentations.
    """
    model_config = ConfigDict(frozen=True)

    sample_id: str
    parent_physical_object_ids: List[str] = Field(default_factory=list)
    ancestor_chain: List[str] = Field(default_factory=list)
    source_observation_ids: List[str] = Field(default_factory=list)
    host_galaxy_id: Optional[str] = None
    host_association_confidence: float = 1.0
    transformation_history: List[str] = Field(default_factory=list)
    lineage_status: LineageStatus = LineageStatus.VERIFIED
    rejection_reason: Optional[str] = None

    @property
    def parent_physical_object_id(self) -> Optional[str]:
        """Convenience property for single-parent backward compatibility."""
        return self.parent_physical_object_ids[0] if self.parent_physical_object_ids else None


class ProvenanceGraphAuditor:
    """
    Audits Dataset B samples against Dataset A test objects to prevent
    information leakage through derived light curves, cutouts, or counterparts.
    Enforces DAG acyclicity, transitive ancestor inspection, and host association governance.
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

    @staticmethod
    def detect_cycles(dag_edges: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
        """
        Validates that the parent-child relationships form a strictly acyclic directed graph (DAG).
        Uses DFS 3-color cycle detection (0=WHITE, 1=GRAY, 2=BLACK).
        Returns (has_cycle, cycle_path).
        """
        color: Dict[str, int] = {}
        parent_map: Dict[str, str] = {}
        cycle_path: List[str] = []

        all_nodes = set(dag_edges.keys())
        for targets in dag_edges.values():
            all_nodes.update(targets)

        for node in all_nodes:
            color[node] = 0

        def dfs(u: str) -> bool:
            color[u] = 1  # GRAY (visiting)
            for v in dag_edges.get(u, []):
                if color.get(v, 0) == 1:
                    # Found cycle
                    curr = u
                    cycle_path.append(v)
                    while curr != v:
                        cycle_path.append(curr)
                        curr = parent_map.get(curr, v)
                    cycle_path.append(v)
                    cycle_path.reverse()
                    return True
                elif color.get(v, 0) == 0:
                    parent_map[v] = u
                    if dfs(v):
                        return True
            color[u] = 2  # BLACK (visited)
            return False

        for node in all_nodes:
            if color[node] == 0:
                if dfs(node):
                    return True, cycle_path

        return False, []

    def resolve_all_ancestors(
        self,
        node: str,
        dag_edges: Dict[str, List[str]],
        visited: Optional[Set[str]] = None,
    ) -> Set[str]:
        """Recursively resolves all upstream ancestors of a sample in the provenance DAG."""
        if visited is None:
            visited = set()
        ancestors: Set[str] = set()
        for parent in dag_edges.get(node, []):
            if parent not in visited:
                visited.add(parent)
                ancestors.add(parent)
                ancestors.update(self.resolve_all_ancestors(parent, dag_edges, visited))
        return ancestors

    def audit_sample(
        self,
        sample: DatasetBSampleLineage,
        dag_edges: Optional[Dict[str, List[str]]] = None,
    ) -> DatasetBSampleLineage:
        """
        Audits a single Dataset B sample against the forbidden test sets,
        including direct parents, recursive transitive ancestors, and source observations.
        """
        # Rule 1: Missing lineage must be quarantined
        has_parents = len(sample.parent_physical_object_ids) > 0
        has_obs = len(sample.source_observation_ids) > 0
        if not has_parents and not has_obs:
            return sample.model_copy(update={
                "lineage_status": LineageStatus.LINEAGE_UNAVAILABLE_QUARANTINED,
                "rejection_reason": "MISSING_PARENT_AND_OBSERVATION_LINEAGE",
            })

        # Rule 2: Cycle check in declared lineage
        all_parents = set(sample.parent_physical_object_ids)
        if sample.sample_id in all_parents:
            return sample.model_copy(update={
                "lineage_status": LineageStatus.CYCLE_DETECTED_REJECTED,
                "rejection_reason": f"SELF_REFERENTIAL_PARENT_CYCLE:{sample.sample_id}",
            })

        # Rule 3: Direct physical parent matching test set
        for p_id in all_parents:
            if p_id in self.forbidden_object_ids:
                return sample.model_copy(update={
                    "lineage_status": LineageStatus.TEST_OVERLAP_REJECTED,
                    "rejection_reason": f"DIRECT_PARENT_IN_TEST_SET:{p_id}",
                })

        # Rule 4: Transitive upstream ancestor matching test set
        transitive_ancestors = set(sample.ancestor_chain)
        if dag_edges:
            transitive_ancestors.update(self.resolve_all_ancestors(sample.sample_id, dag_edges))

        for anc_id in transitive_ancestors:
            if anc_id in self.forbidden_object_ids:
                return sample.model_copy(update={
                    "lineage_status": LineageStatus.TEST_OVERLAP_REJECTED,
                    "rejection_reason": f"TRANSITIVE_ANCESTOR_IN_TEST_SET:{anc_id}",
                })

        # Rule 5: Observation matching test set
        for obs_id in sample.source_observation_ids:
            if obs_id in self.forbidden_observation_ids:
                return sample.model_copy(update={
                    "lineage_status": LineageStatus.TEST_OVERLAP_REJECTED,
                    "rejection_reason": f"SOURCE_OBSERVATION_IN_TEST_SET:{obs_id}",
                })

        # Rule 6: Host galaxy matching test set (governed by host association confidence)
        if sample.host_galaxy_id and sample.host_association_confidence >= 0.5:
            if sample.host_galaxy_id in self.forbidden_host_ids:
                return sample.model_copy(update={
                    "lineage_status": LineageStatus.TEST_OVERLAP_REJECTED,
                    "rejection_reason": f"HOST_GALAXY_IN_TEST_SET:{sample.host_galaxy_id}",
                })

        return sample.model_copy(update={
            "lineage_status": LineageStatus.VERIFIED,
            "rejection_reason": None,
        })

    def audit_batch(
        self,
        samples: List[DatasetBSampleLineage],
        dag_edges: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Audits a collection of Dataset B samples with global DAG cycle validation.
        """
        if dag_edges is None:
            dag_edges = {}
            for s in samples:
                if s.parent_physical_object_ids:
                    dag_edges[s.sample_id] = list(s.parent_physical_object_ids)

        has_cycle, cycle_path = self.detect_cycles(dag_edges)

        verified = []
        rejected = []
        quarantined = []

        for s in samples:
            audited = self.audit_sample(s, dag_edges=dag_edges)
            if has_cycle and s.sample_id in cycle_path:
                audited = audited.model_copy(update={
                    "lineage_status": LineageStatus.CYCLE_DETECTED_REJECTED,
                    "rejection_reason": f"CYCLIC_PROVENANCE_PATH:{'->'.join(cycle_path)}",
                })

            if audited.lineage_status == LineageStatus.VERIFIED:
                verified.append(audited)
            elif audited.lineage_status in [LineageStatus.TEST_OVERLAP_REJECTED, LineageStatus.CYCLE_DETECTED_REJECTED]:
                rejected.append(audited)
            else:
                quarantined.append(audited)

        return {
            "total_samples": len(samples),
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "quarantined_count": len(quarantined),
            "dag_has_cycles": has_cycle,
            "cycle_path": cycle_path,
            "is_clean": len(rejected) == 0 and len(quarantined) == 0 and not has_cycle,
            "rejections": [
                {"sample_id": r.sample_id, "reason": r.rejection_reason}
                for r in rejected
            ],
            "quarantined": [
                {"sample_id": q.sample_id, "reason": q.rejection_reason}
                for q in quarantined
            ],
        }

    def create_pilot_dataset_b_cohort(
        self,
        n_clean: int = 460,
        n_missing: int = 25,
        n_overlap: int = 15,
    ) -> Tuple[List[DatasetBSampleLineage], Dict[str, List[str]]]:
        """
        Constructs a representative pilot Dataset B pretraining cohort ($N=500$)
        simulating temporal sequences from ZTF, PS1, and DECam to demonstrate
        empirical DAG lineage auditing and strict test-set isolation.
        """
        samples: List[DatasetBSampleLineage] = []
        dag_edges: Dict[str, List[str]] = {}

        # 1. Clean pretraining samples derived from training objects and non-overlapping surveys
        for i in range(n_clean):
            sid = f"ASTRAEUS-PRETRAIN-PILOT-{i:05d}"
            parent_id = f"ARCHIVAL-TRAIN-OBJ-{(i % 150):04d}"
            obs_id = f"OBS-ZTF-PRETRAIN-{i:06d}"
            host_id = f"HOST-CATALOG-SDSS-{(i % 80):04d}"
            samples.append(DatasetBSampleLineage(
                sample_id=sid,
                parent_physical_object_ids=[parent_id],
                ancestor_chain=[parent_id],
                source_observation_ids=[obs_id],
                host_galaxy_id=host_id,
                host_association_confidence=0.95,
                transformation_history=["TIME_SERIES_STANDARDIZATION", "PHASE_ALIGNMENT"],
            ))
            dag_edges[sid] = [parent_id]

        # 2. Lineage-unavailable samples (must be quarantined)
        for j in range(n_missing):
            sid = f"ASTRAEUS-PRETRAIN-UNLABELED-{j:04d}"
            samples.append(DatasetBSampleLineage(
                sample_id=sid,
                parent_physical_object_ids=[],
                source_observation_ids=[],
                transformation_history=["UNSOURCED_SCRAPED_LIGHTCURVE"],
            ))

        # 3. Test-overlap samples (must be rejected)
        test_ids_list = sorted(list(self.forbidden_object_ids))
        for k in range(n_overlap):
            target_test_id = test_ids_list[k % len(test_ids_list)]
            sid = f"ASTRAEUS-PRETRAIN-INJECTED-DERIV-{k:04d}"
            if k % 3 == 0:
                # Direct parent overlap
                samples.append(DatasetBSampleLineage(
                    sample_id=sid,
                    parent_physical_object_ids=[target_test_id],
                    source_observation_ids=[f"OBS-DERIV-{k:04d}"],
                    transformation_history=["AUGMENTATION_GAUSSIAN_NOISE"],
                ))
                dag_edges[sid] = [target_test_id]
            elif k % 3 == 1:
                # Transitive 2-hop ancestor overlap: Sample -> Intermediate -> Test Object
                interm_id = f"INTERMEDIATE-AUG-{k:04d}"
                samples.append(DatasetBSampleLineage(
                    sample_id=sid,
                    parent_physical_object_ids=[interm_id],
                    ancestor_chain=[interm_id, target_test_id],
                    source_observation_ids=[f"OBS-DERIV-2HOP-{k:04d}"],
                    transformation_history=["TEMPORAL_SUBSAMPLING", "INTERMEDIATE_CUTOUT"],
                ))
                dag_edges[sid] = [interm_id]
                dag_edges[interm_id] = [target_test_id]
            else:
                # Source observation overlap
                test_obs_list = sorted(list(self.forbidden_observation_ids))
                matched_obs = test_obs_list[0] if test_obs_list else f"OBS-LEAK-{target_test_id}"
                samples.append(DatasetBSampleLineage(
                    sample_id=sid,
                    parent_physical_object_ids=[f"PSEUDO-PARENT-{k:04d}"],
                    source_observation_ids=[matched_obs],
                    transformation_history=["PHOTOMETRIC_RECALIBRATION"],
                ))
                dag_edges[sid] = [f"PSEUDO-PARENT-{k:04d}"]

        return samples, dag_edges
