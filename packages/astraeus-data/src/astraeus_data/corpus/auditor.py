"""
ASTRAEUS Reference Corpus Auditor Module.
Executes the comprehensive 7-tier verification audit across Astronomical Reference Corpus v1.0.
Verifies identity uniqueness, 3-tier leakage independence, boundary buffer quarantine segregation,
cryptographic HMAC seals, and class-balance hygiene before foundation model training.
"""

from typing import List, Dict, Set, Any
from pydantic import BaseModel, Field
import numpy as np

from astraeus_data.corpus.record import ReferenceObjectRecord, DEFAULT_PROVENANCE_SECRET


class CorpusAuditReport(BaseModel):
    """Formal audit report certifying Astronomical Reference Corpus v1.0."""
    passed_all_gates: bool
    total_records_audited: int
    eligible_entities_count: int
    train_count: int
    val_count: int
    test_count: int
    quarantined_boundary_count: int

    # Gate 1: Identity & Alias Leakage
    identity_leakage_count: int
    alias_leakage_count: int
    duplicate_canonical_ids: int

    # Gate 2: Spatial & Host Association Leakage
    spatial_healpix_overlap_count: int
    host_galaxy_cross_split_leakage_count: int

    # Gate 3: Observation & Cross-Survey Leakage
    observation_cross_split_leakage_count: int

    # Gate 4: Boundary Buffer Quarantine Segregation
    boundary_buffer_quarantine_violations: int

    # Gate 5: Temporal Stratification
    temporal_leakage_count: int

    # Gate 6: Origin Integrity & Mislabels
    origin_counts: Dict[str, int]
    origin_misclassification_count: int

    # Gate 7: Cryptographic Seal Authenticity
    cryptographic_failures: int

    # Class Balance & Selection Metrics
    per_class_population_accounting: Dict[str, Dict[str, int]]
    median_snr_per_class: Dict[str, float]
    median_cadence_hours_per_class: Dict[str, float]


class ReferenceCorpusAuditor:
    """Independent auditor verifying reference corpus integrity before pretraining."""

    def __init__(self, provenance_secret: bytes = DEFAULT_PROVENANCE_SECRET) -> None:
        self.secret = provenance_secret

    def audit_corpus(
        self,
        records: List[ReferenceObjectRecord],
        t_split_cutoff_mjd: float = 60000.0,
    ) -> CorpusAuditReport:
        """Executes complete audit across all records."""
        total = len(records)

        # Split separation - Quarantined records are strictly segregated
        train_recs = [r for r in records if r.spatial_partition.split == "train"]
        val_recs = [r for r in records if r.spatial_partition.split == "val"]
        test_recs = [r for r in records if r.spatial_partition.split == "test"]
        quarantined_recs = [r for r in records if r.spatial_partition.split == "quarantine"]

        eligible_count = len(train_recs) + len(val_recs) + len(test_recs)
        quarantine_count = len(quarantined_recs)

        # Gate 1: Canonical Identity & Alias Leakage
        all_canonical_ids = [r.canonical_id for r in records]
        duplicate_ids = len(all_canonical_ids) - len(set(all_canonical_ids))

        train_ids = {r.canonical_id for r in train_recs}
        val_ids = {r.canonical_id for r in val_recs}
        test_ids = {r.canonical_id for r in test_recs}

        id_leakage = (
            len(train_ids.intersection(test_ids))
            + len(train_ids.intersection(val_ids))
            + len(val_ids.intersection(test_ids))
        )

        train_aliases: Set[str] = set()
        for r in train_recs:
            train_aliases.update(r.aliases)
        val_aliases: Set[str] = set()
        for r in val_recs:
            val_aliases.update(r.aliases)
        test_aliases: Set[str] = set()
        for r in test_recs:
            test_aliases.update(r.aliases)

        alias_leakage = (
            len(train_aliases.intersection(test_aliases))
            + len(train_aliases.intersection(val_aliases))
            + len(val_aliases.intersection(test_aliases))
        )

        # Gate 2: Spatial HEALPix & Host Galaxy Cluster Leakage
        train_pixels = {r.spatial_partition.healpix_nside32 for r in train_recs}
        val_pixels = {r.spatial_partition.healpix_nside32 for r in val_recs}
        test_pixels = {r.spatial_partition.healpix_nside32 for r in test_recs}

        spatial_overlap = (
            len(train_pixels.intersection(test_pixels))
            + len(train_pixels.intersection(val_pixels))
            + len(val_pixels.intersection(test_pixels))
        )

        train_hosts = {r.host_leakage_cluster_id for r in train_recs if r.host_leakage_cluster_id}
        val_hosts = {r.host_leakage_cluster_id for r in val_recs if r.host_leakage_cluster_id}
        test_hosts = {r.host_leakage_cluster_id for r in test_recs if r.host_leakage_cluster_id}

        host_leakage = (
            len(train_hosts.intersection(test_hosts))
            + len(train_hosts.intersection(val_hosts))
            + len(val_hosts.intersection(test_hosts))
        )

        # Gate 3: Observation & Cross-Survey Leakage
        train_obs: Set[str] = set()
        for r in train_recs:
            train_obs.update(r.observation_leakage_ids)
        test_obs: Set[str] = set()
        for r in test_recs:
            test_obs.update(r.observation_leakage_ids)

        obs_leakage = len(train_obs.intersection(test_obs))

        # Gate 4: Boundary Buffer Quarantine Segregation
        # Quarantined records must have split == "quarantine" and NEVER enter train/val/test
        boundary_violations = 0
        for r in records:
            if r.spatial_partition.is_boundary_buffer_quarantined and r.spatial_partition.split != "quarantine":
                boundary_violations += 1
            if not r.spatial_partition.is_boundary_buffer_quarantined and r.spatial_partition.split == "quarantine":
                boundary_violations += 1

        # Gate 5: Temporal Cutoff
        temporal_leakage = 0
        for r in train_recs:
            for t in r.times_mjd:
                if t > t_split_cutoff_mjd:
                    temporal_leakage += 1

        # Gate 6: Origin Integrity
        origin_counts: Dict[str, int] = {}
        origin_misclass = 0
        for r in records:
            origin_counts[r.dataset_origin] = origin_counts.get(r.dataset_origin, 0) + 1
            if r.dataset_origin not in ["ARCHIVAL", "SYNTHETIC", "NUMERICALLY_SIMULATED", "INJECTED"]:
                origin_misclass += 1

        # Gate 7: Cryptographic Seals & MAC Verification
        crypto_failures = 0
        for r in records:
            if not r.verify_authenticity(self.secret):
                crypto_failures += 1

        # Class Distribution & Selection Function Accounting Table
        all_classes = sorted(list({r.transient_class for r in records}))
        class_table: Dict[str, Dict[str, int]] = {}
        snr_map: Dict[str, List[float]] = {}
        cadence_map: Dict[str, List[float]] = {}

        for cls in all_classes:
            class_table[cls] = {
                "total": len([r for r in records if r.transient_class == cls]),
                "train": len([r for r in train_recs if r.transient_class == cls]),
                "val": len([r for r in val_recs if r.transient_class == cls]),
                "test": len([r for r in test_recs if r.transient_class == cls]),
                "quarantine": len([r for r in quarantined_recs if r.transient_class == cls]),
            }

        for r in records:
            cls = r.transient_class
            snr_map.setdefault(cls, []).append(r.snr_median)
            cadence_map.setdefault(cls, []).append(r.cadence_median_hours)

        median_snr = {cls: float(np.median(vals)) for cls, vals in snr_map.items()}
        median_cad = {cls: float(np.median(vals)) for cls, vals in cadence_map.items()}

        passed = (
            duplicate_ids == 0
            and id_leakage == 0
            and alias_leakage == 0
            and spatial_overlap == 0
            and host_leakage == 0
            and obs_leakage == 0
            and boundary_violations == 0
            and temporal_leakage == 0
            and origin_misclass == 0
            and crypto_failures == 0
            and eligible_count + quarantine_count == total
        )

        return CorpusAuditReport(
            passed_all_gates=passed,
            total_records_audited=total,
            eligible_entities_count=eligible_count,
            train_count=len(train_recs),
            val_count=len(val_recs),
            test_count=len(test_recs),
            quarantined_boundary_count=quarantine_count,
            identity_leakage_count=id_leakage,
            alias_leakage_count=alias_leakage,
            duplicate_canonical_ids=duplicate_ids,
            spatial_healpix_overlap_count=spatial_overlap,
            host_galaxy_cross_split_leakage_count=host_leakage,
            observation_cross_split_leakage_count=obs_leakage,
            boundary_buffer_quarantine_violations=boundary_violations,
            temporal_leakage_count=temporal_leakage,
            origin_counts=origin_counts,
            origin_misclassification_count=origin_misclass,
            cryptographic_failures=crypto_failures,
            per_class_population_accounting=class_table,
            median_snr_per_class=median_snr,
            median_cadence_hours_per_class=median_cad,
        )
