"""
ASTRAEUS Reference Corpus Auditor Module.
Executes the comprehensive 7-tier verification audit across Astronomical Reference Corpus v1.0.
Verifies identity uniqueness, zero cross-split leakage, boundary buffer quarantine,
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
    train_count: int
    val_count: int
    test_count: int
    quarantined_boundary_count: int

    # Gate 1: Identity & Alias Leakage
    identity_leakage_count: int
    alias_leakage_count: int
    duplicate_canonical_ids: int

    # Gate 2: Spatial HEALPix Isolation
    spatial_healpix_overlap_count: int

    # Gate 3: Boundary Buffer Quarantine
    boundary_buffer_violations: int

    # Gate 4: Temporal Hygiene
    temporal_leakage_count: int

    # Gate 5: Origin Integrity
    origin_counts: Dict[str, int]
    origin_misclassification_count: int

    # Gate 6: Cryptographic Seal Authenticity
    cryptographic_failures: int

    # Gate 7: Class Balance & Selection Metrics
    class_distribution_per_split: Dict[str, Dict[str, int]]
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

        # Split separation
        train_recs = [r for r in records if r.spatial_partition.split == "train"]
        val_recs = [r for r in records if r.spatial_partition.split == "val"]
        test_recs = [r for r in records if r.spatial_partition.split == "test"]
        quarantined = [r for r in records if r.spatial_partition.is_boundary_buffer_quarantined]

        # Gate 1: Identity Uniqueness & Cross-Split Leakage
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
        test_aliases: Set[str] = set()
        for r in test_recs:
            test_aliases.update(r.aliases)

        alias_leakage = len(train_aliases.intersection(test_aliases))

        # Gate 2: Spatial HEALPix Isolation
        train_pixels = {r.spatial_partition.healpix_nside32 for r in train_recs}
        val_pixels = {r.spatial_partition.healpix_nside32 for r in val_recs}
        test_pixels = {r.spatial_partition.healpix_nside32 for r in test_recs}

        spatial_overlap = (
            len(train_pixels.intersection(test_pixels))
            + len(train_pixels.intersection(val_pixels))
            + len(val_pixels.intersection(test_pixels))
        )

        # Gate 3: Boundary Buffer Violations
        # A violation occurs if a record has a conflicting neighbor and is within 5 arcsec but NOT quarantined
        boundary_violations = 0
        for r in records:
            if (
                r.spatial_partition.has_conflicting_neighbor
                and r.spatial_partition.distance_to_boundary_arcsec < 5.0
                and not r.spatial_partition.is_boundary_buffer_quarantined
            ):
                boundary_violations += 1

        # Gate 4: Temporal Cutoff
        temporal_leakage = 0
        for r in train_recs:
            for t in r.times_mjd:
                if t > t_split_cutoff_mjd:
                    temporal_leakage += 1

        # Gate 5: Origin Integrity
        origin_counts: Dict[str, int] = {}
        origin_misclass = 0
        for r in records:
            origin_counts[r.dataset_origin] = origin_counts.get(r.dataset_origin, 0) + 1
            if r.dataset_origin not in ["ARCHIVAL", "SYNTHETIC", "NUMERICALLY_SIMULATED", "INJECTED"]:
                origin_misclass += 1

        # Gate 6: Cryptographic Seals
        crypto_failures = 0
        for r in records:
            if not r.verify_authenticity(self.secret):
                crypto_failures += 1

        # Gate 7: Class Balance & Selection Profiling
        class_dist: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        snr_map: Dict[str, List[float]] = {}
        cadence_map: Dict[str, List[float]] = {}

        for r in records:
            split = r.spatial_partition.split
            cls = r.transient_class
            class_dist[split][cls] = class_dist[split].get(cls, 0) + 1

            snr_map.setdefault(cls, []).append(r.snr_median)
            cadence_map.setdefault(cls, []).append(r.cadence_median_hours)

        median_snr = {cls: float(np.median(vals)) for cls, vals in snr_map.items()}
        median_cad = {cls: float(np.median(vals)) for cls, vals in cadence_map.items()}

        passed = (
            duplicate_ids == 0
            and id_leakage == 0
            and alias_leakage == 0
            and spatial_overlap == 0
            and boundary_violations == 0
            and temporal_leakage == 0
            and origin_misclass == 0
            and crypto_failures == 0
        )

        return CorpusAuditReport(
            passed_all_gates=passed,
            total_records_audited=total,
            train_count=len(train_recs),
            val_count=len(val_recs),
            test_count=len(test_recs),
            quarantined_boundary_count=len(quarantined),
            identity_leakage_count=id_leakage,
            alias_leakage_count=alias_leakage,
            duplicate_canonical_ids=duplicate_ids,
            spatial_healpix_overlap_count=spatial_overlap,
            boundary_buffer_violations=boundary_violations,
            temporal_leakage_count=temporal_leakage,
            origin_counts=origin_counts,
            origin_misclassification_count=origin_misclass,
            cryptographic_failures=crypto_failures,
            class_distribution_per_split=class_dist,
            median_snr_per_class=median_snr,
            median_cadence_hours_per_class=median_cad,
        )
