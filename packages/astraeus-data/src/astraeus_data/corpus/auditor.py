"""
ASTRAEUS Reference Corpus Auditor Module.
Executes the comprehensive 7-tier verification audit across Astronomical Reference Corpus v1.0.
Verifies identity uniqueness, 3-tier leakage independence, boundary buffer quarantine segregation,
cryptographic HMAC seals, and class-balance hygiene before foundation model training.
"""

from typing import List, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field
import numpy as np
from scipy.stats import beta

from astraeus_data.corpus.record import ReferenceObjectRecord, DEFAULT_PROVENANCE_SECRET
from astraeus_data.corpus.geometry import (
    verify_healpix_boundary_convergence,
    validate_healpix_boundary_convergence_global,
)


def compute_clopper_pearson(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson binomial confidence interval [lower, upper]."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - confidence
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (round(lower, 4), round(upper, 4))


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

    # Stage 1 v1.0.4 Additions: Small Test-Set Disclosures & Quarantine Diagnostics
    test_clopper_pearson_intervals: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    boundary_convergence_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    global_convergence_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    quarantine_breakdown_by_class: Dict[str, int] = Field(default_factory=dict)
    quarantine_breakdown_by_original_split: Dict[str, int] = Field(default_factory=dict)
    quarantine_breakdown_by_reason: Dict[str, int] = Field(default_factory=dict)
    quarantine_diagnostic_table: List[Dict[str, Any]] = Field(default_factory=list)
    archival_objects: List[str] = Field(default_factory=list)
    simulated_objects_count: int = 0
    synthetic_objects_count: int = 0


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

        # Stage 1 v1.0.3 Additions: Clopper-Pearson Intervals for Test Set with Explicit (k, n) Integers
        clopper_map: Dict[str, Dict[str, Any]] = {}
        for cls in all_classes:
            n_t = class_table[cls]["test"]
            low_100, up_100 = compute_clopper_pearson(n_t, n_t, 0.95)
            k_m1 = max(0, n_t - 1)
            low_m1, up_m1 = compute_clopper_pearson(k_m1, n_t, 0.95)
            clopper_map[cls] = {
                "n_test": n_t,
                "k_perfect": n_t,
                "recall_100_ci_lower": low_100,
                "recall_100_ci_upper": up_100,
                "k_minus1": k_m1,
                "recall_minus1_ci_lower": low_m1,
                "recall_minus1_ci_upper": up_m1,
            }

        # Quarantine Breakdowns
        quar_by_class: Dict[str, int] = {}
        quar_by_split: Dict[str, int] = {}
        quar_by_reason: Dict[str, int] = {}
        for r in quarantined_recs:
            quar_by_class[r.transient_class] = quar_by_class.get(r.transient_class, 0) + 1
            sp = r.spatial_partition
            quar_by_split[sp.original_assigned_split] = quar_by_split.get(sp.original_assigned_split, 0) + 1
            reason = sp.quarantine_reason or "SPHERICAL_CELL_EDGE_BUFFER_VIOLATION"
            quar_by_reason[reason] = quar_by_reason.get(reason, 0) + 1

        # Diagnostic Table: 4 Illustrative Cases + Closest Actual Objects
        diag_table: List[Dict[str, Any]] = [
            {"object": "Case A (Illustrative)", "distance_to_actual_edge_arcsec": 2.0, "astrometric_uncertainty_arcsec": 0.5, "sigma_perp_arcsec": 0.5, "crossing_probability": 0.00003, "conflicting_neighbour": True, "decision": "Quarantine", "policy": "SPHERICAL_CELL_EDGE_BUFFER_VIOLATION"},
            {"object": "Case B (Illustrative)", "distance_to_actual_edge_arcsec": 500.0, "astrometric_uncertainty_arcsec": 0.5, "sigma_perp_arcsec": 0.5, "crossing_probability": 0.0, "conflicting_neighbour": True, "decision": "Retain", "policy": "SAFE_INTERIOR"},
            {"object": "Case C (Illustrative)", "distance_to_actual_edge_arcsec": 8.0, "astrometric_uncertainty_arcsec": 4.0, "sigma_perp_arcsec": 4.0, "crossing_probability": 0.02275, "conflicting_neighbour": True, "decision": "Quarantine", "policy": "ASTROMETRIC_UNCERTAINTY_CROSSING_RISK"},
            {"object": "Case D (Illustrative)", "distance_to_actual_edge_arcsec": 2.0, "astrometric_uncertainty_arcsec": 0.5, "sigma_perp_arcsec": 0.5, "crossing_probability": 0.00003, "conflicting_neighbour": False, "decision": "Retain", "policy": "INTERNAL_SPLIT_BOUNDARY"},
        ]
        # Sort actual objects by distance to true boundary
        sorted_by_edge = sorted(records, key=lambda r: r.spatial_partition.distance_to_boundary_arcsec)
        for r in sorted_by_edge[:6]:
            sp = r.spatial_partition
            ast_err = max(r.coordinates.uncertainty_mas.get("ra", 30.0), r.coordinates.uncertainty_mas.get("dec", 30.0)) / 1000.0
            dec = "Quarantine" if sp.is_boundary_buffer_quarantined else f"Retain ({sp.split})"
            pol = sp.quarantine_reason if sp.is_boundary_buffer_quarantined else "SAFE_INTERIOR"
            diag_table.append({
                "object": r.canonical_id,
                "distance_to_actual_edge_arcsec": sp.distance_to_boundary_arcsec,
                "astrometric_uncertainty_arcsec": round(ast_err, 3),
                "sigma_perp_arcsec": sp.projected_normal_astrometric_sigma_arcsec,
                "crossing_probability": sp.crossing_probability,
                "conflicting_neighbour": sp.has_conflicting_neighbor,
                "decision": dec,
                "policy": pol,
            })

        # Stage 1 v1.0.5: Empirical HEALPix boundary convergence across polar and equatorial cells
        convergence_diagnostics = [
            verify_healpix_boundary_convergence(pix)
            for pix in [0, 50, 500, 1049, 3201, 6000, 9500, 12287]
        ]
        global_conv = validate_healpix_boundary_convergence_global(num_sample_pixels=128)

        # Origin Cohort Segregation
        archival_ids = [r.canonical_id for r in records if r.dataset_origin == "ARCHIVAL"]
        simulated_count = len([r for r in records if r.dataset_origin == "NUMERICALLY_SIMULATED"])
        synthetic_count = len([r for r in records if r.dataset_origin == "SYNTHETIC"])

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
            test_clopper_pearson_intervals=clopper_map,
            boundary_convergence_diagnostics=convergence_diagnostics,
            global_convergence_diagnostics=global_conv,
            quarantine_breakdown_by_class=quar_by_class,
            quarantine_breakdown_by_original_split=quar_by_split,
            quarantine_breakdown_by_reason=quar_by_reason,
            quarantine_diagnostic_table=diag_table,
            archival_objects=archival_ids,
            simulated_objects_count=simulated_count,
            synthetic_objects_count=synthetic_count,
        )
