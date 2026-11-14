"""
ASTRAEUS Real-Survey 4-Entity Identity Graph, Reconciliation & Gate A Integrity Engine (P0.2).

Implements Sprint 2B scientific-integrity invariants:
1. Explicit 4-entity identity graph separating:
   - `PersistentSkyLocus`: persistent survey/host sky location (`ra`, `dec`, `HEALPix nside=32`)
   - `PhysicalTransientEpisode`: time-localized astrophysical transient or variable episode
   - `CatalogReportRecord`: raw discovery / classification reports (TNS, Gaia DR3, ZTF)
   - `HostCounterpartRecord`: independent galaxy / Gaia nuclear counterpart associations
2. Epoch-aware episode deduplication so distinct outbursts separated in time at the same
   persistent locus (e.g., `SN2021ued` vs `SN2025aghl`) share one `PersistentSkyLocus`
   and spatial split without being transitively merged into one false event.
3. Explicit quarantine of truly concurrent multi-counterpart spatial ambiguities and
   coordinate/alias conflicts.
4. Strict `L1–L5` provenance + zero broker truth in primary labels + zero `AT`-as-negative.
5. HEALPix (`configs/data_splits_healpix.json`) + temporal cutoff split assignment +
   Stage 1/2 reference corpus overlap auditing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from astropy.coordinates import SkyCoord, search_around_sky
import astropy.units as u
from astropy_healpix import HEALPix
import numpy as np
import pyarrow.dataset as ds
from pydantic import BaseModel, ConfigDict, Field

from astraeus_core.coordinates import CelestialCoordinates
from astraeus_data.crossmatch.cross_survey_matcher import CrossSurveyMatcher
from astraeus_data.partitioning import load_split_manifest
from astraeus_data.real_survey.external_catalog_connector import ExternalCatalogRecord
from astraeus_data.real_survey.timestamps import AvailabilityStatus
from astraeus_data.real_survey.tns_connector import (
    CoarseTransientClass,
    EvidenceTier,
    REAL_SURVEY_TAXONOMY_VERSION,
    TnsObjectRecord,
)


class ReconciliationStatus(str, Enum):
    DIRECT_ID_AND_SPATIAL_MATCH = "DIRECT_ID_AND_SPATIAL_MATCH"
    UNIQUE_SPATIAL_MATCH = "UNIQUE_SPATIAL_MATCH"
    EXTERNAL_CATALOG_MATCH = "EXTERNAL_CATALOG_MATCH"
    DETECTION_ARTIFACT_TRACK = "DETECTION_ARTIFACT_TRACK"
    AMBIGUOUS_MULTI_COUNTERPART = "AMBIGUOUS_MULTI_COUNTERPART"
    COORDINATE_ALIAS_CONFLICT = "COORDINATE_ALIAS_CONFLICT"
    UNMATCHED = "UNMATCHED"


class ZtfObjectSummary(BaseModel):
    """Minimal canonical survey candidate record for 4-entity identity reconciliation."""

    model_config = ConfigDict(frozen=True)

    oid: str
    ra_deg: float
    dec_deg: float
    firstmjd: Optional[float] = None
    lastmjd: Optional[float] = None
    ndet: Optional[int] = None
    source_system: str = "ALERCE_ZTF_V1"
    has_msip_alert: bool = False
    has_dr24_lc: bool = False
    has_forced_photometry: bool = False
    has_alerce_access: bool = False
    has_independent_tns: bool = False
    is_detection_artifact: bool = False
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN
    dr24_objectid: Optional[str] = None
    gaia_source_id: Optional[str] = None


class PersistentSkyLocus(BaseModel):
    """Entity 1: Persistent sky position / host locus sharing a single spatial split."""

    model_config = ConfigDict(frozen=True)

    sky_locus_id: str
    ra_deg: float
    dec_deg: float
    healpix_nside32: int
    assigned_base_split: str
    member_oids: List[str] = Field(default_factory=list)
    linked_tns_names: List[str] = Field(default_factory=list)
    linked_external_ids: List[str] = Field(default_factory=list)
    episode_count: int = 1


class PhysicalTransientEpisode(BaseModel):
    """Entity 2: Time-localized physical transient episode (or persistent variable target)."""

    model_config = ConfigDict(frozen=True)

    physical_event_id: str
    sky_locus_id: str
    ra_deg: float
    dec_deg: float
    healpix_nside32: int
    t_start_mjd: Optional[float] = None
    t_end_mjd: Optional[float] = None
    assigned_split: str
    temporal_split: str
    member_oids: List[str] = Field(default_factory=list)
    matched_tns_name: Optional[str] = None
    matched_external_id: Optional[str] = None
    coarse_class: CoarseTransientClass = CoarseTransientClass.UNCLASSIFIED
    fine_class: str = "UNLABELED"
    evidence_tier: EvidenceTier = EvidenceTier.L5_UNCLASSIFIED
    taxonomy_version: str = REAL_SURVEY_TAXONOMY_VERSION
    is_supernova: Optional[bool] = None
    is_quarantined: bool = False
    quarantine_reasons: List[str] = Field(default_factory=list)


class CatalogReportRecord(BaseModel):
    """Entity 3: Individual upstream discovery/classification/catalog report."""

    model_config = ConfigDict(frozen=True)

    report_id: str
    source_system: str
    source_record_id: str
    sky_locus_id: str
    linked_episode_id: Optional[str] = None
    ra_deg: float
    dec_deg: float
    evidence_tier: EvidenceTier
    raw_class: Optional[str] = None
    discovery_mjd_utc: Optional[float] = None
    class_bibcodes: Optional[str] = None
    has_host_redshift_only: bool = False


class CatalogCounterpartRecord(BaseModel):
    """Entity 4: Astronomical catalogue cross-match or host galaxy counterpart at a sky locus."""

    model_config = ConfigDict(frozen=True)

    counterpart_id: str
    sky_locus_id: str
    source_catalog: str
    ra_deg: float
    dec_deg: float
    angular_offset_arcsec: Optional[float] = None
    host_redshift: Optional[float] = None
    gaia_source_id: Optional[str] = None
    is_nuclear_locus: bool = False
    counterpart_type: str = "CATALOG_CROSSMATCH"


# Backward-compatible alias for Entity 4
HostCounterpartRecord = CatalogCounterpartRecord


class ReconciledSurveyEvent(BaseModel):
    """Reconciled survey candidate view linked into the 4-entity Identity Graph."""

    model_config = ConfigDict(frozen=True)

    physical_event_id: str
    sky_locus_id: str
    host_counterpart_id: Optional[str] = None
    ztf_oid: str
    ra_deg: float
    dec_deg: float
    healpix_nside32: int
    assigned_split: str  # "train" | "val" | "test" | "quarantine"
    temporal_split: str  # "historical_dev" | "blind_temporal_test" | "unknown_epoch"
    reconciliation_status: ReconciliationStatus
    matched_tns_name: Optional[str] = None
    matched_external_id: Optional[str] = None
    separation_arcsec: Optional[float] = None
    competing_tns_names: List[str] = Field(default_factory=list)
    competing_ztf_oids: List[str] = Field(default_factory=list)
    coarse_class: CoarseTransientClass = CoarseTransientClass.UNCLASSIFIED
    fine_class: str = "UNLABELED"
    evidence_tier: EvidenceTier = EvidenceTier.L5_UNCLASSIFIED
    label_provenance: EvidenceTier = EvidenceTier.L5_UNCLASSIFIED
    taxonomy_version: str = REAL_SURVEY_TAXONOMY_VERSION
    is_supernova: Optional[bool] = None
    availability_status: AvailabilityStatus = AvailabilityStatus.AVAILABILITY_TIME_UNKNOWN
    has_operational_pub_time: bool = False
    has_msip_alert: bool = False
    has_dr24_lc: bool = False
    has_forced_photometry: bool = False
    has_alerce_access: bool = False
    has_independent_tns: bool = False
    discovery_mjd_utc: Optional[float] = None
    first_alert_mjd_utc: Optional[float] = None
    last_alert_mjd_utc: Optional[float] = None
    reference_corpus_overlap_id: Optional[str] = None
    is_quarantined: bool = False
    quarantine_reasons: List[str] = Field(default_factory=list)


class IdentityGraphBundle(BaseModel):
    """Full serialized 4-entity identity graph."""

    sky_loci: List[PersistentSkyLocus]
    physical_episodes: List[PhysicalTransientEpisode]
    catalog_reports: List[CatalogReportRecord]
    host_counterparts: List[HostCounterpartRecord]
    reconciled_candidates: List[ReconciledSurveyEvent]


class DataIntegrityGateReport(BaseModel):
    """Machine-verifiable Gate A / P0.2 report for a reconciled real-survey cohort."""

    passed_gate_a: bool
    total_ztf_objects: int
    unique_physical_events: int
    unique_sky_loci: int
    total_catalog_reports: int
    total_host_counterparts: int
    multi_episode_loci_count: int
    status_counts: Dict[str, int]
    evidence_tier_counts: Dict[str, int]
    provenance_counts: Dict[str, int]
    availability_status_counts: Dict[str, int]
    split_counts: Dict[str, int]
    physical_event_split_leakage_count: int
    sky_locus_split_leakage_count: int
    spatial_cluster_split_leakage_count: int
    alias_split_leakage_count: int
    unclassified_as_negative_count: int
    broker_truth_in_primary_count: int
    reference_corpus_overlap_count: int
    overlapping_reference_ids: List[str] = Field(default_factory=list)
    quarantined_count: int = 0


def _connected_components_spatial(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    radius_arcsec: float,
) -> np.ndarray:
    """Single-linkage cluster IDs for persistent sky loci within `radius_arcsec`."""
    n = len(ra_deg)
    if n == 0:
        return np.array([], dtype=int)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra_r, rb_r = find(a), find(b)
        if ra_r != rb_r:
            if ra_r < rb_r:
                parent[rb_r] = ra_r
            else:
                parent[ra_r] = rb_r

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    idx1, idx2, _, _ = search_around_sky(coords, coords, radius_arcsec * u.arcsec)
    for i, j in zip(idx1.tolist(), idx2.tolist()):
        if i < j:
            union(i, j)

    roots = [find(i) for i in range(n)]
    remap: Dict[int, int] = {}
    out = np.empty(n, dtype=int)
    for idx, r in enumerate(roots):
        if r not in remap:
            remap[r] = len(remap)
        out[idx] = remap[r]
    return out


def _is_tns_epoch_compatible(
    z: ZtfObjectSummary,
    tns_disc_mjd: Optional[float],
    episode_separation_days: float,
) -> bool:
    """
    Returns True iff a TNS discovery epoch is temporally compatible with a ZTF candidate's
    observed window (or if either epoch is unknown, conservatively assuming possible overlap).
    """
    if tns_disc_mjd is None or z.firstmjd is None:
        return True
    z_start = float(z.firstmjd)
    z_end = float(z.lastmjd) if z.lastmjd is not None else z_start
    if z_start - episode_separation_days <= tns_disc_mjd <= z_end + episode_separation_days:
        return True
    return False


def _windows_overlap(
    z_a: ZtfObjectSummary,
    z_b: ZtfObjectSummary,
    margin_days: float,
) -> bool:
    """True if two ZTF lightcurve windows overlap within `margin_days` (or unknown epoch)."""
    if z_a.firstmjd is None or z_b.firstmjd is None:
        return True
    a0 = float(z_a.firstmjd)
    a1 = float(z_a.lastmjd) if z_a.lastmjd is not None else a0
    b0 = float(z_b.firstmjd)
    b1 = float(z_b.lastmjd) if z_b.lastmjd is not None else b0
    return (a0 - margin_days) <= b1 and (b0 - margin_days) <= a1


def load_reference_corpus_coords(
    ref_corpus_dir: Union[str, Path] = "data/reference_corpus_v1",
) -> Tuple[List[str], Optional[SkyCoord]]:
    """Loads existing Stage 1/2 Reference Corpus v1 IDs and coordinates for overlap auditing."""
    p = Path(ref_corpus_dir)
    if not p.exists():
        return [], None
    table = ds.dataset(str(p), format="parquet", partitioning="hive").to_table(
        columns=["canonical_id", "ra_deg", "dec_deg"]
    )
    if table.num_rows == 0:
        return [], None
    ids = [str(x) for x in table["canonical_id"].to_pylist()]
    coords = SkyCoord(
        ra=np.asarray(table["ra_deg"].to_pylist(), dtype=float) * u.deg,
        dec=np.asarray(table["dec_deg"].to_pylist(), dtype=float) * u.deg,
        frame="icrs",
    )
    return ids, coords


def build_identity_graph_and_reconcile(
    ztf_objects: List[ZtfObjectSummary],
    tns_records: List[TnsObjectRecord],
    external_records: Optional[List[ExternalCatalogRecord]] = None,
    split_manifest_path: Union[str, Path] = "configs/data_splits_healpix.json",
    reference_corpus_dir: Union[str, Path] = "data/reference_corpus_v1",
    match_radius_arcsec: float = 2.0,
    ambiguity_radius_arcsec: float = 5.0,
    dedup_radius_arcsec: float = 2.0,
    episode_separation_days: float = 180.0,
    temporal_cutoff_mjd: float = 60310.0,  # 2024-01-01 00:00 UTC default
) -> Tuple[IdentityGraphBundle, DataIntegrityGateReport]:
    """
    Reconciles survey candidates against TNS + independent external (`L3`) catalogs using an
    explicit 4-entity identity graph (`PersistentSkyLocus`, `PhysicalTransientEpisode`,
    `CatalogReportRecord`, `HostCounterpartRecord`) and verifies Gate A integrity invariants.
    """
    split_map = load_split_manifest(split_manifest_path)
    ref_ids, ref_coords = load_reference_corpus_coords(reference_corpus_dir)
    ext_records = external_records or []

    if not ztf_objects:
        empty_bundle = IdentityGraphBundle(
            sky_loci=[],
            physical_episodes=[],
            catalog_reports=[],
            host_counterparts=[],
            reconciled_candidates=[],
        )
        empty_report = DataIntegrityGateReport(
            passed_gate_a=False,
            total_ztf_objects=0,
            unique_physical_events=0,
            unique_sky_loci=0,
            total_catalog_reports=0,
            total_host_counterparts=0,
            multi_episode_loci_count=0,
            status_counts={},
            evidence_tier_counts={},
            provenance_counts={},
            availability_status_counts={},
            split_counts={},
            physical_event_split_leakage_count=0,
            sky_locus_split_leakage_count=0,
            spatial_cluster_split_leakage_count=0,
            alias_split_leakage_count=0,
            unclassified_as_negative_count=0,
            broker_truth_in_primary_count=0,
            reference_corpus_overlap_count=0,
        )
        return empty_bundle, empty_report

    # Index TNS aliases -> TNS indices
    tns_by_ztf: Dict[str, List[int]] = {}
    for t_idx, rec in enumerate(tns_records):
        for z_id in rec.ztf_ids:
            tns_by_ztf.setdefault(z_id, []).append(t_idx)

    # Index External L3 records by ID + sky coordinates
    ext_by_id: Dict[str, int] = {e.external_id: idx for idx, e in enumerate(ext_records)}

    ztf_ra = np.array([z.ra_deg % 360.0 for z in ztf_objects], dtype=float)
    ztf_dec = np.array([z.dec_deg for z in ztf_objects], dtype=float)
    ztf_coords = SkyCoord(ra=ztf_ra * u.deg, dec=ztf_dec * u.deg, frame="icrs")

    # Entity 1: Persistent sky loci via spatial connected components
    locus_cluster_ids = _connected_components_spatial(ztf_ra, ztf_dec, dedup_radius_arcsec)

    # Spatial search ZTF <-> TNS within ambiguity_radius_arcsec
    ztf_to_tns_cands: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(ztf_objects))}
    tns_to_ztf_cands: Dict[int, List[Tuple[int, float]]] = {}
    if tns_records:
        tns_coords = SkyCoord(
            ra=np.array([r.ra_deg for r in tns_records], dtype=float) * u.deg,
            dec=np.array([r.dec_deg for r in tns_records], dtype=float) * u.deg,
            frame="icrs",
        )
        idx_z, idx_t, sep2d, _ = search_around_sky(
            ztf_coords, tns_coords, ambiguity_radius_arcsec * u.arcsec
        )
        for zi, ti, s_arcsec in zip(idx_z.tolist(), idx_t.tolist(), sep2d.arcsec.tolist()):
            ztf_to_tns_cands[zi].append((ti, float(s_arcsec)))
            tns_to_ztf_cands.setdefault(ti, []).append((zi, float(s_arcsec)))
        for zi in ztf_to_tns_cands:
            ztf_to_tns_cands[zi].sort(key=lambda item: item[1])

    # Spatial search ZTF <-> External L3 Catalog within match_radius_arcsec
    ztf_to_ext_cands: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(len(ztf_objects))}
    if ext_records:
        ext_coords = SkyCoord(
            ra=np.array([e.ra_deg for e in ext_records], dtype=float) * u.deg,
            dec=np.array([e.dec_deg for e in ext_records], dtype=float) * u.deg,
            frame="icrs",
        )
        idx_ze, idx_e, sep_e, _ = search_around_sky(
            ztf_coords, ext_coords, match_radius_arcsec * u.arcsec
        )
        for zi, ei, s_arcsec in zip(idx_ze.tolist(), idx_e.tolist(), sep_e.arcsec.tolist()):
            ztf_to_ext_cands[zi].append((ei, float(s_arcsec)))
        for zi in ztf_to_ext_cands:
            ztf_to_ext_cands[zi].sort(key=lambda item: item[1])

    # Reference corpus overlap audit
    ztf_to_ref: Dict[int, str] = {}
    if ref_coords is not None and len(ref_ids) > 0:
        idx_zr, idx_r, _, _ = search_around_sky(
            ztf_coords, ref_coords, ambiguity_radius_arcsec * u.arcsec
        )
        for zi, ri in zip(idx_zr.tolist(), idx_r.tolist()):
            ztf_to_ref.setdefault(int(zi), ref_ids[int(ri)])

    cluster_to_members: Dict[int, List[int]] = {}
    for zi, cid in enumerate(locus_cluster_ids.tolist()):
        cluster_to_members.setdefault(cid, []).append(zi)

    hp32 = HEALPix(nside=32, order="nested", frame="icrs")

    # Compute canonical SkyLocus metadata per cluster
    cluster_locus_meta: Dict[int, Tuple[str, float, float, int, str]] = {}
    for cid, members in cluster_to_members.items():
        anchor_idx = min(members, key=lambda idx: ztf_objects[idx].oid)
        l_ra = float(ztf_objects[anchor_idx].ra_deg % 360.0)
        l_dec = float(ztf_objects[anchor_idx].dec_deg)
        pix32 = int(hp32.skycoord_to_healpix(SkyCoord(ra=l_ra * u.deg, dec=l_dec * u.deg, frame="icrs")))
        base_split = split_map.get(pix32, "train")
        locus_id = f"locus:hp32_{pix32}:{ztf_objects[anchor_idx].oid}"
        cluster_locus_meta[cid] = (locus_id, l_ra, l_dec, pix32, base_split)

    # First pass: resolve each candidate's TNS / External association with epoch compatibility
    resolved_rows: List[Dict[str, Any]] = []
    for zi, z in enumerate(ztf_objects):
        cid = int(locus_cluster_ids[zi])
        locus_id, l_ra, l_dec, pix32, base_split = cluster_locus_meta[cid]
        cluster_members = cluster_to_members[cid]

        raw_spatial_cands = ztf_to_tns_cands.get(zi, [])
        alias_tns_idxs = tns_by_ztf.get(z.oid, [])

        # Filter spatial TNS candidates by epoch compatibility unless they share direct alias
        compat_spatial_cands = [
            (ti, s)
            for ti, s in raw_spatial_cands
            if (ti in alias_tns_idxs)
            or _is_tns_epoch_compatible(z, tns_records[ti].discovery_mjd_utc, episode_separation_days)
        ]

        reasons: List[str] = []
        matched_rec: Optional[TnsObjectRecord] = None
        matched_ext: Optional[ExternalCatalogRecord] = None
        sep_val: Optional[float] = None
        competing_tns: List[str] = []

        # External Gaia/Host counterpart if present
        ext_cands = ztf_to_ext_cands.get(zi, [])
        if z.gaia_source_id and f"gaia_dr3:{z.gaia_source_id}" in ext_by_id:
            ext_idx = ext_by_id[f"gaia_dr3:{z.gaia_source_id}"]
            host_ext_rec: Optional[ExternalCatalogRecord] = ext_records[ext_idx]
            host_ext_sep: Optional[float] = CrossSurveyMatcher.angular_separation_arcsec(
                z.ra_deg, z.dec_deg, host_ext_rec.ra_deg, host_ext_rec.dec_deg
            )
        elif ext_cands:
            host_ext_rec = ext_records[ext_cands[0][0]]
            host_ext_sep = ext_cands[0][1]
        else:
            host_ext_rec = None
            host_ext_sep = None

        if z.is_detection_artifact:
            status = ReconciliationStatus.DETECTION_ARTIFACT_TRACK
            reasons.append("detection_quality_artifact_track")
        elif alias_tns_idxs:
            # Check competing epoch-compatible TNS reports in neighborhood
            other_compat_tns = {
                tns_records[ti].tns_name
                for ti, _ in compat_spatial_cands
                if ti not in alias_tns_idxs
            }
            if len(alias_tns_idxs) > 1 or other_compat_tns:
                status = ReconciliationStatus.AMBIGUOUS_MULTI_COUNTERPART
                competing_tns = sorted(
                    {tns_records[i].tns_name for i in alias_tns_idxs} | other_compat_tns
                )
                reasons.append("multiple_epoch_compatible_tns_counterparts")
            else:
                ti = alias_tns_idxs[0]
                alias_rec = tns_records[ti]
                sep_alias = CrossSurveyMatcher.angular_separation_arcsec(
                    z.ra_deg, z.dec_deg, alias_rec.ra_deg, alias_rec.dec_deg
                )
                sep_val = sep_alias
                if sep_alias > ambiguity_radius_arcsec:
                    status = ReconciliationStatus.COORDINATE_ALIAS_CONFLICT
                    competing_tns = [alias_rec.tns_name] + [
                        tns_records[k].tns_name for k, _ in compat_spatial_cands
                    ]
                    reasons.append(f"ztf_alias_coordinate_conflict:{sep_alias:.2f}arcsec")
                elif sep_alias <= match_radius_arcsec:
                    matched_rec = alias_rec
                    status = ReconciliationStatus.DIRECT_ID_AND_SPATIAL_MATCH
                else:
                    status = ReconciliationStatus.AMBIGUOUS_MULTI_COUNTERPART
                    competing_tns = [alias_rec.tns_name]
                    reasons.append(f"borderline_separation:{sep_alias:.2f}arcsec")
        elif compat_spatial_cands:
            close_cands = [(ti, s) for ti, s in compat_spatial_cands if s <= match_radius_arcsec]
            if len(compat_spatial_cands) > 1 or len(close_cands) != 1:
                status = ReconciliationStatus.AMBIGUOUS_MULTI_COUNTERPART
                competing_tns = [tns_records[ti].tns_name for ti, _ in compat_spatial_cands]
                sep_val = compat_spatial_cands[0][1]
                reasons.append("ambiguous_epoch_compatible_tns_association")
            else:
                ti, s = close_cands[0]
                matched_rec = tns_records[ti]
                sep_val = s
                status = ReconciliationStatus.UNIQUE_SPATIAL_MATCH
        elif host_ext_rec is not None:
            matched_ext = host_ext_rec
            sep_val = host_ext_sep
            status = ReconciliationStatus.EXTERNAL_CATALOG_MATCH
        else:
            status = ReconciliationStatus.UNMATCHED

        resolved_rows.append(
            {
                "zi": zi,
                "z": z,
                "cid": cid,
                "locus_id": locus_id,
                "locus_pix32": pix32,
                "base_split": base_split,
                "cluster_members": cluster_members,
                "status": status,
                "matched_rec": matched_rec,
                "matched_ext": matched_ext,
                "host_ext_rec": host_ext_rec,
                "host_ext_sep": host_ext_sep,
                "sep_val": sep_val,
                "competing_tns": competing_tns,
                "reasons": reasons,
            }
        )

    # Assign epoch-aware PhysicalTransientEpisode IDs inside each PersistentSkyLocus
    # Distinct temporal episodes at the same locus NEVER collapse into one physical_event_id!
    episode_id_by_zi: Dict[int, str] = {}
    competing_oids_by_zi: Dict[int, List[str]] = {}
    for cid, members in cluster_to_members.items():
        # Group members by matched TNS name first, or by temporal-window overlap for unmatched
        unmatched_subgroups: List[List[int]] = []
        for zi in members:
            row = resolved_rows[zi]
            z = row["z"]
            if row["matched_rec"] is not None:
                episode_id_by_zi[zi] = f"phys:tns:{row['matched_rec'].tns_name}"
            elif row["matched_ext"] is not None:
                episode_id_by_zi[zi] = f"phys:ext:{row['matched_ext'].external_id}"
            else:
                # Merge only epoch-overlapping candidates at this sky locus
                placed = False
                for grp in unmatched_subgroups:
                    if any(
                        _windows_overlap(z, resolved_rows[m]["z"], episode_separation_days)
                        for m in grp
                    ):
                        grp.append(zi)
                        placed = True
                        break
                if not placed:
                    unmatched_subgroups.append([zi])

        for grp in unmatched_subgroups:
            anchor_oid = min(resolved_rows[m]["z"].oid for m in grp)
            for m in grp:
                episode_id_by_zi[m] = f"phys:locus_ep:{anchor_oid}"

        # Track same-episode peer OIDs vs distinct-episode same-locus OIDs
        for zi in members:
            ep_id = episode_id_by_zi[zi]
            same_ep_peers = [
                resolved_rows[m]["z"].oid
                for m in members
                if m != zi and episode_id_by_zi[m] == ep_id
            ]
            competing_oids_by_zi[zi] = same_ep_peers
            if same_ep_peers and resolved_rows[zi]["matched_rec"] is not None:
                resolved_rows[zi]["reasons"].append("multi_oid_merged_to_single_episode")

    reconciled: List[ReconciledSurveyEvent] = []
    catalog_reports: List[CatalogReportRecord] = []
    host_counterparts: List[HostCounterpartRecord] = []
    seen_reports: Set[str] = set()
    seen_hosts: Set[str] = set()

    for row in resolved_rows:
        zi = row["zi"]
        z: ZtfObjectSummary = row["z"]
        status: ReconciliationStatus = row["status"]
        matched_rec: Optional[TnsObjectRecord] = row["matched_rec"]
        matched_ext: Optional[ExternalCatalogRecord] = row["matched_ext"]
        host_ext_rec: Optional[ExternalCatalogRecord] = row["host_ext_rec"]
        reasons: List[str] = list(row["reasons"])
        locus_id: str = row["locus_id"]
        pix32: int = row["locus_pix32"]
        base_split: str = row["base_split"]
        phys_id = episode_id_by_zi[zi]

        is_quarantined = status in (
            ReconciliationStatus.AMBIGUOUS_MULTI_COUNTERPART,
            ReconciliationStatus.COORDINATE_ALIAS_CONFLICT,
        )
        assigned_split = "quarantine" if is_quarantined else base_split

        epoch_ref = (
            z.firstmjd
            if z.firstmjd is not None
            else (matched_rec.discovery_mjd_utc if matched_rec else None)
        )
        if epoch_ref is None:
            temporal_split = "unknown_epoch"
        elif epoch_ref >= temporal_cutoff_mjd:
            temporal_split = "archival_observation_date_holdout"
        else:
            temporal_split = "historical_dev"

        is_tns_spec = matched_rec is not None and matched_rec.evidence_tier in (
            EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            EvidenceTier.L2_REPORTED_CLASSIFICATION_UNVERIFIED,
        )

        if status == ReconciliationStatus.DETECTION_ARTIFACT_TRACK:
            coarse = CoarseTransientClass.DETECTION_ARTIFACT
            fine = "DETECTION_ARTIFACT"
            tier = EvidenceTier.L4_BROKER_OR_PHOTOMETRIC
            is_sn: Optional[bool] = False
            disc_mjd = None
            tns_name = None
            ext_id = None
        elif is_tns_spec:
            # Spectroscopic TNS takes precedence over external catalog
            coarse = matched_rec.coarse_class
            fine = matched_rec.fine_class
            tier = matched_rec.evidence_tier
            is_sn = matched_rec.is_supernova
            disc_mjd = matched_rec.discovery_mjd_utc
            tns_name = matched_rec.tns_name
            ext_id = host_ext_rec.external_id if host_ext_rec else None
            if matched_rec.has_host_redshift_only:
                reasons.append("tns_host_redshift_only_not_transient_spec")
        elif host_ext_rec is not None:
            # Independent Gaia DR3 L3 catalog truth takes precedence over unclassified L5 alerts
            coarse = host_ext_rec.coarse_class
            fine = host_ext_rec.fine_class
            tier = host_ext_rec.evidence_tier
            is_sn = host_ext_rec.is_supernova
            disc_mjd = matched_rec.discovery_mjd_utc if matched_rec else None
            tns_name = matched_rec.tns_name if matched_rec else None
            ext_id = host_ext_rec.external_id
            if matched_rec is not None:
                reasons.append(f"gaia_l3_precedence_over_unclassified_tns:{matched_rec.tns_name}")
        elif matched_rec is not None:
            coarse = matched_rec.coarse_class
            fine = matched_rec.fine_class
            tier = matched_rec.evidence_tier
            is_sn = matched_rec.is_supernova
            disc_mjd = matched_rec.discovery_mjd_utc
            tns_name = matched_rec.tns_name
            ext_id = None
            if matched_rec.has_host_redshift_only:
                reasons.append("tns_host_redshift_only_not_transient_spec")
        else:
            coarse = CoarseTransientClass.UNCLASSIFIED
            fine = "UNLABELED"
            tier = EvidenceTier.L5_UNCLASSIFIED
            is_sn = None
            disc_mjd = None
            tns_name = None
            ext_id = None

        ref_overlap = ztf_to_ref.get(zi)
        if ref_overlap is not None:
            reasons.append(f"overlaps_reference_corpus:{ref_overlap}")

        host_id: Optional[str] = None
        if host_ext_rec is not None:
            host_id = f"host:{host_ext_rec.external_id}"
            if host_id not in seen_hosts:
                seen_hosts.add(host_id)
                host_counterparts.append(
                    HostCounterpartRecord(
                        counterpart_id=host_id,
                        sky_locus_id=locus_id,
                        source_catalog=host_ext_rec.source_catalog,
                        ra_deg=host_ext_rec.ra_deg,
                        dec_deg=host_ext_rec.dec_deg,
                        angular_offset_arcsec=row["host_ext_sep"],
                        host_redshift=host_ext_rec.catalog_redshift,
                        gaia_source_id=host_ext_rec.external_id.split(":")[-1],
                        is_nuclear_locus=(
                            row["host_ext_sep"] is not None and float(row["host_ext_sep"]) <= 1.0
                        ),
                    )
                )

        if matched_rec is not None:
            rep_id = f"report:tns:{matched_rec.objid}"
            if rep_id not in seen_reports:
                seen_reports.add(rep_id)
                catalog_reports.append(
                    CatalogReportRecord(
                        report_id=rep_id,
                        source_system="TNS_PUBLIC_OBJECTS",
                        source_record_id=matched_rec.tns_name,
                        sky_locus_id=locus_id,
                        linked_episode_id=phys_id,
                        ra_deg=matched_rec.ra_deg,
                        dec_deg=matched_rec.dec_deg,
                        evidence_tier=matched_rec.evidence_tier,
                        raw_class=matched_rec.raw_type,
                        discovery_mjd_utc=matched_rec.discovery_mjd_utc,
                        class_bibcodes=matched_rec.class_bibcodes,
                        has_host_redshift_only=matched_rec.has_host_redshift_only,
                    )
                )
        if matched_ext is not None:
            rep_id = f"report:ext:{matched_ext.external_id}"
            if rep_id not in seen_reports:
                seen_reports.add(rep_id)
                catalog_reports.append(
                    CatalogReportRecord(
                        report_id=rep_id,
                        source_system=matched_ext.source_catalog,
                        source_record_id=matched_ext.external_id,
                        sky_locus_id=locus_id,
                        linked_episode_id=phys_id,
                        ra_deg=matched_ext.ra_deg,
                        dec_deg=matched_ext.dec_deg,
                        evidence_tier=matched_ext.evidence_tier,
                        raw_class=matched_ext.fine_class,
                    )
                )

        reconciled.append(
            ReconciledSurveyEvent(
                physical_event_id=phys_id,
                sky_locus_id=locus_id,
                host_counterpart_id=host_id,
                ztf_oid=z.oid,
                ra_deg=z.ra_deg % 360.0,
                dec_deg=z.dec_deg,
                healpix_nside32=pix32,
                assigned_split=assigned_split,
                temporal_split=temporal_split,
                reconciliation_status=status,
                matched_tns_name=tns_name,
                matched_external_id=ext_id,
                separation_arcsec=row["sep_val"],
                competing_tns_names=row["competing_tns"],
                competing_ztf_oids=competing_oids_by_zi[zi],
                coarse_class=coarse,
                fine_class=fine,
                evidence_tier=tier,
                label_provenance=tier,
                taxonomy_version=REAL_SURVEY_TAXONOMY_VERSION,
                is_supernova=is_sn,
                availability_status=z.availability_status,
                has_operational_pub_time=(
                    z.availability_status == AvailabilityStatus.KNOWN_OPERATIONAL
                ),
                has_msip_alert=z.has_msip_alert,
                has_dr24_lc=z.has_dr24_lc,
                has_forced_photometry=z.has_forced_photometry,
                has_alerce_access=z.has_alerce_access,
                has_independent_tns=z.has_independent_tns or (matched_rec is not None),
                discovery_mjd_utc=disc_mjd,
                first_alert_mjd_utc=z.firstmjd,
                last_alert_mjd_utc=z.lastmjd,
                reference_corpus_overlap_id=ref_overlap,
                is_quarantined=is_quarantined,
                quarantine_reasons=reasons,
            )
        )

    # Materialize PersistentSkyLocus + PhysicalTransientEpisode nodes
    ev_by_locus: Dict[str, List[ReconciledSurveyEvent]] = {}
    ev_by_ep: Dict[str, List[ReconciledSurveyEvent]] = {}
    for ev in reconciled:
        ev_by_locus.setdefault(ev.sky_locus_id, []).append(ev)
        ev_by_ep.setdefault(ev.physical_event_id, []).append(ev)

    sky_loci: List[PersistentSkyLocus] = []
    for locus_id, evs in ev_by_locus.items():
        first_ev = evs[0]
        sky_loci.append(
            PersistentSkyLocus(
                sky_locus_id=locus_id,
                ra_deg=first_ev.ra_deg,
                dec_deg=first_ev.dec_deg,
                healpix_nside32=first_ev.healpix_nside32,
                assigned_base_split=split_map.get(first_ev.healpix_nside32, "train"),
                member_oids=[e.ztf_oid for e in evs],
                linked_tns_names=sorted({e.matched_tns_name for e in evs if e.matched_tns_name}),
                linked_external_ids=sorted({e.matched_external_id for e in evs if e.matched_external_id}),
                episode_count=len({e.physical_event_id for e in evs}),
            )
        )

    physical_episodes: List[PhysicalTransientEpisode] = []
    for phys_id, evs in ev_by_ep.items():
        first_ev = evs[0]
        starts = [e.first_alert_mjd_utc for e in evs if e.first_alert_mjd_utc is not None]
        ends = [e.last_alert_mjd_utc for e in evs if e.last_alert_mjd_utc is not None]
        physical_episodes.append(
            PhysicalTransientEpisode(
                physical_event_id=phys_id,
                sky_locus_id=first_ev.sky_locus_id,
                ra_deg=first_ev.ra_deg,
                dec_deg=first_ev.dec_deg,
                healpix_nside32=first_ev.healpix_nside32,
                t_start_mjd=min(starts) if starts else None,
                t_end_mjd=max(ends) if ends else None,
                assigned_split=first_ev.assigned_split,
                temporal_split=first_ev.temporal_split,
                member_oids=[e.ztf_oid for e in evs],
                matched_tns_name=first_ev.matched_tns_name,
                matched_external_id=first_ev.matched_external_id,
                coarse_class=first_ev.coarse_class,
                fine_class=first_ev.fine_class,
                evidence_tier=first_ev.evidence_tier,
                is_supernova=first_ev.is_supernova,
                is_quarantined=any(e.is_quarantined for e in evs),
                quarantine_reasons=sorted({r for e in evs for r in e.quarantine_reasons}),
            )
        )

    # Gate A integrity checks across non-quarantined splits
    phys_to_splits: Dict[str, Set[str]] = {}
    locus_to_splits: Dict[str, Set[str]] = {}
    cluster_to_splits: Dict[int, Set[str]] = {}
    alias_to_splits: Dict[str, Set[str]] = {}
    status_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    avail_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {}
    unclassified_neg = 0
    broker_primary_count = 0
    ref_overlap_ids: Set[str] = set()

    for zi, ev in enumerate(reconciled):
        status_counts[ev.reconciliation_status.value] = (
            status_counts.get(ev.reconciliation_status.value, 0) + 1
        )
        tier_counts[ev.evidence_tier.value] = tier_counts.get(ev.evidence_tier.value, 0) + 1
        avail_counts[ev.availability_status.value] = (
            avail_counts.get(ev.availability_status.value, 0) + 1
        )
        split_counts[ev.assigned_split] = split_counts.get(ev.assigned_split, 0) + 1

        if ev.evidence_tier == EvidenceTier.L5_UNCLASSIFIED and ev.is_supernova is not None:
            unclassified_neg += 1
        if (
            ev.evidence_tier == EvidenceTier.L4_BROKER_OR_PHOTOMETRIC
            and ev.reconciliation_status != ReconciliationStatus.DETECTION_ARTIFACT_TRACK
        ):
            broker_primary_count += 1

        if ev.reference_corpus_overlap_id is not None:
            ref_overlap_ids.add(ev.reference_corpus_overlap_id)

        if ev.assigned_split != "quarantine":
            phys_to_splits.setdefault(ev.physical_event_id, set()).add(ev.assigned_split)
            locus_to_splits.setdefault(ev.sky_locus_id, set()).add(ev.assigned_split)
            cluster_to_splits.setdefault(int(locus_cluster_ids[zi]), set()).add(ev.assigned_split)
            alias_to_splits.setdefault(ev.ztf_oid, set()).add(ev.assigned_split)
            if ev.matched_tns_name:
                alias_to_splits.setdefault(ev.matched_tns_name, set()).add(ev.assigned_split)

    phys_leaks = sum(1 for s in phys_to_splits.values() if len(s) > 1)
    locus_leaks = sum(1 for s in locus_to_splits.values() if len(s) > 1)
    cluster_leaks = sum(1 for s in cluster_to_splits.values() if len(s) > 1)
    alias_leaks = sum(1 for s in alias_to_splits.values() if len(s) > 1)
    multi_ep_loci = sum(1 for l in sky_loci if l.episode_count > 1)

    passed = (
        phys_leaks == 0
        and locus_leaks == 0
        and cluster_leaks == 0
        and alias_leaks == 0
        and unclassified_neg == 0
        and broker_primary_count == 0
    )
    report = DataIntegrityGateReport(
        passed_gate_a=passed,
        total_ztf_objects=len(reconciled),
        unique_physical_events=len(physical_episodes),
        unique_sky_loci=len(sky_loci),
        total_catalog_reports=len(catalog_reports),
        total_host_counterparts=len(host_counterparts),
        multi_episode_loci_count=multi_ep_loci,
        status_counts=status_counts,
        evidence_tier_counts=tier_counts,
        provenance_counts=tier_counts,
        availability_status_counts=avail_counts,
        split_counts=split_counts,
        physical_event_split_leakage_count=phys_leaks,
        sky_locus_split_leakage_count=locus_leaks,
        spatial_cluster_split_leakage_count=cluster_leaks,
        alias_split_leakage_count=alias_leaks,
        unclassified_as_negative_count=unclassified_neg,
        broker_truth_in_primary_count=broker_primary_count,
        reference_corpus_overlap_count=len(ref_overlap_ids),
        overlapping_reference_ids=sorted(ref_overlap_ids),
        quarantined_count=split_counts.get("quarantine", 0),
    )
    bundle = IdentityGraphBundle(
        sky_loci=sky_loci,
        physical_episodes=physical_episodes,
        catalog_reports=catalog_reports,
        host_counterparts=host_counterparts,
        reconciled_candidates=reconciled,
    )
    return bundle, report


def reconcile_ztf_and_tns(
    ztf_objects: List[ZtfObjectSummary],
    tns_records: List[TnsObjectRecord],
    external_records: Optional[List[ExternalCatalogRecord]] = None,
    split_manifest_path: Union[str, Path] = "configs/data_splits_healpix.json",
    reference_corpus_dir: Union[str, Path] = "data/reference_corpus_v1",
    match_radius_arcsec: float = 2.0,
    ambiguity_radius_arcsec: float = 5.0,
    dedup_radius_arcsec: float = 2.0,
    episode_separation_days: float = 180.0,
    temporal_cutoff_mjd: float = 60310.0,  # 2024-01-01 00:00 UTC default
) -> Tuple[List[ReconciledSurveyEvent], DataIntegrityGateReport]:
    """Convenience wrapper returning `(reconciled_candidates, DataIntegrityGateReport)`."""
    bundle, report = build_identity_graph_and_reconcile(
        ztf_objects=ztf_objects,
        tns_records=tns_records,
        external_records=external_records,
        split_manifest_path=split_manifest_path,
        reference_corpus_dir=reference_corpus_dir,
        match_radius_arcsec=match_radius_arcsec,
        ambiguity_radius_arcsec=ambiguity_radius_arcsec,
        dedup_radius_arcsec=dedup_radius_arcsec,
        episode_separation_days=episode_separation_days,
        temporal_cutoff_mjd=temporal_cutoff_mjd,
    )
    return bundle.reconciled_candidates, report
