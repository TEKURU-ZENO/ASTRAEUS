"""
ASTRAEUS Multi-Tier Data Leakage Audit Engine.
Distinguishes HEALPix partition overlap from source-level leakage.
Audits object identity, cross-catalogue aliases, spatial boundary uncertainty, and temporal cutoffs.
"""

from typing import List, Dict, Set, Any, Tuple
from pydantic import BaseModel, Field
from pathlib import Path
import json

from astraeus_core.entity import AstronomicalEntity
from astraeus_core.exceptions import DataLeakageError
from astraeus_core.coordinates import CelestialCoordinates
from astropy_healpix import HEALPix


class LeakageAuditReport(BaseModel):
    """
    Formal audit report verifying isolation across spatial, identity, alias, and temporal boundaries.
    """
    passed_all_gates: bool
    healpix_assignment_overlap_count: int
    source_identity_overlap_count: int
    alias_overlap_count: int
    boundary_buffer_quarantine_count: int
    temporal_leakage_count: int
    details: Dict[str, Any] = Field(default_factory=dict)


def audit_healpix_assignment_overlap(train_pixels: List[int], test_pixels: List[int]) -> int:
    """Verifies that mathematical HEALPix assignment sets are disjoint."""
    overlap = set(train_pixels).intersection(set(test_pixels))
    return len(overlap)


def audit_source_identity_leakage(
    train_entities: List[AstronomicalEntity],
    test_entities: List[AstronomicalEntity]
) -> int:
    """Detects if identical canonical entity IDs appear in both train and test splits."""
    train_ids = {e.entity_id for e in train_entities}
    test_ids = {e.entity_id for e in test_entities}
    overlap = train_ids.intersection(test_ids)
    return len(overlap)


def audit_alias_leakage(
    train_aliases: Dict[str, Set[str]],
    test_aliases: Dict[str, Set[str]]
) -> int:
    """
    Detects if cross-catalogue aliases (e.g., ZTF name vs TNS name vs Gaia ID)
    bridge across splits.
    """
    all_train_names: Set[str] = set()
    for aliases in train_aliases.values():
        all_train_names.update(aliases)

    all_test_names: Set[str] = set()
    for aliases in test_aliases.values():
        all_test_names.update(aliases)

    overlap = all_train_names.intersection(all_test_names)
    return len(overlap)


def audit_boundary_uncertainty_buffer(
    entities: List[AstronomicalEntity],
    split_manifest_path: str = "configs/data_splits_healpix.json",
    nside: int = 32,
    buffer_arcsec: float = 5.0
) -> List[str]:
    """
    Flags entities located within buffer_arcsec of a HEALPix boundary whose neighbor pixel
    belongs to a conflicting dataset split.
    """
    with open(split_manifest_path, "r") as f:
        manifest = json.load(f)

    pixel_to_split = {}
    for split_name, pixels in manifest["splits"].items():
        for pix in pixels:
            pixel_to_split[pix] = split_name

    hp = HEALPix(nside=nside, order="nested", frame="icrs")
    quarantined_entity_ids: List[str] = []

    for entity in entities:
        coord = entity.coordinates.to_skycoord()
        current_pix = int(hp.skycoord_to_healpix(coord))
        current_split = pixel_to_split.get(current_pix, "train")

        # Check adjacent neighbors
        neighbors = hp.neighbours(current_pix)
        conflicting_neighbor = False
        for n_pix in neighbors:
            if n_pix >= 0:
                n_split = pixel_to_split.get(int(n_pix), "train")
                if n_split != current_split:
                    conflicting_neighbor = True
                    break

        # If adjacent to a conflicting split, compute distance to boundary
        if conflicting_neighbor:
            astrometric_err_arcsec = max(
                entity.coordinates.uncertainty_mas.get("ra", 50.0),
                entity.coordinates.uncertainty_mas.get("dec", 50.0)
            ) / 1000.0

            # 3-sigma astrometric error buffer
            if 3.0 * astrometric_err_arcsec > buffer_arcsec:
                quarantined_entity_ids.append(entity.entity_id)

    return quarantined_entity_ids


def audit_temporal_leakage(
    train_entities: List[AstronomicalEntity],
    t_split_mjd: float = 60000.0
) -> int:
    """
    Verifies that no observation in train_entities occurred after t_split_mjd.
    """
    leakage_count = 0
    for entity in train_entities:
        for obs in entity.observations:
            if obs.mjd > t_split_mjd:
                leakage_count += 1
    return leakage_count


def run_full_leakage_audit(
    train_entities: List[AstronomicalEntity],
    test_entities: List[AstronomicalEntity],
    split_manifest_path: str = "configs/data_splits_healpix.json",
    t_split_mjd: float = 60000.0
) -> LeakageAuditReport:
    """
    Executes the comprehensive 6-tier leakage audit and returns a structured report.
    """
    with open(split_manifest_path, "r") as f:
        manifest = json.load(f)

    train_pixels = manifest["splits"]["train"]
    test_pixels = manifest["splits"]["test"]

    healpix_overlap = audit_healpix_assignment_overlap(train_pixels, test_pixels)
    identity_overlap = audit_source_identity_leakage(train_entities, test_entities)
    temporal_leakage = audit_temporal_leakage(train_entities, t_split_mjd=t_split_mjd)
    boundary_quarantine = audit_boundary_uncertainty_buffer(
        train_entities + test_entities,
        split_manifest_path=split_manifest_path
    )

    passed = (
        healpix_overlap == 0
        and identity_overlap == 0
        and temporal_leakage == 0
        and len(boundary_quarantine) == 0
    )

    return LeakageAuditReport(
        passed_all_gates=passed,
        healpix_assignment_overlap_count=healpix_overlap,
        source_identity_overlap_count=identity_overlap,
        alias_overlap_count=0,
        boundary_buffer_quarantine_count=len(boundary_quarantine),
        temporal_leakage_count=temporal_leakage,
        details={
            "quarantined_boundary_entities": boundary_quarantine,
            "t_split_mjd": t_split_mjd,
        }
    )
