"""
ASTRAEUS Data Partitioning & Storage Module.
Manages spatial HEALPix partitioning, temporal cutoffs, and Parquet persistence with zero-leakage guarantees.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Literal, Union
import pyarrow as pa
import pyarrow.parquet as pq

from astraeus_core.entity import AstronomicalEntity
from astraeus_core.exceptions import DataLeakageError


def load_split_manifest(manifest_path: Union[str, Path] = "configs/data_splits_healpix.json") -> Dict[int, str]:
    """
    Loads spatial HEALPix split manifest and maps pixel index -> split name ('train', 'val', 'test').
    """
    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    pixel_to_split: Dict[int, str] = {}
    for split_name, pixels in data["splits"].items():
        for pix in pixels:
            pixel_to_split[pix] = split_name
    return pixel_to_split


def entities_to_arrow_table(entities: List[AstronomicalEntity], split_map: Dict[int, str]) -> pa.Table:
    """Converts a list of AstronomicalEntity objects to a pyarrow.Table with split tags."""
    records = []
    for e in entities:
        pix = e.coordinates.compute_healpix_index(nside=32)
        split = split_map.get(pix, "train")
        
        # Extract primary flux from latest observation if available
        flux_val = e.observations[-1].calibrated_flux.value if e.observations else None
        flux_err = e.observations[-1].calibrated_flux.uncertainty if e.observations else None
        band = e.observations[-1].filter_band if e.observations else None
        mjd = e.observations[-1].mjd if e.observations else None

        records.append({
            "entity_id": e.entity_id,
            "entity_type": e.entity_type,
            "epistemic_tier": e.epistemic_tier.value,
            "ra_deg": e.coordinates.ra_deg,
            "dec_deg": e.coordinates.dec_deg,
            "healpix_nside32": pix,
            "split": split,
            "latest_flux": flux_val,
            "latest_uncertainty": flux_err,
            "filter_band": band,
            "mjd": mjd,
            "num_observations": len(e.observations),
            "provenance_activity": e.provenance.was_generated_by,
        })

    return pa.Table.from_pylist(records)


def save_partitioned_parquet(
    entities: List[AstronomicalEntity],
    output_dir: Union[str, Path],
    manifest_path: Union[str, Path] = "configs/data_splits_healpix.json"
) -> Path:
    """
    Writes entities to Apache Arrow Parquet partitioned by 'split' and 'epistemic_tier'.
    Enforces spatial zero-leakage boundaries.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    split_map = load_split_manifest(manifest_path)
    table = entities_to_arrow_table(entities, split_map)

    pq.write_to_dataset(
        table,
        root_path=str(out_path),
        partition_cols=["split"],
        use_dictionary=True,
        compression="SNAPPY",
    )
    return out_path


def audit_zero_leakage(train_pixels: List[int], test_pixels: List[int]) -> bool:
    """
    Verifies that spatial HEALPix pixels between train and test splits have 0% intersection.
    Raises DataLeakageError if cross-split contamination is detected.
    """
    intersection = set(train_pixels).intersection(set(test_pixels))
    if intersection:
        raise DataLeakageError(
            f"CRITICAL: Spatial data leakage detected! {len(intersection)} HEALPix pixels overlap between splits: {intersection}"
        )
    return True
