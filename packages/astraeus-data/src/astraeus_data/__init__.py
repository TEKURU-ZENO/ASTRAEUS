"""
ASTRAEUS Data Ingestion & Storage Package.
Manages survey connectors, spatial HEALPix partitioning, Parquet writing, and synthetic test suites.
"""

from astraeus_data.partitioning import (
    load_split_manifest,
    entities_to_arrow_table,
    save_partitioned_parquet,
    audit_zero_leakage,
)
from astraeus_data.synthetic import (
    generate_synthetic_entity,
    generate_bazin_lightcurve,
)

__version__ = "0.1.0"

__all__ = [
    "load_split_manifest",
    "entities_to_arrow_table",
    "save_partitioned_parquet",
    "audit_zero_leakage",
    "generate_synthetic_entity",
    "generate_bazin_lightcurve",
]
