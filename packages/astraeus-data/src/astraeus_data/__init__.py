"""
ASTRAEUS Data Package.
Provides ZTF alert schemas, idempotent ingestion engine, probabilistic Gaia cross-matching,
Parquet partitioning, synthetic benchmark datasets, and 6-tier leakage audits.
"""

from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate
from astraeus_data.ingestion.alert_ingester import (
    AlertIngestionEngine,
    IngestionStatus,
    IngestionResult,
)
from astraeus_data.crossmatch.gaia_matcher import (
    ProbabilisticGaiaMatcher,
    GaiaCandidate,
    CrossMatchResult,
    MatchClassification,
)
from astraeus_data.partitioning import (
    load_split_manifest,
    entities_to_arrow_table,
    save_partitioned_parquet,
    audit_zero_leakage,
)
from astraeus_data.leakage_audit import (
    run_full_leakage_audit,
    audit_healpix_assignment_overlap,
    audit_source_identity_leakage,
    audit_alias_leakage,
    audit_boundary_uncertainty_buffer,
    audit_temporal_leakage,
    LeakageAuditReport,
)
from astraeus_data.synthetic import (
    generate_synthetic_entity,
    generate_bazin_lightcurve,
)

__version__ = "0.1.0"

__all__ = [
    "ZtfAlertPacket",
    "ZtfCandidate",
    "AlertIngestionEngine",
    "IngestionStatus",
    "IngestionResult",
    "ProbabilisticGaiaMatcher",
    "GaiaCandidate",
    "CrossMatchResult",
    "MatchClassification",
    "load_split_manifest",
    "entities_to_arrow_table",
    "save_partitioned_parquet",
    "audit_zero_leakage",
    "run_full_leakage_audit",
    "audit_healpix_assignment_overlap",
    "audit_source_identity_leakage",
    "audit_alias_leakage",
    "audit_boundary_uncertainty_buffer",
    "audit_temporal_leakage",
    "LeakageAuditReport",
    "generate_synthetic_entity",
    "generate_bazin_lightcurve",
]
