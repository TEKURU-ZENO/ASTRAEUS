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
from astraeus_data.crossmatch.cross_survey_matcher import (
    CrossSurveyMatcher,
    CrossSurveyMatchStatus,
    SurveyCandidate,
    CrossSurveyAssociation,
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
from astraeus_data.schemas.panstarrs import (
    PanstarrsFilter,
    PanstarrsMeasurement,
    PanstarrsLightCurve,
)
from astraeus_data.schemas.decam import (
    DecamFilter,
    DecamMeasurement,
    DecamLightCurve,
)
from astraeus_data.photometry.harmonization import (
    SURVEY_FILTER_METADATA,
    HarmonizedPhotometricEpoch,
    PhotometricHarmonizer,
)
from astraeus_data.photometry.temporal import TemporalHarmonizer
from astraeus_data.adapters import (
    CanonicalObservationRecord,
    ObservationType,
    AtlasForcedPhotometryAdapter,
    DecamDdfPhotometryAdapter,
    RubinAlertInterfaceAdapter,
)
from astraeus_data.registry import StableEntityRecord, StableEntityRegistry
from astraeus_data.brokers import (
    AtlasRequestState,
    AtlasRequestRecord,
    AtlasForcedPhotometryClient,
)
from astraeus_data.corpus import (
    ReferenceObjectRecord,
    ReferenceCorpusBuilder,
    ReferenceCorpusAuditor,
    CorpusAuditReport,
    FrozenEvaluationProtocol,
    EvaluationTask,
)

from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    UnifiedScientificEvent,
    EventIngestionResult,
)
from astraeus_data.triage.event_triage_engine import (
    PriorityTier,
    AnomalyClass,
    HostGalaxyContext,
    ReferenceStarContext,
    PhotometricFeatures,
    TriageDecision,
    CatalogCrossMatcher,
    PhotometricFeatureExtractor,
    HeuristicAnomalyDetector,
    PrioritizationQueue,
    EventTriageEngine,
)

__version__ = "0.3.0"

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
    "CrossSurveyMatcher",
    "CrossSurveyMatchStatus",
    "SurveyCandidate",
    "CrossSurveyAssociation",
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
    "PanstarrsFilter",
    "PanstarrsMeasurement",
    "PanstarrsLightCurve",
    "DecamFilter",
    "DecamMeasurement",
    "DecamLightCurve",
    "SURVEY_FILTER_METADATA",
    "HarmonizedPhotometricEpoch",
    "PhotometricHarmonizer",
    "TemporalHarmonizer",
    "CanonicalObservationRecord",
    "ObservationType",
    "AtlasForcedPhotometryAdapter",
    "DecamDdfPhotometryAdapter",
    "RubinAlertInterfaceAdapter",
    "StableEntityRecord",
    "StableEntityRegistry",
    "AtlasRequestState",
    "AtlasRequestRecord",
    "AtlasForcedPhotometryClient",
    "ReferenceObjectRecord",
    "ReferenceCorpusBuilder",
    "ReferenceCorpusAuditor",
    "CorpusAuditReport",
    "FrozenEvaluationProtocol",
    "EvaluationTask",
    "ScientificEventIngestionEngine",
    "UnifiedScientificEvent",
    "EventIngestionResult",
    "PriorityTier",
    "AnomalyClass",
    "HostGalaxyContext",
    "ReferenceStarContext",
    "PhotometricFeatures",
    "TriageDecision",
    "CatalogCrossMatcher",
    "PhotometricFeatureExtractor",
    "HeuristicAnomalyDetector",
    "PrioritizationQueue",
    "EventTriageEngine",
]
