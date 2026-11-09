"""
ASTRAEUS Reference Corpus Package.
Phase 2 Stage 1 (Days 91–105): Curates Astronomical Reference Corpus v1.0,
enforces 11-point provenance contracts, and executes comprehensive leakage audits.
"""

from astraeus_data.corpus.record import (
    ReferenceObjectRecord,
    SpatialPartitionInfo,
    CrossMatchPrior,
    PhotometricCalibrationInfo,
    PhotometricRepresentationInfo,
    MeasurementUncertaintyInfo,
    HostAssociationInfo,
    CryptographicSeal,
    DEFAULT_PROVENANCE_SECRET,
    DatasetOrigin,
    TimeStandard,
    RedshiftType,
    SplitType,
)
from astraeus_data.corpus.builder import ReferenceCorpusBuilder
from astraeus_data.corpus.auditor import ReferenceCorpusAuditor, CorpusAuditReport
from astraeus_data.corpus.protocol import FrozenEvaluationProtocol, EvaluationTask
from astraeus_data.corpus.lineage import (
    DatasetBSampleLineage,
    LineageStatus,
    ProvenanceGraphAuditor,
)
from astraeus_data.corpus.time_conversion import (
    compute_tt_from_utc_mjd,
    compute_tdb_from_utc_mjd,
    compute_bmjd_tdb,
    compute_roemer_delay_sec,
    is_observation_visible_at_cutoff,
    parse_operational_utc_epoch,
    OBSERVATORY_GEODETIC,
    TT_TAI_OFFSET_SEC,
)
from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    QualitySeverity,
    NoiseScaleSource,
    SequenceViewType,
    TemporalSequence,
    AkimaPredictiveModel,
    CANONICAL_AB_ZERO_POINT,
    CORPUS_V10_ZERO_POINT,
    CORPUS_V10_TO_NMGY_SCALE,
    SI_FLUX_DENSITY_PER_NANOMAGGY,
    MICROJANSKYS_PER_NANOMAGGY,
)
from astraeus_data.corpus.archival_manifest import (
    ArchivalObservationRecord,
    Track3ArchivalManifest,
    DatasetATestManifest,
)

__all__ = [
    "ReferenceObjectRecord",
    "SpatialPartitionInfo",
    "CrossMatchPrior",
    "PhotometricCalibrationInfo",
    "PhotometricRepresentationInfo",
    "MeasurementUncertaintyInfo",
    "HostAssociationInfo",
    "CryptographicSeal",
    "DEFAULT_PROVENANCE_SECRET",
    "DatasetOrigin",
    "TimeStandard",
    "RedshiftType",
    "SplitType",
    "ReferenceCorpusBuilder",
    "ReferenceCorpusAuditor",
    "CorpusAuditReport",
    "FrozenEvaluationProtocol",
    "EvaluationTask",
    "DatasetBSampleLineage",
    "LineageStatus",
    "ProvenanceGraphAuditor",
    "compute_tt_from_utc_mjd",
    "compute_tdb_from_utc_mjd",
    "compute_bmjd_tdb",
    "compute_roemer_delay_sec",
    "is_observation_visible_at_cutoff",
    "parse_operational_utc_epoch",
    "OBSERVATORY_GEODETIC",
    "TT_TAI_OFFSET_SEC",
    "ObservationRecord",
    "ObservationType",
    "QualitySeverity",
    "NoiseScaleSource",
    "SequenceViewType",
    "TemporalSequence",
    "AkimaPredictiveModel",
    "CANONICAL_AB_ZERO_POINT",
    "CORPUS_V10_ZERO_POINT",
    "CORPUS_V10_TO_NMGY_SCALE",
    "SI_FLUX_DENSITY_PER_NANOMAGGY",
    "MICROJANSKYS_PER_NANOMAGGY",
    "ArchivalObservationRecord",
    "Track3ArchivalManifest",
    "DatasetATestManifest",
]
