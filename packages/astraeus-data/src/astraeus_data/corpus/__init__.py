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
]
