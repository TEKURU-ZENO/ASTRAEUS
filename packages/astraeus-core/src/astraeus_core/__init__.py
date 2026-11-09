"""
ASTRAEUS Core Package.
Provides epistemic stratification schemas, celestial coordinates, IVOA provenance,
conformal uncertainty representations, and composite anomaly scoring.
"""

from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata, ClaimType
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.provenance import (
    ProvAgent,
    ProvActivity,
    ProvEntity,
    create_provenance_record,
)
from astraeus_core.observations import (
    CalibratedMeasurement,
    ObservationRecord,
    PhotometricTimeSeries,
)
from astraeus_core.conformal import ConformalPredictionInterval, UncertaintyDecomposition
from astraeus_core.scoring import (
    CompositeAnomalyScore,
    EmpiricalReferenceCDF,
    AnomalyEvidenceProfile,
    AnomalyRankScore,
)
from astraeus_core.entity import AstronomicalEntity
from astraeus_core.exceptions import (
    AstraeusError,
    EpistemicViolationError,
    ProvenanceMissingError,
    DataLeakageError,
    CoordinateValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "EpistemicTier",
    "EpistemicMetadata",
    "ClaimType",
    "CelestialCoordinates",
    "ProvAgent",
    "ProvActivity",
    "ProvEntity",
    "create_provenance_record",
    "CalibratedMeasurement",
    "ObservationRecord",
    "PhotometricTimeSeries",
    "ConformalPredictionInterval",
    "UncertaintyDecomposition",
    "CompositeAnomalyScore",
    "EmpiricalReferenceCDF",
    "AnomalyEvidenceProfile",
    "AnomalyRankScore",
    "AstronomicalEntity",
    "AstraeusError",
    "EpistemicViolationError",
    "ProvenanceMissingError",
    "DataLeakageError",
    "CoordinateValidationError",
]
