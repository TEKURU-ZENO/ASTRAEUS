"""
ASTRAEUS Core Package.
Provides epistemic stratification schemas, celestial coordinates, IVOA provenance,
conformal uncertainty representations, and composite anomaly scoring.
"""

from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata
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
from astraeus_core.conformal import ConformalPredictionInterval
from astraeus_core.scoring import CompositeAnomalyScore
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
    "CelestialCoordinates",
    "ProvAgent",
    "ProvActivity",
    "ProvEntity",
    "create_provenance_record",
    "CalibratedMeasurement",
    "ObservationRecord",
    "PhotometricTimeSeries",
    "ConformalPredictionInterval",
    "CompositeAnomalyScore",
    "AstronomicalEntity",
    "AstraeusError",
    "EpistemicViolationError",
    "ProvenanceMissingError",
    "DataLeakageError",
    "CoordinateValidationError",
]
