"""
ASTRAEUS Data Discovery Submodule (WS8.2, WS8.5).
Anomaly intelligence, computational experiment engines, and counterfactuals.
"""

from astraeus_data.discovery.anomaly_engine import (
    AnomalySource,
    AnomalyDiagnostic,
    LightCurvePoint,
    AnomalyDiscoveryEngine,
)
from astraeus_data.discovery.experiment_engine import (
    DataProvenanceTag,
    SimulatedObservationPoint,
    ParameterRecoveryResult,
    PosteriorPredictiveCheckResult,
    CounterfactualExperimentResult,
    ComputationalExperimentEngine,
)

__all__ = [
    "AnomalySource",
    "AnomalyDiagnostic",
    "LightCurvePoint",
    "AnomalyDiscoveryEngine",
    "DataProvenanceTag",
    "SimulatedObservationPoint",
    "ParameterRecoveryResult",
    "PosteriorPredictiveCheckResult",
    "CounterfactualExperimentResult",
    "ComputationalExperimentEngine",
]
