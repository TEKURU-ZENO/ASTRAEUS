"""
ASTRAEUS Discovery Agents Submodule (WS8.3, WS10.3, WS10.4).
Hypothesis generation, evolutionary tournaments, evidence fabric, and multimessenger reasoning.
"""

from astraeus_agents.discovery.hypothesis_generator import (
    HypothesisPrediction,
    StructuredHypothesis,
    HypothesisGenerationEngine,
)
from astraeus_agents.discovery.evidence_fabric import (
    EvidenceNodeType,
    EvidenceNode,
    UnifiedEvidenceFabric,
)
from astraeus_agents.discovery.multimessenger_reasoner import (
    CompetingPhysicalModel,
    JeffreysEvidenceScale,
    ModelComparisonEntry,
    ModelSelectionReport,
    TimeSlideScramblingAudit,
    MultiMessengerInferenceEngine,
)

__all__ = [
    "HypothesisPrediction",
    "StructuredHypothesis",
    "HypothesisGenerationEngine",
    "EvidenceNodeType",
    "EvidenceNode",
    "UnifiedEvidenceFabric",
    "CompetingPhysicalModel",
    "JeffreysEvidenceScale",
    "ModelComparisonEntry",
    "ModelSelectionReport",
    "TimeSlideScramblingAudit",
    "MultiMessengerInferenceEngine",
]
