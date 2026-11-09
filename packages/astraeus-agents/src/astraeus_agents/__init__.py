"""
ASTRAEUS Agents Package.
Provides the 4-agent collaborative runtime, deterministic scientific evidence gates,
adversarial skeptic protocols, and auditable scientific investigation dossiers.
"""

from astraeus_agents.contracts import (
    AgentRole,
    AgentTask,
    AgentMessage,
    TerminalInvestigationStatus,
    LiteratureCitation,
    CitationEvidenceType,
    SkepticChallenge,
    ChallengeCategory,
    ChallengeStatus,
)
from astraeus_agents.data_agent import DataCrossMatchAgent, DataAgentOutput
from astraeus_agents.literature_agent import LiteratureContextAgent
from astraeus_agents.skeptic_agent import SkepticValidationAgent
from astraeus_agents.evidence_gates import DeterministicEvidenceGateEngine, GateEvaluationResult
from astraeus_agents.orchestrator import InvestigationOrchestrator
from astraeus_agents.dossier import InvestigationDossier

__version__ = "0.1.0"

__all__ = [
    "AgentRole",
    "AgentTask",
    "AgentMessage",
    "TerminalInvestigationStatus",
    "LiteratureCitation",
    "CitationEvidenceType",
    "SkepticChallenge",
    "ChallengeCategory",
    "ChallengeStatus",
    "DataCrossMatchAgent",
    "DataAgentOutput",
    "LiteratureContextAgent",
    "SkepticValidationAgent",
    "DeterministicEvidenceGateEngine",
    "GateEvaluationResult",
    "InvestigationOrchestrator",
    "InvestigationDossier",
]
