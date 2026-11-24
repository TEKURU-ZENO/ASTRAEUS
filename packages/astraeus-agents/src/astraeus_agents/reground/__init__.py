"""
ASTRAEUS REGROUND Multi-Agent Debate Package (WS7.3).
"""

from astraeus_agents.reground.models import (
    AgentPersona,
    DebateStage,
    ChallengeStatus,
    PhysicalHypothesis,
    DebateTurn,
    SkepticVeto,
    RegroundInvestigationReport,
)
from astraeus_agents.reground.agents import (
    ObservationalAstronomerAgent,
    CosmologistAgent,
    AstrophysicistAgent,
    StatisticalAuditorAgent,
    SkepticRedTeamAgent,
)
from astraeus_agents.reground.protocol import RegroundDebateProtocol

__all__ = [
    "AgentPersona",
    "DebateStage",
    "ChallengeStatus",
    "PhysicalHypothesis",
    "DebateTurn",
    "SkepticVeto",
    "RegroundInvestigationReport",
    "ObservationalAstronomerAgent",
    "CosmologistAgent",
    "AstrophysicistAgent",
    "StatisticalAuditorAgent",
    "SkepticRedTeamAgent",
    "RegroundDebateProtocol",
]
