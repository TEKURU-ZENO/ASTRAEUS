"""
ASTRAEUS REGROUND Multi-Agent Scientific Investigation Models (WS7.3).
Part of Phase 7 Closed-Loop Autonomous Scientific Intelligence.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class AgentPersona(str, Enum):
    ASTROPHYSICIST = "ASTROPHYSICIST"                   # Physical transient modeling (SNe, KNe, TDEs)
    COSMOLOGIST = "COSMOLOGIST"                         # Large-scale structure, Hubble context, host environment
    OBSERVATIONAL_ASTRONOMER = "OBSERVATIONAL_ASTRONOMER" # Calibrations, filter systematics, extinction, seeing
    STATISTICAL_AUDITOR = "STATISTICAL_AUDITOR"         # Likelihoods, degrees of freedom, priors, BIC, identifiability
    SKEPTIC_RED_TEAM = "SKEPTIC_RED_TEAM"               # Adversarial falsification, premise challenging, veto


class DebateStage(str, Enum):
    EVIDENCE_GROUNDING = "EVIDENCE_GROUNDING"
    HYPOTHESIS_FORMULATION = "HYPOTHESIS_FORMULATION"
    COMPUTATIONAL_SIMULATION = "COMPUTATIONAL_SIMULATION"
    SKEPTIC_CHALLENGE = "SKEPTIC_CHALLENGE"
    BAYESIAN_POSTERIOR_UPDATE = "BAYESIAN_POSTERIOR_UPDATE"
    FINAL_REPORT_SEALING = "FINAL_REPORT_SEALING"


class ChallengeStatus(str, Enum):
    VALIDATED_WITH_QUALIFICATIONS = "VALIDATED_WITH_QUALIFICATIONS"
    CHALLENGE_ACTIVE = "CHALLENGE_ACTIVE"
    VETO_DISQUALIFIED = "VETO_DISQUALIFIED"


class HypothesisEvaluationStatus(str, Enum):
    """
    Principled 4-tier scientific evaluation taxonomy for transient hypotheses.
    Distinguishes statistical likelihood deficits from physical contradiction.
    """
    STRONGER_EVIDENCE = "STRONGER_EVIDENCE"
    # Observation increases relative posterior probability and reduces parameter variance.

    MODEL_REJECTED_UNDER_ASSUMPTIONS = "MODEL_REJECTED_UNDER_ASSUMPTIONS"
    # Parameterized model disfavored by delta BIC > 10, but broader progenitor class may survive.

    PHYSICALLY_FALSIFIED = "PHYSICALLY_FALSIFIED"
    # Progenitor hypothesis physically ruled out by fundamental conservation laws or definitive data.

    UNRESOLVED_ALTERNATIVE = "UNRESOLVED_ALTERNATIVE"
    # Competing alternative hypothesis remains plausible within observational noise and degeneracies.


class PhysicalHypothesis(BaseModel):
    """Structured physical hypothesis with Bayesian probabilities and falsification boundaries."""
    hypothesis_id: str
    name: str
    progenitor_class: str
    prior_prob: float
    posterior_prob: float
    log_likelihood: float = -999.0
    bic: float = 999.0
    degrees_of_freedom_k: int = 4
    reduced_chi2: Optional[float] = None
    physical_parameters: Dict[str, Any] = Field(default_factory=dict)
    param_uncertainties: Dict[str, float] = Field(default_factory=dict)
    falsification_criteria: List[str] = Field(default_factory=list)
    untested_falsification_criteria: List[str] = Field(default_factory=list)
    evaluation_status: HypothesisEvaluationStatus = HypothesisEvaluationStatus.UNRESOLVED_ALTERNATIVE
    is_falsified: bool = False
    is_leading: bool = False
    epistemic_qualifications: List[str] = Field(default_factory=list)


class DebateTurn(BaseModel):
    """Individual recorded turn in the REGROUND multi-agent scientific debate."""
    turn_number: int
    persona: AgentPersona
    stage: DebateStage
    argument_summary: str
    evidence_sources: List[str] = Field(default_factory=list)
    engine_invoked: Optional[str] = None
    engine_tier: Optional[str] = None
    timestamp_iso: str


class SkepticVeto(BaseModel):
    """Adversarial challenge and falsification test raised by the Skeptic Red-Team."""
    veto_id: str
    target_hypothesis_id: str
    challenge_title: str
    adversarial_critique: str
    instrumental_artifact_checked: str
    status: ChallengeStatus
    required_discriminating_observation: str


class RegroundInvestigationReport(BaseModel):
    """Comprehensive, reproducible final report compiled from REGROUND debate."""
    investigation_id: str
    event_id: str
    target_coordinates: Dict[str, float]
    healpix_pixel_nside32: int
    hypotheses: List[PhysicalHypothesis]
    leading_hypothesis: Optional[PhysicalHypothesis] = None
    debate_turns: List[DebateTurn] = Field(default_factory=list)
    skeptic_challenges: List[SkepticVeto] = Field(default_factory=list)
    recommended_followup_filter: str
    expected_information_gain: float
    scientific_conclusion: str
    provenance_sealed_digest: str
    is_quarantine_compliant: bool = True
