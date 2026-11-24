"""
ASTRAEUS Observatory Intelligence - Human Approval & Governance Interface (WS7.5).
Enforces the mandatory human-in-the-loop authorization gate before any observational
pointing recommendation can be dispatched to an astronomical facility or queue.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import time
import math
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from astraeus_agents.observatory.schedulers import ScheduledObservation, TargetState


class ApprovalStatus(str, Enum):
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    AUTHORIZED_BY_INVESTIGATOR = "AUTHORIZED_BY_INVESTIGATOR"
    REJECTED_BY_INVESTIGATOR = "REJECTED_BY_INVESTIGATOR"


class AutonomousDispatchProhibitedError(PermissionError):
    """Raised when an automated agent or pipeline attempts to dispatch pointing commands without human authorization."""
    pass


class ObservationRecommendation(BaseModel):
    """Formal human-facing observation dispatch recommendation."""
    recommendation_id: str
    target_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    exposure_time_sec: float
    scheduled_mjd: float
    expected_eig_nats: float
    prior_entropy_nats: float
    expected_posterior_entropy_nats: float
    entropy_reduction_pct: float
    competing_hypotheses: Dict[str, float]
    scientific_justification: str
    physical_constraints_audit: Dict[str, Any]
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_HUMAN_REVIEW
    authorized_investigator: Optional[str] = None
    authorization_timestamp_utc: Optional[float] = None
    rejection_reason: Optional[str] = None
    cryptographic_proposal_hash: str


class HumanApprovalGate:
    """
    Mandatory governance gate prohibiting autonomous dispatch of observational pointings.
    All telescope queue modifications must be explicitly signed by the Lead Investigator.
    """

    def __init__(self):
        self._recommendations: Dict[str, ObservationRecommendation] = {}

    def create_recommendation(
        self,
        obs: ScheduledObservation,
        target: TargetState,
        scientific_justification: str,
    ) -> ObservationRecommendation:
        """Packages a scheduled observation into a human review proposal."""
        # Prior entropy
        ent_prior = 0.0
        for p in target.hypothesis_posteriors.values():
            if p > 1e-12:
                ent_prior -= p * math.log(p)

        ent_post = max(0.0, ent_prior - obs.expected_eig_nats)
        reduction_pct = (obs.expected_eig_nats / max(1e-4, ent_prior)) * 100.0

        raw_id = f"{obs.target_id}:{obs.facility_id}:{obs.filter_id}:{obs.start_mjd:.5f}"
        rec_id = f"rec:{hashlib.sha256(raw_id.encode()).hexdigest()[:16]}"

        raw_content = f"{rec_id}:{target.target_id}:{obs.facility_id}:{obs.instrument_id}:{obs.filter_id}:{obs.exposure_time_sec}:{obs.start_mjd}"
        prop_hash = hashlib.sha256(raw_content.encode()).hexdigest()

        rec = ObservationRecommendation(
            recommendation_id=rec_id,
            target_id=target.target_id,
            facility_id=obs.facility_id,
            instrument_id=obs.instrument_id,
            filter_id=obs.filter_id,
            exposure_time_sec=obs.exposure_time_sec,
            scheduled_mjd=obs.start_mjd,
            expected_eig_nats=round(obs.expected_eig_nats, 4),
            prior_entropy_nats=round(ent_prior, 4),
            expected_posterior_entropy_nats=round(ent_post, 4),
            entropy_reduction_pct=round(reduction_pct, 2),
            competing_hypotheses={k: round(v, 4) for k, v in target.hypothesis_posteriors.items()},
            scientific_justification=scientific_justification,
            physical_constraints_audit={
                "airmass": obs.airmass,
                "slew_time_sec": obs.slew_time_sec,
                "weather_score": obs.weather_score,
                "hard_violations": obs.constraint_violations,
            },
            approval_status=ApprovalStatus.PENDING_HUMAN_REVIEW,
            cryptographic_proposal_hash=prop_hash,
        )
        self._recommendations[rec_id] = rec
        return rec

    def authorize_dispatch(
        self,
        recommendation_id: str,
        investigator_email: str,
    ) -> ObservationRecommendation:
        """Authorizes dispatch with human investigator signature."""
        rec = self._recommendations.get(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found.")

        rec.approval_status = ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
        rec.authorized_investigator = investigator_email
        rec.authorization_timestamp_utc = time.time()
        return rec

    def reject_dispatch(
        self,
        recommendation_id: str,
        reason: str,
    ) -> ObservationRecommendation:
        """Rejects dispatch recommendation with justification."""
        rec = self._recommendations.get(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found.")

        rec.approval_status = ApprovalStatus.REJECTED_BY_INVESTIGATOR
        rec.rejection_reason = reason
        return rec

    def assert_authorized_for_dispatch(self, recommendation_id: str) -> None:
        """
        Enforces execution gate: raises AutonomousDispatchProhibitedError
        if recommendation is not officially approved.
        """
        rec = self._recommendations.get(recommendation_id)
        if rec is None or rec.approval_status != ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR:
            raise AutonomousDispatchProhibitedError(
                f"GOVERNANCE GATE VIOLATION: Recommendation {recommendation_id} is in state "
                f"'{rec.approval_status if rec else 'UNKNOWN'}'. Autonomous telescope dispatch is strictly prohibited."
            )
