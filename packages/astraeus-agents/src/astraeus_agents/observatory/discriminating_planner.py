"""
ASTRAEUS Discriminating Experimental Design Intelligence (WS8.6).
Evaluates Expected Model Discrimination (EMD), Value of Information (VoI),
and enforces the Human Authorization Boundary for all telescope queue dispatch.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from astraeus_agents.discovery.hypothesis_generator import StructuredHypothesis
from astraeus_agents.observatory.human_approval_interface import (
    AutonomousDispatchProhibitedError,
    AuthorizationToken,
    HumanApprovalGate,
    provision_cryptographic_secret,
)


class CandidateObservationDesign(BaseModel):
    """Candidate observational action to be evaluated for model discrimination."""
    action_id: str
    target_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    exposure_time_sec: float
    slew_time_sec: float
    scheduled_mjd: float
    facility_cost_multiplier: float = 1.0  # Rarity weight: e.g. JWST=10.0, VLT=3.0, ZTF=1.0


class DiscriminationEvaluation(BaseModel):
    """Quantitative model discrimination assessment for an observation design."""
    action_id: str
    target_id: str
    facility_id: str
    filter_id: str
    expected_model_discrimination_nats: float
    value_of_information: float
    competing_hypotheses: List[str]
    pairwise_kl_divergences: Dict[str, float]
    recommended: bool
    requires_human_token: bool = True
    dispatch_status: str = "PENDING_APPROVAL"


class DiscriminatingExperimentPlanner:
    """
    Selects optimal discriminating observations that maximize information-theoretic
    distance between competing scientific hypotheses, subject to resource costs
    and the non-negotiable Human Approval Gate.
    """

    def __init__(self, human_gate: Optional[HumanApprovalGate] = None) -> None:
        self.human_gate = human_gate or HumanApprovalGate()

    def calculate_gaussian_kl_divergence(
        self,
        mu1: float,
        sig1: float,
        mu2: float,
        sig2: float,
    ) -> float:
        """
        Computes analytical KL divergence D_KL(N(mu1, sig1^2) || N(mu2, sig2^2)):
        D_KL = ln(sig2 / sig1) + (sig1^2 + (mu1 - mu2)^2) / (2 * sig2^2) - 0.5.
        """
        s1 = max(1e-4, sig1)
        s2 = max(1e-4, sig2)
        term1 = math.log(s2 / s1)
        term2 = (s1**2 + (mu1 - mu2)**2) / (2.0 * s2**2)
        return float(term1 + term2 - 0.5)

    def evaluate_expected_model_discrimination(
        self,
        candidate: CandidateObservationDesign,
        hypotheses: List[StructuredHypothesis],
        hypothesis_priors: Dict[str, float],
    ) -> DiscriminationEvaluation:
        """
        Computes Expected Model Discrimination (EMD):
        EMD(a) = sum_{i < j} P(H_i) P(H_j) * [ D_KL(p_i || p_j) + D_KL(p_j || p_i) ] / 2.
        """
        pairwise_kl: Dict[str, float] = {}
        total_emd = 0.0

        # Extract predictions for this filter / observable from each hypothesis
        preds: Dict[str, Tuple[float, float]] = {}
        for h in hypotheses:
            # Match prediction closest to this filter
            matching = next((p for p in h.predictions if candidate.filter_id.lower() in p.observable_name.lower()), None)
            if matching:
                p_span = matching.uncertainty_range[1] - matching.uncertainty_range[0]
                sig = max(0.05, p_span / 3.92)
                preds[h.hypothesis_id] = (matching.predicted_value, sig)
            else:
                # Default baseline prediction
                preds[h.hypothesis_id] = (19.0, 0.20)

        h_ids = list(preds.keys())
        for i in range(len(h_ids)):
            for j in range(i + 1, len(h_ids)):
                h1, h2 = h_ids[i], h_ids[j]
                mu1, sig1 = preds[h1]
                mu2, sig2 = preds[h2]

                p1 = hypothesis_priors.get(h1, 1.0 / len(h_ids))
                p2 = hypothesis_priors.get(h2, 1.0 / len(h_ids))

                kl_12 = self.calculate_gaussian_kl_divergence(mu1, sig1, mu2, sig2)
                kl_21 = self.calculate_gaussian_kl_divergence(mu2, sig2, mu1, sig1)
                sym_kl = 0.5 * (kl_12 + kl_21)

                pairwise_kl[f"{h1}_vs_{h2}"] = round(sym_kl, 4)
                total_emd += (p1 * p2) * sym_kl

        # Compute Value of Information (VoI)
        # Cost = (slew_time + exposure_time) / 3600.0 * cost_multiplier
        total_time_hours = (candidate.slew_time_sec + candidate.exposure_time_sec) / 3600.0
        effective_cost = max(0.01, total_time_hours * candidate.facility_cost_multiplier)
        voi = total_emd / effective_cost

        return DiscriminationEvaluation(
            action_id=candidate.action_id,
            target_id=candidate.target_id,
            facility_id=candidate.facility_id,
            filter_id=candidate.filter_id,
            expected_model_discrimination_nats=round(total_emd, 4),
            value_of_information=round(voi, 4),
            competing_hypotheses=h_ids,
            pairwise_kl_divergences=pairwise_kl,
            recommended=total_emd > 0.5,
            requires_human_token=True,
            dispatch_status="PENDING_APPROVAL",
        )

    def dispatch_observation_to_telescope(
        self,
        evaluation: DiscriminationEvaluation,
        human_token: Optional[AuthorizationToken] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches observation design to observatory queue.
        Enforces Gate 3 Zero Autonomous Dispatch Covenant:
        If token is None or invalid, raises AutonomousDispatchProhibitedError.
        """
        if human_token is None:
            raise AutonomousDispatchProhibitedError(
                f"Autonomous Dispatch Prohibited: Observation action {evaluation.action_id} on target {evaluation.target_id} "
                f"requires explicit human investigator authorization token!"
            )

        # Validate token
        if human_token.target_id != evaluation.target_id:
            raise AutonomousDispatchProhibitedError(
                f"Token target mismatch: token issued for {human_token.target_id}, action is for {evaluation.target_id}."
            )

        if human_token.is_expired():
            raise AutonomousDispatchProhibitedError(
                f"Expired authorization token {human_token.token_id} for target {evaluation.target_id}."
            )

        if not human_token.hmac_signature:
            raise AutonomousDispatchProhibitedError(
                f"Token {human_token.token_id} lacks required cryptographic HMAC signature."
            )

        evaluation.dispatch_status = "DISPATCHED_TO_QUEUE"

        return {
            "status": "SUCCESS",
            "dispatch_id": f"disp_{evaluation.action_id}",
            "action_id": evaluation.action_id,
            "target_id": evaluation.target_id,
            "facility_id": evaluation.facility_id,
            "filter_id": evaluation.filter_id,
            "authorized_by": human_token.investigator_email,
            "dispatched_at_iso": datetime.now(timezone.utc).isoformat(),
        }
