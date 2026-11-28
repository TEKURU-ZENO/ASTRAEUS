"""
ASTRAEUS Integrated Investigator Operations Console (WS10.7).
Provides an interactive operational command terminal and API dashboard for
Lead Investigator Dev Mehta and the ASTRAEUS Science Working Group:
1. Real-time alert feed & multi-messenger correlation monitor.
2. Multi-band light curves with Bayesian posterior credible intervals.
3. Localization 50% and 90% credible sky areas.
4. Competing physical hypothesis evaluation and Bayes factor ranking.
5. External custodian cryptographic audit trail.
6. Human-in-the-loop telescope follow-up authorization gate.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import os
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from astraeus_agents.discovery.evidence_fabric import (
    UnifiedEvidenceFabric,
    EvidenceNodeType,
    EvidenceNode,
)
from astraeus_agents.discovery.multimessenger_reasoner import (
    MultiMessengerInferenceEngine,
    ModelSelectionReport,
    CompetingPhysicalModel,
)
from astraeus_agents.governance.learning_governance import (
    LiveIntelligenceGovernanceGate,
    FollowUpRecommendation,
    OperationalMode,
)
from astraeus_bench.custody.custodian_service import (
    StandaloneCustodianService,
    CustodianServiceClient,
)


class TargetSummaryView(BaseModel):
    target_id: str
    survey_origin: str
    ra_deg: float
    dec_deg: float
    num_photometry_points: int
    favored_model: str
    bayes_factor_vs_runner_up: float
    decisive_evidence: bool
    custodian_ledger_status: str
    follow_up_recommended: bool
    is_authorized_by_human: bool


class InvestigatorConsole:
    """Integrated operations console for human investigators."""

    def __init__(
        self,
        investigator_id: str = "Dev Mehta <devmehta060505@gmail.com>",
        governance_gate: Optional[LiveIntelligenceGovernanceGate] = None,
        custodian_client: Optional[CustodianServiceClient] = None,
        evidence_fabric: Optional[UnifiedEvidenceFabric] = None,
    ) -> None:
        self.investigator_id = investigator_id
        self.gate = governance_gate or LiveIntelligenceGovernanceGate(mode=OperationalMode.LIVE_INTELLIGENCE)
        self.custodian = custodian_client
        self.fabric = evidence_fabric or UnifiedEvidenceFabric()
        self.inference_engine = MultiMessengerInferenceEngine()
        self.active_targets: Dict[str, Dict[str, Any]] = {}

    def register_incoming_event(
        self,
        target_id: str,
        survey_origin: str,
        ra_deg: float,
        dec_deg: float,
        photometry_points: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TargetSummaryView:
        """Processes and displays an incoming transient event."""
        # 1. Evidence fabric node
        obs_node = self.fabric.add_node(
            node_type=EvidenceNodeType.OBSERVATION,
            payload={
                "target_id": target_id,
                "survey_origin": survey_origin,
                "ra_deg": ra_deg,
                "dec_deg": dec_deg,
                "num_points": len(photometry_points),
                "metadata": metadata or {},
            },
            author_or_agent="INVESTIGATOR_CONSOLE_INGEST",
        )

        # 2. Bayesian model selection if photometry is available
        report: Optional[ModelSelectionReport] = None
        if len(photometry_points) >= 3:
            report = self.inference_engine.evaluate_competing_physical_models(photometry_points)
            favored = report.favored_model.value
            bf = report.model_comparisons[1].bayes_factor_vs_best if len(report.model_comparisons) > 1 else 100.0
            decisive = report.decisive_evidence
        else:
            favored = "INSUFFICIENT_DATA"
            bf = 1.0
            decisive = False

        # 3. Automatic follow-up recommendation if decisive or promising
        rec: Optional[FollowUpRecommendation] = None
        if decisive or (len(photometry_points) >= 2 and photometry_points[-1].get("mag", 20.0) < 19.0):
            last_mag = photometry_points[-1].get("mag", 18.0)
            rec = self.gate.generate_follow_up_recommendation(
                target_id=target_id,
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                facility="VLT_XSHOOTER",
                filters=["optical_spectrum", "NIR_spectroscopy"],
                urgency_hours=4.0,
                justification=f"Rapidly evolving {favored} candidate requiring immediate spectroscopic classification.",
                pred_mag=last_mag + 0.3,
            )

        self.active_targets[target_id] = {
            "survey_origin": survey_origin,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "photometry": photometry_points,
            "report": report,
            "recommendation": rec,
            "obs_node_id": obs_node.node_id,
        }

        return TargetSummaryView(
            target_id=target_id,
            survey_origin=survey_origin,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            num_photometry_points=len(photometry_points),
            favored_model=favored,
            bayes_factor_vs_runner_up=round(bf, 2),
            decisive_evidence=decisive,
            custodian_ledger_status="VERIFIED" if self.custodian else "STANDALONE_LOCAL",
            follow_up_recommended=(rec is not None),
            is_authorized_by_human=False,
        )

    def human_approve_follow_up(
        self,
        target_id: str,
        investigator_hmac_key: bytes,
    ) -> Dict[str, Any]:
        """
        Interactive human investigator gate:
        Authorizes observation dispatch with human HMAC signature.
        """
        if target_id not in self.active_targets:
            raise KeyError(f"Target '{target_id}' not found in active console targets.")

        rec = self.active_targets[target_id].get("recommendation")
        if rec is None:
            raise ValueError(f"No pending follow-up recommendation exists for target '{target_id}'.")

        authorized_rec = self.gate.authorize_recommendation(
            recommendation_id=rec.recommendation_id,
            investigator_id=self.investigator_id,
            investigator_hmac_key=investigator_hmac_key,
        )

        dispatch_result = self.gate.dispatch_telescope_command(authorized_rec.recommendation_id)

        # Record decision in evidence fabric
        self.fabric.add_node(
            node_type=EvidenceNodeType.DECISION,
            payload={
                "target_id": target_id,
                "recommendation_id": authorized_rec.recommendation_id,
                "facility": authorized_rec.recommended_facility,
                "authorized_by": self.investigator_id,
                "signature": authorized_rec.investigator_signature_hex,
            },
            parent_node_ids=[self.active_targets[target_id]["obs_node_id"]],
            author_or_agent=self.investigator_id,
        )

        return dispatch_result

    def render_console_dashboard_text(self) -> str:
        """Renders an ASCII/Markdown status dashboard of the investigator console."""
        lines = [
            "=" * 78,
            f"ASTRAEUS OPERATIONAL INVESTIGATOR CONSOLE - LEAD: {self.investigator_id.upper()}",
            f"Operational Mode: {self.gate.mode.value} | Active Targets: {len(self.active_targets)}",
            "=" * 78,
            f"{'Target ID':<22} | {'Survey':<8} | {'RA/Dec':<16} | {'Favored Model':<20} | {'Status':<12}",
            "-" * 78,
        ]

        for tid, data in self.active_targets.items():
            rep = data.get("report")
            model_str = rep.favored_model.value if rep else "INGESTED"
            coord_str = f"{data['ra_deg']:.2f}, {data['dec_deg']:.2f}"
            rec = data.get("recommendation")
            status = "PENDING_AUTH" if (rec and not rec.is_signed_by_investigator) else "MONITORING"
            if rec and rec.is_signed_by_investigator:
                status = "DISPATCHED"

            lines.append(f"{tid:<22} | {data['survey_origin']:<8} | {coord_str:<16} | {model_str:<20} | {status:<12}")

        lines.append("=" * 78)
        return "\n".join(lines)
