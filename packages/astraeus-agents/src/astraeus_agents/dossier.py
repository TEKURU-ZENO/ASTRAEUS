"""
ASTRAEUS Scientific Investigation Dossier Compiler.
Compiles the comprehensive, auditable candidate investigation dossier
in both structured JSON and rich Markdown formats, adhering strictly to
epistemic boundaries, IVOA provenance, and human-in-the-loop scientific oversight.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import json
from pydantic import BaseModel, Field, ConfigDict, model_validator

from astraeus_agents.contracts import (
    TerminalInvestigationStatus,
    LiteratureCitation,
    SkepticChallenge,
)
from astraeus_agents.evidence_gates import GateEvaluationResult
from astraeus_models.physics.engine import PhysicsFitResult
from astraeus_data.crossmatch.cross_survey_matcher import CrossSurveyAssociation
from astraeus_core.provenance import ProvEntity


class InvestigationDossier(BaseModel):
    """
    Immutable scientific candidate investigation dossier.
    Enforces deterministic gate alignment and bypass resistance at the system boundary.
    """
    model_config = ConfigDict(frozen=True)

    dossier_id: str
    target_entity_id: str
    ra_deg: float
    dec_deg: float
    timestamp_utc: str
    terminal_status: TerminalInvestigationStatus
    recommended_action: str
    data_summary: Dict[str, Any]
    cross_match: CrossSurveyAssociation
    physical_modeling: Dict[str, Any]
    literature_context: List[LiteratureCitation]
    skeptic_challenges: List[SkepticChallenge]
    gate_evaluation: GateEvaluationResult
    provenance: ProvEntity
    epistemic_disclaimer: str = (
        "ASTRAEUS Phase 1 Operational Invariant: This dossier represents an evidence-calibrated "
        "autonomous investigation. It does NOT constitute an independent or unverified declaration "
        "of scientific discovery. Final physical claims require human spectroscopic confirmation."
    )

    @model_validator(mode="after")
    def verify_gate_bypass_resistance(self) -> "InvestigationDossier":
        """
        Guarantees that no caller or agent can bypass the Deterministic Evidence Gate.
        """
        if self.terminal_status != self.gate_evaluation.terminal_status:
            raise ValueError(
                f"Gate bypass detected: dossier terminal_status ({self.terminal_status.value}) "
                f"does not match gate_evaluation terminal_status ({self.gate_evaluation.terminal_status.value})."
            )

        if self.terminal_status == TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP:
            if not self.gate_evaluation.data_integrity_passed:
                raise ValueError("Gate bypass detected: Follow-up recommended despite failed data integrity gate.")
            if not self.gate_evaluation.artifact_filter_passed:
                raise ValueError("Gate bypass detected: Follow-up recommended despite failed artifact filter gate.")
            if not self.gate_evaluation.physical_plausibility_passed:
                raise ValueError("Gate bypass detected: Follow-up recommended despite failed physical plausibility gate.")
            if len(self.gate_evaluation.unresolved_fatal_objections) > 0:
                raise ValueError("Gate bypass detected: Follow-up recommended despite unresolved fatal skeptic objections.")

        return self

    def to_markdown(self) -> str:
        """Renders the dossier as an auditable scientific Markdown report."""
        lines = []
        lines.append(f"# ASTRAEUS Scientific Investigation Dossier: `{self.target_entity_id}`")
        lines.append(f"**Dossier ID:** `{self.dossier_id}` | **Generated:** `{self.timestamp_utc}`")
        lines.append(f"**Coordinates:** RA = {self.ra_deg:.6f}°, Dec = {self.dec_deg:.6f}°")
        lines.append(f"**Terminal Scientific Status:** `{self.terminal_status.value}`")
        lines.append(f"**Action Recommended:** {self.recommended_action}\n")

        lines.append("## 1. Observational Photometry & Cross-Match Summary")
        lines.append(f"- **Epochs:** {self.data_summary.get('num_detections', 0)} detections, {self.data_summary.get('num_upper_limits', 0)} upper limits over {self.data_summary.get('time_baseline_days', 0.0):.1f} days.")
        lines.append(f"- **Bayesian Counterpart Status:** `{self.cross_match.match_status.value}` (Separation: {self.cross_match.separation_arcsec:.2f}\", Posterior P: {self.cross_match.posterior_match_probability:.3f}).\n")

        lines.append("## 2. Decoupled Physical Model Inference")
        lines.append(f"- **Preferred Model:** `{self.physical_modeling.get('preferred_model', 'None')}`")
        lines.append(f"- **Model Fit Quality (reduced chi2):** {self.physical_modeling.get('model_fit_quality', 'N/A')}")
        lines.append(f"- **Parameter-Space Deviation (D_phys):** {self.physical_modeling.get('parameter_space_deviation', 0.0):.2f} sigma against reference SNe.")
        lines.append(f"- **Inferred Parameters:** `{json.dumps(self.physical_modeling.get('physical_parameters', {}))}`\n")

        lines.append("## 3. Literature Grounding & Archival Context")
        if self.literature_context:
            for cit in self.literature_context:
                direct_str = "Direct Measurement" if cit.is_direct_evidence else "Inferred / Hypothesis"
                lines.append(f"- **[{cit.citation_id}]** *{cit.title}* ({cit.authors}, {cit.publication_year})")
                lines.append(f"  - **Evidence Type:** `{cit.evidence_type.value}` ({direct_str})")
                lines.append(f"  - **Excerpt:** \"{cit.relevant_excerpt}\"")
                lines.append(f"  - **Source:** [{cit.bibcode_or_doi}]({cit.source_url})")
        else:
            lines.append("- *No prior cataloged literature matches found in archival database.*")
        lines.append("")

        lines.append("## 4. Adversarial Skeptic Audit")
        for chk in self.skeptic_challenges:
            status_symbol = "RESOLVED" if chk.status.value == "RESOLVED" else "FATAL OBJECTION" if chk.status.value == "FATAL_DISQUALIFICATION" else "WARNING RAISED"
            lines.append(f"- **[{chk.category.value}]** `{status_symbol}`: {chk.assertion}")
            if chk.resolution_notes:
                lines.append(f"  - *Resolution:* {chk.resolution_notes}")
        lines.append("")

        lines.append("## 5. Deterministic Evidence Gate Evaluation")
        lines.append(f"- **Integrity Gate:** `{'PASSED' if self.gate_evaluation.data_integrity_passed else 'FAILED'}`")
        lines.append(f"- **Artifact Filter:** `{'PASSED' if self.gate_evaluation.artifact_filter_passed else 'REJECTED'}`")
        lines.append(f"- **Physical Plausibility:** `{'PASSED' if self.gate_evaluation.physical_plausibility_passed else 'FAILED'}`")
        lines.append(f"- **Gate Justifications:**")
        for just in self.gate_evaluation.gate_justifications:
            lines.append(f"  - {just}")
        lines.append("")

        lines.append("## 6. IVOA Provenance & Epistemic Boundaries")
        lines.append(f"- **Activity ID (wasGeneratedBy):** `{self.provenance.was_generated_by}`")
        lines.append(f"- **Agent ID (wasAssociatedWith):** `{self.provenance.was_associated_with}`")
        lines.append(f"- **Disclaimer:** *{self.epistemic_disclaimer}*\n")

        return "\n".join(lines)
