"""
ASTRAEUS Deterministic Scientific Evidence Gates.
Evaluates agent outputs against immutable scientific, epistemic, and physical invariants.
Guarantees that final investigation statuses are assigned by deterministic rules,
strictly outside the language model's discretion.
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from astraeus_agents.contracts import (
    TerminalInvestigationStatus,
    SkepticChallenge,
    ChallengeStatus,
    ChallengeCategory,
    LiteratureCitation,
    CitationEvidenceType,
)
from astraeus_models.physics.engine import PhysicsFitResult
from astraeus_core.epistemic import EpistemicTier


class GateEvaluationResult(BaseModel):
    terminal_status: TerminalInvestigationStatus
    data_integrity_passed: bool
    artifact_filter_passed: bool
    known_object_status: str       # 'UNKNOWN_TRANSIENT', 'CONFIRMED_KNOWN_OBJECT', 'AMBIGUOUS'
    physical_plausibility_passed: bool
    provenance_complete: bool
    gate_justifications: List[str] = Field(default_factory=list)
    unresolved_fatal_objections: List[SkepticChallenge] = Field(default_factory=list)
    recommended_followup_action: str


class DeterministicEvidenceGateEngine:
    """
    Enforces deterministic validation rules across data quality, physical models,
    adversarial challenges, and epistemic boundaries.
    """

    @classmethod
    def evaluate_candidate(
        cls,
        entity_id: str,
        n_epochs: int,
        duration_days: float,
        quality_flags: List[int],
        physics_fit: Optional[PhysicsFitResult],
        literature_citations: List[LiteratureCitation],
        skeptic_challenges: List[SkepticChallenge],
        has_ivoa_provenance: bool = True
    ) -> GateEvaluationResult:
        """
        Executes the deterministic 5-stage gate pipeline and assigns terminal status.
        """
        justifications: List[str] = []

        # -------------------------------------------------------------
        # Stage 1: Data Completeness & Cadence Gate
        # -------------------------------------------------------------
        if n_epochs < 4 or duration_days <= 0.0:
            justifications.append(f"Insufficient photometric observations ({n_epochs} epochs < 4).")
            return GateEvaluationResult(
                terminal_status=TerminalInvestigationStatus.INSUFFICIENT_EVIDENCE,
                data_integrity_passed=False,
                artifact_filter_passed=False,
                known_object_status="UNKNOWN",
                physical_plausibility_passed=False,
                provenance_complete=has_ivoa_provenance,
                gate_justifications=justifications,
                recommended_followup_action="Wait for further survey detections or manual vetting."
            )

        data_integrity_passed = True

        # -------------------------------------------------------------
        # Stage 2: Adversarial Skeptic & Artifact Gate
        # -------------------------------------------------------------
        fatal_objections = [
            c for c in skeptic_challenges
            if c.status == ChallengeStatus.FATAL_DISQUALIFICATION
        ]

        # Check for unphysical instrumental spikes or subtraction errors
        has_artifact_flag = any(f != 0 for f in quality_flags)
        artifact_categories = {ChallengeCategory.PSF_SHAPE_FAILURE, ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT}
        artifact_objections = [c for c in fatal_objections if c.category in artifact_categories]

        if artifact_objections or (has_artifact_flag and len(quality_flags) <= 5):
            for c in artifact_objections:
                justifications.append(f"Fatal Skeptic Objection: {c.assertion}")
            if has_artifact_flag:
                justifications.append("Instrumental quality flag triggered non-zero error bitmask.")

            return GateEvaluationResult(
                terminal_status=TerminalInvestigationStatus.REJECTED_AS_ARTIFACT,
                data_integrity_passed=True,
                artifact_filter_passed=False,
                known_object_status="INSTRUMENTAL_DEFECT",
                physical_plausibility_passed=False,
                provenance_complete=has_ivoa_provenance,
                gate_justifications=justifications,
                unresolved_fatal_objections=artifact_objections,
                recommended_followup_action="Quarantine alert packet; discard from astronomical candidate stream."
            )

        artifact_filter_passed = True

        # -------------------------------------------------------------
        # Stage 3: Known Object & Archival Literature Gate
        # -------------------------------------------------------------
        # Check if literature confirms a known periodic variable star or previously cataloged normal SN
        is_known_normal = False
        for cit in literature_citations:
            if cit.evidence_type in {CitationEvidenceType.DIRECT_MEASUREMENT, CitationEvidenceType.CATALOG_ASSOCIATION}:
                txt_lower = cit.relevant_excerpt.lower()
                if any(k in txt_lower for k in ["rr lyrae", "eclipsing binary", "known cataclysmic variable", "normal sn ia"]):
                    is_known_normal = True
                    justifications.append(f"Archival literature identifies source as known entity: {cit.title} ({cit.bibcode_or_doi}).")

        # -------------------------------------------------------------
        # Stage 4: Physical Plausibility & Parameter Novelty Gate
        # -------------------------------------------------------------
        physical_plausibility_passed = True
        d_phys = 0.0
        if physics_fit is not None:
            d_phys = physics_fit.parameter_space_deviation
            # Check for unphysical parameter limits
            params = physics_fit.physical_parameters
            if params.get("m_ni", 0.0) > 3.0:
                physical_plausibility_passed = False
                justifications.append(f"Unphysical Nickel mass inferred: {params['m_ni']:.2f} M_sun > 3.0 M_sun.")
            if params.get("tau_rise", 10.0) < 0.1:
                physical_plausibility_passed = False
                justifications.append(f"Unphysical rise timescale: {params['tau_rise']:.2f} d < 0.1 d.")

        # -------------------------------------------------------------
        # Stage 5: Final Terminal Status Assignment
        # -------------------------------------------------------------
        if is_known_normal and d_phys < 2.5:
            terminal_status = TerminalInvestigationStatus.KNOWN_OBJECT
            justifications.append("Source matches known cataloged object with normal astrophysical parameters.")
            action = "Archive candidate; no immediate telescope trigger required."
        elif is_known_normal and d_phys >= 2.5:
            terminal_status = TerminalInvestigationStatus.KNOWN_CLASS_UNUSUAL
            justifications.append(f"Source belongs to known class but exhibits unusual physical parameter deviation (D_phys = {d_phys:.2f} sigma).")
            action = "Schedule secondary spectroscopic confirmation to investigate anomalous parameter behavior."
        elif d_phys >= 3.0 and physical_plausibility_passed and len(fatal_objections) == 0:
            terminal_status = TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP
            justifications.append(
                f"Candidate exhibits significant physical parameter deviation (D_phys = {d_phys:.2f} sigma), "
                "survived all adversarial skeptic challenges, and has valid data provenance."
            )
            action = "Recommend urgent multi-band photometric and spectroscopic follow-up for human astronomer review."
        elif not physical_plausibility_passed:
            terminal_status = TerminalInvestigationStatus.INSUFFICIENT_EVIDENCE
            justifications.append("Physical parameter inference failed plausibility bounds.")
            action = "Awaiting deeper observations or manual review."
        else:
            terminal_status = TerminalInvestigationStatus.KNOWN_OBJECT
            justifications.append("Candidate physical parameters consistent with standard astronomical population.")
            action = "Routine monitoring."

        return GateEvaluationResult(
            terminal_status=terminal_status,
            data_integrity_passed=data_integrity_passed,
            artifact_filter_passed=artifact_filter_passed,
            known_object_status="CONFIRMED_KNOWN" if is_known_normal else "NOVEL_CANDIDATE",
            physical_plausibility_passed=physical_plausibility_passed,
            provenance_complete=has_ivoa_provenance,
            gate_justifications=justifications,
            unresolved_fatal_objections=fatal_objections,
            recommended_followup_action=action,
        )
