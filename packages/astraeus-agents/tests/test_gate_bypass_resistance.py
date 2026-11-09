"""
Unit tests for ASTRAEUS Deterministic Evidence Gate Bypass Resistance.
Verifies that no agent, API shortcut, or downstream mutation can circumvent
the deterministic evidence gate invariants at the system boundary.
"""

import pytest
from pydantic import ValidationError

from astraeus_agents.contracts import (
    TerminalInvestigationStatus,
    SkepticChallenge,
    ChallengeCategory,
    ChallengeStatus,
)
from astraeus_agents.evidence_gates import GateEvaluationResult
from astraeus_agents.dossier import InvestigationDossier
from astraeus_data.crossmatch.cross_survey_matcher import CrossSurveyAssociation, CrossSurveyMatchStatus
from astraeus_core.provenance import ProvEntity


def dummy_cross_match() -> CrossSurveyAssociation:
    return CrossSurveyAssociation(
        query_entity_id="test_entity",
        match_status=CrossSurveyMatchStatus.NO_RELIABLE_MATCH,
        separation_arcsec=999.0,
        posterior_match_probability=0.0,
        chance_coincidence_probability=1.0,
        local_source_density_per_sq_arcsec=0.001,
        is_training_overlap_quarantined=False,
    )


def dummy_provenance() -> ProvEntity:
    return ProvEntity(
        entity_id="test_entity",
        entity_type="astronomical_candidate",
        was_generated_by="activity:test",
        was_associated_with="agent:test",
    )


def test_bypass_blocked_when_terminal_status_mismatches_gate():
    """Attempting to assign CANDIDATE_REQUIRES_FOLLOWUP when gate returned REJECTED_AS_ARTIFACT must fail."""
    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.REJECTED_AS_ARTIFACT,
        data_integrity_passed=True,
        artifact_filter_passed=False,
        known_object_status="DEFECT",
        physical_plausibility_passed=False,
        provenance_complete=True,
        recommended_followup_action="Quarantine alert.",
    )

    with pytest.raises(ValidationError) as exc_info:
        InvestigationDossier(
            dossier_id="dossier:bypass:001",
            target_entity_id="test_entity",
            ra_deg=244.0,
            dec_deg=22.0,
            timestamp_utc="2026-11-09T00:00:00Z",
            terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,  # Bypassed claim
            recommended_action="Trigger ToO follow-up.",
            data_summary={"num_detections": 6},
            cross_match=dummy_cross_match(),
            physical_modeling={},
            literature_context=[],
            skeptic_challenges=[],
            gate_evaluation=gate_res,
            provenance=dummy_provenance(),
        )

    assert "Gate bypass detected: dossier terminal_status" in str(exc_info.value)


def test_bypass_blocked_when_followup_claims_failed_artifact_gate():
    """Attempting to claim follow-up when artifact filter failed must be rejected."""
    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
        data_integrity_passed=True,
        artifact_filter_passed=False,  # Failed artifact filter!
        known_object_status="NOVEL",
        physical_plausibility_passed=True,
        provenance_complete=True,
        recommended_followup_action="Trigger ToO.",
    )

    with pytest.raises(ValidationError) as exc_info:
        InvestigationDossier(
            dossier_id="dossier:bypass:002",
            target_entity_id="test_entity",
            ra_deg=244.0,
            dec_deg=22.0,
            timestamp_utc="2026-11-09T00:00:00Z",
            terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
            recommended_action="Trigger ToO.",
            data_summary={"num_detections": 6},
            cross_match=dummy_cross_match(),
            physical_modeling={},
            literature_context=[],
            skeptic_challenges=[],
            gate_evaluation=gate_res,
            provenance=dummy_provenance(),
        )

    assert "Follow-up recommended despite failed artifact filter gate" in str(exc_info.value)


def test_bypass_blocked_when_fatal_objections_unresolved():
    """Attempting to claim follow-up with unresolved fatal skeptic objections must be rejected."""
    fatal_challenge = SkepticChallenge(
        challenge_id="chk:fatal:001",
        category=ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT,
        assertion="Fatal dipole residual.",
        evidence_data={},
        status=ChallengeStatus.FATAL_DISQUALIFICATION,
    )
    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
        data_integrity_passed=True,
        artifact_filter_passed=True,
        known_object_status="NOVEL",
        physical_plausibility_passed=True,
        provenance_complete=True,
        unresolved_fatal_objections=[fatal_challenge],  # Fatal objection!
        recommended_followup_action="Trigger ToO.",
    )

    with pytest.raises(ValidationError) as exc_info:
        InvestigationDossier(
            dossier_id="dossier:bypass:003",
            target_entity_id="test_entity",
            ra_deg=244.0,
            dec_deg=22.0,
            timestamp_utc="2026-11-09T00:00:00Z",
            terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
            recommended_action="Trigger ToO.",
            data_summary={"num_detections": 6},
            cross_match=dummy_cross_match(),
            physical_modeling={},
            literature_context=[],
            skeptic_challenges=[fatal_challenge],
            gate_evaluation=gate_res,
            provenance=dummy_provenance(),
        )

    assert "Follow-up recommended despite unresolved fatal skeptic objections" in str(exc_info.value)


def test_dossier_is_frozen_against_downstream_mutation():
    """InvestigationDossier instances must be immutable (frozen). Mutation attempts must raise ValidationError."""
    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.KNOWN_OBJECT,
        data_integrity_passed=True,
        artifact_filter_passed=True,
        known_object_status="KNOWN",
        physical_plausibility_passed=True,
        provenance_complete=True,
        recommended_followup_action="Routine logging.",
    )

    dossier = InvestigationDossier(
        dossier_id="dossier:frozen:001",
        target_entity_id="test_entity",
        ra_deg=180.0,
        dec_deg=10.0,
        timestamp_utc="2026-11-09T00:00:00Z",
        terminal_status=TerminalInvestigationStatus.KNOWN_OBJECT,
        recommended_action="Routine logging.",
        data_summary={"num_detections": 8},
        cross_match=dummy_cross_match(),
        physical_modeling={},
        literature_context=[],
        skeptic_challenges=[],
        gate_evaluation=gate_res,
        provenance=dummy_provenance(),
    )

    with pytest.raises(ValidationError):
        dossier.terminal_status = TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP  # Illegal mutation!
