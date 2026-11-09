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


def test_queue_blocks_direct_agent_writes():
    """Agents must NOT have permission to write directly to the telescope follow-up queue."""
    from astraeus_agents.queue import FollowUpQueueManager, UnauthorizedQueueSubmissionError

    queue_mgr = FollowUpQueueManager()
    with pytest.raises(UnauthorizedQueueSubmissionError) as exc_info:
        queue_mgr.direct_agent_enqueue(candidate_id="unauthorized_transient_001")

    assert "Direct agent queue writes are prohibited" in str(exc_info.value)


def test_queue_authorizes_genuine_followup_with_cryptographic_digest():
    """A valid candidate passing all deterministic gates must be enqueued with a valid SHA-256 digest."""
    from astraeus_agents.queue import FollowUpQueueManager

    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
        data_integrity_passed=True,
        artifact_filter_passed=True,
        known_object_status="NOVEL",
        physical_plausibility_passed=True,
        provenance_complete=True,
        recommended_followup_action="Trigger ToO spectroscopy.",
        gate_justifications=["Multi-band rise confirmed.", "No subtraction artifacts."],
    )

    dossier = InvestigationDossier(
        dossier_id="dossier:valid:001",
        target_entity_id="FBOT-2018cow-replay",
        ra_deg=244.000917,
        dec_deg=22.268031,
        timestamp_utc="2026-11-09T00:00:00Z",
        terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
        recommended_action="Trigger ToO spectroscopy.",
        data_summary={"num_detections": 6},
        cross_match=dummy_cross_match(),
        physical_modeling={"parameter_space_deviation": 3.92},
        literature_context=[],
        skeptic_challenges=[],
        gate_evaluation=gate_res,
        provenance=dummy_provenance(),
    )

    queue_mgr = FollowUpQueueManager()
    record = queue_mgr.authorize_and_enqueue(dossier)

    assert record.target_entity_id == "FBOT-2018cow-replay"
    assert record.is_authoritative is True
    assert len(record.gate_digest_sha256) == 64
    assert len(queue_mgr.get_queue()) == 1
    assert queue_mgr.verify_queue_integrity() is True


def test_queue_rejects_non_followup_terminal_statuses():
    """Attempting to enqueue a KNOWN_OBJECT or REJECTED_AS_ARTIFACT dossier must be blocked at queue boundary."""
    from astraeus_agents.queue import FollowUpQueueManager, GateValidationMismatchError

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
        dossier_id="dossier:known:001",
        target_entity_id="SN-2026abc-known",
        ra_deg=150.0,
        dec_deg=2.0,
        timestamp_utc="2026-11-09T00:00:00Z",
        terminal_status=TerminalInvestigationStatus.KNOWN_OBJECT,
        recommended_action="Routine logging.",
        data_summary={"num_detections": 15},
        cross_match=dummy_cross_match(),
        physical_modeling={},
        literature_context=[],
        skeptic_challenges=[],
        gate_evaluation=gate_res,
        provenance=dummy_provenance(),
    )

    queue_mgr = FollowUpQueueManager()
    with pytest.raises(GateValidationMismatchError) as exc_info:
        queue_mgr.authorize_and_enqueue(dossier)

    assert "Cannot enqueue candidate with terminal status 'KNOWN_OBJECT'" in str(exc_info.value)
    assert len(queue_mgr.get_queue()) == 0


def test_queue_independent_gate_reevaluation_catches_spoof():
    """If independent gate re-evaluation indicates artifact, queue insertion must be rejected even if dossier claims clean."""
    from astraeus_agents.queue import FollowUpQueueManager, GateValidationMismatchError
    from astraeus_agents.evidence_gates import DeterministicEvidenceGateEngine
    from astraeus_agents.contracts import ChallengeCategory

    # Spoofed dossier claims gate passed
    gate_res = GateEvaluationResult(
        terminal_status=TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP,
        data_integrity_passed=True,
        artifact_filter_passed=True,
        known_object_status="NOVEL",
        physical_plausibility_passed=True,
        provenance_complete=True,
        recommended_followup_action="Trigger ToO.",
    )

    dossier = InvestigationDossier(
        dossier_id="dossier:spoof:001",
        target_entity_id="Artifact-Dipole-Spoof",
        ra_deg=100.0,
        dec_deg=15.0,
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

    # Candidate data actually has a fatal dipole challenge in independent evaluation
    fatal_challenge = SkepticChallenge(
        challenge_id="chk:dipole:999",
        category=ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT,
        assertion="Subtraction dipole artifact detected.",
        evidence_data={},
        status=ChallengeStatus.FATAL_DISQUALIFICATION,
    )
    raw_candidate_data = {
        "data_summary": {"num_detections": 6, "has_image_subtraction_defect": True},
        "cross_match": dummy_cross_match(),
        "physics_result": None,
        "skeptic_challenges": [fatal_challenge],
        "provenance": dummy_provenance(),
    }

    queue_mgr = FollowUpQueueManager()
    gate_engine = DeterministicEvidenceGateEngine()

    with pytest.raises(GateValidationMismatchError) as exc_info:
        queue_mgr.authorize_and_enqueue(dossier, independent_gate_engine=gate_engine, candidate_data=raw_candidate_data)

    assert "Independent gate re-evaluation refused follow-up" in str(exc_info.value)
    assert len(queue_mgr.get_queue()) == 0

