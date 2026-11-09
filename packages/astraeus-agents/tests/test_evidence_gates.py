"""
Unit tests for ASTRAEUS Deterministic Evidence Gates.
Verifies that terminal statuses are assigned by deterministic rules,
strictly outside the language model's discretion.
"""

import pytest

from astraeus_agents.contracts import (
    TerminalInvestigationStatus,
    SkepticChallenge,
    ChallengeCategory,
    ChallengeStatus,
    LiteratureCitation,
    CitationEvidenceType,
)
from astraeus_agents.evidence_gates import DeterministicEvidenceGateEngine, GateEvaluationResult
from astraeus_models.physics.engine import PhysicsFitResult
from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.bazin import BazinFitResult
from astraeus_models.physics.arnett import ArnettFitResult
from astraeus_models.physics.shock_cooling import ShockCoolingFitResult


def dummy_physics_result(d_phys: float = 0.0, params: dict = None) -> PhysicsFitResult:
    return PhysicsFitResult(
        preferred_model="bazin",
        best_bic=15.0,
        model_fit_quality=1.1,
        physical_parameters=params or {"tau_rise": 15.0, "tau_fall": 35.0},
        parameter_space_deviation=d_phys,
        physical_residual=1.1,
        bazin_result=BazinFitResult(status=FitStatus.FIT_SUCCESS, reduced_chi2=1.1),
        arnett_result=ArnettFitResult(status=FitStatus.FIT_SUCCESS, reduced_chi2=1.2),
        shock_cooling_result=ShockCoolingFitResult(status=FitStatus.MODEL_NOT_APPLICABLE),
    )


def test_gate_rejects_artifact():
    # Candidate with a fatal subtraction dipole objection
    fatal_obj = SkepticChallenge(
        challenge_id="chk:psf_01",
        category=ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT,
        assertion="Subtraction dipole detected.",
        status=ChallengeStatus.FATAL_DISQUALIFICATION,
    )

    res = DeterministicEvidenceGateEngine.evaluate_candidate(
        entity_id="candidate_artifact_01",
        n_epochs=10,
        duration_days=20.0,
        quality_flags=[1, 0, 0, 0],
        physics_fit=None,
        literature_citations=[],
        skeptic_challenges=[fatal_obj],
    )

    assert res.terminal_status == TerminalInvestigationStatus.REJECTED_AS_ARTIFACT
    assert res.artifact_filter_passed is False
    assert len(res.unresolved_fatal_objections) == 1


def test_gate_insufficient_evidence():
    # Only 2 epochs
    res = DeterministicEvidenceGateEngine.evaluate_candidate(
        entity_id="sparse_cand_01",
        n_epochs=2,
        duration_days=1.0,
        quality_flags=[0, 0],
        physics_fit=None,
        literature_citations=[],
        skeptic_challenges=[],
    )

    assert res.terminal_status == TerminalInvestigationStatus.INSUFFICIENT_EVIDENCE
    assert res.data_integrity_passed is False


def test_gate_known_normal_object():
    # Literature identifies as known RR Lyrae
    citation = LiteratureCitation(
        citation_id="cit:gaia_rrl",
        bibcode_or_doi="2023A&A...674A..18C",
        title="Gaia DR3 Variable Star Summary",
        authors="Clementini et al.",
        publication_year=2023,
        source_url="https://doi.org/10.1051/0004-6361/202243940",
        retrieval_timestamp_iso="2026-11-09T00:00:00Z",
        evidence_type=CitationEvidenceType.CATALOG_ASSOCIATION,
        relevant_excerpt="Confirmed RR Lyrae periodic pulsating variable with period P = 0.55 d.",
        is_direct_evidence=True,
    )

    res = DeterministicEvidenceGateEngine.evaluate_candidate(
        entity_id="rrl_01",
        n_epochs=15,
        duration_days=30.0,
        quality_flags=[0] * 15,
        physics_fit=dummy_physics_result(d_phys=0.4),
        literature_citations=[citation],
        skeptic_challenges=[],
    )

    assert res.terminal_status == TerminalInvestigationStatus.KNOWN_OBJECT
    assert res.known_object_status == "CONFIRMED_KNOWN"


def test_gate_candidate_requires_followup():
    # Peculiar transient with high parameter deviation (D_phys = 4.2 sigma) and zero fatal objections
    phys = dummy_physics_result(d_phys=4.2, params={"tau_rise": 1.1, "tau_fall": 5.0})

    res = DeterministicEvidenceGateEngine.evaluate_candidate(
        entity_id="at2018cow_analog",
        n_epochs=12,
        duration_days=15.0,
        quality_flags=[0] * 12,
        physics_fit=phys,
        literature_citations=[],
        skeptic_challenges=[],
    )

    assert res.terminal_status == TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP
    assert res.data_integrity_passed is True
    assert res.artifact_filter_passed is True
    assert "Recommend urgent multi-band photometric and spectroscopic follow-up" in res.recommended_followup_action
