"""
ASTRAEUS Phase 7 WS7.3 REGROUND Multi-Agent Scientific Investigation Test Suite.
Tests:
1. 5 Specialized Agent Personas initialization
2. Observational Astronomer extinction & SNR audit
3. Cosmologist host environment & distance modulus calculation
4. Astrophysicist multi-hypothesis formulation (at least 3 hypotheses)
5. Statistical Auditor BIC calculation & normalized posterior model selection
6. Skeptic Red-Team adversarial challenge & artifact auditing
7. Epistemic promotion violation prevention on experimental engines
8. End-to-end REGROUND debate protocol with EIG follow-up and Prov-DM SHA-256 seal
9. Gate 3 Quarantine Firewall blocking of investigations in quarantined footprint
"""

import pytest
from datetime import datetime, timezone

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
from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from astraeus_data.triage.event_triage_engine import (
    TriageDecision,
    PriorityTier,
    AnomalyClass,
    HostGalaxyContext,
    ReferenceStarContext,
    PhotometricFeatures,
    EventTriageEngine,
    CatalogCrossMatcher,
)
from engines.capability_registry import (
    ScientificCapabilityRegistry,
    EngineCapabilityTier,
    EpistemicPromotionViolation,
)
from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


@pytest.fixture
def sample_kilonova_event():
    hp = QuarantineFirewall.get_healpix()
    coord = hp.healpix_to_skycoord(0)
    ra, dec = float(coord.ra.deg), float(coord.dec.deg)
    return UnifiedScientificEvent(
        event_id="KN-CAND-2026A",
        survey_origin="RUBIN",
        ra_deg=ra,
        dec_deg=dec,
        healpix_pixel_nside32=0,
        event_time_mjd=60002.5,
        photometry_points=[
            {"mjd": 60000.0, "mag": 19.8, "magerr": 0.05, "filter_band": "g"},
            {"mjd": 60000.0, "mag": 19.6, "magerr": 0.04, "filter_band": "r"},
            {"mjd": 60001.0, "mag": 17.9, "magerr": 0.03, "filter_band": "g"},
            {"mjd": 60001.0, "mag": 18.0, "magerr": 0.03, "filter_band": "r"},
            {"mjd": 60002.0, "mag": 19.5, "magerr": 0.06, "filter_band": "g"},
            {"mjd": 60002.0, "mag": 18.7, "magerr": 0.05, "filter_band": "r"},
        ],
        event_metadata={"filter": "r"},
        raw_payload_hash="knpayloadhash999",
        quarantine_cleared=True,
    )


@pytest.fixture
def sample_triage_decision(sample_kilonova_event):
    matcher = CatalogCrossMatcher()
    matcher.add_host_galaxy(
        name="NGC 4993-Like",
        ra=sample_kilonova_event.ra_deg,
        dec=sample_kilonova_event.dec_deg + 10.0 / 3600.0,
        redshift=0.0098,
        catalog="GLADE+"
    )
    engine = EventTriageEngine(cross_matcher=matcher)
    return engine.triage_event(sample_kilonova_event)


def test_reground_agents_initialization():
    obs = ObservationalAstronomerAgent()
    cosmo = CosmologistAgent()
    astro = AstrophysicistAgent()
    stats = StatisticalAuditorAgent()
    skeptic = SkepticRedTeamAgent()
    assert obs is not None
    assert cosmo is not None
    assert astro is not None
    assert stats is not None
    assert skeptic is not None


def test_observational_astronomer_audit(sample_kilonova_event, sample_triage_decision):
    obs = ObservationalAstronomerAgent()
    turn, metrics = obs.evaluate(sample_kilonova_event, sample_triage_decision)
    assert turn.persona == AgentPersona.OBSERVATIONAL_ASTRONOMER
    assert turn.stage == DebateStage.EVIDENCE_GROUNDING
    assert "extinction_a_v" in metrics
    assert "mean_snr" in metrics
    assert metrics["mean_snr"] > 10.0
    assert 0.0 < metrics["extinction_a_v"] < 2.0


def test_cosmologist_evaluation(sample_kilonova_event, sample_triage_decision):
    cosmo = CosmologistAgent()
    turn, metrics = cosmo.evaluate(sample_kilonova_event, sample_triage_decision)
    assert turn.persona == AgentPersona.COSMOLOGIST
    assert turn.stage == DebateStage.EVIDENCE_GROUNDING
    assert metrics["host_matched"] is True
    assert metrics["redshift_z"] == pytest.approx(0.0098)
    assert 32.0 < metrics["dist_mod"] < 35.0


def test_astrophysicist_hypothesis_formulation(sample_kilonova_event, sample_triage_decision):
    obs = ObservationalAstronomerAgent()
    _, obs_metrics = obs.evaluate(sample_kilonova_event, sample_triage_decision)
    cosmo = CosmologistAgent()
    _, cosmo_metrics = cosmo.evaluate(sample_kilonova_event, sample_triage_decision)

    astro = AstrophysicistAgent()
    turn, hypotheses = astro.formulate_hypotheses(
        event=sample_kilonova_event,
        triage=sample_triage_decision,
        obs_metrics=obs_metrics,
        cosmo_metrics=cosmo_metrics
    )

    assert turn.persona == AgentPersona.ASTROPHYSICIST
    assert turn.stage == DebateStage.HYPOTHESIS_FORMULATION
    assert len(hypotheses) >= 3

    h_ids = [h.hypothesis_id for h in hypotheses]
    assert "H1_KILONOVA" in h_ids
    assert "H2_FAST_CCSN" in h_ids
    assert "H3_TDE_AGN" in h_ids

    for h in hypotheses:
        assert len(h.falsification_criteria) > 0
        assert len(h.physical_parameters) > 0


def test_statistical_auditor_bic_and_posterior(sample_kilonova_event, sample_triage_decision):
    obs = ObservationalAstronomerAgent()
    _, obs_metrics = obs.evaluate(sample_kilonova_event, sample_triage_decision)
    cosmo = CosmologistAgent()
    _, cosmo_metrics = cosmo.evaluate(sample_kilonova_event, sample_triage_decision)
    astro = AstrophysicistAgent()
    _, hypotheses = astro.formulate_hypotheses(
        event=sample_kilonova_event,
        triage=sample_triage_decision,
        obs_metrics=obs_metrics,
        cosmo_metrics=cosmo_metrics
    )

    stats = StatisticalAuditorAgent()
    turn, updated = stats.audit_and_update_posteriors(hypotheses, num_data_points=6)

    assert turn.persona == AgentPersona.STATISTICAL_AUDITOR
    assert turn.stage == DebateStage.BAYESIAN_POSTERIOR_UPDATE

    total_prob = sum(h.posterior_prob for h in updated)
    assert pytest.approx(total_prob, abs=1e-5) == 1.0

    leading = [h for h in updated if h.is_leading]
    assert len(leading) == 1
    # Minimum BIC must correspond to leading hypothesis
    min_bic_h = min(updated, key=lambda h: h.bic)
    assert leading[0].hypothesis_id == min_bic_h.hypothesis_id


def test_skeptic_challenge_and_qualification(sample_kilonova_event, sample_triage_decision):
    obs = ObservationalAstronomerAgent()
    _, obs_metrics = obs.evaluate(sample_kilonova_event, sample_triage_decision)
    cosmo = CosmologistAgent()
    _, cosmo_metrics = cosmo.evaluate(sample_kilonova_event, sample_triage_decision)
    astro = AstrophysicistAgent()
    _, hypotheses = astro.formulate_hypotheses(sample_kilonova_event, sample_triage_decision, obs_metrics, cosmo_metrics)
    stats = StatisticalAuditorAgent()
    _, updated = stats.audit_and_update_posteriors(hypotheses, num_data_points=6)

    skeptic = SkepticRedTeamAgent()
    turn, veto = skeptic.challenge(sample_kilonova_event, sample_triage_decision, updated)

    assert turn.persona == AgentPersona.SKEPTIC_RED_TEAM
    assert turn.stage == DebateStage.SKEPTIC_CHALLENGE
    assert veto.target_hypothesis_id == "H1_KILONOVA"
    assert veto.status in (ChallengeStatus.VALIDATED_WITH_QUALIFICATIONS, ChallengeStatus.VETO_DISQUALIFIED)
    assert len(veto.required_discriminating_observation) > 0


def test_epistemic_guardrail_prevents_unearned_discovery():
    registry = ScientificCapabilityRegistry()
    hydro = registry.get("PINO3D_HYDRO_SURROGATE")
    assert hydro.tier == EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE

    # Attempting to cite an experimental prototype for validated scientific discovery must raise error
    with pytest.raises(EpistemicPromotionViolation, match="EPISTEMIC VIOLATION"):
        hydro.assert_usage_authorized(
            intended_use="Asserting definitive discovery of cosmological structure",
            claiming_validated_discovery=True
        )


def test_reground_protocol_full_lifecycle(sample_kilonova_event, sample_triage_decision):
    protocol = RegroundDebateProtocol(random_seed=12345)
    report = protocol.run_investigation(
        event=sample_kilonova_event,
        triage=sample_triage_decision,
        claiming_validated_discovery=False
    )

    assert isinstance(report, RegroundInvestigationReport)
    assert report.event_id == sample_kilonova_event.event_id
    assert len(report.hypotheses) >= 3
    assert report.leading_hypothesis is not None
    assert report.leading_hypothesis.hypothesis_id == "H1_KILONOVA"
    assert report.recommended_followup_filter in ("z", "i", "y")
    assert report.expected_information_gain > 0.0
    assert len(report.provenance_sealed_digest) == 64  # SHA-256 length
    assert report.is_quarantine_compliant is True


def test_quarantine_firewall_blocks_investigation_in_restricted_pixels(sample_triage_decision):
    firewall = QuarantineFirewall()
    quarantined_pixels = firewall.get_quarantined_pixels()
    quarantined_pixel = list(quarantined_pixels)[0]
    hp = firewall.get_healpix()
    q_coord = hp.healpix_to_skycoord(quarantined_pixel)

    quarantined_event = UnifiedScientificEvent(
        event_id="FORBIDDEN-EVENT-001",
        survey_origin="RUBIN",
        ra_deg=float(q_coord.ra.deg),
        dec_deg=float(q_coord.dec.deg),
        healpix_pixel_nside32=quarantined_pixel,
        event_time_mjd=60000.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="payload999",
        quarantine_cleared=False
    )

    protocol = RegroundDebateProtocol(quarantine_firewall=firewall)

    with pytest.raises(QuarantineViolationError):
        protocol.run_investigation(
            event=quarantined_event,
            triage=sample_triage_decision
        )
