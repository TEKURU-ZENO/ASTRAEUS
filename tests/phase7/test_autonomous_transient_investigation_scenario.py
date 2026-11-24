"""
ASTRAEUS Phase 7 Master Acceptance Scenario: Autonomous Transient Investigation.
Tests the complete 8-stage closed-loop autonomous scientific intelligence pipeline:
1. Event Arrival: Rubin optical alert packet ingestion & normalization (WS7.1)
2. Validation & Firewall: Gate 3 Zero-Touch Quarantine Firewall verification (WS7.1)
3. Intelligent Triage: Multi-catalogue cross-match & anomaly classification (WS7.2)
4. Evidence Retrieval: Environmental context, host distance & dust extinction (WS7.3)
5. Hypothesis Formulation: Generation of 3 competing physical hypotheses (WS7.3, WS7.4)
6. Skeptic Challenge & EIG Planning: Adversarial challenge & optimal filter selection (WS7.3, WS7.5)
7. Closed-Loop Follow-Up: Targeted observation ingestion & Bayesian posterior revision (WS7.5, WS7.6)
8. Reproducible Reporting: IVOA Prov-DM v1.0 sealed digest & verifiable report (WS7.6)

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import os
import json
import hashlib
from datetime import datetime, timezone

from astraeus_data.schemas.rubin_alert import RubinAlertPacket, RubinDiaSource, RubinDiaObject
from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    UnifiedScientificEvent,
    IngestionStatus,
)
from astraeus_data.triage.event_triage_engine import (
    CatalogCrossMatcher,
    EventTriageEngine,
    PriorityTier,
    AnomalyClass,
    TriageDecision,
)
from astraeus_agents.reground.protocol import RegroundDebateProtocol
from astraeus_agents.reground.models import (
    RegroundInvestigationReport,
    ChallengeStatus,
    HypothesisEvaluationStatus,
    AgentPersona,
    DebateStage,
)
from engines.capability_registry import ScientificCapabilityRegistry, EngineCapabilityTier
from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


def test_autonomous_transient_investigation_closed_loop_scenario(tmp_path):
    """
    Master Acceptance Scenario for Phase 7 Closed-Loop Autonomous Scientific Intelligence.
    """
    # =========================================================================
    # Stage 1: Event Arrival & Alert Ingestion (WS7.1)
    # =========================================================================
    firewall = QuarantineFirewall()
    hp = firewall.get_healpix()
    # Choose a coordinate guaranteed to be outside the 1,808 quarantined test pixels
    train_pixel = 0
    skycoord = hp.healpix_to_skycoord(train_pixel)
    ra_cand = float(skycoord.ra.deg)
    dec_cand = float(skycoord.dec.deg)

    # Construct Rubin LSST alert packet for rapid optical transient
    rubin_alert = RubinAlertPacket(
        alertId=778899001122,
        diaSource=RubinDiaSource(
            diaSourceId=1001,
            diaObjectId=2002,
            midPointTai=60002.0,
            filterName="r",
            ra=ra_cand,
            dec=dec_cand,
            raErr=0.0001,
            decErr=0.0001,
            psfFlux=7585.8,           # ~19.2 AB mag (rapid decay from 17.6)
            psfFluxErr=500.0,
            snr=15.1,
            isDiffPos=True,
        ),
        prvDiaSources=[
            RubinDiaSource(
                diaSourceId=1000,
                diaObjectId=2002,
                midPointTai=60000.0,
                filterName="r",
                ra=ra_cand,
                dec=dec_cand,
                raErr=0.0001,
                decErr=0.0001,
                psfFlux=5754.4,       # ~19.5 AB mag
                psfFluxErr=800.0,
                snr=7.2,
                isDiffPos=True,
            ),
            RubinDiaSource(
                diaSourceId=999,
                diaObjectId=2002,
                midPointTai=60001.0,
                filterName="r",
                ra=ra_cand,
                dec=dec_cand,
                raErr=0.0001,
                decErr=0.0001,
                psfFlux=33113.1,      # ~17.6 AB mag (peak r)
                psfFluxErr=900.0,
                snr=36.8,
                isDiffPos=True,
            ),
            RubinDiaSource(
                diaSourceId=998,
                diaObjectId=2002,
                midPointTai=60001.0,
                filterName="g",
                ra=ra_cand,
                dec=dec_cand,
                raErr=0.0001,
                decErr=0.0001,
                psfFlux=36307.8,      # ~17.5 AB mag (peak g)
                psfFluxErr=900.0,
                snr=40.3,
                isDiffPos=True,
            ),
            RubinDiaSource(
                diaSourceId=997,
                diaObjectId=2002,
                midPointTai=60002.0,
                filterName="g",
                ra=ra_cand,
                dec=dec_cand,
                raErr=0.0001,
                decErr=0.0001,
                psfFlux=2290.9,       # ~20.5 AB mag (rapid reddening g-r = 1.3)
                psfFluxErr=300.0,
                snr=7.6,
                isDiffPos=True,
            ),
        ],
        diaObject=RubinDiaObject(
            diaObjectId=2002,
            ra=ra_cand,
            dec=dec_cand,
            nDiaSources=5
        )
    )

    ingestion_engine = ScientificEventIngestionEngine(
        storage_root=tmp_path / "events"
    )
    ingestion_res = ingestion_engine.ingest_rubin_packet(rubin_alert)

    assert ingestion_res.status == IngestionStatus.INGESTED
    assert ingestion_res.unified_event is not None
    event = ingestion_res.unified_event
    assert event.event_id == f"evt:rubin:{rubin_alert.diaSource.diaSourceId}"
    assert event.survey_origin == "RUBIN"
    assert len(event.photometry_points) == 5

    # =========================================================================
    # Stage 2: Validation & Quarantine Firewall Verification (WS7.1)
    # =========================================================================
    assert event.quarantine_cleared is True
    # Verify firewall directly accepts safe coordinates
    firewall.assert_not_quarantined(event.ra_deg, event.dec_deg)

    # Verify firewall actively blocks quarantined pixel coordinates
    q_pixel = list(firewall.get_quarantined_pixels())[0]
    q_coord = hp.healpix_to_skycoord(q_pixel)
    with pytest.raises(QuarantineViolationError):
        firewall.assert_not_quarantined(float(q_coord.ra.deg), float(q_coord.dec.deg))

    # =========================================================================
    # Stage 3: Intelligent Triage & Anomaly Classification (WS7.2)
    # =========================================================================
    cross_matcher = CatalogCrossMatcher()
    # Associate NGC 4993 host galaxy at 15 arcsec offset (~3.0 kpc)
    cross_matcher.add_host_galaxy(
        name="NGC 4993",
        ra=ra_cand,
        dec=dec_cand + 15.0 / 3600.0,
        redshift=0.00973,
        catalog="GLADE+"
    )

    triage_engine = EventTriageEngine(cross_matcher=cross_matcher)
    decision = triage_engine.triage_event(event)

    assert decision.priority_tier in (PriorityTier.CRITICAL_TOO, PriorityTier.HIGH_PRIORITY)
    assert decision.host_context.matched is True
    assert decision.host_context.galaxy_name == "NGC 4993"
    assert decision.host_context.physical_offset_kpc is not None
    assert decision.host_context.physical_offset_kpc < 5.0  # within 5 kpc
    assert len(triage_engine.queue) == 1

    # =========================================================================
    # Stage 4 & 5: REGROUND Multi-Agent Investigation & Engine Bounds (WS7.3, WS7.4)
    # =========================================================================
    capability_registry = ScientificCapabilityRegistry()
    protocol = RegroundDebateProtocol(
        capability_registry=capability_registry,
        quarantine_firewall=firewall,
        random_seed=4242
    )

    report = protocol.run_investigation(
        event=event,
        triage=decision,
        claiming_validated_discovery=False
    )

    assert isinstance(report, RegroundInvestigationReport)
    assert len(report.hypotheses) == 3
    assert report.leading_hypothesis is not None

    # Check the 3 competing hypotheses exist
    hypo_ids = {h.hypothesis_id for h in report.hypotheses}
    assert "H1_KILONOVA" in hypo_ids
    assert "H2_FAST_CCSN" in hypo_ids
    assert "H3_TDE_AGN" in hypo_ids

    # Verify debate turns were recorded from all specialized personas
    personas_participated = {t.persona for t in report.debate_turns}
    assert AgentPersona.OBSERVATIONAL_ASTRONOMER in personas_participated
    assert AgentPersona.COSMOLOGIST in personas_participated
    assert AgentPersona.ASTROPHYSICIST in personas_participated
    assert AgentPersona.STATISTICAL_AUDITOR in personas_participated
    assert AgentPersona.SKEPTIC_RED_TEAM in personas_participated

    # =========================================================================
    # Stage 6: Skeptic Challenge & EIG Follow-up Recommendation (WS7.3, WS7.5)
    # =========================================================================
    assert len(report.skeptic_challenges) > 0
    veto = report.skeptic_challenges[0]
    assert veto.target_hypothesis_id == report.leading_hypothesis.hypothesis_id
    assert veto.status == ChallengeStatus.VALIDATED_WITH_QUALIFICATIONS

    # EIG recommendation should prioritize infrared/red band (z or i or y)
    assert report.recommended_followup_filter in ("z", "i", "y")
    assert report.expected_information_gain > 0.0

    # =========================================================================
    # Stage 7: Closed-Loop Follow-Up Ingestion & Bayesian Update (WS7.5, WS7.6)
    # =========================================================================
    # Simulated follow-up observation in recommended z-band at MJD 60003.5
    followup_observation = [
        {
            "mjd": 60003.5,
            "mag": 19.1,
            "magerr": 0.04,
            "filter_band": "z",
            "flux_nJy": 8317.6,
            "flux_err_nJy": 300.0
        }
    ]

    closed_loop_report = protocol.update_with_followup(
        event=event,
        triage=decision,
        previous_report=report,
        followup_observations=followup_observation,
        claiming_validated_discovery=False
    )

    # Check updated Bayesian posteriors under 4-tier taxonomy
    assert closed_loop_report.leading_hypothesis is not None
    assert closed_loop_report.leading_hypothesis.hypothesis_id == "H1_KILONOVA"
    assert closed_loop_report.leading_hypothesis.posterior_prob > 0.90
    assert closed_loop_report.leading_hypothesis.evaluation_status == HypothesisEvaluationStatus.STRONGER_EVIDENCE

    # Check 4-tier evaluation on competing models
    h2 = next(h for h in closed_loop_report.hypotheses if h.hypothesis_id == "H2_FAST_CCSN")
    assert h2.evaluation_status == HypothesisEvaluationStatus.MODEL_REJECTED_UNDER_ASSUMPTIONS

    h3 = next(h for h in closed_loop_report.hypotheses if h.hypothesis_id == "H3_TDE_AGN")
    assert h3.evaluation_status == HypothesisEvaluationStatus.PHYSICALLY_FALSIFIED
    assert h3.is_falsified is True

    # Skeptic challenge resolved
    resolved_veto = closed_loop_report.skeptic_challenges[0]
    assert "Passed" in resolved_veto.adversarial_critique or "confirmed" in resolved_veto.adversarial_critique

    # =========================================================================
    # Stage 8: Sealed Reproducible Prov-DM Report Verification (WS7.6)
    # =========================================================================
    digest = closed_loop_report.provenance_sealed_digest
    assert len(digest) == 64
    assert int(digest, 16) > 0  # Valid hex digest

    # Verify report can be serialized to JSON and reloaded without loss
    report_dict = closed_loop_report.model_dump()
    serialized = json.dumps(report_dict, indent=2)
    deserialized = json.loads(serialized)
    assert deserialized["investigation_id"] == closed_loop_report.investigation_id
    assert deserialized["provenance_sealed_digest"] == digest
    assert deserialized["is_quarantine_compliant"] is True
