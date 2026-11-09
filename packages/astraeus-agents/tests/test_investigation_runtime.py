"""
Unit tests for ASTRAEUS Multi-Agent Investigation Runtime & Orchestrator.
Tests end-to-end execution of the 4-agent graph and dossier generation.
"""

import numpy as np
import pytest

from astraeus_agents.orchestrator import InvestigationOrchestrator
from astraeus_agents.contracts import TerminalInvestigationStatus
from astraeus_agents.dossier import InvestigationDossier
from astraeus_models.physics.bazin import bazin_function


def test_orchestrator_investigates_fbot_candidate():
    orchestrator = InvestigationOrchestrator()

    # Generate FBOT-like light curve (rapid rise, rapid decay)
    t = np.linspace(60000.0, 60015.0, 12)
    flux_profile = bazin_function(t, t0=60002.0, a=80.0, b=1.0, tau_rise=1.1, tau_fall=4.5)
    mags = -2.5 * np.log10(np.clip(flux_profile, 1e-4, None)) + 25.0
    errs = np.full_like(mags, 0.04)
    bands = ["ZTF_g" if i % 2 == 0 else "ZTF_r" for i in range(len(t))]

    dossier = orchestrator.investigate_candidate(
        entity_id="candidate_fbot_at2018cow_analog",
        ra_deg=244.000917,
        dec_deg=22.268031,
        times_mjd=t,
        magnitudes=mags,
        errors=errs,
        bands=bands,
        quality_flags=[0] * len(t),
    )

    assert isinstance(dossier, InvestigationDossier)
    assert dossier.target_entity_id == "candidate_fbot_at2018cow_analog"
    assert dossier.terminal_status == TerminalInvestigationStatus.CANDIDATE_REQUIRES_FOLLOWUP
    assert "Recommend urgent multi-band photometric and spectroscopic follow-up" in dossier.recommended_action

    # Check Markdown report formatting
    md = dossier.to_markdown()
    assert "# ASTRAEUS Scientific Investigation Dossier" in md
    assert "**Terminal Scientific Status:** `CANDIDATE_REQUIRES_FOLLOWUP`" in md
    assert "Bayesian Counterpart Status:" in md
    assert "Decoupled Physical Model Inference" in md
    assert "IVOA Provenance & Epistemic Boundaries" in md


def test_orchestrator_rejects_subtraction_artifact():
    orchestrator = InvestigationOrchestrator()

    t = np.array([60000.0, 60001.0, 60002.0, 60003.0, 60004.0])
    mags = np.array([21.0, 21.0, 14.5, 21.0, 21.0]) # Single-epoch 6.5 mag jump
    errs = np.array([0.1, 0.1, 0.05, 0.1, 0.1])
    flags = [0, 0, 1, 0, 0] # Flag 1 indicates subtraction dipole defect

    dossier = orchestrator.investigate_candidate(
        entity_id="artifact_dipole_099",
        ra_deg=120.0,
        dec_deg=45.0,
        times_mjd=t,
        magnitudes=mags,
        errors=errs,
        bands=["ZTF_g"] * 5,
        quality_flags=flags,
    )

    assert dossier.terminal_status == TerminalInvestigationStatus.REJECTED_AS_ARTIFACT
    assert "Quarantine alert packet" in dossier.recommended_action
