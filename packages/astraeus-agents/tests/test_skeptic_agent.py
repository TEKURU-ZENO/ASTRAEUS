"""
Unit tests for ASTRAEUS Skeptic & Adversarial Validation Agent.
Verifies structured challenges against dipole artifacts, crowded fields, and unphysical parameters.
"""

import numpy as np
import pytest

from astraeus_agents.skeptic_agent import SkepticValidationAgent
from astraeus_agents.contracts import ChallengeCategory, ChallengeStatus
from astraeus_models.physics.engine import PhysicsFitResult
from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.bazin import BazinFitResult
from astraeus_models.physics.arnett import ArnettFitResult
from astraeus_models.physics.shock_cooling import ShockCoolingFitResult


def test_skeptic_detects_subtraction_dipole():
    skeptic = SkepticValidationAgent()
    times = np.array([60000.0, 60001.0, 60002.0])
    mags = np.array([19.0, 15.0, 19.0])
    errs = np.array([0.05, 0.05, 0.05])
    flags = [0, 1, 0] # Flag 1 indicates subtraction dipole

    challenges = skeptic.challenge_candidate(
        entity_id="cand_dipole",
        times_mjd=times,
        magnitudes=mags,
        errors=errs,
        quality_flags=flags,
        local_source_density_sq_arcsec=0.01,
        physics_fit=None,
        literature_citations=[],
    )

    categories = [c.category for c in challenges]
    assert ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT in categories
    fatal = [c for c in challenges if c.status == ChallengeStatus.FATAL_DISQUALIFICATION]
    assert len(fatal) >= 1


def test_skeptic_disqualifies_unphysical_nickel_mass():
    skeptic = SkepticValidationAgent()
    times = np.linspace(60000.0, 60040.0, 15)
    mags = np.linspace(18.0, 15.0, 15)
    errs = np.full_like(mags, 0.05)

    # Inferred Nickel mass = 3.5 M_sun (exceeds 2.5 M_sun Chandrasekhar limit)
    phys_fit = PhysicsFitResult(
        preferred_model="arnett",
        best_bic=20.0,
        model_fit_quality=1.2,
        physical_parameters={"m_ni": 3.5, "tau_d": 15.0},
        parameter_space_deviation=5.0,
        physical_residual=1.2,
        bazin_result=BazinFitResult(status=FitStatus.FIT_SUCCESS),
        arnett_result=ArnettFitResult(status=FitStatus.FIT_SUCCESS, nickel_mass_msun=3.5),
        shock_cooling_result=ShockCoolingFitResult(status=FitStatus.MODEL_NOT_APPLICABLE),
    )

    challenges = skeptic.challenge_candidate(
        entity_id="cand_unphysical_mni",
        times_mjd=times,
        magnitudes=mags,
        errors=errs,
        quality_flags=[0] * 15,
        local_source_density_sq_arcsec=0.005,
        physics_fit=phys_fit,
        literature_citations=[],
    )

    fatal_mni = [c for c in challenges if c.category == ChallengeCategory.UNPHYSICAL_PARAMETERS and c.status == ChallengeStatus.FATAL_DISQUALIFICATION]
    assert len(fatal_mni) == 1
    assert "Chandrasekhar mass limit" in fatal_mni[0].assertion
