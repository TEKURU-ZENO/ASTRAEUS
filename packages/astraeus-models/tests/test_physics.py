"""
Unit tests for ASTRAEUS Physics Forward Fitting, Applicability Matrix, and BIC Engine.
"""

import numpy as np
import pytest

from astraeus_models.physics.fit_status import FitStatus
from astraeus_models.physics.applicability import ModelApplicabilityMatrix
from astraeus_models.physics.bazin import bazin_function, fit_bazin, BazinFitResult
from astraeus_models.physics.arnett import (
    arnett_bolometric_luminosity,
    arnett_photometric_flux,
    fit_arnett,
    ArnettFitResult,
)
from astraeus_models.physics.shock_cooling import (
    shock_cooling_luminosity,
    shock_cooling_photometric_flux,
    fit_shock_cooling,
    ShockCoolingFitResult,
)
from astraeus_models.physics.engine import PhysicalModelEngine, PhysicsFitResult


def test_fit_status_enum():
    assert FitStatus.FIT_SUCCESS == "FIT_SUCCESS"
    assert FitStatus.MODEL_NOT_APPLICABLE == "MODEL_NOT_APPLICABLE"
    assert FitStatus.INSUFFICIENT_OBSERVATIONS == "INSUFFICIENT_OBSERVATIONS"
    assert FitStatus.FIT_POOR_QUALITY == "FIT_POOR_QUALITY"


def test_applicability_matrix_gating():
    matrix = ModelApplicabilityMatrix()
    times = np.linspace(60000.0, 60030.0, 15)
    mags = np.linspace(19.0, 17.0, 15)

    # 1. Artifact candidate should reject all physical models
    res_art = matrix.evaluate_applicability("detector_artifact", times, mags)
    assert res_art["bazin"][0] is False
    assert res_art["arnett"][0] is False
    assert res_art["shock_cooling"][0] is False

    # 2. Explosive candidate with 30-day baseline
    res_sn = matrix.evaluate_applicability("supernova_ia", times, mags)
    assert res_sn["bazin"][0] is True
    assert res_sn["arnett"][0] is True

    # 3. Insufficient epochs (< 5)
    short_times = np.array([60000.0, 60001.0, 60002.0])
    short_mags = np.array([19.0, 18.5, 18.0])
    res_short = matrix.evaluate_applicability("supernova_ia", short_times, short_mags)
    assert res_short["bazin"][0] is False
    assert res_short["arnett"][0] is False


def test_bazin_fitting_converges():
    # Generate synthetic Bazin profile
    t = np.linspace(0.0, 40.0, 25)
    f_true = bazin_function(t, t0=15.0, a=50.0, b=2.0, tau_rise=3.0, tau_fall=12.0)
    noise = np.random.RandomState(42).normal(0.0, 0.5, size=len(t))
    f_obs = f_true + noise
    f_err = np.full_like(f_obs, 0.5)

    res = fit_bazin(t + 60000.0, f_obs, f_err)
    assert res.status == FitStatus.FIT_SUCCESS
    assert res.best_fit_params["tau_rise"] > 0.0
    assert res.best_fit_params["tau_fall"] > 0.0
    assert res.reduced_chi2 < 5.0
    assert res.bic > 0.0


def test_arnett_fitting_converges():
    # Generate synthetic SN Ia Arnett profile
    t = np.linspace(60000.0, 60050.0, 30)
    t0_true = 60000.0
    f_true = arnett_photometric_flux(t, t0=t0_true, m_ni=0.6, tau_d=16.0, flux_scale=50.0)
    noise = np.random.RandomState(42).normal(0.0, 0.2, size=len(t))
    f_obs = np.clip(f_true + noise, 0.0, None)
    f_err = np.full_like(f_obs, 0.3)

    res = fit_arnett(t, f_obs, f_err)
    assert res.status == FitStatus.FIT_SUCCESS
    assert res.nickel_mass_msun is not None
    assert 0.1 <= res.nickel_mass_msun <= 1.5
    assert res.diffusion_time_days is not None
    assert res.reduced_chi2 < 5.0


def test_shock_cooling_fitting_converges():
    # Early-time light curve (dt <= 5 days)
    t = np.linspace(60000.1, 60003.5, 8)
    t0_true = 60000.0
    f_true = shock_cooling_photometric_flux(t, t0=t0_true, r_rsun=100.0, v_shock=1.2, flux_scale=10.0)
    noise = np.random.RandomState(42).normal(0.0, 0.05, size=len(t))
    f_obs = np.clip(f_true + noise, 0.0, None)
    f_err = np.full_like(f_obs, 0.05)

    res = fit_shock_cooling(t, f_obs, f_err)
    assert res.status == FitStatus.FIT_SUCCESS
    assert res.progenitor_radius_rsun is not None
    assert res.progenitor_radius_rsun > 0.0
    assert res.reduced_chi2 < 5.0


def test_physical_model_engine():
    engine = PhysicalModelEngine()
    # SN-like curve
    t = np.linspace(60000.0, 60040.0, 20)
    f = bazin_function(t, t0=60015.0, a=40.0, b=1.0, tau_rise=4.0, tau_fall=15.0)
    err = np.full_like(f, 0.5)

    res = engine.fit_all_applicable(t, f, err, candidate_hypothesis="supernova_ia")
    assert isinstance(res, PhysicsFitResult)
    assert res.preferred_model in {"bazin", "arnett"}
    assert res.best_bic is not None
    assert res.model_fit_quality is not None
    assert isinstance(res.physical_parameters, dict)
    assert res.parameter_space_deviation >= 0.0
    assert res.physical_residual > 0.0
    assert "bazin" in res.model_comparison
    assert res.summary_status == "CONVERGED"


def test_physical_model_engine_artifact_rejection():
    engine = PhysicalModelEngine()
    t = np.array([60000.0, 60001.0, 60002.0, 60003.0, 60004.0])
    f = np.array([1.0, 1.0, 50.0, 1.0, 1.0]) # Single-epoch spike
    err = np.full_like(f, 0.5)

    res = engine.fit_all_applicable(t, f, err, candidate_hypothesis="detector_artifact")
    assert res.preferred_model is None
    assert res.bazin_result.status == FitStatus.MODEL_NOT_APPLICABLE
    assert res.arnett_result.status == FitStatus.MODEL_NOT_APPLICABLE
    assert res.summary_status == "NO_MODEL_CONVERGED"
