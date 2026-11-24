"""
ASTRAEUS Phase 7 - Workstream 7.5 Observational Planning & Multi-Facility Scheduling Tests.
Verifies:
1. Versioned facility and instrument registry (optical, NIR, UV, X-ray, radio).
2. Astrometric visibility engine (LST, altitude, azimuth, Kasten & Young airmass, twilight).
3. Space telescope constraints (JWST Sun-Earth L2 field of regard, Swift Sun avoidance).
4. Weather model with Kolmogorov seeing and forecast uncertainty propagation.
5. Physics-based exposure calculator and SNR inversion.
6. Generative hypothesis EIG calculation and non-negative mutual information.
7. Multi-facility schedulers (Random, Greedy, Cost-Aware, Rolling Horizon, CSP, MCTS, Hybrid).
8. Human-in-the-loop approval gate preventing autonomous dispatch.
9. Gate 3 Zero-Touch Quarantine Firewall enforcement.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import math
from pathlib import Path

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_agents.observatory.facility_registry import (
    FacilityRegistry,
    FacilityAccessType,
    WavelengthRegime,
)
from astraeus_agents.observatory.visibility_engine import (
    VisibilityEngine,
    TargetVisibilityState,
)
from astraeus_agents.observatory.weather_model import WeatherModel
from astraeus_agents.observatory.exposure_calculator import ExposureCalculator
from astraeus_agents.observatory.predictive_eig import (
    ObservationAction,
    PredictiveEIGEngine,
)
from astraeus_agents.observatory.schedulers import (
    TargetState,
    RandomFeasibleScheduler,
    GreedyEIGPlanner,
    CostAwareGreedyPlanner,
    RollingHorizonOptimizer,
    ConstraintSatisfactionOptimizer,
    MCTSPlanner,
    HybridOptimizer,
)
from astraeus_agents.observatory.human_approval_interface import (
    ApprovalStatus,
    AutonomousDispatchProhibitedError,
    HumanApprovalGate,
)


def test_facility_registry_and_regimes():
    """Verifies that all 9 multi-facility observatories are configured with correct regimes and access classes."""
    reg = FacilityRegistry()
    facilities = reg.list_all_facilities()
    assert len(facilities) >= 9

    fac_ids = {f.facility_id for f in facilities}
    expected_ids = {"ZTF", "RUBIN", "SWIFT", "ASTROSAT", "VLT", "GEMINI", "JWST", "MEERKAT", "VLA"}
    assert expected_ids.issubset(fac_ids)

    # Verify access types
    rubin = reg.get_facility("RUBIN")
    assert rubin.access_type == FacilityAccessType.STREAMING_SURVEY_BROKER

    swift = reg.get_facility("SWIFT")
    assert swift.access_type == FacilityAccessType.RAPID_TOO_TARGETABLE
    assert swift.location.is_space_based is True

    jwst = reg.get_facility("JWST")
    assert jwst.access_type == FacilityAccessType.PROPOSAL_GATED
    assert jwst.proposal_id_required is True

    meerkat = reg.get_facility("MEERKAT")
    assert meerkat.access_type == FacilityAccessType.QUEUE_SCHEDULED

    # Verify regime filters
    radio_facs = reg.list_facilities_by_regime(WavelengthRegime.RADIO)
    assert any(f.facility_id == "MEERKAT" for f in radio_facs)
    assert any(f.facility_id == "VLA" for f in radio_facs)


def test_visibility_engine_coordinates_and_airmass():
    """Verifies Local Sidereal Time, Alt/Az, Kasten & Young airmass, and horizon checks."""
    reg = FacilityRegistry()
    paranal = reg.get_facility("VLT")

    epoch_mjd = 60000.0  # arbitrary test epoch
    lst_hours = VisibilityEngine.calculate_lst_hours(epoch_mjd, paranal.location.longitude_deg)
    assert 0.0 <= lst_hours <= 24.0

    # Zenith target (RA = LST * 15, Dec = Observatory Latitude)
    zenith_ra = (lst_hours * 15.0) % 360.0
    zenith_dec = paranal.location.latitude_deg
    alt, az = VisibilityEngine.calculate_alt_az(zenith_ra, zenith_dec, paranal.location.latitude_deg, lst_hours)
    assert alt >= 89.9  # must be essentially 90 deg

    airmass = VisibilityEngine.calculate_refraction_airmass(alt)
    assert abs(airmass - 1.0) < 0.01  # Airmass at zenith is 1.0

    # Horizon limit test (target far below horizon)
    nadir_dec = -paranal.location.latitude_deg
    nadir_ra = (zenith_ra + 180.0) % 360.0
    vis = VisibilityEngine.evaluate_target_visibility(paranal, nadir_ra, nadir_dec, epoch_mjd)
    assert vis.is_observable is False
    assert any("below horizon" in r for r in vis.rejection_reasons)


def test_space_observatory_field_of_regard():
    """Verifies JWST Sun-Earth L2 elongation [85, 135] deg and Swift Sun avoidance constraints."""
    reg = FacilityRegistry()
    jwst = reg.get_facility("JWST")
    swift = reg.get_facility("SWIFT")

    epoch_mjd = 60000.0
    sun_ra, sun_dec = VisibilityEngine.calculate_solar_ephemeris(epoch_mjd)

    # 1. Target pointing directly at Sun (0 deg elongation) -> JWST and Swift must reject!
    vis_sun = VisibilityEngine.evaluate_target_visibility(jwst, sun_ra, sun_dec, epoch_mjd)
    assert vis_sun.is_observable is False
    assert any("Field of Regard violation" in r for r in vis_sun.rejection_reasons)

    vis_swift_sun = VisibilityEngine.evaluate_target_visibility(swift, sun_ra, sun_dec, epoch_mjd)
    assert vis_swift_sun.is_observable is False
    assert any("Sun pointing constraint" in r for r in vis_swift_sun.rejection_reasons)

    # 2. Target 95 deg from Sun -> within JWST Field of Regard
    target_ra = (sun_ra + 95.0) % 360.0
    target_dec = sun_dec
    vis_jwst_valid = VisibilityEngine.evaluate_target_visibility(jwst, target_ra, target_dec, epoch_mjd)
    assert vis_jwst_valid.is_observable is True


def test_weather_model_sampling_and_dome_closure():
    """Verifies Kolmogorov seeing sampling, closure thresholds, and space observatory immunity."""
    wx = WeatherModel(random_seed=42)

    # Ground observatory
    cond = wx.sample_weather("VLT", 60000.0, lookahead_hours=2.0)
    assert 0.35 <= cond.seeing_fwhm_arcsec <= 3.5
    assert 0.0 <= cond.forecast_confidence <= 1.0
    assert 0.0 <= cond.quality_score <= 1.0

    # Space observatory (immune to terrestrial seeing and clouds)
    space_cond = wx.sample_weather("JWST", 60000.0, lookahead_hours=6.0)
    assert space_cond.is_dome_open is True
    assert space_cond.cloud_extinction_mag == 0.0
    assert space_cond.quality_score == 1.0


def test_exposure_calculator_snr_and_limiting_magnitude():
    """Verifies photometric SNR calculation, 5-sigma limiting magnitude, and required exposure time inversion."""
    reg = FacilityRegistry()
    vlt = reg.get_facility("VLT")
    xshooter = vlt.get_instrument("XSHOOTER")
    vis_filter = xshooter.get_filter("VIS_SPEC")

    # Evaluate bright source (m = 18.0) in 300s
    est = ExposureCalculator.compute_snr(
        instrument=xshooter,
        filter_obj=vis_filter,
        target_apparent_mag=18.0,
        airmass=1.1,
        seeing_fwhm_arcsec=0.7,
        exposure_time_sec=300.0,
    )
    assert est.snr > 15.0
    assert est.is_detected_5sigma is True
    assert est.photometric_uncertainty_mag < 0.10

    # Invert exposure time for target SNR = 20.0
    req_sec = ExposureCalculator.calculate_required_exposure_sec(
        instrument=xshooter,
        filter_obj=vis_filter,
        target_apparent_mag=19.5,
        airmass=1.2,
        seeing_fwhm_arcsec=0.8,
        target_snr=20.0,
    )
    assert req_sec >= xshooter.min_exposure_sec
    assert req_sec <= xshooter.max_exposure_sec


def test_predictive_eig_engine_calculation():
    """Verifies that EIG is non-negative and quantitatively distinguishes hypotheses across wavebands."""
    reg = FacilityRegistry()
    vlt = reg.get_facility("VLT")
    xshooter = vlt.get_instrument("XSHOOTER")
    nir_filter = xshooter.get_filter("NIR_SPEC")

    # Competing: Kilonova vs. Supernova Ia
    prior = {"KILONOVA": 0.50, "SUPERNOVA_IA": 0.50}

    action = ObservationAction(
        action_id="act_test_01",
        target_id="tgt_kn_test",
        target_ra_deg=180.0,
        target_dec_deg=-20.0,
        facility=vlt,
        instrument=xshooter,
        filter_obj=nir_filter,
        exposure_time_sec=600.0,
        scheduled_mjd=60002.0,  # 2 days post-trigger
        target_hypothesis_names=["KILONOVA", "SUPERNOVA_IA"],
    )

    eig_res = PredictiveEIGEngine.evaluate_eig(action, prior, airmass=1.2, seeing_fwhm_arcsec=0.7)
    # EIG must be strictly positive since Kilonova and SN Ia have very different NIR fluxes at 2 days
    assert eig_res.expected_information_gain_nats > 0.05
    assert eig_res.expected_posterior_entropy_nats < eig_res.prior_entropy_nats
    assert eig_res.discriminative_power_score > 1.0


def test_all_schedulers_generate_feasible_schedules():
    """Verifies that all 7 schedulers run successfully and generate zero hard constraint violations."""
    reg = FacilityRegistry()
    facilities = [reg.get_facility("ZTF"), reg.get_facility("VLT"), reg.get_facility("SWIFT")]

    targets = [
        TargetState(
            target_id="tgt_01",
            ra_deg=150.0,
            dec_deg=10.0,
            trigger_mjd=60000.0,
            base_magnitude=18.5,
            hypothesis_posteriors={"KILONOVA": 0.5, "SUPERNOVA_IA": 0.5},
            is_too=True,
        ),
        TargetState(
            target_id="tgt_02",
            ra_deg=180.0,
            dec_deg=-15.0,
            trigger_mjd=60000.0,
            base_magnitude=19.0,
            hypothesis_posteriors={"TDE": 0.6, "SUPERNOVA_CC": 0.4},
            is_too=False,
        ),
    ]

    schedulers = [
        RandomFeasibleScheduler(),
        GreedyEIGPlanner(),
        CostAwareGreedyPlanner(),
        RollingHorizonOptimizer(horizon_depth=2),
        ConstraintSatisfactionOptimizer(),
        MCTSPlanner(n_rollouts=5),
        HybridOptimizer(),
    ]

    start_mjd = 60001.0
    duration_hours = 4.0

    for sched in schedulers:
        res = sched.plan_schedule(targets, facilities, start_mjd, duration_hours)
        assert res.num_hard_violations == 0, f"Scheduler {sched.__class__.__name__} had hard violations!"
        assert res.total_time_used_hours <= duration_hours + 0.1
        assert res.total_expected_eig_nats >= 0.0


def test_human_approval_gate_enforces_review_and_prevents_autonomous_dispatch():
    """Verifies that recommendations require explicit human authorization and autonomous dispatch is blocked."""
    gate = HumanApprovalGate()

    target = TargetState(
        target_id="tgt_too_alert",
        ra_deg=200.0,
        dec_deg=5.0,
        trigger_mjd=60000.0,
        base_magnitude=17.5,
        hypothesis_posteriors={"KILONOVA": 0.70, "SUPERNOVA_CC": 0.30},
    )

    from astraeus_agents.observatory.schedulers import ScheduledObservation
    obs = ScheduledObservation(
        target_id=target.target_id,
        facility_id="VLT",
        instrument_id="XSHOOTER",
        filter_id="NIR_SPEC",
        start_mjd=60000.5,
        end_mjd=60000.52,
        exposure_time_sec=600.0,
        slew_time_sec=45.0,
        readout_time_sec=20.0,
        airmass=1.15,
        expected_eig_nats=0.35,
        weather_score=0.95,
    )

    # 1. Generate recommendation
    rec = gate.create_recommendation(
        obs=obs,
        target=target,
        scientific_justification="Confirm NIR lanthanide opacity peak to definitively separate Kilonova from CCSN.",
    )
    assert rec.approval_status == ApprovalStatus.PENDING_HUMAN_REVIEW

    # 2. Attempting to dispatch unapproved recommendation raises AutonomousDispatchProhibitedError
    with pytest.raises(AutonomousDispatchProhibitedError):
        gate.assert_authorized_for_dispatch(rec.recommendation_id)

    # 3. Human authorization by Lead Investigator
    auth_rec = gate.authorize_dispatch(rec.recommendation_id, investigator_email="devmehta060505@gmail.com")
    assert auth_rec.approval_status == ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
    assert auth_rec.authorized_investigator == "devmehta060505@gmail.com"

    # 4. Now assertion passes
    gate.assert_authorized_for_dispatch(rec.recommendation_id)


def test_gate3_quarantine_enforcement_in_observatory_planning():
    """Verifies that Gate 3 quarantined coordinates trigger QuarantineViolationError in planning workflows."""
    QuarantineFirewall._initialize()
    q_pix = next(iter(QuarantineFirewall.get_quarantined_pixels()))
    coord = QuarantineFirewall._HP.healpix_to_skycoord(q_pix)
    q_ra = float(coord.ra.deg)
    q_dec = float(coord.dec.deg)

    with pytest.raises(QuarantineViolationError, match="GATE 3 QUARANTINE VIOLATION"):
        QuarantineFirewall.assert_not_quarantined(q_ra, q_dec, identifier="quarantine_target_test")
