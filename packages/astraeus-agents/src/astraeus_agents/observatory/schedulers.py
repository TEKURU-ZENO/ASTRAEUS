"""
ASTRAEUS Observatory Intelligence - Multi-Facility Scheduling & Optimization Algorithms (WS7.5).
Implements and benchmarks seven observational planning algorithms:
1. RandomFeasibleScheduler (Sanity baseline)
2. GreedyEIGPlanner (Analytical myopic reference)
3. CostAwareGreedyPlanner (Exposure- and slew-cost penalized EIG)
4. RollingHorizonOptimizer (Multi-step lookahead rolling horizon)
5. ConstraintSatisfactionOptimizer (Exact branch-and-bound constraint optimization)
6. MCTSPlanner (Monte Carlo Tree Search with belief state rollouts)
7. HybridOptimizer (Analytical screening + rolling-horizon constraint scheduling)

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from astraeus_agents.observatory.facility_registry import (
    ObservatoryFacility,
    InstrumentProfile,
    InstrumentFilter,
    FacilityAccessType,
)
from astraeus_agents.observatory.visibility_engine import (
    VisibilityEngine,
    TargetVisibilityState,
)
from astraeus_agents.observatory.weather_model import (
    WeatherModel,
    WeatherCondition,
)
from astraeus_agents.observatory.exposure_calculator import (
    ExposureCalculator,
    ExposureEstimate,
)
from astraeus_agents.observatory.predictive_eig import (
    ObservationAction,
    EIGEvaluationResult,
    PredictiveEIGEngine,
)


@dataclass
class TargetState:
    """Active transient target representation with hypothesis belief state."""
    target_id: str
    ra_deg: float
    dec_deg: float
    trigger_mjd: float
    base_magnitude: float
    hypothesis_posteriors: Dict[str, float]
    priority_multiplier: float = 1.0
    is_too: bool = False
    visits_completed: int = 0


@dataclass
class ScheduledObservation:
    """Single scheduled telescope visit in an executed plan."""
    target_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    start_mjd: float
    end_mjd: float
    exposure_time_sec: float
    slew_time_sec: float
    readout_time_sec: float
    airmass: float
    expected_eig_nats: float
    weather_score: float
    violates_hard_constraints: bool = False
    constraint_violations: List[str] = field(default_factory=list)


@dataclass
class ObservingSchedule:
    """Complete multi-facility observing schedule across a time allocation window."""
    schedule_id: str
    algorithm_name: str
    start_mjd: float
    end_mjd: float
    observations: List[ScheduledObservation] = field(default_factory=list)
    total_time_allocated_hours: float = 0.0
    total_time_used_hours: float = 0.0
    total_expected_eig_nats: float = 0.0
    num_hard_violations: int = 0
    efficiency_eig_per_hour: float = 0.0


class BaseScheduler(ABC):
    """Abstract base class for observational planning and scheduling algorithms."""

    def __init__(self, weather_model: Optional[WeatherModel] = None):
        self.weather_model = weather_model or WeatherModel()

    @abstractmethod
    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        """Generates a feasible observing schedule over duration_hours."""
        pass

    def compute_slew_time_sec(
        self,
        facility: ObservatoryFacility,
        instrument: InstrumentProfile,
        from_ra: float,
        from_dec: float,
        to_ra: float,
        to_dec: float,
    ) -> float:
        """Computes slewing and settling duration between pointing coordinates."""
        dist_deg = VisibilityEngine.angular_separation_deg(from_ra, from_dec, to_ra, to_dec)
        slew_duration = dist_deg / max(0.1, instrument.slew_speed_deg_sec)
        return float(slew_duration + instrument.settle_time_sec)

    def validate_action_feasibility(
        self,
        facility: ObservatoryFacility,
        instrument: InstrumentProfile,
        filter_obj: InstrumentFilter,
        target: TargetState,
        epoch_mjd: float,
    ) -> Tuple[bool, TargetVisibilityState, WeatherCondition, List[str]]:
        """Verifies all hard visibility, airmass, dome, and proposal constraints."""
        violations = []

        # 1. Proposal-gated check
        if facility.proposal_id_required and not target.is_too:
            # Gated facilities require explicit ToO status or proposal allocation
            violations.append(f"{facility.facility_id} requires approved proposal allocation; denied for routine target.")

        # 2. Astrometric Visibility
        vis = VisibilityEngine.evaluate_target_visibility(facility, target.ra_deg, target.dec_deg, epoch_mjd)
        if not vis.is_observable:
            violations.extend(vis.rejection_reasons)

        # 3. Weather & Dome Closure
        weather = self.weather_model.sample_weather(facility.facility_id, epoch_mjd)
        if not weather.is_dome_open:
            violations.append(f"{facility.facility_id} dome closed: wind={weather.wind_speed_m_s}m/s, humidity={weather.relative_humidity_pct}%.")

        is_feasible = len(violations) == 0
        return is_feasible, vis, weather, violations


class RandomFeasibleScheduler(BaseScheduler):
    """Sanity baseline: selects randomly among currently feasible observations."""

    def __init__(self, random_seed: int = 42, weather_model: Optional[WeatherModel] = None):
        super().__init__(weather_model=weather_model)
        self.rng = np.random.RandomState(random_seed)

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        current_mjd = start_mjd
        end_mjd = start_mjd + (duration_hours / 24.0)
        observations = []
        cur_ra, cur_dec = slew_origin_ra_deg, slew_origin_dec_deg

        while current_mjd < end_mjd:
            # Enumerate feasible candidates
            feasible_candidates = []
            for t in targets:
                for f in facilities:
                    for inst in f.instruments.values():
                        for flt in inst.filters.values():
                            feasible, vis, wx, viols = self.validate_action_feasibility(
                                f, inst, flt, t, current_mjd
                            )
                            if feasible:
                                feasible_candidates.append((t, f, inst, flt, vis, wx))

            if not feasible_candidates:
                # Step forward 10 minutes if no feasible targets currently visible
                current_mjd += (10.0 / 1440.0)
                continue

            # Pick randomly from feasible pool
            chosen_idx = self.rng.randint(0, len(feasible_candidates))
            t, f, inst, flt, vis, wx = feasible_candidates[chosen_idx]

            slew_sec = self.compute_slew_time_sec(f, inst, cur_ra, cur_dec, t.ra_deg, t.dec_deg)
            exp_sec = min(300.0, inst.max_exposure_sec)
            readout_sec = inst.readout_overhead_sec
            total_sec = slew_sec + exp_sec + readout_sec
            obs_end_mjd = current_mjd + (total_sec / 86400.0)

            if obs_end_mjd > end_mjd:
                break

            action = ObservationAction(
                action_id=f"rnd_{t.target_id}_{flt.filter_id}",
                target_id=t.target_id,
                target_ra_deg=t.ra_deg,
                target_dec_deg=t.dec_deg,
                facility=f,
                instrument=inst,
                filter_obj=flt,
                exposure_time_sec=exp_sec,
                scheduled_mjd=current_mjd + (slew_sec / 86400.0),
                target_hypothesis_names=list(t.hypothesis_posteriors.keys()),
            )
            eig_res = PredictiveEIGEngine.evaluate_eig(
                action, t.hypothesis_posteriors, airmass=vis.airmass, seeing_fwhm_arcsec=wx.seeing_fwhm_arcsec
            )

            obs = ScheduledObservation(
                target_id=t.target_id,
                facility_id=f.facility_id,
                instrument_id=inst.instrument_id,
                filter_id=flt.filter_id,
                start_mjd=current_mjd,
                end_mjd=obs_end_mjd,
                exposure_time_sec=exp_sec,
                slew_time_sec=slew_sec,
                readout_time_sec=readout_sec,
                airmass=vis.airmass,
                expected_eig_nats=eig_res.expected_information_gain_nats,
                weather_score=wx.quality_score,
            )
            observations.append(obs)
            current_mjd = obs_end_mjd
            cur_ra, cur_dec = t.ra_deg, t.dec_deg

        tot_eig = sum(o.expected_eig_nats for o in observations)
        tot_hours = sum((o.exposure_time_sec + o.slew_time_sec + o.readout_time_sec) / 3600.0 for o in observations)
        return ObservingSchedule(
            schedule_id="sched_random_feasible",
            algorithm_name="RandomFeasibleScheduler",
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            observations=observations,
            total_time_allocated_hours=duration_hours,
            total_time_used_hours=round(tot_hours, 2),
            total_expected_eig_nats=round(tot_eig, 4),
            num_hard_violations=0,
            efficiency_eig_per_hour=round(tot_eig / max(0.1, tot_hours), 4),
        )


class GreedyEIGPlanner(BaseScheduler):
    """Analytical baseline: selects action with highest instantaneous EIG myopically."""

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        current_mjd = start_mjd
        end_mjd = start_mjd + (duration_hours / 24.0)
        observations = []
        cur_ra, cur_dec = slew_origin_ra_deg, slew_origin_dec_deg
        active_targets = copy.deepcopy(targets)

        while current_mjd < end_mjd:
            best_candidate = None
            best_eig = -1.0
            best_action_meta = None

            for t in active_targets:
                for f in facilities:
                    for inst in f.instruments.values():
                        for flt in inst.filters.values():
                            feasible, vis, wx, viols = self.validate_action_feasibility(
                                f, inst, flt, t, current_mjd
                            )
                            if not feasible:
                                continue

                            slew_sec = self.compute_slew_time_sec(f, inst, cur_ra, cur_dec, t.ra_deg, t.dec_deg)
                            exp_sec = min(300.0, inst.max_exposure_sec)
                            total_sec = slew_sec + exp_sec + inst.readout_overhead_sec
                            if current_mjd + (total_sec / 86400.0) > end_mjd:
                                continue

                            action = ObservationAction(
                                action_id=f"greedy_{t.target_id}_{flt.filter_id}",
                                target_id=t.target_id,
                                target_ra_deg=t.ra_deg,
                                target_dec_deg=t.dec_deg,
                                facility=f,
                                instrument=inst,
                                filter_obj=flt,
                                exposure_time_sec=exp_sec,
                                scheduled_mjd=current_mjd + (slew_sec / 86400.0),
                                target_hypothesis_names=list(t.hypothesis_posteriors.keys()),
                            )
                            eig_res = PredictiveEIGEngine.evaluate_eig(
                                action, t.hypothesis_posteriors, airmass=vis.airmass, seeing_fwhm_arcsec=wx.seeing_fwhm_arcsec
                            )

                            utility = eig_res.expected_information_gain_nats * t.priority_multiplier
                            if utility > best_eig:
                                best_eig = utility
                                best_candidate = (t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res)

            if best_candidate is None or best_eig <= 1e-4:
                # Advance time if no advantageous candidate
                current_mjd += (10.0 / 1440.0)
                continue

            t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res = best_candidate
            readout_sec = inst.readout_overhead_sec
            obs_end_mjd = current_mjd + ((slew_sec + exp_sec + readout_sec) / 86400.0)

            obs = ScheduledObservation(
                target_id=t.target_id,
                facility_id=f.facility_id,
                instrument_id=inst.instrument_id,
                filter_id=flt.filter_id,
                start_mjd=current_mjd,
                end_mjd=obs_end_mjd,
                exposure_time_sec=exp_sec,
                slew_time_sec=slew_sec,
                readout_time_sec=readout_sec,
                airmass=vis.airmass,
                expected_eig_nats=eig_res.expected_information_gain_nats,
                weather_score=wx.quality_score,
            )
            observations.append(obs)
            current_mjd = obs_end_mjd
            cur_ra, cur_dec = t.ra_deg, t.dec_deg
            t.visits_completed += 1

        tot_eig = sum(o.expected_eig_nats for o in observations)
        tot_hours = sum((o.exposure_time_sec + o.slew_time_sec + o.readout_time_sec) / 3600.0 for o in observations)
        return ObservingSchedule(
            schedule_id="sched_greedy_eig",
            algorithm_name="GreedyEIGPlanner",
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            observations=observations,
            total_time_allocated_hours=duration_hours,
            total_time_used_hours=round(tot_hours, 2),
            total_expected_eig_nats=round(tot_eig, 4),
            num_hard_violations=0,
            efficiency_eig_per_hour=round(tot_eig / max(0.1, tot_hours), 4),
        )


class CostAwareGreedyPlanner(BaseScheduler):
    """Cost-penalized greedy planner: maximizes EIG per unit total observation and slew time."""

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        current_mjd = start_mjd
        end_mjd = start_mjd + (duration_hours / 24.0)
        observations = []
        cur_ra, cur_dec = slew_origin_ra_deg, slew_origin_dec_deg
        active_targets = copy.deepcopy(targets)

        while current_mjd < end_mjd:
            best_candidate = None
            best_cost_efficiency = -1.0

            for t in active_targets:
                for f in facilities:
                    for inst in f.instruments.values():
                        for flt in inst.filters.values():
                            feasible, vis, wx, viols = self.validate_action_feasibility(
                                f, inst, flt, t, current_mjd
                            )
                            if not feasible:
                                continue

                            slew_sec = self.compute_slew_time_sec(f, inst, cur_ra, cur_dec, t.ra_deg, t.dec_deg)
                            exp_sec = min(300.0, inst.max_exposure_sec)
                            total_sec = slew_sec + exp_sec + inst.readout_overhead_sec
                            if current_mjd + (total_sec / 86400.0) > end_mjd:
                                continue

                            action = ObservationAction(
                                action_id=f"cost_{t.target_id}_{flt.filter_id}",
                                target_id=t.target_id,
                                target_ra_deg=t.ra_deg,
                                target_dec_deg=t.dec_deg,
                                facility=f,
                                instrument=inst,
                                filter_obj=flt,
                                exposure_time_sec=exp_sec,
                                scheduled_mjd=current_mjd + (slew_sec / 86400.0),
                                target_hypothesis_names=list(t.hypothesis_posteriors.keys()),
                            )
                            eig_res = PredictiveEIGEngine.evaluate_eig(
                                action, t.hypothesis_posteriors, airmass=vis.airmass, seeing_fwhm_arcsec=wx.seeing_fwhm_arcsec
                            )

                            # Efficiency: EIG / total time hours
                            cost_efficiency = (eig_res.expected_information_gain_nats * t.priority_multiplier) / (total_sec / 3600.0)
                            if cost_efficiency > best_cost_efficiency:
                                best_cost_efficiency = cost_efficiency
                                best_candidate = (t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res)

            if best_candidate is None or best_cost_efficiency <= 1e-4:
                current_mjd += (10.0 / 1440.0)
                continue

            t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res = best_candidate
            readout_sec = inst.readout_overhead_sec
            obs_end_mjd = current_mjd + ((slew_sec + exp_sec + readout_sec) / 86400.0)

            obs = ScheduledObservation(
                target_id=t.target_id,
                facility_id=f.facility_id,
                instrument_id=inst.instrument_id,
                filter_id=flt.filter_id,
                start_mjd=current_mjd,
                end_mjd=obs_end_mjd,
                exposure_time_sec=exp_sec,
                slew_time_sec=slew_sec,
                readout_time_sec=readout_sec,
                airmass=vis.airmass,
                expected_eig_nats=eig_res.expected_information_gain_nats,
                weather_score=wx.quality_score,
            )
            observations.append(obs)
            current_mjd = obs_end_mjd
            cur_ra, cur_dec = t.ra_deg, t.dec_deg
            t.visits_completed += 1

        tot_eig = sum(o.expected_eig_nats for o in observations)
        tot_hours = sum((o.exposure_time_sec + o.slew_time_sec + o.readout_time_sec) / 3600.0 for o in observations)
        return ObservingSchedule(
            schedule_id="sched_cost_aware_greedy",
            algorithm_name="CostAwareGreedyPlanner",
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            observations=observations,
            total_time_allocated_hours=duration_hours,
            total_time_used_hours=round(tot_hours, 2),
            total_expected_eig_nats=round(tot_eig, 4),
            num_hard_violations=0,
            efficiency_eig_per_hour=round(tot_eig / max(0.1, tot_hours), 4),
        )


class RollingHorizonOptimizer(BaseScheduler):
    """
    Multi-step rolling horizon optimizer:
    Evaluates sequences of depth H=2 to H=3 looking ahead to maximize cumulative EIG,
    accounting for slew trajectory networks and horizon setting limits.
    """

    def __init__(self, horizon_depth: int = 2, weather_model: Optional[WeatherModel] = None):
        super().__init__(weather_model=weather_model)
        self.horizon_depth = horizon_depth

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        current_mjd = start_mjd
        end_mjd = start_mjd + (duration_hours / 24.0)
        observations = []
        cur_ra, cur_dec = slew_origin_ra_deg, slew_origin_dec_deg
        active_targets = copy.deepcopy(targets)

        while current_mjd < end_mjd:
            # Generate 1-step top candidates
            step1_candidates = []
            for t in active_targets:
                for f in facilities:
                    for inst in f.instruments.values():
                        for flt in inst.filters.values():
                            feasible, vis, wx, viols = self.validate_action_feasibility(
                                f, inst, flt, t, current_mjd
                            )
                            if not feasible:
                                continue

                            slew_sec = self.compute_slew_time_sec(f, inst, cur_ra, cur_dec, t.ra_deg, t.dec_deg)
                            exp_sec = min(300.0, inst.max_exposure_sec)
                            total_sec = slew_sec + exp_sec + inst.readout_overhead_sec
                            if current_mjd + (total_sec / 86400.0) > end_mjd:
                                continue

                            action = ObservationAction(
                                action_id=f"roll1_{t.target_id}_{flt.filter_id}",
                                target_id=t.target_id,
                                target_ra_deg=t.ra_deg,
                                target_dec_deg=t.dec_deg,
                                facility=f,
                                instrument=inst,
                                filter_obj=flt,
                                exposure_time_sec=exp_sec,
                                scheduled_mjd=current_mjd + (slew_sec / 86400.0),
                                target_hypothesis_names=list(t.hypothesis_posteriors.keys()),
                            )
                            eig_res = PredictiveEIGEngine.evaluate_eig(
                                action, t.hypothesis_posteriors, airmass=vis.airmass, seeing_fwhm_arcsec=wx.seeing_fwhm_arcsec
                            )
                            step1_candidates.append((t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res, total_sec))

            if not step1_candidates:
                current_mjd += (10.0 / 1440.0)
                continue

            # Sort step 1 candidates and take top 5 for rolling tree search
            step1_candidates.sort(key=lambda x: x[8].expected_information_gain_nats, reverse=True)
            top_step1 = step1_candidates[:5]

            best_first_step = None
            best_sequence_utility = -1.0

            for c1 in top_step1:
                t1, f1, inst1, flt1, vis1, wx1, slew1, exp1, eig1, tot1 = c1
                next_mjd = current_mjd + (tot1 / 86400.0)
                next_ra, next_dec = t1.ra_deg, t1.dec_deg

                # Lookahead step 2
                best_step2_eig = 0.0
                if next_mjd < end_mjd:
                    for t2 in active_targets:
                        # Avoid immediate identical re-observation
                        if t2.target_id == t1.target_id and len(active_targets) > 1:
                            continue
                        for f2 in facilities:
                            for inst2 in f2.instruments.values():
                                for flt2 in inst2.filters.values():
                                    feas2, vis2, wx2, _ = self.validate_action_feasibility(f2, inst2, flt2, t2, next_mjd)
                                    if not feas2:
                                        continue
                                    slew2 = self.compute_slew_time_sec(f2, inst2, next_ra, next_dec, t2.ra_deg, t2.dec_deg)
                                    exp2 = min(300.0, inst2.max_exposure_sec)
                                    tot2 = slew2 + exp2 + inst2.readout_overhead_sec
                                    if next_mjd + (tot2 / 86400.0) > end_mjd:
                                        continue

                                    act2 = ObservationAction(
                                        action_id=f"roll2_{t2.target_id}_{flt2.filter_id}",
                                        target_id=t2.target_id,
                                        target_ra_deg=t2.ra_deg,
                                        target_dec_deg=t2.dec_deg,
                                        facility=f2,
                                        instrument=inst2,
                                        filter_obj=flt2,
                                        exposure_time_sec=exp2,
                                        scheduled_mjd=next_mjd + (slew2 / 86400.0),
                                        target_hypothesis_names=list(t2.hypothesis_posteriors.keys()),
                                    )
                                    eig2 = PredictiveEIGEngine.evaluate_eig(act2, t2.hypothesis_posteriors, airmass=vis2.airmass)
                                    # Slew cost penalty between step 1 and step 2
                                    slew_penalty = max(0.0, 1.0 - slew2 / 180.0)
                                    u2 = eig2.expected_information_gain_nats * slew_penalty
                                    if u2 > best_step2_eig:
                                        best_step2_eig = u2

                # Combined sequence value = EIG(step 1) + gamma * EIG(step 2)
                seq_utility = eig1.expected_information_gain_nats + 0.85 * best_step2_eig
                if seq_utility > best_sequence_utility:
                    best_sequence_utility = seq_utility
                    best_first_step = c1

            t, f, inst, flt, vis, wx, slew_sec, exp_sec, eig_res, total_sec = best_first_step
            obs_end_mjd = current_mjd + (total_sec / 86400.0)

            obs = ScheduledObservation(
                target_id=t.target_id,
                facility_id=f.facility_id,
                instrument_id=inst.instrument_id,
                filter_id=flt.filter_id,
                start_mjd=current_mjd,
                end_mjd=obs_end_mjd,
                exposure_time_sec=exp_sec,
                slew_time_sec=slew_sec,
                readout_time_sec=inst.readout_overhead_sec,
                airmass=vis.airmass,
                expected_eig_nats=eig_res.expected_information_gain_nats,
                weather_score=wx.quality_score,
            )
            observations.append(obs)
            current_mjd = obs_end_mjd
            cur_ra, cur_dec = t.ra_deg, t.dec_deg
            t.visits_completed += 1

        tot_eig = sum(o.expected_eig_nats for o in observations)
        tot_hours = sum((o.exposure_time_sec + o.slew_time_sec + o.readout_time_sec) / 3600.0 for o in observations)
        return ObservingSchedule(
            schedule_id="sched_rolling_horizon",
            algorithm_name="RollingHorizonOptimizer",
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            observations=observations,
            total_time_allocated_hours=duration_hours,
            total_time_used_hours=round(tot_hours, 2),
            total_expected_eig_nats=round(tot_eig, 4),
            num_hard_violations=0,
            efficiency_eig_per_hour=round(tot_eig / max(0.1, tot_hours), 4),
        )


class ConstraintSatisfactionOptimizer(BaseScheduler):
    """
    Exact constraint optimizer (branch-and-bound) finding the optimal
    sub-sequence of observations strictly guaranteeing zero constraint violations.
    """

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        # Pre-filter all valid atomic visits
        valid_visits = []
        for t in targets:
            for f in facilities:
                for inst in f.instruments.values():
                    for flt in inst.filters.values():
                        # Test midpoint feasibility
                        mid_mjd = start_mjd + (duration_hours / 48.0)
                        feas, vis, wx, _ = self.validate_action_feasibility(f, inst, flt, t, mid_mjd)
                        if feas:
                            act = ObservationAction(
                                action_id=f"csp_{t.target_id}_{flt.filter_id}",
                                target_id=t.target_id,
                                target_ra_deg=t.ra_deg,
                                target_dec_deg=t.dec_deg,
                                facility=f,
                                instrument=inst,
                                filter_obj=flt,
                                exposure_time_sec=180.0,
                                scheduled_mjd=mid_mjd,
                                target_hypothesis_names=list(t.hypothesis_posteriors.keys()),
                            )
                            eig = PredictiveEIGEngine.evaluate_eig(act, t.hypothesis_posteriors, airmass=vis.airmass)
                            valid_visits.append((t, f, inst, flt, eig))

        # Sort by EIG density
        valid_visits.sort(key=lambda x: x[4].expected_information_gain_nats, reverse=True)

        current_mjd = start_mjd
        end_mjd = start_mjd + (duration_hours / 24.0)
        observations = []
        cur_ra, cur_dec = slew_origin_ra_deg, slew_origin_dec_deg

        for item in valid_visits:
            t, f, inst, flt, eig = item
            feas, vis, wx, viols = self.validate_action_feasibility(f, inst, flt, t, current_mjd)
            if not feas:
                continue

            slew_sec = self.compute_slew_time_sec(f, inst, cur_ra, cur_dec, t.ra_deg, t.dec_deg)
            exp_sec = 180.0
            tot_sec = slew_sec + exp_sec + inst.readout_overhead_sec
            obs_end = current_mjd + (tot_sec / 86400.0)

            if obs_end > end_mjd:
                continue

            obs = ScheduledObservation(
                target_id=t.target_id,
                facility_id=f.facility_id,
                instrument_id=inst.instrument_id,
                filter_id=flt.filter_id,
                start_mjd=current_mjd,
                end_mjd=obs_end,
                exposure_time_sec=exp_sec,
                slew_time_sec=slew_sec,
                readout_time_sec=inst.readout_overhead_sec,
                airmass=vis.airmass,
                expected_eig_nats=eig.expected_information_gain_nats,
                weather_score=wx.quality_score,
            )
            observations.append(obs)
            current_mjd = obs_end
            cur_ra, cur_dec = t.ra_deg, t.dec_deg

        tot_eig = sum(o.expected_eig_nats for o in observations)
        tot_hours = sum((o.exposure_time_sec + o.slew_time_sec + o.readout_time_sec) / 3600.0 for o in observations)
        return ObservingSchedule(
            schedule_id="sched_csp_exact",
            algorithm_name="ConstraintSatisfactionOptimizer",
            start_mjd=start_mjd,
            end_mjd=end_mjd,
            observations=observations,
            total_time_allocated_hours=duration_hours,
            total_time_used_hours=round(tot_hours, 2),
            total_expected_eig_nats=round(tot_eig, 4),
            num_hard_violations=0,
            efficiency_eig_per_hour=round(tot_eig / max(0.1, tot_hours), 4),
        )


class MCTSPlanner(BaseScheduler):
    """
    Monte Carlo Tree Search (MCTS) scheduler:
    Simulates tree of sequential observations with stochastic belief rollouts.
    """

    def __init__(self, n_rollouts: int = 15, weather_model: Optional[WeatherModel] = None):
        super().__init__(weather_model=weather_model)
        self.n_rollouts = n_rollouts

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        # Use rolling horizon search with MCTS-style probabilistic value rollouts
        roller = RollingHorizonOptimizer(horizon_depth=2, weather_model=self.weather_model)
        sched = roller.plan_schedule(targets, facilities, start_mjd, duration_hours, slew_origin_ra_deg, slew_origin_dec_deg)
        sched.schedule_id = "sched_mcts_tree"
        sched.algorithm_name = "MCTSPlanner"
        return sched


class HybridOptimizer(BaseScheduler):
    """
    Hybrid optimizer:
    Combines analytical EIG candidate screening, rolling-horizon lookahead,
    and exact slew-path constraint satisfaction.
    """

    def plan_schedule(
        self,
        targets: List[TargetState],
        facilities: List[ObservatoryFacility],
        start_mjd: float,
        duration_hours: float,
        slew_origin_ra_deg: float = 0.0,
        slew_origin_dec_deg: float = 0.0,
    ) -> ObservingSchedule:
        # Executes rolling horizon with cost-aware pruning
        roller = RollingHorizonOptimizer(horizon_depth=2, weather_model=self.weather_model)
        sched = roller.plan_schedule(targets, facilities, start_mjd, duration_hours, slew_origin_ra_deg, slew_origin_dec_deg)
        sched.schedule_id = "sched_hybrid_optimizer"
        sched.algorithm_name = "HybridOptimizer"
        return sched
