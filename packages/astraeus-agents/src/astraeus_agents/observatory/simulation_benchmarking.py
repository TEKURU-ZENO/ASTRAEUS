"""
ASTRAEUS Observatory Intelligence - 100-Scenario Paired Simulation Benchmark (WS7.5).
Executes rigorous Monte Carlo simulation studies across 100+ independent paired transient scenarios,
comparing Rolling-Horizon Optimization, Cost-Aware Greedy, Analytical Greedy EIG, and Random baselines.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import numpy as np
import scipy.stats as stats
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_agents.observatory.facility_registry import (
    FacilityRegistry,
    ObservatoryFacility,
)
from astraeus_agents.observatory.weather_model import WeatherModel
from astraeus_agents.observatory.schedulers import (
    TargetState,
    ObservingSchedule,
    RandomFeasibleScheduler,
    GreedyEIGPlanner,
    CostAwareGreedyPlanner,
    RollingHorizonOptimizer,
    ConstraintSatisfactionOptimizer,
    HybridOptimizer,
)


@dataclass
class ScenarioBenchmarkResult:
    """Statistical summary across 100+ paired Monte Carlo scenario runs."""
    num_scenarios: int
    mean_eig_random: float
    mean_eig_greedy: float
    mean_eig_cost_aware: float
    mean_eig_rolling_horizon: float
    mean_eig_hybrid: float
    std_eig_greedy: float
    std_eig_rolling_horizon: float
    p_value_rolling_vs_greedy: float
    rolling_horizon_superiority_percentage: float
    zero_hard_violation_rate: float
    gate3_quarantine_breaches: int


class MultiFacilitySimulationBenchmark:
    """
    Automated benchmark testbed executing paired synthetic transient observing runs.
    """

    def __init__(self, n_scenarios: int = 100, random_seed: int = 1001):
        self.n_scenarios = n_scenarios
        self.random_seed = random_seed
        self.registry = FacilityRegistry()
        self.weather_model = WeatherModel(random_seed=random_seed)

    def generate_synthetic_scenario(self, seed: int) -> Tuple[List[TargetState], List[ObservatoryFacility], float, float]:
        """
        Generates a paired, unquarantined multi-target scenario with competing hypotheses.
        """
        rng = np.random.RandomState(seed)
        start_mjd = 60000.0 + rng.uniform(0.0, 300.0)
        duration_hours = 6.0  # 6-hour half-night allocation

        # Generate 4 distinct transient targets outside Gate 3 quarantine
        targets: List[TargetState] = []
        target_types = ["KILONOVA_FAST", "SUPERNOVA_IA", "SUPERNOVA_CC", "TDE_BLUE"]

        QuarantineFirewall._initialize()

        for idx, t_type in enumerate(target_types):
            # Find an unquarantined coordinate
            while True:
                ra = float(rng.uniform(0.0, 360.0))
                dec = float(rng.uniform(-30.0, 60.0))
                pix = QuarantineFirewall.compute_healpix_nside32(ra, dec)
                if not QuarantineFirewall.is_quarantined_pixel(pix):
                    break

            base_mag = float(rng.uniform(17.5, 20.0))

            # Prior distribution with uncertainty
            if t_type == "KILONOVA_FAST":
                posts = {"KILONOVA": 0.45, "SUPERNOVA_IA": 0.25, "SUPERNOVA_CC": 0.20, "TDE": 0.10}
                is_too = True
                p_mult = 1.5
            elif t_type == "TDE_BLUE":
                posts = {"TDE": 0.50, "SUPERNOVA_IA": 0.25, "KILONOVA": 0.15, "SUPERNOVA_CC": 0.10}
                is_too = True
                p_mult = 1.2
            elif t_type == "SUPERNOVA_IA":
                posts = {"SUPERNOVA_IA": 0.55, "SUPERNOVA_CC": 0.35, "KILONOVA": 0.05, "TDE": 0.05}
                is_too = False
                p_mult = 1.0
            else:
                posts = {"SUPERNOVA_CC": 0.50, "SUPERNOVA_IA": 0.40, "KILONOVA": 0.05, "TDE": 0.05}
                is_too = False
                p_mult = 1.0

            targets.append(TargetState(
                target_id=f"sim_tgt_{seed}_{idx}_{t_type}",
                ra_deg=ra,
                dec_deg=dec,
                trigger_mjd=start_mjd - rng.uniform(0.5, 3.0),
                base_magnitude=base_mag,
                hypothesis_posteriors=posts,
                priority_multiplier=p_mult,
                is_too=is_too,
            ))

        # Select a diverse subset of ground and space facilities
        facilities = [
            self.registry.get_facility("ZTF"),
            self.registry.get_facility("VLT"),
            self.registry.get_facility("SWIFT"),
            self.registry.get_facility("GEMINI"),
        ]
        # Filter None
        facilities = [f for f in facilities if f is not None]

        return targets, facilities, start_mjd, duration_hours

    def run_benchmark(self) -> ScenarioBenchmarkResult:
        """
        Executes paired benchmark across all 100 independent scenarios.
        """
        eigs_random: List[float] = []
        eigs_greedy: List[float] = []
        eigs_cost_aware: List[float] = []
        eigs_rolling: List[float] = []
        eigs_hybrid: List[float] = []

        total_violations = 0
        quarantine_breaches = 0

        # Instantiate schedulers
        sched_random = RandomFeasibleScheduler(weather_model=self.weather_model)
        sched_greedy = GreedyEIGPlanner(weather_model=self.weather_model)
        sched_cost = CostAwareGreedyPlanner(weather_model=self.weather_model)
        sched_rolling = RollingHorizonOptimizer(horizon_depth=2, weather_model=self.weather_model)
        sched_hybrid = HybridOptimizer(weather_model=self.weather_model)

        for s_idx in range(self.n_scenarios):
            seed = self.random_seed + s_idx
            targets, facilities, start_mjd, duration = self.generate_synthetic_scenario(seed)

            # Audit quarantine compliance
            for t in targets:
                pix = QuarantineFirewall.compute_healpix_nside32(t.ra_deg, t.dec_deg)
                if QuarantineFirewall.is_quarantined_pixel(pix):
                    quarantine_breaches += 1

            # Run paired algorithms under identical conditions
            s_rnd = sched_random.plan_schedule(targets, facilities, start_mjd, duration)
            s_grd = sched_greedy.plan_schedule(targets, facilities, start_mjd, duration)
            s_cst = sched_cost.plan_schedule(targets, facilities, start_mjd, duration)
            s_rol = sched_rolling.plan_schedule(targets, facilities, start_mjd, duration)
            s_hyb = sched_hybrid.plan_schedule(targets, facilities, start_mjd, duration)

            total_violations += (
                s_rnd.num_hard_violations +
                s_grd.num_hard_violations +
                s_cst.num_hard_violations +
                s_rol.num_hard_violations +
                s_hyb.num_hard_violations
            )

            eigs_random.append(s_rnd.total_expected_eig_nats)
            eigs_greedy.append(s_grd.total_expected_eig_nats)
            eigs_cost_aware.append(s_cst.total_expected_eig_nats)
            eigs_rolling.append(s_rol.total_expected_eig_nats)
            eigs_hybrid.append(s_hyb.total_expected_eig_nats)

        # Statistical analysis
        mean_rnd = float(np.mean(eigs_random))
        mean_grd = float(np.mean(eigs_greedy))
        mean_cst = float(np.mean(eigs_cost_aware))
        mean_rol = float(np.mean(eigs_rolling))
        mean_hyb = float(np.mean(eigs_hybrid))

        std_grd = float(np.std(eigs_greedy))
        std_rol = float(np.std(eigs_rolling))

        # Paired t-test: Rolling Horizon vs. Greedy EIG
        diffs = np.array(eigs_rolling) - np.array(eigs_greedy)
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        # Fraction where rolling horizon was >= greedy
        superior_pct = float(np.mean(np.array(eigs_rolling) >= np.array(eigs_greedy)) * 100.0)

        return ScenarioBenchmarkResult(
            num_scenarios=self.n_scenarios,
            mean_eig_random=round(mean_rnd, 4),
            mean_eig_greedy=round(mean_grd, 4),
            mean_eig_cost_aware=round(mean_cst, 4),
            mean_eig_rolling_horizon=round(mean_rol, 4),
            mean_eig_hybrid=round(mean_hyb, 4),
            std_eig_greedy=round(std_grd, 4),
            std_eig_rolling_horizon=round(std_rol, 4),
            p_value_rolling_vs_greedy=float(p_val),
            rolling_horizon_superiority_percentage=round(superior_pct, 1),
            zero_hard_violation_rate=100.0 if total_violations == 0 else 0.0,
            gate3_quarantine_breaches=quarantine_breaches,
        )
