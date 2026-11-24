"""
ASTRAEUS Phase 7 - Workstream 7.5 100-Scenario Paired Simulation Benchmark.
Executes 100 independent multi-facility observing scenarios with independent random seeds,
benchmarking Rolling Horizon Optimization, Cost-Aware Greedy, Analytical Greedy EIG, and Random baselines.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
from astraeus_agents.observatory.simulation_benchmarking import (
    MultiFacilitySimulationBenchmark,
    ScenarioBenchmarkResult,
)


def test_100_scenario_multi_facility_simulation_benchmark():
    """
    Executes 100 paired synthetic observing scenarios across multi-facility instruments.
    Validates:
    1. Zero hard-constraint violations across all 100 scenarios.
    2. Zero Gate 3 quarantine breaches.
    3. Positive EIG over random baseline across all scenarios.
    4. Demonstrated superiority of Rolling-Horizon / Hybrid Optimizer over myopic Greedy EIG.
    """
    benchmark = MultiFacilitySimulationBenchmark(n_scenarios=100, random_seed=20261124)
    result = benchmark.run_benchmark()

    # 1. Scope requirement: exactly 100 scenarios
    assert result.num_scenarios == 100

    # 2. Hard constraint enforcement: 100% zero-violation rate
    assert result.zero_hard_violation_rate == 100.0, "Hard constraint violations detected during scheduling!"

    # 3. Gate 3 Zero-Touch Quarantine Covenant: 0 breaches
    assert result.gate3_quarantine_breaches == 0, f"Quarantine breach detected: {result.gate3_quarantine_breaches} targets violated Gate 3!"

    # 4. Information gain ordering:
    # Random < Greedy <= Cost-Aware / Rolling-Horizon
    assert result.mean_eig_random >= 0.0
    assert result.mean_eig_greedy > result.mean_eig_random, (
        f"Greedy EIG ({result.mean_eig_greedy:.4f}) failed to exceed Random baseline ({result.mean_eig_random:.4f})"
    )

    # 5. Demonstrated benefit of multi-step rolling horizon over myopic 1-step greedy
    assert result.mean_eig_rolling_horizon >= result.mean_eig_greedy, (
        f"Rolling horizon ({result.mean_eig_rolling_horizon:.4f}) should meet or exceed Greedy ({result.mean_eig_greedy:.4f})"
    )
    assert result.rolling_horizon_superiority_percentage >= 50.0

    print("\n" + "=" * 80)
    print("ASTRAEUS WS7.5 100-SCENARIO BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Scenarios evaluated:                  {result.num_scenarios}")
    print(f"Mean EIG (Random Feasible):            {result.mean_eig_random:.4f} nats")
    print(f"Mean EIG (Myopic Greedy EIG):          {result.mean_eig_greedy:.4f} nats")
    print(f"Mean EIG (Cost-Aware Greedy):          {result.mean_eig_cost_aware:.4f} nats")
    print(f"Mean EIG (Rolling Horizon Optimizer):  {result.mean_eig_rolling_horizon:.4f} nats")
    print(f"Mean EIG (Hybrid Optimizer):           {result.mean_eig_hybrid:.4f} nats")
    print(f"Rolling Horizon Superiority Rate:     {result.rolling_horizon_superiority_percentage:.1f}%")
    print(f"Hard Constraint Violation Rate:        0.0% (100% compliant)")
    print(f"Gate 3 Quarantine Breaches:           0 (100% clean)")
    print("=" * 80)
