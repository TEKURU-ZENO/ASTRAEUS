"""
ASTRAEUS Phase 6: Workstream 1 Test Suite — Scientific Computing Foundation.
Validates background cosmology, particle-mesh Poisson solver, symplectic KDK leapfrog,
physical conservation invariants, and analytical convergence against the Zel'dovich pancake.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import numpy as np
import pytest

from engines.cosmology.constants import (
    CosmologicalParameters,
    SPEED_OF_LIGHT_KM_S,
    GRAVITATIONAL_CONSTANT_KPC,
    HUBBLE_DISTANCE_MPC,
)
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.mesh import ParticleMeshSolver3D
from engines.cosmology.integrator import SimulationState, SymplecticKDKLeapfrog
from engines.cosmology.analytic_tests import ZeldovichPancakeSolution, KeplerTwoBodySolution


class TestCosmologicalParametersAndBackground:
    """Validates CosmologicalParameters and BackgroundCosmology analytical solvers."""

    def test_cosmological_parameters_validation(self):
        # Default Planck 2018
        cosmo = CosmologicalParameters()
        assert math.isclose(cosmo.h, 0.6766, rel_tol=1e-3)
        assert math.isclose(cosmo.Omega_m, 0.3111, rel_tol=1e-3)
        assert math.isclose(cosmo.Omega_b, 0.0490, rel_tol=1e-3)
        assert math.isclose(cosmo.Omega_k, 0.0, abs_tol=1e-5)
        assert math.isclose(cosmo.H0, 67.66, rel_tol=1e-3)

        # Einstein-de Sitter universe
        eds = CosmologicalParameters.einstein_de_sitter(h=0.7)
        assert eds.Omega_m == 1.0
        assert eds.Omega_lambda == 0.0
        assert eds.Omega_k == 0.0
        assert eds.H0 == 70.0

        # Invalid parameters should raise
        with pytest.raises(AssertionError):
            CosmologicalParameters(Omega_m=-0.1)
        with pytest.raises(AssertionError):
            CosmologicalParameters(h=-0.5)

    def test_background_expansion_functions(self):
        cosmo = CosmologicalParameters(Omega_m=0.3, Omega_lambda=0.7, Omega_r=0.0, Omega_k=0.0, h=0.7)
        bg = BackgroundCosmology(cosmo)

        # At z = 0 (a = 1), E(a) = 1.0
        assert math.isclose(bg.E(1.0), 1.0, rel_tol=1e-6)
        assert math.isclose(bg.H(1.0), 70.0, rel_tol=1e-6)

        # Monotonicity of H(a) with respect to redshift (higher at earlier times / smaller a)
        assert bg.H(0.5) > bg.H(1.0)
        assert bg.H(0.1) > bg.H(0.5)

        # Analytical dE/da consistency against finite difference
        a_test = 0.6
        da = 1e-6
        dE_analytical = bg.dE_da(a_test)
        dE_numerical = (bg.E(a_test + da) - bg.E(a_test - da)) / (2.0 * da)
        assert math.isclose(dE_analytical, dE_numerical, rel_tol=1e-4)

    def test_cosmological_distance_measures_and_duality(self):
        cosmo = CosmologicalParameters()
        bg = BackgroundCosmology(cosmo)

        # At z = 0, distances must be zero
        assert math.isclose(bg.comoving_distance_chi(0.0), 0.0, abs_tol=1e-8)
        assert math.isclose(bg.luminosity_distance_dL(0.0), 0.0, abs_tol=1e-8)

        # At z = 1.0, distance reciprocity / Etherington relation: d_L = (1+z)^2 * d_A
        z = 1.0
        dL = bg.luminosity_distance_dL(z)
        dA = bg.angular_diameter_distance_dA(z)
        chi = bg.comoving_distance_chi(z)

        assert dL > chi > dA > 0.0
        expected_dL = ((1.0 + z) ** 2) * dA
        assert math.isclose(dL, expected_dL, rel_tol=1e-6)

    def test_linear_growth_factor_and_growth_rate(self):
        # In Einstein-de Sitter: D_1(a) = a, and f(a) = dlnD/dlna = 1.0 exactly
        eds = CosmologicalParameters.einstein_de_sitter(h=0.7)
        bg_eds = BackgroundCosmology(eds)

        # Check normalization D_1(1.0) = 1.0
        assert math.isclose(bg_eds.linear_growth_factor(1.0), 1.0, rel_tol=1e-3)
        assert math.isclose(bg_eds.linear_growth_factor(0.5), 0.5, rel_tol=1e-2)
        assert math.isclose(bg_eds.growth_rate_f(1.0), 1.0, rel_tol=1e-2)
        assert math.isclose(bg_eds.growth_rate_f(0.5), 1.0, rel_tol=1e-2)

        # In flat LambdaCDM (Planck 2018): f(z=0) ~ Omega_m^0.55 ~ 0.3111^0.55 ~ 0.526
        planck = CosmologicalParameters()
        bg_planck = BackgroundCosmology(planck)
        f_0 = bg_planck.growth_rate_f(1.0)
        expected_f0 = planck.Omega_m ** 0.55
        assert math.isclose(f_0, expected_f0, rel_tol=0.05)


class TestParticleMeshSolver:
    """Validates Cloud-in-Cell mass deposit, 3D Poisson Green's function, and force interpolation."""

    def test_cic_mass_conservation(self):
        solver = ParticleMeshSolver3D(box_size=100.0, n_grid=32)
        rng = np.random.RandomState(42)

        n_p = 500
        positions = rng.uniform(0.0, 100.0, size=(n_p, 3))
        masses = rng.uniform(1.0, 5.0, size=n_p)
        total_mass = float(np.sum(masses))

        mass_grid, _ = solver.cic_deposit(positions, masses)
        grid_mass = float(np.sum(mass_grid))

        # Mass conservation must hold to machine precision
        assert math.isclose(grid_mass, total_mass, rel_tol=1e-14)

    def test_cic_periodic_wrapping(self):
        solver = ParticleMeshSolver3D(box_size=100.0, n_grid=16)

        # Particles placed right at boundaries and slightly outside
        positions = np.array([
            [0.0, 0.0, 0.0],
            [99.9999, 50.0, 50.0],
            [-0.0001, 10.0, 20.0],
            [100.0001, 80.0, 30.0],
        ])
        masses = np.array([10.0, 10.0, 10.0, 10.0])

        mass_grid, _ = solver.cic_deposit(positions, masses)
        assert math.isclose(float(np.sum(mass_grid)), 40.0, rel_tol=1e-14)

    def test_density_contrast_zero_mean(self):
        solver = ParticleMeshSolver3D(box_size=50.0, n_grid=16)
        rng = np.random.RandomState(123)

        positions = rng.uniform(0.0, 50.0, size=(200, 3))
        masses = rng.uniform(0.5, 2.0, size=200)

        mass_grid, _ = solver.cic_deposit(positions, masses)
        delta_grid = solver.compute_density_contrast(mass_grid)

        # Average density contrast must sum to 0 across grid nodes
        assert math.isclose(float(np.mean(delta_grid)), 0.0, abs_tol=1e-14)

    def test_poisson_solver_greens_function_and_zero_mode(self):
        solver = ParticleMeshSolver3D(box_size=64.0, n_grid=16)
        delta_grid = np.zeros((16, 16, 16), dtype=np.float64)

        # Dipole-like perturbation with zero mean
        delta_grid[4, 8, 8] = 1.0
        delta_grid[12, 8, 8] = -1.0

        phi_grid = solver.solve_poisson(delta_grid, a=1.0)

        # Zero-mode exclusion: mean potential across cosmological periodic volume must be zero
        assert math.isclose(float(np.mean(phi_grid)), 0.0, abs_tol=1e-14)

        # Potential must be antisymmetric reflecting the dipole
        assert math.isclose(phi_grid[4, 8, 8], -phi_grid[12, 8, 8], rel_tol=1e-6)

    def test_momentum_conservation_newton_third_law(self):
        """
        In a periodic Particle-Mesh solver with matched CIC mass deposit and force interpolation,
        the net force on all particles MUST sum to zero: sum m_i * g_i = 0.
        """
        solver = ParticleMeshSolver3D(box_size=100.0, n_grid=32)
        rng = np.random.RandomState(999)

        n_p = 300
        positions = rng.uniform(0.0, 100.0, size=(n_p, 3))
        masses = rng.uniform(1.0, 10.0, size=n_p)

        accelerations, _, _ = solver.compute_forces(positions, masses, a=0.5)

        # Net force vector
        net_force = np.sum(masses[:, None] * accelerations, axis=0)
        net_force_norm = float(np.linalg.norm(net_force))
        total_force_magnitude = float(np.sum(masses * np.linalg.norm(accelerations, axis=1)))

        # Relative momentum violation must be bounded by machine/grid precision
        rel_momentum_error = net_force_norm / max(total_force_magnitude, 1e-10)
        assert rel_momentum_error < 1e-11, f"Momentum conservation violated: {rel_momentum_error}"


class TestSymplecticIntegratorAndInvariants:
    """Validates Symplectic KDK Leapfrog integrator, phase space reversibility, and invariants."""

    def test_symplectic_time_reversibility(self):
        """
        A symplectic integrator is time-reversible: stepping forward by +da then backward by -da
        recovers the initial phase space position to high precision.
        """
        solver = ParticleMeshSolver3D(box_size=50.0, n_grid=16)
        integrator = SymplecticKDKLeapfrog(solver)

        rng = np.random.RandomState(456)
        n_p = 50
        pos_init = rng.uniform(10.0, 40.0, size=(n_p, 3))
        mom_init = rng.normal(0.0, 1.0, size=(n_p, 3))
        masses = np.ones(n_p)

        state_init = SimulationState(
            positions=pos_init.copy(),
            momenta=mom_init.copy(),
            masses=masses,
            a=0.1,
            box_size=50.0,
            n_grid=16,
        )

        delta_a = 0.005

        # Forward step
        state_fwd, _, _ = integrator.step(state_init, delta_a)

        # Backward step (stepping with -delta_a)
        state_rev, _, _ = integrator.step(state_fwd, -delta_a)

        # Reversibility error
        pos_diff = np.abs(state_rev.positions - state_init.positions)
        max_pos_err = float(np.max(pos_diff))
        rel_pos_err = max_pos_err / state_init.box_size

        assert rel_pos_err < 1e-8, f"Symplectic reversibility failed: {rel_pos_err}"

    def test_layzer_irvine_residual_monitoring(self):
        """Tests multi-step evolution and Layzer-Irvine cosmic energy equation residual calculation."""
        solver = ParticleMeshSolver3D(box_size=100.0, n_grid=16)
        integrator = SymplecticKDKLeapfrog(solver)

        rng = np.random.RandomState(789)
        n_p = 100
        positions = rng.uniform(0.0, 100.0, size=(n_p, 3))
        momenta = rng.normal(0.0, 0.5, size=(n_p, 3))
        masses = np.full(n_p, 10.0)

        initial_state = SimulationState(
            positions=positions,
            momenta=momenta,
            masses=masses,
            a=0.1,
            box_size=100.0,
            n_grid=16,
        )

        # Evolve from a=0.1 to a=0.12 in 5 steps
        final_state, history = integrator.evolve(initial_state, target_a=0.12, n_steps=5)

        assert len(history) == 5
        assert math.isclose(final_state.a, 0.12, rel_tol=1e-5)

        # Check mass conservation across all steps
        for diag in history:
            assert diag["mass_error"] < 1e-14

        # Compute Layzer-Irvine residuals
        residuals = SymplecticKDKLeapfrog.compute_layzer_irvine_residuals(history)
        assert len(residuals) == 4
        # In a smoothly evolving cosmologial simulation, residuals should be finite and bounded
        assert np.all(np.isfinite(residuals))


class TestAnalyticalBenchmarks:
    """Validates analytical benchmarks: Zel'dovich pancake and Kepler two-body orbit."""

    def test_kepler_two_body_orbit_invariants(self):
        kepler = KeplerTwoBodySolution(m1=1.0, m2=1.0, separation=2.0, G=1.0)
        t_orbit = kepler.period

        # Trajectory at t = 0 and t = T (full orbit)
        sol_0 = kepler.exact_trajectory(0.0)
        sol_T = kepler.exact_trajectory(t_orbit)

        # After one complete period, positions must return to start
        assert np.allclose(sol_0["pos1"], sol_T["pos1"], atol=1e-12)
        assert np.allclose(sol_0["pos2"], sol_T["pos2"], atol=1e-12)
        assert np.allclose(sol_0["vel1"], sol_T["vel1"], atol=1e-12)
        assert np.allclose(sol_0["vel2"], sol_T["vel2"], atol=1e-12)

    def test_zeldovich_pancake_analytical_convergence(self):
        """
        Zel'dovich pancake test:
        Compares 3D particle-mesh simulation of a 1D plane wave against the exact analytical trajectory.
        Prior to caustic formation (shell crossing), the Zel'dovich solution is exact.
        We assert that the numerical relative position error is < 2.0%.
        """
        box_size = 64.0
        n_grid = 32
        n_p_1d = 32
        a_init = 0.02
        a_target = 0.06
        a_caustic = 0.20

        # Einstein-de Sitter background for clean linear growth D_1(a) = a
        eds = CosmologicalParameters.einstein_de_sitter(h=0.7)
        bg = BackgroundCosmology(eds)

        pancake = ZeldovichPancakeSolution(
            box_size=box_size,
            a_init=a_init,
            a_caustic=a_caustic,
            cosmo=eds,
            background=bg,
        )

        initial_state, q_x = pancake.create_initial_state(
            n_particles_1d=n_p_1d,
            n_grid=n_grid,
            total_mass=1.0e6,
        )

        solver = ParticleMeshSolver3D(box_size=box_size, n_grid=n_grid, cosmo=eds)
        integrator = SymplecticKDKLeapfrog(solver, background=bg)

        # Evolve from a=0.02 to a=0.06 over 10 steps
        n_steps = 10
        final_state, _ = integrator.evolve(initial_state, target_a=a_target, n_steps=n_steps)

        # Evaluate discrepancy against exact Zel'dovich solution
        error_metrics = pancake.evaluate_error(final_state, q_x)

        rel_l1_pos = error_metrics["rel_l1_pos"]
        rel_linf_pos = error_metrics["rel_linf_pos"]

        # Rigorous acceptance thresholds
        assert rel_l1_pos < 0.02, f"Pancake L1 error exceeds 2.0%: {rel_l1_pos:.4f}"
        assert rel_linf_pos < 0.05, f"Pancake Linf error exceeds 5.0%: {rel_linf_pos:.4f}"


class TestGovernanceAndFirewallIntegrity:
    """Validates epistemic firewall, Phase 5 immutable checkpoints, and hardware declarations."""

    def test_phase5_continuation_ledger_exists_and_tracks_open_obligations(self):
        ledger_path = Path("docs/phase6/PHASE5_CONTINUATION_LEDGER.md")
        assert ledger_path.exists(), "Phase 5 continuation ledger must exist"
        content = ledger_path.read_text(encoding="utf-8")

        # Must record frozen checkpoint 0601beb
        assert "0601beb" in content
        # Must track open empirical obligations: H1, H3, H4, H5
        assert "H1" in content
        assert "H3" in content
        assert "H4" in content
        assert "H5" in content
        # Must record H2 falsification preservation
        assert "H2" in content
        # Must enforce Gate 3 quarantine covenant
        assert "Gate 3" in content or "gate3" in content.lower()

    def test_hardware_feasibility_spec_exists(self):
        hw_spec = Path("docs/phase6/PHASE6_HARDWARE_AND_COMPUTE_FEASIBILITY.md")
        assert hw_spec.exists(), "Hardware feasibility spec must exist"
        content = hw_spec.read_text(encoding="utf-8")

        # Must explicitly declare RTX 5050 Laptop GPU and 3-tier compute model
        assert "RTX 5050" in content
        assert "tier 1" in content.lower()
        assert "tier 2" in content.lower()
        assert "tier 3" in content.lower()
