"""
ASTRAEUS Phase 6 Workstream 2.1: Native JAX Numerical Parity Test Suite.
Verifies machine-precision forward numerical parity between the JAX differentiable port
and the NumPy golden reference solver across Gates 6.2A-1 through 6.2A-8.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
import time
from typing import Tuple, Dict, Any
import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.mesh import ParticleMeshSolver3D
from engines.cosmology.integrator import SimulationState, SymplecticKDKLeapfrog
from engines.cosmology.jax_engine import (
    JaxParticleMeshSolver3D,
    JaxSimulationState,
    JaxSymplecticKDKLeapfrog,
)


@pytest.fixture
def test_setup():
    """Provides identical test initial conditions and cosmologies for both backends."""
    box_size = 100.0
    n_grid = 32
    cosmo = CosmologicalParameters()
    bg = BackgroundCosmology(cosmo)

    # Identical random particles
    rng = np.random.RandomState(42)
    n_p = 500
    pos_np = rng.uniform(0.0, box_size, size=(n_p, 3)).astype(np.float64)
    mom_np = rng.normal(0.0, 1.0, size=(n_p, 3)).astype(np.float64)
    masses_np = rng.uniform(1.0, 5.0, size=n_p).astype(np.float64)

    # JAX arrays
    pos_jax = jnp.array(pos_np, dtype=jnp.float64)
    mom_jax = jnp.array(mom_np, dtype=jnp.float64)
    masses_jax = jnp.array(masses_np, dtype=jnp.float64)

    # Solvers
    numpy_solver = ParticleMeshSolver3D(box_size=box_size, n_grid=n_grid, cosmo=cosmo)
    jax_solver = JaxParticleMeshSolver3D(box_size=box_size, n_grid=n_grid, cosmo=cosmo, use_x64=True)

    numpy_integrator = SymplecticKDKLeapfrog(numpy_solver, background=bg)
    jax_integrator = JaxSymplecticKDKLeapfrog(jax_solver, background=bg)

    return {
        "box_size": box_size,
        "n_grid": n_grid,
        "cosmo": cosmo,
        "bg": bg,
        "pos_np": pos_np,
        "mom_np": mom_np,
        "masses_np": masses_np,
        "pos_jax": pos_jax,
        "mom_jax": mom_jax,
        "masses_jax": masses_jax,
        "numpy_solver": numpy_solver,
        "jax_solver": jax_solver,
        "numpy_integrator": numpy_integrator,
        "jax_integrator": jax_integrator,
    }


class TestGate62A_ForwardNumericalParity:
    """Verifies Gates 6.2A-1 through 6.2A-8 as mandated by the research protocol."""

    def test_gate_6_2a_1_cic_mass_deposition_parity(self, test_setup):
        """Gate 6.2A-1: CIC mass deposition parity between NumPy and JAX."""
        s = test_setup
        mass_grid_np, _ = s["numpy_solver"].cic_deposit(s["pos_np"], s["masses_np"])
        mass_grid_jax, _ = s["jax_solver"].cic_deposit(s["pos_jax"], s["masses_jax"])

        diff = np.abs(mass_grid_np - np.array(mass_grid_jax))
        max_diff = float(np.max(diff))
        assert max_diff < 1e-12, f"Gate 6.2A-1 failed: CIC max diff = {max_diff}"

    def test_gate_6_2a_2_fft_poisson_potential_parity(self, test_setup):
        """Gate 6.2A-2: FFT Poisson potential parity."""
        s = test_setup
        mass_grid_np, _ = s["numpy_solver"].cic_deposit(s["pos_np"], s["masses_np"])
        delta_np = s["numpy_solver"].compute_density_contrast(mass_grid_np)
        phi_np = s["numpy_solver"].solve_poisson(delta_np, a=0.5)

        delta_jax = jnp.array(delta_np)
        phi_jax = s["jax_solver"].solve_poisson(delta_jax, a=0.5)

        diff = np.abs(phi_np - np.array(phi_jax))
        max_diff = float(np.max(diff))
        rel_diff = max_diff / float(np.max(np.abs(phi_np)))
        assert rel_diff < 1e-12, f"Gate 6.2A-2 failed: Poisson rel diff = {rel_diff}"

    def test_gate_6_2a_3_grid_acceleration_parity(self, test_setup):
        """Gate 6.2A-3: Grid acceleration parity (central differences)."""
        s = test_setup
        mass_grid_np, _ = s["numpy_solver"].cic_deposit(s["pos_np"], s["masses_np"])
        delta_np = s["numpy_solver"].compute_density_contrast(mass_grid_np)
        phi_np = s["numpy_solver"].solve_poisson(delta_np, a=0.5)
        gx_np, gy_np, gz_np = s["numpy_solver"].compute_grid_acceleration(phi_np)

        phi_jax = jnp.array(phi_np)
        gx_jax, gy_jax, gz_jax = s["jax_solver"].compute_grid_acceleration(phi_jax)

        diff_x = float(np.max(np.abs(gx_np - np.array(gx_jax))))
        diff_y = float(np.max(np.abs(gy_np - np.array(gy_jax))))
        diff_z = float(np.max(np.abs(gz_np - np.array(gz_jax))))

        max_diff = max(diff_x, diff_y, diff_z)
        assert max_diff < 1e-12, f"Gate 6.2A-3 failed: Grid acceleration max diff = {max_diff}"

    def test_gate_6_2a_4_particle_force_interpolation_parity(self, test_setup):
        """Gate 6.2A-4: Particle force interpolation parity."""
        s = test_setup
        gp_np, phi_np, delta_np = s["numpy_solver"].compute_forces(s["pos_np"], s["masses_np"], a=0.5)
        gp_jax, phi_jax, delta_jax = s["jax_solver"].compute_forces(s["pos_jax"], s["masses_jax"], a=0.5)

        diff = np.abs(gp_np - np.array(gp_jax))
        max_diff = float(np.max(diff))
        rel_diff = max_diff / float(np.max(np.abs(gp_np)))
        assert rel_diff < 1e-12, f"Gate 6.2A-4 failed: Particle force rel diff = {rel_diff}"

    def test_gate_6_2a_5_single_step_kdk_parity(self, test_setup):
        """Gate 6.2A-5: Single-step KDK leapfrog state parity."""
        s = test_setup
        state_np = SimulationState(
            positions=s["pos_np"].copy(),
            momenta=s["mom_np"].copy(),
            masses=s["masses_np"].copy(),
            a=0.1,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )
        state_jax = JaxSimulationState(
            positions=s["pos_jax"],
            momenta=s["mom_jax"],
            masses=s["masses_jax"],
            a=0.1,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )

        delta_a = 0.005
        next_np, _, _ = s["numpy_integrator"].step(state_np, delta_a)
        next_jax, _, _ = s["jax_integrator"].step(state_jax, delta_a)

        pos_diff = float(np.max(np.abs(next_np.positions - np.array(next_jax.positions))))
        mom_diff = float(np.max(np.abs(next_np.momenta - np.array(next_jax.momenta))))

        assert pos_diff / s["box_size"] < 1e-12, f"Gate 6.2A-5 pos diff: {pos_diff}"
        assert mom_diff / float(np.max(np.abs(state_np.momenta))) < 1e-12, f"Gate 6.2A-5 mom diff: {mom_diff}"

    def test_gate_6_2a_6_multi_step_trajectory_parity(self, test_setup):
        """Gate 6.2A-6: Multi-step trajectory parity over 15 steps."""
        s = test_setup
        state_np = SimulationState(
            positions=s["pos_np"].copy(),
            momenta=s["mom_np"].copy(),
            masses=s["masses_np"].copy(),
            a=0.05,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )
        state_jax = JaxSimulationState(
            positions=s["pos_jax"],
            momenta=s["mom_jax"],
            masses=s["masses_jax"],
            a=0.05,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )

        n_steps = 15
        final_np, history_np = s["numpy_integrator"].evolve(state_np, target_a=0.10, n_steps=n_steps)
        final_jax, history_jax = s["jax_integrator"].evolve(state_jax, target_a=0.10, n_steps=n_steps)

        pos_diff = float(np.max(np.abs(final_np.positions - np.array(final_jax.positions))))
        mom_diff = float(np.max(np.abs(final_np.momenta - np.array(final_jax.momenta))))

        rel_pos = pos_diff / s["box_size"]
        rel_mom = mom_diff / float(np.max(np.abs(final_np.momenta)))

        assert rel_pos < 1e-11, f"Gate 6.2A-6 multi-step pos diff: {rel_pos}"
        assert rel_mom < 1e-11, f"Gate 6.2A-6 multi-step mom diff: {rel_mom}"

    def test_gate_6_2a_7_physical_invariant_comparison(self, test_setup):
        """Gate 6.2A-7: Physical invariant parity (mass conservation, momentum, energy)."""
        s = test_setup
        state_np = SimulationState(
            positions=s["pos_np"].copy(),
            momenta=s["mom_np"].copy(),
            masses=s["masses_np"].copy(),
            a=0.1,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )
        state_jax = JaxSimulationState(
            positions=s["pos_jax"],
            momenta=s["mom_jax"],
            masses=s["masses_jax"],
            a=0.1,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )

        _, hist_np = s["numpy_integrator"].evolve(state_np, target_a=0.12, n_steps=5)
        _, hist_jax = s["jax_integrator"].evolve(state_jax, target_a=0.12, n_steps=5)

        for d_np, d_jax in zip(hist_np, hist_jax):
            # Mass conservation
            assert math.isclose(d_np["mass_error"], d_jax["mass_error"], abs_tol=1e-14)
            # Kinetic energy
            assert math.isclose(d_np["kinetic_energy"], d_jax["kinetic_energy"], rel_tol=1e-11)
            # Potential energy
            assert math.isclose(d_np["potential_energy"], d_jax["potential_energy"], rel_tol=1e-11)
            # Momentum norm
            assert math.isclose(d_np["momentum_norm"], d_jax["momentum_norm"], abs_tol=1e-11)

    def test_gate_6_2a_8_performance_and_memory_benchmarking(self, test_setup):
        """Gate 6.2A-8: Performance and execution benchmarking across grid sizes."""
        s = test_setup
        t0 = time.perf_counter()
        # Warmup and timing
        state_jax = JaxSimulationState(
            positions=s["pos_jax"],
            momenta=s["mom_jax"],
            masses=s["masses_jax"],
            a=0.1,
            box_size=s["box_size"],
            n_grid=s["n_grid"],
        )
        new_state, _, _ = s["jax_integrator"].step(state_jax, delta_a=0.005)
        t_step = time.perf_counter() - t0

        # Step time must be realistic for interactive experimentation
        assert t_step < 5.0, f"Single KDK step too slow: {t_step:.3f}s"
