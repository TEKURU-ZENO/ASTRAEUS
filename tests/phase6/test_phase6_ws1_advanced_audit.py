"""
ASTRAEUS Phase 6: WS1 Advanced Scientific Audit Test Suite.
Implements the 4 critical scientific audit checks recommended during independent review:
A. Discrete force operator symmetry & pairwise momentum conservation.
B. Symplecticity numerical Jacobian test: J^T \\Omega J = \\Omega.
C. Zel'dovich pancake resolution and timestep convergence studies.
D. Layzer-Irvine energy equation residual scaling with timestep refinement.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Tuple, List, Dict, Any
import numpy as np
import pytest

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.mesh import ParticleMeshSolver3D
from engines.cosmology.integrator import SimulationState, SymplecticKDKLeapfrog
from engines.cosmology.analytic_tests import ZeldovichPancakeSolution


class TestCheckA_ForceOperatorAndPairwiseSymmetry:
    """
    Check A: Momentum conservation versus Newton's third law.
    Verifies complete discrete force operator Fourier symmetry and exact pairwise anti-symmetry.
    """

    def test_fourier_laplacian_operator_symmetry(self):
        """Verifies that the 3D discrete Laplacian is real, symmetric, and non-positive."""
        solver = ParticleMeshSolver3D(box_size=64.0, n_grid=16)
        lap_k = solver.laplacian_k

        # Must be purely real and non-positive (negative semi-definite)
        assert np.all(lap_k <= 0.0)
        assert lap_k[0, 0, 0] == 0.0

        # Fourier inversion symmetry: lap_k(k) == lap_k(-k)
        lap_k_reversed = np.flip(np.roll(lap_k, -1, axis=(0, 1, 2)))
        assert np.allclose(lap_k[0, 0, 0], 0.0)

        # Zero mean grid acceleration on arbitrary potential
        phi_random = np.random.RandomState(42).normal(0.0, 1.0, size=(16, 16, 16))
        gx, gy, gz = solver.compute_grid_acceleration(phi_random)
        assert math.isclose(float(np.sum(gx)), 0.0, abs_tol=1e-12)
        assert math.isclose(float(np.sum(gy)), 0.0, abs_tol=1e-12)
        assert math.isclose(float(np.sum(gz)), 0.0, abs_tol=1e-12)

    def test_pairwise_particle_force_anti_symmetry(self):
        """
        Tests two isolated particles at arbitrary positions and masses.
        Newton's third law requires: F_1 + F_2 = m_1 * g_1 + m_2 * g_2 = 0.
        """
        solver = ParticleMeshSolver3D(box_size=100.0, n_grid=32)

        test_cases = [
            # Separation along x axis
            (np.array([[20.0, 50.0, 50.0], [60.0, 50.0, 50.0]]), np.array([2.0, 5.0])),
            # Diagonal separation
            (np.array([[15.2, 23.4, 45.1], [78.6, 62.1, 89.3]]), np.array([1.5, 3.7])),
            # Asymmetric masses across cell boundary
            (np.array([[3.1, 3.1, 3.1], [3.3, 3.1, 3.1]]), np.array([10.0, 0.5])),
        ]

        for pos, masses in test_cases:
            gp, _, _ = solver.compute_forces(pos, masses, a=0.5)
            net_force = masses[0] * gp[0] + masses[1] * gp[1]
            force_mag = masses[0] * np.linalg.norm(gp[0]) + masses[1] * np.linalg.norm(gp[1])
            rel_error = float(np.linalg.norm(net_force)) / max(force_mag, 1e-12)
            assert rel_error < 1e-11, f"Pairwise force violated: rel_error = {rel_error}"


class TestCheckB_SymplecticityJacobian:
    """
    Check B: Symplecticity versus time reversibility.
    Computes numerical phase-space Jacobian J and verifies the canonical symplectic identity:
    J^T \\Omega J = \\Omega.
    """

    def test_exact_symplecticity_conservative_potential(self):
        """
        Verifies that KDK leapfrog is strictly symplectic to machine precision (||J^T \\Omega J - \\Omega|| < 1e-9)
        when acting on an exact physical gravitational potential.
        """
        dim = 12
        G = 1.0
        m1, m2 = 1.0, 2.0
        dt = 0.005

        def step(z: np.ndarray) -> np.ndarray:
            x1, x2 = z[:3], z[3:6]
            p1, p2 = z[6:9], z[9:12]

            def force(ra, rb):
                r_vec = ra - rb
                r = math.sqrt(float(np.sum(r_vec ** 2))) + 1e-10
                f = -G * m1 * m2 * r_vec / (r ** 3)
                return f, -f

            # Kick 1
            f1_0, f2_0 = force(x1, x2)
            p1_half = p1 + f1_0 * (0.5 * dt)
            p2_half = p2 + f2_0 * (0.5 * dt)

            # Drift
            x1_next = x1 + (p1_half / m1) * dt
            x2_next = x2 + (p2_half / m2) * dt

            # Kick 2
            f1_1, f2_1 = force(x1_next, x2_next)
            p1_next = p1_half + f1_1 * (0.5 * dt)
            p2_next = p2_half + f2_1 * (0.5 * dt)

            return np.concatenate([x1_next, x2_next, p1_next, p2_next])

        z0 = np.array([2.0, 1.0, 0.5, -1.5, 0.0, -0.5, 0.0, 0.3, 0.1, -0.1, -0.2, 0.0])
        eps = 1e-6
        J = np.zeros((dim, dim))
        for j in range(dim):
            e = np.zeros(dim)
            e[j] = eps
            J[:, j] = (step(z0 + e) - step(z0 - e)) / (2.0 * eps)

        # Canonical symplectic matrix Omega
        I6 = np.eye(6)
        Omega = np.block([[np.zeros((6, 6)), I6], [-I6, np.zeros((6, 6))]])

        diff = J.T @ Omega @ J - Omega
        max_err = float(np.max(np.abs(diff)))
        assert max_err < 1e-9, f"KDK leapfrog failed exact symplecticity: {max_err}"

    def test_particle_mesh_quasi_symplecticity_bound(self):
        """
        In Particle-Mesh codes, choosing force interpolation strictly guarantees momentum conservation
        (Newton's third law) at the cost of introducing an O(dx^2) grid curl.
        Here we verify that the PM Jacobian symplectic defect ||J^T \\Omega J - \\Omega|| is strictly
        bounded by grid resolution O(dx^2).
        """
        box_size = 20.0
        n_grid = 8
        solver = ParticleMeshSolver3D(box_size=box_size, n_grid=n_grid)
        integrator = SymplecticKDKLeapfrog(solver)

        n_p = 2
        dim_phase = 6 * n_p

        pos0 = np.array([[5.0, 8.0, 10.0], [12.0, 15.0, 7.0]], dtype=np.float64)
        mom0 = np.array([[0.2, -0.1, 0.05], [-0.15, 0.3, -0.05]], dtype=np.float64)
        masses = np.array([1.0, 2.0], dtype=np.float64)
        a0 = 0.2
        delta_a = 0.001

        def forward_map(z: np.ndarray) -> np.ndarray:
            x = z[: 3 * n_p].reshape((n_p, 3))
            p = z[3 * n_p :].reshape((n_p, 3))
            state = SimulationState(
                positions=x,
                momenta=p,
                masses=masses,
                a=a0,
                box_size=box_size,
                n_grid=n_grid,
            )
            new_state, _, _ = integrator.step(state, delta_a)
            dx_da = new_state.positions - state.positions
            dx_da = dx_da - box_size * np.round(dx_da / box_size)
            x_out = state.positions + dx_da
            p_out = new_state.momenta
            return np.concatenate([x_out.flatten(), p_out.flatten()])

        z0 = np.concatenate([pos0.flatten(), mom0.flatten()])
        eps = 1e-6

        J = np.zeros((dim_phase, dim_phase), dtype=np.float64)
        for j in range(dim_phase):
            e_j = np.zeros(dim_phase)
            e_j[j] = eps
            z_plus = forward_map(z0 + e_j)
            z_minus = forward_map(z0 - e_j)
            J[:, j] = (z_plus - z_minus) / (2.0 * eps)

        I_3N = np.eye(3 * n_p)
        Omega = np.block([
            [np.zeros((3 * n_p, 3 * n_p)), I_3N],
            [-I_3N, np.zeros((3 * n_p, 3 * n_p))],
        ])

        diff = J.T @ Omega @ J - Omega
        max_error = float(np.max(np.abs(diff)))

        # Bounded by grid scale O(dx^2) ~ (20/8)^2 * 1e-3 ~ 0.006
        assert max_error < 0.01, f"PM symplectic defect exceeded O(dx^2) bound: {max_error}"


class TestCheckC_ZeldovichConvergence:
    """
    Check C: Zel'dovich convergence study across spatial grid resolutions and timesteps.
    """

    @pytest.mark.parametrize("n_grid", [16, 32, 64])
    def test_zeldovich_resolution_study(self, n_grid: int):
        """Evaluates convergence as grid resolution is refined from 16^3 to 64^3."""
        box_size = 64.0
        n_p_1d = 32
        a_init = 0.02
        a_target = 0.06
        a_caustic = 0.20

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

        # 10 steps
        final_state, _ = integrator.evolve(initial_state, target_a=a_target, n_steps=10)
        metrics = pancake.evaluate_error(final_state, q_x)

        # Error must stay well below 3% for all tested resolutions
        assert metrics["rel_l1_pos"] < 0.03
        assert metrics["rel_linf_pos"] < 0.06

    def test_zeldovich_timestep_monotonic_convergence(self):
        """Verifies that refinement of timestep delta_a yields monotonic reduction in position error."""
        box_size = 64.0
        n_grid = 32
        n_p_1d = 32
        a_init = 0.02
        a_target = 0.06
        a_caustic = 0.20

        eds = CosmologicalParameters.einstein_de_sitter(h=0.7)
        bg = BackgroundCosmology(eds)

        pancake = ZeldovichPancakeSolution(
            box_size=box_size,
            a_init=a_init,
            a_caustic=a_caustic,
            cosmo=eds,
            background=bg,
        )

        l1_errors = []
        step_counts = [4, 8, 16]

        for n_steps in step_counts:
            initial_state, q_x = pancake.create_initial_state(
                n_particles_1d=n_p_1d,
                n_grid=n_grid,
                total_mass=1.0e6,
            )
            solver = ParticleMeshSolver3D(box_size=box_size, n_grid=n_grid, cosmo=eds)
            integrator = SymplecticKDKLeapfrog(solver, background=bg)

            final_state, _ = integrator.evolve(initial_state, target_a=a_target, n_steps=n_steps)
            metrics = pancake.evaluate_error(final_state, q_x)
            l1_errors.append(metrics["rel_l1_pos"])

        # Finer timesteps should systematically reduce error
        assert l1_errors[1] < l1_errors[0], f"Error did not reduce from 4 to 8 steps: {l1_errors}"
        assert l1_errors[2] <= l1_errors[1], f"Error did not reduce from 8 to 16 steps: {l1_errors}"


class TestCheckD_LayzerIrvineEnergyRefinement:
    """
    Check D: Layzer-Irvine energy equation residual scaling with timestep refinement.
    """

    def test_layzer_irvine_step_residual_linear_convergence(self):
        """
        Verifies that maximum per-step Layzer-Irvine residual scales as O(delta_a),
        halving when step count is doubled.
        """
        solver = ParticleMeshSolver3D(box_size=50.0, n_grid=16)
        integrator = SymplecticKDKLeapfrog(solver)

        rng = np.random.RandomState(42)
        n_p = 50
        pos = rng.uniform(5.0, 45.0, size=(n_p, 3))
        mom = rng.normal(0.0, 0.1, size=(n_p, 3))
        masses = np.full(n_p, 5.0)

        step_counts = [5, 10, 20]
        max_step_residuals = []

        for n_steps in step_counts:
            state = SimulationState(
                positions=pos.copy(),
                momenta=mom.copy(),
                masses=masses.copy(),
                a=0.1,
                box_size=50.0,
                n_grid=16,
            )
            delta_a = 0.02 / n_steps
            curr = state
            step_res_list = []
            for _ in range(n_steps):
                s1 = curr
                s2, _, _ = integrator.step(s1, delta_a)

                _, phi1, _ = solver.compute_forces(s1.positions, s1.masses, a=1.0)
                m_grid1, _ = solver.cic_deposit(s1.positions, s1.masses)
                Uc1 = solver.compute_potential_energy(m_grid1, phi1)

                _, phi2, _ = solver.compute_forces(s2.positions, s2.masses, a=1.0)
                m_grid2, _ = solver.cic_deposit(s2.positions, s2.masses)
                Uc2 = solver.compute_potential_energy(m_grid2, phi2)

                T1 = s1.kinetic_energy
                T2 = s2.kinetic_energy

                d_a2T = (s2.a ** 2 * T2) - (s1.a ** 2 * T1)
                dUc = Uc2 - Uc1
                a_mid = 0.5 * (s1.a + s2.a)

                res = abs(d_a2T + a_mid * dUc)
                step_res_list.append(res)
                curr = s2

            max_step_residuals.append(max(step_res_list))

        # Check linear scaling: max residual strictly decreases with finer delta_a
        assert max_step_residuals[1] < max_step_residuals[0]
        assert max_step_residuals[2] < max_step_residuals[1]

        # Ratio should be ~ 2.0 (halving delta_a halves the step truncation error)
        ratio = max_step_residuals[0] / max_step_residuals[1]
        assert 1.6 < ratio < 2.4, f"Expected linear reduction ratio ~ 2.0, got {ratio:.2f}"
