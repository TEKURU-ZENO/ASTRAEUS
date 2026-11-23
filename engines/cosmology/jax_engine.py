"""
ASTRAEUS Cosmology Engine: Native JAX Particle-Mesh Solver & Symplectic Integrator.
Workstream 2.1: Native JAX Numerical Port & Parity Architecture.

Provides differentiable tensor operations for CIC mass deposition, FFT Poisson solving,
central difference acceleration, and symplectic KDK leapfrog integration.
Guarantees machine-level forward numerical parity with the NumPy golden reference.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

import jax
# Enable 64-bit floating point precision by default for high-precision scientific computing
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology


@dataclass
class JaxSimulationState:
    """
    Cosmological simulation state with JAX tensor arrays.
    """
    positions: jnp.ndarray       # Comoving coordinates x in [0, L)^3, shape (N_p, 3)
    momenta: jnp.ndarray         # Canonical comoving momenta p = m * a * u, shape (N_p, 3)
    masses: jnp.ndarray          # Particle masses m_p, shape (N_p,)
    a: float                     # Scale factor a
    box_size: float              # Box size L
    n_grid: int                  # Grid dimension N_grid

    @property
    def n_particles(self) -> int:
        return len(self.positions)

    @property
    def total_mass(self) -> float:
        return float(jnp.sum(self.masses))

    @property
    def peculiar_velocities(self) -> jnp.ndarray:
        """Peculiar velocity u = p / (m * a)."""
        m_denom = jnp.maximum(self.masses[:, None] * self.a, 1e-30)
        return self.momenta / m_denom

    @property
    def kinetic_energy(self) -> float:
        """Total comoving kinetic energy T = 0.5 * sum m * u^2."""
        u = self.peculiar_velocities
        u_sq = jnp.sum(u ** 2, axis=1)
        return 0.5 * float(jnp.sum(self.masses * u_sq))

    @property
    def total_momentum(self) -> jnp.ndarray:
        """Total canonical momentum vector P = sum p_i."""
        return jnp.sum(self.momenta, axis=0)

    @property
    def total_momentum_norm(self) -> float:
        return float(jnp.linalg.norm(self.total_momentum))

    def copy(self) -> JaxSimulationState:
        return JaxSimulationState(
            positions=jnp.copy(self.positions),
            momenta=jnp.copy(self.momenta),
            masses=jnp.copy(self.masses),
            a=float(self.a),
            box_size=float(self.box_size),
            n_grid=int(self.n_grid),
        )


class JaxParticleMeshSolver3D:
    """
    Differentiable 3D Particle-Mesh (PM) gravity solver implemented in JAX.
    Strictly preserves comoving momentum and mirrors the NumPy reference implementation.
    """

    def __init__(
        self,
        box_size: float,
        n_grid: int,
        cosmo: Optional[CosmologicalParameters] = None,
        use_x64: bool = True,
    ):
        """
        Args:
            box_size: Comoving box size L in Mpc/h.
            n_grid: Grid resolution along each axis (N_grid x N_grid x N_grid).
            cosmo: Cosmological parameters (default: Planck 2018).
            use_x64: Whether to enforce float64 precision.
        """
        assert box_size > 0, "Box size must be positive"
        assert n_grid >= 4 and (n_grid % 2 == 0), "n_grid must be an even integer >= 4"

        if use_x64:
            jax.config.update("jax_enable_x64", True)

        self.box_size = float(box_size)
        self.n_grid = int(n_grid)
        self.dx = self.box_size / self.n_grid
        self.cosmo = cosmo if cosmo is not None else CosmologicalParameters()
        self.dtype = jnp.float64 if use_x64 else jnp.float32

        self._precompute_fourier_grid()

    def _precompute_fourier_grid(self):
        """Precomputes 3D discrete 7-point Laplacian and Green's function in k-space."""
        N = self.n_grid
        dx = self.dx

        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

        sin_sq_sum = (
            np.sin(KX * dx * 0.5) ** 2
            + np.sin(KY * dx * 0.5) ** 2
            + np.sin(KZ * dx * 0.5) ** 2
        )
        lap_np = -4.0 / (dx ** 2) * sin_sq_sum

        inv_lap_np = np.zeros_like(lap_np)
        mask = lap_np != 0
        inv_lap_np[mask] = 1.0 / lap_np[mask]

        self.laplacian_k = jnp.array(lap_np, dtype=self.dtype)
        self.inv_laplacian_k = jnp.array(inv_lap_np, dtype=self.dtype)

    def cic_deposit(
        self, positions: jnp.ndarray, masses: jnp.ndarray
    ) -> Tuple[jnp.ndarray, list]:
        """
        Differentiable Cloud-in-Cell (CIC) trilinear mass assignment from particles to grid.
        Returns:
            mass_grid: (N, N, N) accumulated cell masses.
            corners: Precomputed corner indices and weights for matching force interpolation.
        """
        N = self.n_grid
        L = self.box_size
        dx = self.dx

        # Periodic wrapping
        pos_wrapped = jnp.mod(positions, L)
        x_norm = pos_wrapped / dx

        # Grid cell indices and offsets
        i = jnp.floor(x_norm).astype(jnp.int64)
        d = x_norm - i

        i0 = i[:, 0] % N
        j0 = i[:, 1] % N
        k0 = i[:, 2] % N

        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        k1 = (k0 + 1) % N

        dx0 = 1.0 - d[:, 0]
        dx1 = d[:, 0]
        dy0 = 1.0 - d[:, 1]
        dy1 = d[:, 1]
        dz0 = 1.0 - d[:, 2]
        dz1 = d[:, 2]

        corners = [
            (i0, j0, k0, dx0 * dy0 * dz0),
            (i1, j0, k0, dx1 * dy0 * dz0),
            (i0, j1, k0, dx0 * dy1 * dz0),
            (i1, j1, k0, dx1 * dy1 * dz0),
            (i0, j0, k1, dx0 * dy0 * dz1),
            (i1, j0, k1, dx1 * dy0 * dz1),
            (i0, j1, k1, dx0 * dy1 * dz1),
            (i1, j1, k1, dx1 * dy1 * dz1),
        ]

        mass_grid = jnp.zeros((N, N, N), dtype=self.dtype)
        for ci, cj, ck, w in corners:
            mass_grid = mass_grid.at[ci, cj, ck].add(masses * w)

        return mass_grid, corners

    def compute_density_contrast(
        self, mass_grid: jnp.ndarray, total_mass: Optional[float] = None
    ) -> jnp.ndarray:
        """Computes dimensionless density contrast delta = rho / rho_bar - 1."""
        if total_mass is None:
            total_mass = float(jnp.sum(mass_grid))

        mean_mass_per_cell = total_mass / (self.n_grid ** 3)
        delta_grid = (mass_grid / mean_mass_per_cell) - 1.0
        return delta_grid

    def solve_poisson(
        self, delta_grid: jnp.ndarray, a: float = 1.0, G_source_coeff: Optional[float] = None
    ) -> jnp.ndarray:
        """
        Solves comoving Poisson equation nabla^2 phi = (3/2) * Omega_m * H0^2 * (1/a) * delta
        using 3D FFT and the 7-point discrete Green's function.
        """
        if G_source_coeff is None:
            G_source_coeff = 1.5 * self.cosmo.Omega_m * (self.cosmo.H0 ** 2) / max(a, 1e-10)

        delta_k = jnp.fft.fftn(delta_grid)
        phi_k = delta_k * self.inv_laplacian_k * G_source_coeff
        # Zero-mode exclusion (zero mean cosmological potential)
        phi_k = phi_k.at[0, 0, 0].set(0.0)

        phi_grid = jnp.fft.ifftn(phi_k).real
        return phi_grid

    def compute_grid_acceleration(
        self, phi_grid: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Computes grid accelerations via 2nd-order central differences."""
        dx = self.dx
        gx = -(jnp.roll(phi_grid, -1, axis=0) - jnp.roll(phi_grid, 1, axis=0)) / (2.0 * dx)
        gy = -(jnp.roll(phi_grid, -1, axis=1) - jnp.roll(phi_grid, 1, axis=1)) / (2.0 * dx)
        gz = -(jnp.roll(phi_grid, -1, axis=2) - jnp.roll(phi_grid, 1, axis=2)) / (2.0 * dx)
        return gx, gy, gz

    def interpolate_particle_acceleration(
        self,
        gx: jnp.ndarray,
        gy: jnp.ndarray,
        gz: jnp.ndarray,
        corners: list,
        n_particles: int,
    ) -> jnp.ndarray:
        """Interpolates grid accelerations to particles using identical CIC weights."""
        gp_x = jnp.zeros(n_particles, dtype=self.dtype)
        gp_y = jnp.zeros(n_particles, dtype=self.dtype)
        gp_z = jnp.zeros(n_particles, dtype=self.dtype)

        for ci, cj, ck, w in corners:
            gp_x = gp_x + gx[ci, cj, ck] * w
            gp_y = gp_y + gy[ci, cj, ck] * w
            gp_z = gp_z + gz[ci, cj, ck] * w

        return jnp.stack([gp_x, gp_y, gp_z], axis=1)

    def compute_forces(
        self,
        positions: jnp.ndarray,
        masses: jnp.ndarray,
        a: float = 1.0,
        G_source_coeff: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Complete forward Particle-Mesh force pipeline:
        Particles -> CIC deposit -> delta -> Poisson solve -> Central diff -> CIC interpolate.
        """
        n_p = len(positions)
        mass_grid, corners = self.cic_deposit(positions, masses)
        total_mass = float(jnp.sum(masses))
        delta_grid = self.compute_density_contrast(mass_grid, total_mass=total_mass)
        phi_grid = self.solve_poisson(delta_grid, a=a, G_source_coeff=G_source_coeff)
        gx, gy, gz = self.compute_grid_acceleration(phi_grid)
        gp = self.interpolate_particle_acceleration(gx, gy, gz, corners, n_particles=n_p)
        return gp, phi_grid, delta_grid

    def compute_potential_energy(
        self, mass_grid: jnp.ndarray, phi_grid: jnp.ndarray
    ) -> float:
        """Computes gravitational potential energy U = 0.5 * sum_cell M_cell * phi_cell."""
        U = 0.5 * float(jnp.sum(mass_grid * phi_grid))
        return U


class JaxSymplecticKDKLeapfrog:
    """
    Symplectic Kick-Drift-Kick (KDK) leapfrog time integrator in JAX.
    Provides identical dynamics to NumPy reference.
    """

    def __init__(
        self,
        mesh_solver: JaxParticleMeshSolver3D,
        background: Optional[BackgroundCosmology] = None,
    ):
        self.mesh = mesh_solver
        self.cosmo = mesh_solver.cosmo
        self.background = background if background is not None else BackgroundCosmology(self.cosmo)

    def _force_factor(self, a: float) -> float:
        """dp/da = (dp/dt) / (da/dt) = m * g / H(a) -> force_factor = 1.0 / H(a)."""
        h_val = self.background.H(a)
        return 1.0 / max(h_val, 1e-10)

    def _drift_factor(self, a_half: float) -> float:
        """dx/da = p / (m * a^3 * H(a)) -> drift_factor = 1.0 / (a^3 * H(a))."""
        h_val = self.background.H(a_half)
        return 1.0 / max((a_half ** 3) * h_val, 1e-20)

    def step(
        self,
        state: JaxSimulationState,
        delta_a: float,
        cached_accel: Optional[jnp.ndarray] = None,
    ) -> Tuple[JaxSimulationState, jnp.ndarray, Dict[str, Any]]:
        """
        Advances the simulation state by delta_a using Kick-Drift-Kick (KDK) leapfrog.
        """
        a_n = state.a
        a_half = a_n + 0.5 * delta_a
        a_next = a_n + delta_a

        # 1. Compute forces at current step if not cached
        if cached_accel is not None:
            g_n = cached_accel
        else:
            g_n, phi_n, _ = self.mesh.compute_forces(state.positions, state.masses, a=a_n)

        # 2. First Half-Kick: p_{n+1/2} = p_n + dp/da * (delta_a / 2)
        force_fact_n = self._force_factor(a_n)
        dp_da_n = state.masses[:, None] * g_n * force_fact_n
        p_half = state.momenta + dp_da_n * (0.5 * delta_a)

        # 3. Full Drift: x_{n+1} = x_n + dx/da * delta_a
        drift_fact = self._drift_factor(a_half)
        dx_da = (p_half / state.masses[:, None]) * drift_fact
        x_next = jnp.mod(state.positions + dx_da * delta_a, state.box_size)

        # 4. Recompute forces at x_{n+1}, a_{n+1}
        g_next, phi_next, delta_next = self.mesh.compute_forces(x_next, state.masses, a=a_next)

        # 5. Second Half-Kick: p_{n+1} = p_{n+1/2} + dp/da * (delta_a / 2)
        force_fact_next = self._force_factor(a_next)
        dp_da_next = state.masses[:, None] * g_next * force_fact_next
        p_next = p_half + dp_da_next * (0.5 * delta_a)

        new_state = JaxSimulationState(
            positions=x_next,
            momenta=p_next,
            masses=state.masses,
            a=a_next,
            box_size=state.box_size,
            n_grid=state.n_grid,
        )

        mass_grid, _ = self.mesh.cic_deposit(x_next, state.masses)
        U_next = self.mesh.compute_potential_energy(mass_grid, phi_next)
        T_next = new_state.kinetic_energy

        diagnostics = {
            "a": a_next,
            "kinetic_energy": T_next,
            "potential_energy": U_next,
            "total_energy": T_next + U_next,
            "momentum_norm": new_state.total_momentum_norm,
            "mass_error": abs(float(jnp.sum(mass_grid)) - state.total_mass) / state.total_mass,
            "delta_variance": float(jnp.var(delta_next)),
            "delta_max": float(jnp.max(delta_next)),
            "delta_min": float(jnp.min(delta_next)),
        }

        return new_state, g_next, diagnostics

    def evolve(
        self, state: JaxSimulationState, target_a: float, n_steps: int
    ) -> Tuple[JaxSimulationState, List[Dict[str, Any]]]:
        """Evolves state from state.a to target_a over n_steps."""
        assert target_a > state.a, "target_a must be greater than state.a"
        assert n_steps >= 1, "n_steps must be at least 1"

        delta_a = (target_a - state.a) / n_steps
        current_state = state.copy()
        history: List[Dict[str, Any]] = []

        g_cached = None
        for step_idx in range(n_steps):
            current_state, g_cached, diag = self.step(current_state, delta_a, cached_accel=g_cached)
            diag["step"] = step_idx + 1
            history.append(diag)

        return current_state, history
