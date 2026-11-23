"""
ASTRAEUS Cosmology Engine: Symplectic KDK Leapfrog Integrator & Physical Invariants.
Workstream 1: Scientific Computing Foundation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.mesh import ParticleMeshSolver3D


@dataclass
class SimulationState:
    """
    Cosmological N-body simulation state at scale factor a.
    """
    positions: np.ndarray       # Comoving coordinates x in [0, L)^3, shape (N_p, 3)
    momenta: np.ndarray         # Canonical comoving momenta p = m * a^2 * x_dot = m * a * u, shape (N_p, 3)
    masses: np.ndarray          # Particle masses m_p, shape (N_p,)
    a: float                    # Scale factor a
    box_size: float             # Box size L
    n_grid: int                 # Grid dimension N_grid

    @property
    def n_particles(self) -> int:
        return len(self.positions)

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def peculiar_velocities(self) -> np.ndarray:
        """Peculiar velocity u = dx/dt * a = p / (m * a)."""
        m_denom = np.maximum(self.masses[:, None] * self.a, 1e-30)
        return self.momenta / m_denom

    @property
    def kinetic_energy(self) -> float:
        """Total comoving kinetic energy T = 0.5 * sum m * u^2."""
        u = self.peculiar_velocities
        u_sq = np.sum(u ** 2, axis=1)
        return 0.5 * float(np.sum(self.masses * u_sq))

    @property
    def total_momentum(self) -> np.ndarray:
        """Total canonical momentum vector P = sum p_i."""
        return np.sum(self.momenta, axis=0)

    @property
    def total_momentum_norm(self) -> float:
        return float(np.linalg.norm(self.total_momentum))

    def copy(self) -> SimulationState:
        return SimulationState(
            positions=self.positions.copy(),
            momenta=self.momenta.copy(),
            masses=self.masses.copy(),
            a=float(self.a),
            box_size=float(self.box_size),
            n_grid=int(self.n_grid),
        )


class SymplecticKDKLeapfrog:
    """
    Symplectic Kick-Drift-Kick (KDK) leapfrog time integrator in comoving FLRW coordinates.
    Strictly preserves phase-space volume (Liouville's theorem) and monitors comoving momentum
    and the cosmic energy equation (Layzer-Irvine equation).
    """

    def __init__(
        self,
        mesh_solver: ParticleMeshSolver3D,
        background: Optional[BackgroundCosmology] = None,
    ):
        self.mesh = mesh_solver
        self.cosmo = mesh_solver.cosmo
        self.background = background if background is not None else BackgroundCosmology(self.cosmo)

    def _force_factor(self, a: float) -> float:
        """
        Conversion factor from acceleration g to dp/da:
        dp/da = (dp/dt) / (da/dt) = (m * a * g) / (a * H(a)) = m * g / H(a).
        Here we define force_factor = 1.0 / H(a) so that dp/da = m * g * force_factor.
        """
        h_val = self.background.H(a)
        return 1.0 / max(h_val, 1e-10)

    def _drift_factor(self, a_half: float) -> float:
        """
        Conversion factor from momentum p to dx/da:
        dx/da = (dx/dt) / (da/dt) = (p / (m * a^2)) / (a * H(a)) = p / (m * a^3 * H(a)).
        Here drift_factor = 1.0 / (a^3 * H(a)).
        """
        h_val = self.background.H(a_half)
        return 1.0 / max((a_half ** 3) * h_val, 1e-20)

    def step(
        self, state: SimulationState, delta_a: float, cached_accel: Optional[np.ndarray] = None
    ) -> Tuple[SimulationState, np.ndarray, Dict[str, Any]]:
        """
        Advances the simulation state by delta_a using Kick-Drift-Kick (KDK) leapfrog.

        Returns:
            new_state: Updated SimulationState at a + delta_a
            new_accel: Acceleration at final positions (can be cached for next step)
            diagnostics: Step metrics including energy and momentum invariants
        """
        a_n = state.a
        a_half = a_n + 0.5 * delta_a
        a_next = a_n + delta_a

        # 1. Compute forces at current step if not cached
        if cached_accel is not None:
            g_n = cached_accel
        else:
            g_n, phi_n, delta_n = self.mesh.compute_forces(state.positions, state.masses, a=a_n)

        # 2. First Half-Kick: p_{n+1/2} = p_n + dp/da * (delta_a / 2)
        # dp/da = m * g * (1 / H(a))
        force_fact_n = self._force_factor(a_n)
        dp_da_n = state.masses[:, None] * g_n * force_fact_n
        p_half = state.momenta + dp_da_n * (0.5 * delta_a)

        # 3. Full Drift: x_{n+1} = x_n + dx/da * delta_a
        # dx/da = p_{n+1/2} / (m * a_{n+1/2}^3 * H(a_{n+1/2}))
        drift_fact = self._drift_factor(a_half)
        dx_da = (p_half / state.masses[:, None]) * drift_fact
        x_next = np.mod(state.positions + dx_da * delta_a, state.box_size)

        # 4. Recompute forces at x_{n+1}, a_{n+1}
        g_next, phi_next, delta_next = self.mesh.compute_forces(x_next, state.masses, a=a_next)

        # 5. Second Half-Kick: p_{n+1} = p_{n+1/2} + dp/da * (delta_a / 2)
        force_fact_next = self._force_factor(a_next)
        dp_da_next = state.masses[:, None] * g_next * force_fact_next
        p_next = p_half + dp_da_next * (0.5 * delta_a)

        # Assemble new state
        new_state = SimulationState(
            positions=x_next,
            momenta=p_next,
            masses=state.masses.copy(),
            a=a_next,
            box_size=state.box_size,
            n_grid=state.n_grid,
        )

        # Compute step diagnostics
        mass_grid, _ = self.mesh.cic_deposit(x_next, state.masses)
        U_next = self.mesh.compute_potential_energy(mass_grid, phi_next)
        T_next = new_state.kinetic_energy

        diagnostics = {
            "a": a_next,
            "kinetic_energy": T_next,
            "potential_energy": U_next,
            "total_energy": T_next + U_next,
            "momentum_norm": new_state.total_momentum_norm,
            "mass_error": abs(float(np.sum(mass_grid)) - state.total_mass) / state.total_mass,
            "delta_variance": float(np.var(delta_next)),
            "delta_max": float(np.max(delta_next)),
            "delta_min": float(np.min(delta_next)),
        }

        return new_state, g_next, diagnostics

    def evolve(
        self, state: SimulationState, target_a: float, n_steps: int
    ) -> Tuple[SimulationState, List[Dict[str, Any]]]:
        """
        Evolves state from state.a to target_a over n_steps equal intervals.
        """
        assert target_a > state.a, "target_a must be greater than initial a"
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

    @staticmethod
    def compute_layzer_irvine_residuals(history: List[Dict[str, Any]]) -> np.ndarray:
        """
        Computes Layzer-Irvine cosmic energy equation discrete residuals:
        Residual_n = (E_{n+1} - E_n) + (a_{n+1} - a_n) * (2*T_{n+1/2} + U_{n+1/2}) / a_{n+1/2}
        """
        if len(history) < 2:
            return np.array([0.0])

        residuals = []
        for i in range(len(history) - 1):
            s1 = history[i]
            s2 = history[i + 1]

            da = s2["a"] - s1["a"]
            a_mid = 0.5 * (s1["a"] + s2["a"])

            T_mid = 0.5 * (s1["kinetic_energy"] + s2["kinetic_energy"])
            U_mid = 0.5 * (s1["potential_energy"] + s2["potential_energy"])

            dE = s2["total_energy"] - s1["total_energy"]
            li_term = da * (2.0 * T_mid + U_mid) / a_mid

            res = abs(dE + li_term) / max(abs(s1["total_energy"]), 1e-15)
            residuals.append(res)

        return np.array(residuals)
