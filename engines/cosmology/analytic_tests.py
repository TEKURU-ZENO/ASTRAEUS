"""
ASTRAEUS Cosmology Engine: Analytical Benchmarks & Exact Reference Solutions.
Workstream 1: Scientific Computing Foundation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.integrator import SimulationState


class ZeldovichPancakeSolution:
    """
    Exact analytical solution for the 1D Zel'dovich pancake collapse in an expanding FLRW universe.
    
    In 1D plane symmetry, the Zel'dovich approximation is an EXACT solution of the
    collisionless Vlasov-Poisson system prior to caustic formation (shell crossing) at a = a_c.
    
    Analytical Trajectory:
        x(q, a) = q - (D_1(a) / D_1(a_c)) * (1 / k) * sin(k * q)
        u(q, a) = -a * H(a) * f(a) * (D_1(a) / D_1(a_c)) * (1 / k) * sin(k * q)
        p(q, a) = m * a * u(q, a)
        1 + delta(q, a) = 1 / [1 - (D_1(a) / D_1(a_c)) * cos(k * q)]
    
    where k = 2 * pi / L, q is the Lagrangian coordinate, and D_1(a) is the linear growth factor.
    """

    def __init__(
        self,
        box_size: float = 100.0,
        a_init: float = 0.02,
        a_caustic: float = 0.20,
        cosmo: Optional[CosmologicalParameters] = None,
        background: Optional[BackgroundCosmology] = None,
    ):
        """
        Args:
            box_size: Comoving box size L in Mpc/h.
            a_init: Initial scale factor.
            a_caustic: Scale factor at which caustic formation / shell crossing occurs.
            cosmo: Cosmological parameters.
            background: Background cosmology instance.
        """
        assert box_size > 0, "Box size must be positive"
        assert 0 < a_init < a_caustic, "Must have 0 < a_init < a_caustic"

        self.box_size = float(box_size)
        self.a_init = float(a_init)
        self.a_caustic = float(a_caustic)
        self.k = 2.0 * np.pi / self.box_size

        self.cosmo = cosmo if cosmo is not None else CosmologicalParameters()
        self.background = background if background is not None else BackgroundCosmology(self.cosmo)

        # Precompute growth factor at caustic
        self.D_c = self.background.linear_growth_factor(self.a_caustic)

    def growth_ratio(self, a: float) -> float:
        """Returns D_1(a) / D_1(a_c)."""
        return self.background.linear_growth_factor(a) / self.D_c

    def exact_position(self, q: np.ndarray, a: float) -> np.ndarray:
        """
        Exact Eulerian comoving position x(q, a) for Lagrangian coordinate q.
        """
        growth_fac = self.growth_ratio(a)
        disp = (growth_fac / self.k) * np.sin(self.k * q)
        x = q - disp
        return np.mod(x, self.box_size)

    def exact_velocity(self, q: np.ndarray, a: float) -> np.ndarray:
        """
        Exact comoving peculiar velocity u(q, a) = a * dx/dt.
        """
        growth_fac = self.growth_ratio(a)
        f_val = self.background.growth_rate_f(a)
        H_val = self.background.H(a)
        u = -a * H_val * f_val * (growth_fac / self.k) * np.sin(self.k * q)
        return u

    def exact_momentum(self, q: np.ndarray, a: float, mass: float) -> np.ndarray:
        """
        Exact canonical momentum p(q, a) = m * a * u.
        """
        u = self.exact_velocity(q, a)
        return mass * a * u

    def exact_density_contrast(self, q: np.ndarray, a: float) -> np.ndarray:
        """
        Exact density contrast delta(q, a) prior to shell crossing.
        """
        growth_fac = self.growth_ratio(a)
        denom = 1.0 - growth_fac * np.cos(self.k * q)
        assert np.all(denom > 0), "Shell crossing has occurred; density diverges at caustic"
        return (1.0 / denom) - 1.0

    def create_initial_state(
        self,
        n_particles_1d: int = 64,
        n_grid: int = 64,
        total_mass: float = 1.0e5,
    ) -> Tuple[SimulationState, np.ndarray]:
        """
        Creates a 3D simulation state representing a 1D plane wave perturbation along the x-axis.
        Particles are placed on a regular 1D grid along x and replicated in y and z to ensure plane symmetry.

        Returns:
            sim_state: Initialized SimulationState at a_init.
            q_x: 1D array of unperturbed Lagrangian coordinates along x.
        """
        assert n_particles_1d >= 8, "Need at least 8 particles along 1D axis"

        # 1D Lagrangian grid along x
        dx_q = self.box_size / n_particles_1d
        q_x = np.arange(n_particles_1d, dtype=np.float64) * dx_q

        # For a full 3D simulation, we use a grid of n_particles_1d along x,
        # with minimal sampling in y and z (e.g. 4x4 or 8x8) to maintain 1D planar symmetry.
        n_transverse = 4
        n_total = n_particles_1d * n_transverse * n_transverse
        mass_per_particle = total_mass / n_total

        # Generate meshgrid of Lagrangian coordinates
        qx_mesh, qy_mesh, qz_mesh = np.meshgrid(
            q_x,
            np.linspace(0.0, self.box_size, n_transverse, endpoint=False),
            np.linspace(0.0, self.box_size, n_transverse, endpoint=False),
            indexing="ij",
        )

        qx_flat = qx_mesh.flatten()
        qy_flat = qy_mesh.flatten()
        qz_flat = qz_mesh.flatten()

        # Perturb only along x
        x_init = self.exact_position(qx_flat, self.a_init)
        y_init = qy_flat
        z_init = qz_flat

        positions = np.stack([x_init, y_init, z_init], axis=1)

        # Momenta only along x
        p_x_init = self.exact_momentum(qx_flat, self.a_init, mass_per_particle)
        p_y_init = np.zeros_like(p_x_init)
        p_z_init = np.zeros_like(p_x_init)

        momenta = np.stack([p_x_init, p_y_init, p_z_init], axis=1)
        masses = np.full(n_total, mass_per_particle, dtype=np.float64)

        state = SimulationState(
            positions=positions,
            momenta=momenta,
            masses=masses,
            a=self.a_init,
            box_size=self.box_size,
            n_grid=n_grid,
        )

        return state, qx_flat

    def evaluate_error(
        self, state: SimulationState, q_x: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates discrepancy between simulation state and exact analytical solution.
        
        Returns:
            Dictionary containing L1, L2, and Linf relative errors for position and velocity.
        """
        a_current = state.a
        x_exact = self.exact_position(q_x, a_current)
        u_exact = self.exact_velocity(q_x, a_current)

        x_sim = state.positions[:, 0]
        u_sim = state.peculiar_velocities[:, 0]

        # Account for periodic boundary conditions when computing position difference
        dx = x_sim - x_exact
        dx = dx - self.box_size * np.round(dx / self.box_size)

        # Error norms
        l1_pos = float(np.mean(np.abs(dx)))
        l2_pos = float(np.sqrt(np.mean(dx ** 2)))
        linf_pos = float(np.max(np.abs(dx)))

        rel_l1_pos = l1_pos / self.box_size
        rel_linf_pos = linf_pos / self.box_size

        du = u_sim - u_exact
        u_scale = max(float(np.max(np.abs(u_exact))), 1e-10)
        l1_vel = float(np.mean(np.abs(du)))
        linf_vel = float(np.max(np.abs(du)))
        rel_l1_vel = l1_vel / u_scale

        return {
            "a": a_current,
            "l1_pos": l1_pos,
            "l2_pos": l2_pos,
            "linf_pos": linf_pos,
            "rel_l1_pos": rel_l1_pos,
            "rel_linf_pos": rel_linf_pos,
            "l1_vel": l1_vel,
            "linf_vel": linf_vel,
            "rel_l1_vel": rel_l1_vel,
        }


class KeplerTwoBodySolution:
    """
    Analytical Keplerian two-body circular orbit solver for non-cosmological
    orbital dynamics and symplectic conservation benchmarking.
    """

    def __init__(self, m1: float = 1.0, m2: float = 1.0, separation: float = 1.0, G: float = 1.0):
        """
        Args:
            m1: Mass of body 1.
            m2: Mass of body 2.
            separation: Initial orbital radius r.
            G: Gravitational constant.
        """
        assert m1 > 0 and m2 > 0 and separation > 0 and G > 0
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.M = self.m1 + self.m2
        self.mu = (self.m1 * self.m2) / self.M
        self.r = float(separation)
        self.G = float(G)

        # Circular orbital velocity: v = sqrt(G * M / r)
        self.v_rel = math.sqrt(self.G * self.M / self.r)
        self.omega = self.v_rel / self.r
        self.period = 2.0 * math.pi / self.omega

        # Invariants
        self.total_energy = -0.5 * self.G * self.m1 * self.m2 / self.r
        self.angular_momentum = self.mu * self.r * self.v_rel

    def exact_trajectory(self, t: float) -> Dict[str, np.ndarray]:
        """
        Computes exact positions and velocities of both bodies in the center-of-mass frame.
        """
        theta = self.omega * t
        r1 = (self.m2 / self.M) * self.r
        r2 = (self.m1 / self.M) * self.r

        # Body 1
        pos1 = np.array([r1 * math.cos(theta), r1 * math.sin(theta), 0.0])
        vel1 = np.array([-r1 * self.omega * math.sin(theta), r1 * self.omega * math.cos(theta), 0.0])

        # Body 2 (opposite phase)
        pos2 = np.array([-r2 * math.cos(theta), -r2 * math.sin(theta), 0.0])
        vel2 = np.array([r2 * self.omega * math.sin(theta), -r2 * self.omega * math.cos(theta), 0.0])

        return {
            "pos1": pos1,
            "vel1": vel1,
            "pos2": pos2,
            "vel2": vel2,
            "energy": self.total_energy,
            "period": self.period,
        }
