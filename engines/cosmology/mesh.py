"""
ASTRAEUS Cosmology Engine: 3D Particle-Mesh (PM) Gravity & Mesh Operations.
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


class ParticleMeshSolver3D:
    """
    3D Particle-Mesh (PM) gravity solver using Cloud-in-Cell (CIC) mass assignment,
    discretized Green's function FFT Poisson solver, and 2nd-order central difference force interpolation.
    Strictly preserves total mass and total comoving momentum.
    """

    def __init__(self, box_size: float, n_grid: int, cosmo: Optional[CosmologicalParameters] = None):
        """
        Args:
            box_size: Comoving box size L in Mpc/h (or arbitrary comoving length units).
            n_grid: Grid resolution along each axis (N_grid x N_grid x N_grid).
            cosmo: Cosmological parameters (default: Planck 2018).
        """
        assert box_size > 0, "Box size must be positive"
        assert n_grid >= 4 and (n_grid % 2 == 0), "n_grid must be an even integer >= 4"

        self.box_size = float(box_size)
        self.n_grid = int(n_grid)
        self.dx = self.box_size / self.n_grid
        self.cosmo = cosmo if cosmo is not None else CosmologicalParameters()

        # Precompute k-vectors and discrete 7-point finite difference Laplacian in Fourier space
        self._precompute_fourier_grid()

    def _precompute_fourier_grid(self):
        """Precomputes 3D discrete Laplacian in k-space for the 7-point stencil."""
        N = self.n_grid
        dx = self.dx
        L = self.box_size

        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

        # Discrete finite-difference Laplacian in Fourier space:
        # nabla^2_disc = -4/dx^2 * [sin^2(kx*dx/2) + sin^2(ky*dx/2) + sin^2(kz*dx/2)]
        sin_sq_sum = (
            np.sin(KX * dx * 0.5) ** 2
            + np.sin(KY * dx * 0.5) ** 2
            + np.sin(KZ * dx * 0.5) ** 2
        )
        self.laplacian_k = -4.0 / (dx ** 2) * sin_sq_sum

        # Green's function G(k) = 1 / laplacian_k, with G(0) = 0
        self.inv_laplacian_k = np.zeros_like(self.laplacian_k)
        mask = self.laplacian_k != 0
        self.inv_laplacian_k[mask] = 1.0 / self.laplacian_k[mask]

    def cic_deposit(
        self, positions: np.ndarray, masses: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Cloud-in-Cell (CIC) trilinear mass assignment from particles onto 3D grid.
        Returns:
            mass_grid: (N, N, N) float64 array of accumulated cell masses.
            corners: Precomputed corner indices and weights for matching force interpolation.
        """
        N = self.n_grid
        L = self.box_size
        dx = self.dx

        # Comoving periodic wrap to [0, L)
        pos_wrapped = np.mod(positions, L)
        x_norm = pos_wrapped / dx

        # Base grid node coordinates
        i = np.floor(x_norm).astype(np.int64)
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

        mass_grid = np.zeros((N, N, N), dtype=np.float64)
        for ci, cj, ck, w in corners:
            np.add.at(mass_grid, (ci, cj, ck), masses * w)

        return mass_grid, corners

    def compute_density_contrast(
        self, mass_grid: np.ndarray, total_mass: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes the dimensionless density contrast delta(x) = rho(x)/rho_bar - 1.
        Mean of delta across all grid cells is guaranteed to be 0.
        """
        if total_mass is None:
            total_mass = float(np.sum(mass_grid))

        assert total_mass > 0, "Total mass must be strictly positive"
        mean_mass_per_cell = total_mass / (self.n_grid ** 3)
        delta_grid = (mass_grid / mean_mass_per_cell) - 1.0
        return delta_grid

    def solve_poisson(
        self, delta_grid: np.ndarray, a: float = 1.0, G_source_coeff: Optional[float] = None
    ) -> np.ndarray:
        """
        Solves comoving Poisson equation nabla^2 phi = (3/2) * Omega_m * H0^2 * (1/a) * delta.
        Returns peculiar gravitational potential phi on grid nodes.
        """
        if G_source_coeff is None:
            # Standard comoving Poisson source factor: 1.5 * Omega_m * H0^2 / a
            G_source_coeff = 1.5 * self.cosmo.Omega_m * (self.cosmo.H0 ** 2) / max(a, 1e-10)

        delta_k = np.fft.fftn(delta_grid)
        phi_k = delta_k * self.inv_laplacian_k * G_source_coeff
        # Zero-mode exclusion: net cosmological potential has zero mean
        phi_k[0, 0, 0] = 0.0

        phi_grid = np.fft.ifftn(phi_k).real
        return phi_grid

    def compute_grid_acceleration(self, phi_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes gravitational acceleration g = -nabla phi on grid nodes via 2nd-order central differences.
        """
        dx = self.dx
        gx = -(np.roll(phi_grid, -1, axis=0) - np.roll(phi_grid, 1, axis=0)) / (2.0 * dx)
        gy = -(np.roll(phi_grid, -1, axis=1) - np.roll(phi_grid, 1, axis=1)) / (2.0 * dx)
        gz = -(np.roll(phi_grid, -1, axis=2) - np.roll(phi_grid, 1, axis=2)) / (2.0 * dx)
        return gx, gy, gz

    def interpolate_particle_acceleration(
        self,
        gx: np.ndarray,
        gy: np.ndarray,
        gz: np.ndarray,
        corners: list,
        n_particles: int,
    ) -> np.ndarray:
        """
        Interpolates grid acceleration onto particles using identical CIC trilinear weights.
        Returns: (n_particles, 3) float64 array of particle accelerations.
        """
        gp = np.zeros((n_particles, 3), dtype=np.float64)
        for ci, cj, ck, w in corners:
            gp[:, 0] += gx[ci, cj, ck] * w
            gp[:, 1] += gy[ci, cj, ck] * w
            gp[:, 2] += gz[ci, cj, ck] * w
        return gp

    def compute_forces(
        self, positions: np.ndarray, masses: np.ndarray, a: float = 1.0, G_source_coeff: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete end-to-end forward force computation pipeline:
        Particles -> CIC deposit -> delta -> Poisson solve -> nabla phi -> CIC interpolate.
        Returns:
            particle_accelerations: (N_p, 3) array
            phi_grid: (N, N, N) potential grid
            delta_grid: (N, N, N) density contrast grid
        """
        n_p = len(positions)
        mass_grid, corners = self.cic_deposit(positions, masses)
        total_mass = float(np.sum(masses))
        delta_grid = self.compute_density_contrast(mass_grid, total_mass=total_mass)
        phi_grid = self.solve_poisson(delta_grid, a=a, G_source_coeff=G_source_coeff)
        gx, gy, gz = self.compute_grid_acceleration(phi_grid)
        gp = self.interpolate_particle_acceleration(gx, gy, gz, corners, n_particles=n_p)
        return gp, phi_grid, delta_grid

    def compute_potential_energy(
        self, mass_grid: np.ndarray, phi_grid: np.ndarray
    ) -> float:
        """
        Computes gravitational potential energy U = 0.5 * sum_cell M_cell * phi_cell.
        """
        U = 0.5 * float(np.sum(mass_grid * phi_grid))
        return U
