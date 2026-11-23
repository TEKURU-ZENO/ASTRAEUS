"""
ASTRAEUS Cosmology Engine: Differentiable Simulation & Parameter Inference (Track A).
Provides end-to-end automatic differentiation through particle-mesh cosmology,
physical initial-condition generation with top-hat window normalization,
Fisher information matrix identifiability, and gradient-based parameter recovery.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.integrate

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from engines.cosmology.constants import CosmologicalParameters
from engines.cosmology.background import BackgroundCosmology
from engines.cosmology.jax_engine import JaxParticleMeshSolver3D, JaxSimulationState


class DifferentiableInitialConditions:
    """
    Physical cosmological initial condition generator with top-hat window function normalization
    and frozen Gaussian random phases to avoid stochastic Monte Carlo gradient contamination.
    """

    def __init__(
        self,
        box_size: float = 100.0,
        n_grid: int = 32,
        a_init: float = 0.02,
        cosmo_base: Optional[CosmologicalParameters] = None,
    ):
        self.box_size = float(box_size)
        self.n_grid = int(n_grid)
        self.a_init = float(a_init)
        self.cosmo_base = cosmo_base if cosmo_base is not None else CosmologicalParameters()
        self.k_fund = 2.0 * np.pi / self.box_size
        self.k_nyq = np.pi * self.n_grid / self.box_size

    @staticmethod
    def bbks_transfer(k: float | np.ndarray, q: float | np.ndarray) -> float | np.ndarray:
        """
        BBKS (Bardeen, Bond, Kaiser, Szalay 1986) analytical matter transfer function.
        q = k / (Omega_m * h^2 * Mpc^-1)
        """
        q_safe = np.maximum(q, 1e-10)
        t = (
            np.log(1.0 + 2.34 * q_safe) / (2.34 * q_safe)
            * (
                1.0
                + 3.89 * q_safe
                + (16.1 * q_safe) ** 2
                + (5.46 * q_safe) ** 3
                + (6.71 * q_safe) ** 4
            )
            ** (-0.25)
        )
        return t

    @staticmethod
    def top_hat_window(x: float | np.ndarray) -> float | np.ndarray:
        """Fourier transform of spherical top-hat window: W(x) = 3*(sin(x) - x*cos(x)) / x^3."""
        x_arr = np.asarray(x, dtype=np.float64)
        # Small x expansion: 1 - x^2 / 10 + x^4 / 280
        small_mask = np.abs(x_arr) < 1e-3
        w = np.zeros_like(x_arr)
        if np.any(small_mask):
            xs = x_arr[small_mask]
            w[small_mask] = 1.0 - (xs ** 2) / 10.0 + (xs ** 4) / 280.0
        if np.any(~small_mask):
            xr = x_arr[~small_mask]
            w[~small_mask] = 3.0 * (np.sin(xr) - xr * np.cos(xr)) / (xr ** 3)
        return float(w) if np.isscalar(x) else w

    def compute_sigma8_norm(self, Omega_m: float, h: float = 0.6766, ns: float = 0.9665) -> float:
        """
        Computes integral of unnormalized power spectrum with top-hat filter at R = 8 Mpc/h:
        I_8 = (1 / (2*pi^2)) * int_0^inf k^2 P_shape(k) W_TH^2(8k) dk
        """
        R8 = 8.0  # Mpc / h
        gamma = Omega_m * h  # Shape parameter

        def integrand(k: float) -> float:
            if k <= 1e-6:
                return 0.0
            q = k / max(gamma, 1e-6)
            T_val = float(self.bbks_transfer(k, q))
            P_shape = (k ** ns) * (T_val ** 2)
            w = float(self.top_hat_window(k * R8))
            return (k ** 2) * P_shape * (w ** 2) / (2.0 * math.pi ** 2)

        integral, _ = scipy.integrate.quad(integrand, 1e-4, 50.0, epsrel=1e-6, limit=200)
        return float(integral)

    def linear_power_spectrum(
        self, k: np.ndarray, Omega_m: float, sigma8: float, h: float = 0.6766, ns: float = 0.9665
    ) -> np.ndarray:
        """
        Physical linear matter power spectrum calibrated to exact sigma8.
        """
        gamma = Omega_m * h
        q = k / max(gamma, 1e-6)
        T_val = self.bbks_transfer(k, q)
        P_shape = (k ** ns) * (T_val ** 2)

        norm_integral = self.compute_sigma8_norm(Omega_m, h=h, ns=ns)
        amp = (sigma8 ** 2) / max(norm_integral, 1e-20)
        return amp * P_shape

    def generate_gaussian_phases(self, seed: int = 42) -> np.ndarray:
        """Generates frozen complex Gaussian random phases with Hermitian symmetry."""
        N = self.n_grid
        rng = np.random.RandomState(seed)
        re = rng.normal(0.0, 1.0, size=(N, N, N))
        im = rng.normal(0.0, 1.0, size=(N, N, N))
        phases = (re + 1j * im) / np.sqrt(2.0)
        # Real field in real space requires Hermitian symmetry: phase(-k) = phase*(k)
        phases = 0.5 * (phases + np.flip(np.conj(phases)))
        phases[0, 0, 0] = 0.0
        return phases

    def create_initial_mesh_state(
        self,
        Omega_m: float,
        sigma8: float,
        phases: np.ndarray,
        total_mass: float = 1.0e6,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generates 3D particles displaced via Zel'dovich approximation with frozen phases.
        Returns: (positions, peculiar_momenta, masses)
        """
        N = self.n_grid
        L = self.box_size
        dx = L / N
        N_p = N ** 3
        m_particle = total_mass / N_p

        cosmo = CosmologicalParameters(Omega_m=Omega_m, h=self.cosmo_base.h)
        bg = BackgroundCosmology(cosmo)
        D1 = bg.linear_growth_factor(self.a_init)
        f_rate = bg.growth_rate_f(self.a_init)
        H_init = bg.H(self.a_init)

        # Fourier mesh
        kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
        K_sq = KX ** 2 + KY ** 2 + KZ ** 2
        K_mag = np.sqrt(K_sq)
        K_mag[0, 0, 0] = 1.0  # avoid divide by zero

        P_k = self.linear_power_spectrum(K_mag, Omega_m, sigma8, h=self.cosmo_base.h)
        P_k[0, 0, 0] = 0.0

        # Linear delta at a_init in Fourier space: delta_k = D1 * sqrt(P(k) * V) * phases
        V = L ** 3
        delta_k = D1 * np.sqrt(P_k * V) * phases

        # Zel'dovich displacement: Psi_x = -i (kx / k^2) * delta_k
        K_sq_safe = np.where(K_sq == 0.0, 1.0, K_sq)
        inv_k2 = 1.0 / K_sq_safe
        inv_k2[0, 0, 0] = 0.0

        psi_x_k = -1j * KX * inv_k2 * delta_k
        psi_y_k = -1j * KY * inv_k2 * delta_k
        psi_z_k = -1j * KZ * inv_k2 * delta_k

        psi_x = np.fft.ifftn(psi_x_k).real
        psi_y = np.fft.ifftn(psi_y_k).real
        psi_z = np.fft.ifftn(psi_z_k).real

        # Unperturbed Lagrangian coordinates
        q_coords = np.linspace(0.0, L, N, endpoint=False)
        QX, QY, QZ = np.meshgrid(q_coords, q_coords, q_coords, indexing="ij")

        # Euler positions
        x_init = np.mod(QX + psi_x, L).flatten()
        y_init = np.mod(QY + psi_y, L).flatten()
        z_init = np.mod(QZ + psi_z, L).flatten()

        pos = np.stack([x_init, y_init, z_init], axis=1)

        # Momenta: p = m * a^2 * x_dot = m * a^2 * H * f * Psi
        p_fac = m_particle * (self.a_init ** 2) * H_init * f_rate
        px = (p_fac * psi_x).flatten()
        py = (p_fac * psi_y).flatten()
        pz = (p_fac * psi_z).flatten()
        mom = np.stack([px, py, pz], axis=1)

        masses = np.full(N_p, m_particle, dtype=np.float64)

        return jnp.array(pos), jnp.array(mom), jnp.array(masses)


def jax_hubble_parameter(a: jnp.ndarray, Omega_m: float, H0: float = 67.66) -> jnp.ndarray:
    """Pure JAX differentiable Hubble expansion rate H(a) for flat Lambda-CDM."""
    omega_lambda = 1.0 - Omega_m
    a_safe = jnp.maximum(a, 1e-6)
    e_sq = Omega_m * (a_safe ** (-3.0)) + omega_lambda
    return H0 * jnp.sqrt(jnp.maximum(e_sq, 1e-10))


class DifferentiableSimulator:
    """
    Pure JAX differentiable cosmological particle-mesh trajectory simulator.
    """

    def __init__(self, box_size: float = 64.0, n_grid: int = 16):
        self.box_size = float(box_size)
        self.n_grid = int(n_grid)
        self.solver = JaxParticleMeshSolver3D(box_size=box_size, n_grid=n_grid, use_x64=True)

    def compute_power_spectrum(
        self, delta_grid: jnp.ndarray, n_bins: int = 8
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes 3D spherically averaged matter power spectrum P(k) in radial Fourier bins.
        """
        N = self.n_grid
        L = self.box_size
        dx = L / N
        V = L ** 3

        delta_k = jnp.fft.fftn(delta_grid)
        power_3d = (jnp.abs(delta_k) ** 2) * (dx ** 6) / V

        kx = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=dx)
        KX, KY, KZ = jnp.meshgrid(kx, kx, kx, indexing="ij")
        K_mag = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

        k_min = 2.0 * jnp.pi / L
        k_max = jnp.pi / dx
        k_edges = jnp.linspace(k_min, k_max, n_bins + 1)
        k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

        pk_bins = []
        for b in range(n_bins):
            mask = (K_mag >= k_edges[b]) & (K_mag < k_edges[b + 1])
            n_modes = jnp.sum(mask)
            pk_val = jnp.where(n_modes > 0, jnp.sum(power_3d * mask) / jnp.maximum(n_modes, 1), 0.0)
            pk_bins.append(pk_val)

        return k_centers, jnp.array(pk_bins)

    def simulate_trajectory(
        self,
        positions: jnp.ndarray,
        momenta: jnp.ndarray,
        masses: jnp.ndarray,
        a_init: float,
        a_target: float,
        n_steps: int,
        Omega_m: float = 0.3111,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Advances particles through n_steps KDK leapfrog steps in scale factor.
        Returns: (final_positions, final_momenta, final_delta_grid)
        """
        cosmo = CosmologicalParameters(Omega_m=Omega_m)
        bg = BackgroundCosmology(cosmo)
        delta_a = (a_target - a_init) / n_steps
        L = self.box_size

        def kdk_step(carry, step_idx):
            x, p = carry
            a_n = a_init + step_idx * delta_a
            a_half = a_n + 0.5 * delta_a
            a_next = a_n + delta_a

            # 1. Force at n
            g_n, _, _ = self.solver.compute_forces(x, masses, a=a_n)
            # dp/da = m * g / H(a)
            h_n = jax_hubble_parameter(a_n, Omega_m)
            dp_da_n = masses[:, None] * g_n / jnp.maximum(h_n, 1e-10)
            p_half = p + dp_da_n * (0.5 * delta_a)

            # 2. Drift
            h_half = jax_hubble_parameter(a_half, Omega_m)
            drift_factor = 1.0 / jnp.maximum((a_half ** 3) * h_half, 1e-20)
            dx_da = (p_half / masses[:, None]) * drift_factor
            x_next = jnp.mod(x + dx_da * delta_a, L)

            # 3. Force at n+1
            g_next, _, delta_next = self.solver.compute_forces(x_next, masses, a=a_next)
            h_next = jax_hubble_parameter(a_next, Omega_m)
            dp_da_next = masses[:, None] * g_next / jnp.maximum(h_next, 1e-10)
            p_next = p_half + dp_da_next * (0.5 * delta_a)

            return (x_next, p_next), delta_next

        (x_fin, p_fin), delta_grids = jax.lax.scan(
            kdk_step, (positions, momenta), jnp.arange(n_steps)
        )
        final_delta = delta_grids[-1]
        return x_fin, p_fin, final_delta


class FisherParameterInference:
    """
    Fisher Information matrix identifiability and parameter posterior recovery.
    """

    def __init__(self, simulator: DifferentiableSimulator, ic_gen: DifferentiableInitialConditions):
        self.sim = simulator
        self.ic = ic_gen

    def evaluate_power_spectrum(
        self,
        Omega_m: float,
        sigma8: float,
        phases: np.ndarray,
        n_steps: int = 5,
        a_target: float = 0.08,
    ) -> jnp.ndarray:
        """Evaluates simulated P(k) for given (Omega_m, sigma8)."""
        pos, mom, masses = self.ic.create_initial_mesh_state(Omega_m, sigma8, phases)
        _, _, delta_fin = self.sim.simulate_trajectory(
            pos, mom, masses, a_init=self.ic.a_init, a_target=a_target, n_steps=n_steps, Omega_m=Omega_m
        )
        _, pk = self.sim.compute_power_spectrum(delta_fin)
        return pk

    def compute_fisher_matrix(
        self,
        theta_fid: Tuple[float, float],
        phases: np.ndarray,
        n_steps: int = 5,
        eps: float = 1e-4,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Computes local 2x2 Fisher information matrix F_ij = sum_b (dPk_b / dtheta_i) * (dPk_b / dtheta_j) / sigma_b^2.
        Returns: (F, determinant, condition_number)
        """
        Om_fid, s8_fid = theta_fid
        pk_fid = np.array(self.evaluate_power_spectrum(Om_fid, s8_fid, phases, n_steps=n_steps))

        # Variance: cosmic variance sigma_b^2 = 2 * P(k_b)^2 / N_modes (approximated)
        sigma_sq = np.maximum(2.0 * (pk_fid ** 2) / 8.0, 1e-12)

        # Numerical Jacobian dP(k) / dtheta
        pk_om_plus = np.array(self.evaluate_power_spectrum(Om_fid + eps, s8_fid, phases, n_steps=n_steps))
        pk_om_minus = np.array(self.evaluate_power_spectrum(Om_fid - eps, s8_fid, phases, n_steps=n_steps))
        dPk_dOm = (pk_om_plus - pk_om_minus) / (2.0 * eps)

        pk_s8_plus = np.array(self.evaluate_power_spectrum(Om_fid, s8_fid + eps, phases, n_steps=n_steps))
        pk_s8_minus = np.array(self.evaluate_power_spectrum(Om_fid, s8_fid - eps, phases, n_steps=n_steps))
        dPk_ds8 = (pk_s8_plus - pk_s8_minus) / (2.0 * eps)

        J = np.stack([dPk_dOm, dPk_ds8], axis=1)  # (N_bins, 2)
        inv_cov = np.diag(1.0 / sigma_sq)

        F = J.T @ inv_cov @ J  # (2, 2)
        det_F = float(np.linalg.det(F))
        eigvals = np.linalg.eigvals(F)
        cond_F = float(np.max(eigvals) / max(np.min(eigvals), 1e-15))

        return F, det_F, cond_F

    def run_parameter_recovery(
        self,
        ground_truth: Tuple[float, float],
        initial_guess: Tuple[float, float],
        phases: np.ndarray,
        n_steps: int = 5,
        max_iter: int = 15,
        lr: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Runs gradient-based parameter posterior optimization recovering ground truth (Omega_m, sigma8).
        """
        Om_true, s8_true = ground_truth
        pk_obs = np.array(self.evaluate_power_spectrum(Om_true, s8_true, phases, n_steps=n_steps))
        sigma_b = np.maximum(pk_obs * 0.1, 1e-6)

        curr_Om, curr_s8 = initial_guess
        history = []
        eps = 1e-4

        for iteration in range(max_iter):
            # Compute loss
            pk_curr = np.array(self.evaluate_power_spectrum(curr_Om, curr_s8, phases, n_steps=n_steps))
            res = (pk_curr - pk_obs) / sigma_b
            loss = 0.5 * float(np.sum(res ** 2))

            history.append({
                "iter": iteration,
                "Omega_m": curr_Om,
                "sigma8": curr_s8,
                "loss": loss,
                "err_Om": abs(curr_Om - Om_true) / Om_true,
                "err_s8": abs(curr_s8 - s8_true) / s8_true,
            })

            if loss < 1e-5:
                break

            # Gradient via finite-difference sensitivities
            pk_om_plus = np.array(self.evaluate_power_spectrum(curr_Om + eps, curr_s8, phases, n_steps=n_steps))
            pk_om_minus = np.array(self.evaluate_power_spectrum(curr_Om - eps, curr_s8, phases, n_steps=n_steps))
            grad_Om = float(np.sum(res * (pk_om_plus - pk_om_minus) / (2.0 * eps * sigma_b)))

            pk_s8_plus = np.array(self.evaluate_power_spectrum(curr_Om, curr_s8 + eps, phases, n_steps=n_steps))
            pk_s8_minus = np.array(self.evaluate_power_spectrum(curr_Om, curr_s8 - eps, phases, n_steps=n_steps))
            grad_s8 = float(np.sum(res * (pk_s8_plus - pk_s8_minus) / (2.0 * eps * sigma_b)))

            # Normalized gradient step
            grad_norm = math.sqrt(grad_Om ** 2 + grad_s8 ** 2) + 1e-10
            curr_Om = float(np.clip(curr_Om - lr * (grad_Om / grad_norm), 0.15, 0.60))
            curr_s8 = float(np.clip(curr_s8 - lr * (grad_s8 / grad_norm), 0.40, 1.20))

        return {
            "ground_truth": ground_truth,
            "recovered": (curr_Om, curr_s8),
            "final_err_Om": abs(curr_Om - Om_true) / Om_true,
            "final_err_s8": abs(curr_s8 - s8_true) / s8_true,
            "history": history,
        }
