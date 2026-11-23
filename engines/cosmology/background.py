"""
ASTRAEUS Cosmology Engine: Background Cosmology & Linear Perturbation Growth.
Workstream 1: Scientific Computing Foundation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, Callable
import numpy as np
from scipy.integrate import quad, solve_ivp

from engines.cosmology.constants import (
    CosmologicalParameters,
    SPEED_OF_LIGHT_KM_S,
    MPC_TO_KM,
    SEC_PER_YEAR,
)


class BackgroundCosmology:
    """
    Cosmological background expansion, conformal time, distance measures,
    and linear perturbation growth factor solver.
    """

    def __init__(self, params: Optional[CosmologicalParameters] = None):
        self.params = params if params is not None else CosmologicalParameters()
        self._norm_growth_at_1: Optional[float] = None
        self._init_growth_normalizer()

    def E(self, a: float | np.ndarray) -> float | np.ndarray:
        """
        Dimensionless expansion rate E(a) = H(a) / H0.
        E^2(a) = Omega_m a^-3 + Omega_r a^-4 + Omega_k a^-2 + Omega_lambda a^-3(1+w0+wa) exp[3 wa (a - 1)].
        """
        p = self.params
        a_arr = np.asarray(a, dtype=np.float64)

        term_m = p.Omega_m * (a_arr ** (-3))
        term_r = p.Omega_r * (a_arr ** (-4)) if p.Omega_r > 0 else 0.0
        term_k = p.Omega_k * (a_arr ** (-2)) if abs(p.Omega_k) > 1e-10 else 0.0

        if p.wa == 0.0 and p.w0 == -1.0:
            term_de = p.Omega_lambda
        else:
            w_exponent = -3.0 * (1.0 + p.w0 + p.wa)
            term_de = p.Omega_lambda * (a_arr ** w_exponent) * np.exp(3.0 * p.wa * (a_arr - 1.0))

        e2 = term_m + term_r + term_k + term_de
        # Prevent non-positive values due to extreme floating point rounding
        e2 = np.maximum(e2, 1e-30)
        e_val = np.sqrt(e2)

        return float(e_val) if np.isscalar(a) else e_val

    def H(self, a: float | np.ndarray) -> float | np.ndarray:
        """Hubble parameter in km / s / Mpc at scale factor a."""
        return self.params.H0 * self.E(a)

    def dE_da(self, a: float | np.ndarray) -> float | np.ndarray:
        """Analytical derivative dE/da of dimensionless expansion rate."""
        p = self.params
        a_arr = np.asarray(a, dtype=np.float64)
        e_val = self.E(a_arr)

        # d(E^2)/da = -3 Omega_m a^-4 - 4 Omega_r a^-5 - 2 Omega_k a^-3 + d(term_de)/da
        de2_da = -3.0 * p.Omega_m * (a_arr ** (-4))
        if p.Omega_r > 0:
            de2_da -= 4.0 * p.Omega_r * (a_arr ** (-5))
        if abs(p.Omega_k) > 1e-10:
            de2_da -= 2.0 * p.Omega_k * (a_arr ** (-3))

        if p.wa == 0.0 and p.w0 == -1.0:
            pass  # derivative of constant Omega_lambda is 0
        else:
            w_exp = -3.0 * (1.0 + p.w0 + p.wa)
            de_base = p.Omega_lambda * (a_arr ** w_exp) * np.exp(3.0 * p.wa * (a_arr - 1.0))
            de2_da += de_base * (w_exp / a_arr + 3.0 * p.wa)

        # dE/da = (1 / 2E) * d(E^2)/da
        res = (0.5 / e_val) * de2_da
        return float(res) if np.isscalar(a) else res

    def dH_da(self, a: float | np.ndarray) -> float | np.ndarray:
        """Derivative dH/da in (km/s/Mpc) / scale_factor."""
        return self.params.H0 * self.dE_da(a)

    def omega_m_a(self, a: float | np.ndarray) -> float | np.ndarray:
        """Time-dependent matter density parameter Omega_m(a)."""
        p = self.params
        a_arr = np.asarray(a, dtype=np.float64)
        e_val = self.E(a_arr)
        res = p.Omega_m * (a_arr ** (-3)) / (e_val ** 2)
        return float(res) if np.isscalar(a) else res

    def conformal_time_mpc(self, a: float) -> float:
        """
        Conformal time tau(a) = c * int_0^a da' / (a'^2 H(a')) in comoving Mpc.
        """
        if a <= 0:
            return 0.0

        def integrand(a_prime: float) -> float:
            return SPEED_OF_LIGHT_KM_S / (self.params.H0 * (a_prime ** 2) * self.E(a_prime))

        val, _ = quad(integrand, 0.0, a, epsrel=1e-8, limit=100)
        return float(val)

    def cosmic_time_gyr(self, a: float) -> float:
        """
        Cosmic time t(a) = int_0^a da' / (a' H(a')) in Gyr.
        """
        if a <= 0:
            return 0.0

        def integrand(a_prime: float) -> float:
            # 1 / H(a') in seconds
            h_sec = (self.params.H0 * self.E(a_prime)) / MPC_TO_KM
            return 1.0 / (a_prime * h_sec)

        sec, _ = quad(integrand, 0.0, a, epsrel=1e-8, limit=100)
        gyr = sec / (SEC_PER_YEAR * 1e9)
        return float(gyr)

    def comoving_distance_mpc(self, z: float) -> float:
        """
        Line-of-sight comoving distance chi(z) = c * int_0^z dz' / H(z') in Mpc.
        """
        if z <= 0:
            return 0.0

        def integrand(z_prime: float) -> float:
            a_prime = 1.0 / (1.0 + z_prime)
            return SPEED_OF_LIGHT_KM_S / (self.params.H0 * self.E(a_prime))

        val, _ = quad(integrand, 0.0, z, epsrel=1e-8, limit=100)
        return float(val)

    def comoving_distance_chi(self, z: float) -> float:
        """Comoving distance chi(z) in Mpc."""
        return self.comoving_distance_mpc(z)

    def angular_diameter_distance_dA(self, z: float) -> float:
        """Angular diameter distance d_A(z) = chi(z) / (1 + z) in Mpc."""
        if z <= 0:
            return 0.0
        return self.comoving_distance_mpc(z) / (1.0 + z)

    def luminosity_distance_dL(self, z: float) -> float:
        """Luminosity distance d_L(z) = (1 + z) * chi(z) in Mpc."""
        if z <= 0:
            return 0.0
        return (1.0 + z) * self.comoving_distance_mpc(z)

    def _init_growth_normalizer(self):
        """Precomputes unnormalized growth at a=1 to calibrate D1(1) = 1."""
        if abs(self.params.Omega_m - 1.0) < 1e-6 and self.params.Omega_lambda == 0:
            self._norm_growth_at_1 = 1.0  # Einstein-de Sitter D(a) = a exactly
            return

        a_init = 1e-4

        def odefunc(a_val: float, y: np.ndarray) -> np.ndarray:
            D, dD = y
            e_val = self.E(a_val)
            de_val = self.dE_da(a_val)
            d2D = -(3.0 / a_val + de_val / e_val) * dD + (1.5 * self.params.Omega_m / (a_val ** 5 * e_val ** 2)) * D
            return np.array([dD, d2D], dtype=np.float64)

        sol = solve_ivp(odefunc, [a_init, 1.0], [a_init, 1.0], rtol=1e-10, atol=1e-12)
        self._norm_growth_at_1 = float(sol.y[0, -1])

    def linear_growth_factor(self, a: float | np.ndarray) -> float | np.ndarray:
        """
        Linear growing mode D1(a) normalized to D1(a=1) = 1.
        In Einstein-de Sitter, D1(a) = a.
        """
        a_arr = np.asarray(a, dtype=np.float64)
        if abs(self.params.Omega_m - 1.0) < 1e-6 and self.params.Omega_lambda == 0:
            return float(a_arr) if np.isscalar(a) else a_arr

        scalar = np.isscalar(a)
        flat_a = np.atleast_1d(a_arr)
        results = np.zeros_like(flat_a)

        a_init = 1e-4

        def odefunc(a_val: float, y: np.ndarray) -> np.ndarray:
            D, dD = y
            e_val = self.E(a_val)
            de_val = self.dE_da(a_val)
            d2D = -(3.0 / a_val + de_val / e_val) * dD + (1.5 * self.params.Omega_m / (a_val ** 5 * e_val ** 2)) * D
            return np.array([dD, d2D], dtype=np.float64)

        # Sort evaluation targets
        sort_idx = np.argsort(flat_a)
        sorted_a = flat_a[sort_idx]

        # Integrate forward to max_a
        max_a = max(1.0, float(np.max(sorted_a)))
        t_eval = np.unique(np.clip(sorted_a, a_init, max_a))
        sol = solve_ivp(odefunc, [a_init, max_a], [a_init, 1.0], t_eval=t_eval, rtol=1e-10, atol=1e-12)

        # Map back
        val_map = {t: val / self._norm_growth_at_1 for t, val in zip(sol.t, sol.y[0])}
        for i, a_i in enumerate(flat_a):
            if a_i <= a_init:
                results[i] = a_i / self._norm_growth_at_1
            else:
                # nearest or interpolated
                results[i] = val_map.get(a_i, np.interp(a_i, sol.t, sol.y[0] / self._norm_growth_at_1))

        return float(results[0]) if scalar else results

    def growth_rate_f(self, a: float) -> float:
        """
        Logarithmic growth rate f(a) = d ln D1 / d ln a = (a / D1) * dD1 / da.
        Approximates Omega_m(a)^0.55.
        """
        if abs(self.params.Omega_m - 1.0) < 1e-6 and self.params.Omega_lambda == 0:
            return 1.0

        da = 1e-4 * a
        d1 = self.linear_growth_factor(a)
        d1_plus = self.linear_growth_factor(a + da)
        d1_minus = self.linear_growth_factor(a - da)
        dD_da = (d1_plus - d1_minus) / (2.0 * da)
        return float((a / d1) * dD_da)
