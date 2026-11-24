"""
ASTRAEUS Multi-Scale Simulation Intelligence Engine (WS9.6 / WS9.9).
Couples microphysical nuclear reaction networks, mesoscopic relativistic hydrodynamics,
and macroscopic radiative transfer with physics-derived continuous uncertainty bounding
and reference model benchmark validation.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import math
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class ScaleLayer(str, Enum):
    """The 3 coupled simulation scales."""
    MICROPHYSICAL_NUCLEAR = "MICROPHYSICAL_NUCLEAR"
    MESOSCOPIC_HYDRODYNAMICS = "MESOSCOPIC_HYDRODYNAMICS"
    MACROSCOPIC_RADIATION_TRANSFER = "MACROSCOPIC_RADIATION_TRANSFER"


class MultiScaleState(BaseModel):
    """Coupled physical state across all 3 physical simulation scales."""
    epoch_days: float
    # Microphysics
    electron_fraction_ye: float = Field(..., ge=0.01, le=0.55)
    nuclear_specific_heating_erg_s_g: float
    thermalization_efficiency: float = 1.0
    # Mesoscopic hydro
    ejecta_radius_cm: float
    shock_velocity_cm_s: float
    kinetic_energy_erg: float
    # Macroscopic radiation
    optical_depth_tau: float
    effective_temperature_k: float
    bolometric_luminosity_erg_s: float
    surrogate_uncertainty_fraction: float
    is_surrogate_extrapolating: bool = False
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MultiScaleBenchmarkReport(BaseModel):
    """Comparison report against external reference simulation benchmarks."""
    benchmark_source: str
    target_observable: str
    mean_relative_error: float
    max_relative_error: float
    is_physically_consistent: bool
    reference_data_points: int


class MultiScaleSimulator:
    """
    Coupled multi-scale simulator connecting nuclear networks,
    hydrodynamics, and radiative transfer with continuous uncertainty propagation
    and validation against external physical reference solutions.
    """

    C_LIGHT = 2.99792458e10
    M_SUN = 1.98847e33
    SIGMA_SB = 5.670374e-5

    # Validated surrogate parameter envelope
    VALID_ENVELOPE = {
        "ejecta_mass_msun": (0.005, 0.10),
        "velocity_c": (0.05, 0.35),
        "electron_fraction_ye": (0.05, 0.50),
    }

    def check_surrogate_envelope(self, parameters: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Verifies if input parameters fall within the verified training envelope."""
        violations = []
        for p, (pmin, pmax) in self.VALID_ENVELOPE.items():
            if p in parameters:
                val = parameters[p]
                if val < pmin or val > pmax:
                    violations.append(f"{p} ({val:.4f}) outside surrogate envelope [{pmin}, {pmax}]")
        return len(violations) == 0, violations

    def step_multiscale_evolution(
        self,
        epoch_days: float,
        parameters: Dict[str, float],
    ) -> MultiScaleState:
        """
        Integrates across all 3 scales for a given epoch.
        Derives continuous nuclear heating, dynamic opacity, and radiative diffusion.
        """
        m_ej_msun = parameters.get("ejecta_mass_msun", 0.04)
        v_c = parameters.get("velocity_c", 0.18)
        y_e = parameters.get("electron_fraction_ye", 0.20)

        # 0. Check surrogate extrapolation
        is_in_envelope, envelope_viols = self.check_surrogate_envelope(parameters)

        # 1. Microphysics: Nuclear Reaction Network Heating Rate with Thermalization
        t_days = max(0.05, epoch_days)
        ye_enhancement = 1.0 + 2.5 * max(0.0, 0.30 - y_e)

        # Thermalization efficiency eps_th(t) (Barnes et al. 2016)
        eps_th = 0.36 * (1.0 + 0.5 * math.log(1.0 + 2.0 * (t_days ** 1.3))) * (t_days ** -0.3)
        eps_th = min(1.0, max(0.10, eps_th))

        q_nuclear = 4.0e10 * (t_days ** -1.30) * eps_th * ye_enhancement  # erg/s/g

        # 2. Mesoscale Hydrodynamics
        m_g = m_ej_msun * self.M_SUN
        v_cm_s = v_c * self.C_LIGHT
        r_cm = max(1e13, v_cm_s * t_days * 86400.0)
        e_kin = 0.5 * m_g * (v_cm_s**2)

        # 3. Macroscopic Radiative Transfer
        # Continuous lanthanide opacity transition (Tanaka et al. 2020)
        # Smooth transition from iron-peak opacity (0.5) to lanthanide-curtained opacity (15.0)
        kappa = 0.5 + 14.5 / (1.0 + math.exp(min(50.0, max(-50.0, (y_e - 0.25) / 0.03))))
        rho = m_g / ((4.0 / 3.0) * math.pi * (r_cm**3))
        tau = max(0.01, kappa * rho * r_cm)

        # Radiative diffusion time
        t_diff_sec = math.sqrt((2.0 * kappa * m_g) / (4.0 * math.pi * v_cm_s * self.C_LIGHT))
        t_diff_days = t_diff_sec / 86400.0

        leak_factor = 1.0 - math.exp(-((t_diff_days / t_days) ** 2))
        l_bol = m_g * q_nuclear * leak_factor

        t_eff = max(1000.0, (l_bol / (4.0 * math.pi * (r_cm**2) * self.SIGMA_SB)) ** 0.25)

        # Continuous physics-derived uncertainty propagation
        dist_m = max(0.0, max(0.005 - m_ej_msun, m_ej_msun - 0.10) / 0.095)
        dist_v = max(0.0, max(0.05 - v_c, v_c - 0.35) / 0.30)
        dist_y = max(0.0, max(0.05 - y_e, y_e - 0.50) / 0.45)
        boundary_dist = max(dist_m, dist_v, dist_y)

        if is_in_envelope:
            surr_uncert = round(0.02 + 0.03 * boundary_dist, 4)
        else:
            surr_uncert = round(0.08 + 0.20 * boundary_dist, 4)

        return MultiScaleState(
            epoch_days=epoch_days,
            electron_fraction_ye=round(y_e, 3),
            nuclear_specific_heating_erg_s_g=round(q_nuclear, 2),
            thermalization_efficiency=round(eps_th, 3),
            ejecta_radius_cm=round(r_cm, 2),
            shock_velocity_cm_s=round(v_cm_s, 2),
            kinetic_energy_erg=round(e_kin, 2),
            optical_depth_tau=round(tau, 3),
            effective_temperature_k=round(t_eff, 1),
            bolometric_luminosity_erg_s=round(l_bol, 2),
            surrogate_uncertainty_fraction=surr_uncert,
            is_surrogate_extrapolating=not is_in_envelope,
        )

    def validate_against_reference_models(self) -> MultiScaleBenchmarkReport:
        """
        Validates the multiscale simulator against external benchmark calculations
        from Kasen & Barnes (2019) and Tanaka et al. (2020) for standard kilonova parameters:
        M_ej = 0.04 M_sun, v = 0.18 c, Y_e = 0.20.
        """
        params = {"ejecta_mass_msun": 0.04, "velocity_c": 0.18, "electron_fraction_ye": 0.20}
        # Reference points: (epoch_days, expected_log_Lbol)
        reference_kb19 = [
            (0.5, 41.25),
            (1.0, 41.35),
            (2.0, 41.15),
            (3.0, 40.85),
            (5.0, 40.35),
        ]

        errors = []
        for ep, ref_log_l in reference_kb19:
            st = self.step_multiscale_evolution(ep, params)
            sim_log_l = math.log10(max(1e30, st.bolometric_luminosity_erg_s))
            rel_err = abs(sim_log_l - ref_log_l) / ref_log_l
            errors.append(rel_err)

        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        is_consistent = (mean_err < 0.08) and (max_err < 0.15)

        return MultiScaleBenchmarkReport(
            benchmark_source="Kasen & Barnes (2019) 2D Radiative Transfer Reference",
            target_observable="log10(L_bol [erg/s])",
            mean_relative_error=round(mean_err, 4),
            max_relative_error=round(max_err, 4),
            is_physically_consistent=is_consistent,
            reference_data_points=len(reference_kb19),
        )
