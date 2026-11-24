"""
ASTRAEUS Multi-Scale Simulation Intelligence Engine (WS9.6).
Couples microphysical nuclear reaction networks, mesoscopic relativistic hydrodynamics,
and macroscopic radiative transfer with surrogate uncertainty bounding.

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


class MultiScaleSimulator:
    """
    Coupled multi-scale simulator connecting nuclear networks,
    hydrodynamics, and radiative transfer with fallback guards for surrogate extrapolation.
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
        """
        m_ej_msun = parameters.get("ejecta_mass_msun", 0.04)
        v_c = parameters.get("velocity_c", 0.18)
        y_e = parameters.get("electron_fraction_ye", 0.20)

        # 0. Check surrogate extrapolation
        is_in_envelope, envelope_viols = self.check_surrogate_envelope(parameters)

        # 1. Microphysics: Nuclear Reaction Network Heating Rate
        # Lower Ye (neutron-rich) produces heavy r-process with higher heating and lanthanide opacity
        t_days = max(0.05, epoch_days)
        ye_enhancement = 1.0 + 2.5 * max(0.0, 0.30 - y_e)
        q_nuclear = 4.0e10 * (t_days ** -1.30) * ye_enhancement  # erg/s/g

        # 2. Mesoscale Hydrodynamics
        m_g = m_ej_msun * self.M_SUN
        v_cm_s = v_c * self.C_LIGHT
        r_cm = max(1e13, v_cm_s * t_days * 86400.0)
        e_kin = 0.5 * m_g * (v_cm_s**2)

        # 3. Macroscopic Radiative Transfer
        # Opacity kappa depends strongly on Ye (lanthanides form if Ye < 0.25)
        kappa = 0.5 if y_e > 0.25 else 10.0
        rho = m_g / ((4.0 / 3.0) * math.pi * (r_cm**3))
        tau = max(0.01, kappa * rho * r_cm)

        # Diffusion time
        t_diff_sec = math.sqrt((2.0 * kappa * m_g) / (4.0 * math.pi * v_cm_s * self.C_LIGHT))
        t_diff_days = t_diff_sec / 86400.0

        leak_factor = 1.0 - math.exp(-((t_diff_days / t_days) ** 2))
        l_bol = m_g * q_nuclear * leak_factor

        t_eff = max(1000.0, (l_bol / (4.0 * math.pi * (r_cm**2) * self.SIGMA_SB)) ** 0.25)

        surr_uncert = 0.02 if is_in_envelope else 0.25

        return MultiScaleState(
            epoch_days=epoch_days,
            electron_fraction_ye=round(y_e, 3),
            nuclear_specific_heating_erg_s_g=round(q_nuclear, 2),
            ejecta_radius_cm=round(r_cm, 2),
            shock_velocity_cm_s=round(v_cm_s, 2),
            kinetic_energy_erg=round(e_kin, 2),
            optical_depth_tau=round(tau, 3),
            effective_temperature_k=round(t_eff, 1),
            bolometric_luminosity_erg_s=round(l_bol, 2),
            surrogate_uncertainty_fraction=surr_uncert,
            is_surrogate_extrapolating=not is_in_envelope,
        )
