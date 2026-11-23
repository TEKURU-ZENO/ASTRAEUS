"""
ASTRAEUS Cosmology Engine: Cosmological Parameters & Physical Constants.
Workstream 1: Scientific Computing Foundation.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Any


# Physical constants in SI and Astrophysical units
SPEED_OF_LIGHT_KM_S: float = 299792.458  # km / s
SPEED_OF_LIGHT_M_S: float = 299792458.0   # m / s
GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11  # m^3 / (kg s^2)

# Astrophysical conversion constants
MPC_TO_KM: float = 3.08567758149137e19    # km per Mpc
KPC_TO_KM: float = 3.08567758149137e16    # km per kpc
MSUN_TO_KG: float = 1.98847e30            # kg per Solar Mass
SEC_PER_YEAR: float = 31557600.0          # seconds in Julian year

# Gravitational constant in comoving astrophysical units: (km/s)^2 * Mpc / M_sun
# G * M_sun / (Mpc * (km/s)^2)
GRAVITATIONAL_CONSTANT_ASTRO: float = (
    GRAVITATIONAL_CONSTANT_SI * MSUN_TO_KG / (MPC_TO_KM * 1e6)
)  # ~ 4.30091e-9 (km/s)^2 Mpc / M_sun

# Gravitational constant in (km/s)^2 * kpc / M_sun
GRAVITATIONAL_CONSTANT_KPC: float = GRAVITATIONAL_CONSTANT_ASTRO * 1000.0  # ~ 4.30091e-6

# Hubble distance c / H0 for H0 = 100 km/s/Mpc
HUBBLE_DISTANCE_MPC: float = SPEED_OF_LIGHT_KM_S / 100.0  # 2997.92458 Mpc/h


@dataclass(frozen=True)
class CosmologicalParameters:
    """
    Standard Flat/Curved Lambda-CDM / w0-wa Cosmological Parameter Specification.
    Default parameters follow Planck 2018 (A&A 641, A6, 2020) fiducial baseline.
    """
    h: float = 0.6766               # Dimensionless Hubble parameter H0 / (100 km/s/Mpc)
    Omega_m: float = 0.3111         # Total matter density parameter at z = 0
    Omega_b: float = 0.0490         # Baryon density parameter at z = 0
    Omega_lambda: float = 0.6889    # Dark energy density parameter at z = 0
    Omega_r: float = 8.6e-5         # Radiation density parameter at z = 0
    Omega_k: float = 0.0            # Spatial curvature parameter (0.0 = flat)
    w0: float = -1.0                # Dark energy equation of state parameter
    wa: float = 0.0                 # Dark energy equation of state time derivative
    sigma8: float = 0.8102          # Matter power spectrum normalization at 8 Mpc/h
    ns: float = 0.9665              # Primordial scalar spectral index

    def __post_init__(self):
        assert self.h > 0, "h must be strictly positive"
        assert self.Omega_m >= 0, "Omega_m must be non-negative"
        assert self.Omega_b >= 0, "Omega_b must be non-negative"

    @property
    def H0(self) -> float:
        """Hubble constant in km / s / Mpc."""
        return self.h * 100.0

    @property
    def H0_per_sec(self) -> float:
        """Hubble constant in 1 / s."""
        return self.H0 / MPC_TO_KM

    @property
    def Omega_cdm(self) -> float:
        """Cold dark matter density parameter."""
        return self.Omega_m - self.Omega_b

    @property
    def critical_density_0_kg_m3(self) -> float:
        """Critical density rho_crit,0 = 3 H0^2 / (8 pi G) in kg / m^3."""
        h0_sec = self.H0_per_sec
        return (3.0 * h0_sec ** 2) / (8.0 * math.pi * GRAVITATIONAL_CONSTANT_SI)

    @property
    def critical_density_0_Msun_Mpc3(self) -> float:
        """Critical density in M_sun / Mpc^3."""
        kg_m3 = self.critical_density_0_kg_m3
        return kg_m3 * (MPC_TO_KM * 1000.0) ** 3 / MSUN_TO_KG

    @property
    def mean_matter_density_0_Msun_Mpc3(self) -> float:
        """Mean comoving matter density rho_bar = Omega_m * rho_crit,0 in M_sun / Mpc^3."""
        return self.Omega_m * self.critical_density_0_Msun_Mpc3

    @classmethod
    def ed_s(cls, h: float = 0.5) -> CosmologicalParameters:
        """Einstein-de Sitter (EdS) universe: Omega_m = 1.0, Omega_lambda = 0.0."""
        return cls(
            h=h,
            Omega_m=1.0,
            Omega_b=0.05,
            Omega_lambda=0.0,
            Omega_r=0.0,
            Omega_k=0.0,
            w0=-1.0,
            wa=0.0,
        )

    @classmethod
    def einstein_de_sitter(cls, h: float = 0.5) -> CosmologicalParameters:
        """Alias for ed_s: Einstein-de Sitter (EdS) universe."""
        return cls.ed_s(h=h)

    @classmethod
    def planck18(cls) -> CosmologicalParameters:
        """Planck 2018 fiducial flat Lambda-CDM."""
        return cls()
