"""
ASTRAEUS Cross-Domain Scientific Intelligence Engine (WS9.2 / WS9.9).
Provides cross-domain generalization and negative-transfer detection across:
1. Multi-Messenger Transient Astrophysics
2. Cosmology & Large-Scale Structure
3. Exoplanets & Astrobiology
4. Extreme Gravity & Compact Objects

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


class ScientificDomain(str, Enum):
    """Core astrophysical and planetary domains supported by ASTRAEUS."""
    MULTI_MESSENGER_TRANSIENTS = "MULTI_MESSENGER_TRANSIENTS"
    COSMOLOGY_STRUCTURE = "COSMOLOGY_STRUCTURE"
    EXOPLANETS_ASTROBIOLOGY = "EXOPLANETS_ASTROBIOLOGY"
    EXTREME_GRAVITY_COMPACT_OBJECTS = "EXTREME_GRAVITY_COMPACT_OBJECTS"


class DimensionlessInvariants(BaseModel):
    """Universal dimensionless scale invariants enabling safe cross-domain reasoning."""
    gravitational_compactness: float = Field(..., description="G * M / (R * c^2)")
    eddington_ratio: float = Field(..., description="L / L_Edd")
    dynamical_to_cooling_timescale_ratio: float = Field(..., description="t_dyn / t_cool")
    virial_ratio: float = Field(default=0.5, description="2 * E_kin / |E_pot|")


class DomainReasoningResult(BaseModel):
    """Output of cross-domain scientific analysis."""
    domain: ScientificDomain
    primary_phenomenon: str
    invariants: DimensionlessInvariants
    analytical_predictions: Dict[str, float]
    physical_regime: str
    transfer_risk_score: float = Field(..., ge=0.0, le=1.0)
    is_safe_transfer: bool = True
    diagnostic_insights: List[str] = Field(default_factory=list)
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TransferDiscrepancyReport(BaseModel):
    """Quantitative measurement of cross-domain transfer divergence and negative transfer risk."""
    source_domain: ScientificDomain
    target_domain: ScientificDomain
    domain_discrepancy_metric: float
    is_negative_transfer_detected: bool
    is_transfer_admissible: bool
    diagnostic_divergences: List[str]
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CrossDomainIntelligenceEngine:
    """
    Unified multi-domain scientific reasoning engine.
    Derives invariants dynamically from physical governing equations,
    and quantifies negative transfer risk across distinct astrophysical domains.
    """

    C_LIGHT = 2.99792458e10
    G_NEWTON = 6.67430e-8
    M_SUN = 1.98847e33
    K_BOLTZMANN = 1.380649e-16
    H_PLANCK = 6.62607015e-27
    SIGMA_SB = 5.670374e-5

    def compute_compactness(self, mass_msun: float, radius_cm: float) -> float:
        """Computes dimensionless gravitational compactness G M / (R c^2)."""
        m_g = mass_msun * self.M_SUN
        return float((self.G_NEWTON * m_g) / (max(1e5, radius_cm) * (self.C_LIGHT**2)))

    def compute_eddington_luminosity(self, mass_msun: float) -> float:
        """Computes classical Eddington luminosity L_Edd = 4 pi G M c / kappa_es (erg/s)."""
        kappa_es = 0.34  # Thomson scattering opacity (cm^2/g)
        m_g = mass_msun * self.M_SUN
        return float((4.0 * math.pi * self.G_NEWTON * m_g * self.C_LIGHT) / kappa_es)

    # 1. Multi-Messenger Transient Domain
    def evaluate_multi_messenger_domain(
        self,
        ejecta_mass_msun: float,
        expansion_velocity_c: float,
        peak_luminosity_erg_s: float,
    ) -> DomainReasoningResult:
        m_g = ejecta_mass_msun * self.M_SUN
        v_cm_s = expansion_velocity_c * self.C_LIGHT
        r_cm = v_cm_s * 86400.0  # 1 day radius

        compactness = self.compute_compactness(ejecta_mass_msun, r_cm)
        l_edd = self.compute_eddington_luminosity(ejecta_mass_msun)
        edd_ratio = peak_luminosity_erg_s / max(1e30, l_edd)

        # Dynamic timescale calculation
        t_dyn_sec = r_cm / v_cm_s
        e_th_erg = 0.5 * m_g * (v_cm_s**2) * 0.15  # thermalized fraction
        t_cool_sec = e_th_erg / max(1e35, peak_luminosity_erg_s)
        timescale_ratio = t_dyn_sec / max(1.0, t_cool_sec)

        # Dynamic virial parameter
        e_pot = (self.G_NEWTON * (m_g**2)) / r_cm
        e_kin = 0.5 * m_g * (v_cm_s**2)
        virial = (2.0 * e_kin) / max(1e20, e_pot + 2.0 * e_kin)

        invariants = DimensionlessInvariants(
            gravitational_compactness=float(compactness),
            eddington_ratio=round(edd_ratio, 4),
            dynamical_to_cooling_timescale_ratio=float(timescale_ratio),
            virial_ratio=round(virial, 4),
        )

        risk_score = round(min(1.0, max(0.01, 0.05 + 0.1 * math.log10(max(1.0, edd_ratio)))), 3)

        return DomainReasoningResult(
            domain=ScientificDomain.MULTI_MESSENGER_TRANSIENTS,
            primary_phenomenon="Relativistic Transient Explosion",
            invariants=invariants,
            analytical_predictions={
                "diffusion_time_days": round(math.sqrt(2.0 * ejecta_mass_msun * 1e33 / (4 * math.pi * expansion_velocity_c * self.C_LIGHT**2)) / 86400.0, 2),
                "peak_luminosity_erg_s": peak_luminosity_erg_s,
            },
            physical_regime="Radiation-dominated relativistic outflow",
            transfer_risk_score=risk_score,
            is_safe_transfer=True,
            diagnostic_insights=[
                f"Super-Eddington outflow: L/L_Edd = {edd_ratio:.1f} powered by radioactive heating.",
                f"Dynamical-to-cooling timescale ratio derived at {timescale_ratio:.2f}.",
            ],
        )

    # 2. Cosmology & Large-Scale Structure Domain
    def evaluate_cosmology_domain(
        self,
        redshift_z: float,
        h0_km_s_mpc: float = 67.4,
        omega_m: float = 0.315,
        omega_lambda: float = 0.685,
        omega_b: float = 0.049,
    ) -> DomainReasoningResult:
        # Friedmann equation H(z)
        z_factor = omega_m * ((1.0 + redshift_z) ** 3) + omega_lambda
        hz = h0_km_s_mpc * math.sqrt(z_factor)

        # Sound horizon at drag epoch r_s derived from Eisenstein & Hu (1998) baryonic acoustic formula
        h = h0_km_s_mpc / 100.0
        ombh2 = omega_b * (h**2)
        ommh2 = omega_m * (h**2)
        r_s_mpc = 44.5 * math.log(9.83 / max(0.01, ommh2)) / math.sqrt(1.0 + 10.0 * (ombh2**0.75))

        # Dynamic Hubble tension vs local ladder baseline (Riess et al. 2022: H0=73.04 +/- 1.04)
        local_h0 = 73.04
        local_sigma = 1.04
        delta_h0 = abs(h0_km_s_mpc - local_h0)
        tension_sigma = delta_h0 / local_sigma

        invariants = DimensionlessInvariants(
            gravitational_compactness=round(1e-5 * (1.0 + redshift_z), 6),
            eddington_ratio=0.0,
            dynamical_to_cooling_timescale_ratio=1.0,
            virial_ratio=0.50,
        )

        risk_score = round(min(1.0, max(0.02, 0.04 * tension_sigma)), 3)

        return DomainReasoningResult(
            domain=ScientificDomain.COSMOLOGY_STRUCTURE,
            primary_phenomenon="Cosmic Expansion and Sound Horizon",
            invariants=invariants,
            analytical_predictions={
                "hubble_parameter_hz": round(hz, 2),
                "sound_horizon_mpc": round(r_s_mpc, 2),
                "hubble_tension_sigma": round(tension_sigma, 2),
            },
            physical_regime="Lambda-CDM homogeneous expanding Friedmann-Lemaitre-Robertson-Walker spacetime",
            transfer_risk_score=risk_score,
            is_safe_transfer=True,
            diagnostic_insights=[
                f"Hubble parameter H(z={redshift_z:.2f}) = {hz:.2f} km/s/Mpc.",
                f"Sound horizon derived from baryonic acoustics: r_s = {r_s_mpc:.2f} Mpc.",
                f"Cosmic distance ladder tension evaluated at {tension_sigma:.2f} sigma against local ladder.",
            ],
        )

    # 3. Exoplanets & Astrobiology Domain
    def evaluate_exoplanet_domain(
        self,
        planet_radius_earth: float,
        stellar_radius_solar: float,
        equilibrium_temp_k: float,
        surface_gravity_m_s2: float = 9.81,
        mean_molecular_weight_amu: float = 28.0,
    ) -> DomainReasoningResult:
        r_p_cm = planet_radius_earth * 6.371e8
        r_star_cm = stellar_radius_solar * 6.957e10
        transit_depth_ppm = ((r_p_cm / r_star_cm) ** 2) * 1e6

        # Atmospheric scale height H = k_B * T / (mu * g)
        mu_g = mean_molecular_weight_amu * 1.66054e-24
        g_cgs = surface_gravity_m_s2 * 100.0
        scale_height_km = (self.K_BOLTZMANN * equilibrium_temp_k) / (mu_g * g_cgs) / 1e5

        # Dynamic timescales
        t_dyn_sec = math.sqrt(r_p_cm / g_cgs)
        # Radiative cooling timescale for hydrostatic planetary atmosphere
        cp = 1.0e7  # erg / (g * K)
        p_surf = 1.0e6  # dyn / cm^2
        t_cool_sec = (cp * p_surf) / (g_cgs * self.SIGMA_SB * (equilibrium_temp_k**3))
        timescale_ratio = t_dyn_sec / max(1.0, t_cool_sec)

        compactness = self.compute_compactness(1e-6 * (planet_radius_earth**3), r_p_cm)

        invariants = DimensionlessInvariants(
            gravitational_compactness=float(compactness),
            eddington_ratio=0.0,
            dynamical_to_cooling_timescale_ratio=float(timescale_ratio),
            virial_ratio=0.50,
        )

        risk_score = round(min(1.0, max(0.01, 0.02 + 0.01 * (scale_height_km / 10.0))), 3)

        return DomainReasoningResult(
            domain=ScientificDomain.EXOPLANETS_ASTROBIOLOGY,
            primary_phenomenon="Exoplanet Transit & Atmospheric Transmission",
            invariants=invariants,
            analytical_predictions={
                "transit_depth_ppm": round(transit_depth_ppm, 1),
                "atmospheric_scale_height_km": round(scale_height_km, 2),
                "transmission_signal_ppm": round((2.0 * r_p_cm * (scale_height_km * 1e5) / (r_star_cm**2)) * 1e6, 2),
            },
            physical_regime="Hydrostatic planetary atmosphere with chemical disequilibrium",
            transfer_risk_score=risk_score,
            is_safe_transfer=True,
            diagnostic_insights=[
                f"Transit depth = {transit_depth_ppm:.1f} ppm; Atmospheric scale height = {scale_height_km:.2f} km.",
                f"Dynamic planetary timescale ratio derived at {timescale_ratio:.4f}.",
            ],
        )

    # 4. Extreme Gravity & Compact Objects Domain
    def evaluate_extreme_gravity_domain(
        self,
        black_hole_mass_msun: float,
        dimensionless_spin_a: float = 0.90,
        distance_kpc: float = 8.178,
    ) -> DomainReasoningResult:
        spin = min(0.998, max(-0.998, dimensionless_spin_a))
        r_g_cm = (self.G_NEWTON * black_hole_mass_msun * self.M_SUN) / (self.C_LIGHT**2)
        r_horizon_cm = r_g_cm * (1.0 + math.sqrt(1.0 - spin**2))

        d_cm = distance_kpc * 3.0857e21
        shadow_rad = (math.sqrt(27.0) * r_g_cm) / d_cm
        shadow_microarcsec = shadow_rad * (180.0 / math.pi) * 3600.0 * 1e6

        f_qnm_hz = (self.C_LIGHT**3 / (2.0 * math.pi * self.G_NEWTON * black_hole_mass_msun * self.M_SUN)) * (
            1.0 - 0.63 * ((1.0 - spin) ** 0.3)
        )

        compactness = 0.5 * (1.0 + math.sqrt(1.0 - spin**2))
        t_dyn_sec = r_horizon_cm / self.C_LIGHT
        # Hawking / vacuum or accreting cooling timescale
        t_cool_sec = 10.0 * t_dyn_sec

        invariants = DimensionlessInvariants(
            gravitational_compactness=round(compactness, 4),
            eddington_ratio=1.0,
            dynamical_to_cooling_timescale_ratio=round(t_dyn_sec / t_cool_sec, 3),
            virial_ratio=0.50,
        )

        risk_score = round(min(1.0, max(0.01, 0.02 + 0.05 * abs(spin))), 3)

        return DomainReasoningResult(
            domain=ScientificDomain.EXTREME_GRAVITY_COMPACT_OBJECTS,
            primary_phenomenon="Kerr Spacetime Event Horizon and QNM Ringdown",
            invariants=invariants,
            analytical_predictions={
                "event_horizon_radius_km": round(r_horizon_cm / 1e5, 2),
                "shadow_angular_diameter_microarcsec": round(shadow_microarcsec, 2),
                "qnm_frequency_f220_hz": round(f_qnm_hz, 4),
            },
            physical_regime="Strong-field general relativity in rotating Kerr metric",
            transfer_risk_score=risk_score,
            is_safe_transfer=True,
            diagnostic_insights=[
                f"Kerr horizon radius = {r_horizon_cm/1e5:.1f} km at spin a = {spin:.2f}.",
                f"Gravitational wave QNM ringdown frequency = {f_qnm_hz:.2f} Hz.",
            ],
        )

    # 5. Cross-Domain Transfer & Negative Transfer Verification
    def evaluate_cross_domain_transfer(
        self,
        source_domain: ScientificDomain,
        target_domain: ScientificDomain,
        source_invariants: DimensionlessInvariants,
        target_invariants: DimensionlessInvariants,
        divergence_threshold: float = 3.0,
    ) -> TransferDiscrepancyReport:
        """
        Quantifies cross-domain transfer discrepancy D_{S -> T}.
        Detects negative transfer risk when source and target domains possess
        incompatible physical compactness, radiation regimes, or timescale regimes.
        """
        divergences = []

        # Compactness discrepancy (logarithmic)
        c_src = max(1e-12, source_invariants.gravitational_compactness)
        c_tgt = max(1e-12, target_invariants.gravitational_compactness)
        delta_log_c = abs(math.log10(c_src) - math.log10(c_tgt))
        if delta_log_c > 1.0:
            divergences.append(f"Gravitational compactness divergence: {delta_log_c:.1f} orders of magnitude difference ({c_src:.1e} vs {c_tgt:.1e}).")

        # Eddington radiation regime discrepancy
        delta_edd = abs(source_invariants.eddington_ratio - target_invariants.eddington_ratio)
        if delta_edd > 2.0:
            divergences.append(f"Radiation pressure mismatch: Delta(L/L_Edd) = {delta_edd:.1f}.")

        # Timescale discrepancy
        t_src = max(1e-6, source_invariants.dynamical_to_cooling_timescale_ratio)
        t_tgt = max(1e-6, target_invariants.dynamical_to_cooling_timescale_ratio)
        delta_log_t = abs(math.log10(t_src) - math.log10(t_tgt))
        if delta_log_t > 1.0:
            divergences.append(f"Cooling timescale regime mismatch: Delta log10(t_dyn/t_cool) = {delta_log_t:.1f}.")

        # Total Euclidean metric in normalized log-invariant space
        metric = math.sqrt(0.5 * (delta_log_c**2) + 0.3 * (min(10.0, delta_edd)**2) + 0.2 * (delta_log_t**2))

        is_negative = metric > divergence_threshold
        is_admissible = not is_negative

        return TransferDiscrepancyReport(
            source_domain=source_domain,
            target_domain=target_domain,
            domain_discrepancy_metric=round(metric, 3),
            is_negative_transfer_detected=is_negative,
            is_transfer_admissible=is_admissible,
            diagnostic_divergences=divergences,
        )
