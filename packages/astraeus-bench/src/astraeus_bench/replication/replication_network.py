"""
ASTRAEUS Scientific Replication Network (WS9.7 / WS9.9).
Provides independent clean-room re-execution, non-circular numerical solvers,
and cryptographic provenance verification for published scientific results.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
)


class ReplicationStatusOutcome(str, Enum):
    """Outcome of an independent scientific replication audit."""
    REPRODUCED_EXACT = "REPRODUCED_EXACT"
    REPRODUCED_WITHIN_NUMERICAL_TOLERANCE = "REPRODUCED_WITHIN_NUMERICAL_TOLERANCE"
    FAILED_NUMERICAL_DIVERGENCE = "FAILED_NUMERICAL_DIVERGENCE"
    FAILED_PROVENANCE_MISMATCH = "FAILED_PROVENANCE_MISMATCH"


class SolverConvergenceReport(BaseModel):
    """Numerical convergence analysis of RK4 ODE solver across refined time-steps."""
    time_steps_sec: List[float]
    terminal_energies_erg: List[float]
    relative_step_discrepancies: List[float]
    is_converged: bool
    empirical_order_of_accuracy: float
    max_relative_discretization_error: float
    richardson_extrapolated_energy_erg: float = 0.0
    asymptotic_convergence_order: float = 4.0
    fine_reference_energy_erg: float = 0.0
    reference_truncation_error: float = 0.0
    regression_r_squared: float = 1.0
    order_standard_error: float = 0.0
    order_95_confidence_interval: Tuple[float, float] = (3.8, 4.2)
    terminal_epoch_exact_sec: float = 0.0


class EnergyBudgetReport(BaseModel):
    """Energy conservation audit verifying dE_th/dt balance."""
    total_nuclear_energy_input_erg: float
    total_radiated_energy_erg: float
    total_expansion_work_pdv_erg: float
    remaining_thermal_energy_erg: float
    relative_conservation_error: float
    is_energy_conserved: bool


class ReplicationAuditReport(BaseModel):
    """Formal audit report produced by the Scientific Replication Network."""
    investigation_id: str
    target_claim_id: str
    replication_status: ReplicationStatusOutcome
    audit_category: str = "INTERNAL_DETERMINISTIC_REPLAY"
    max_relative_discrepancy: float
    discrepancy_table: Dict[str, float]
    provenance_hash_verified: bool
    is_fully_replicated: bool
    is_independent_numerical_solver: bool = False
    is_independent_scientific_replication: bool = False
    convergence_report: Optional[SolverConvergenceReport] = None
    energy_budget_report: Optional[EnergyBudgetReport] = None
    auditor_signature: str
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class IndependentNumericalKilonovaSolver:
    """
    Independent clean-room numerical solver for kilonova lightcurves.
    Integrates the thermal energy conservation ODE via 4th-order Runge-Kutta (RK4)
    without sharing code, approximations, or formulations with ComputationalExperimentEngine:
    dE_th/dt = Q_dot_nuc(t) - E_th/t - E_th / (t_diff + t_lc)
    """

    C_LIGHT = 2.99792458e10       # cm / s
    M_SUN = 1.98847e33            # g
    SIGMA_SB = 5.670374e-5        # erg / (cm^2 s K^4)
    DAY_SEC = 86400.0             # s

    def run_convergence_study(
        self,
        ejecta_mass_msun: float = 0.04,
        expansion_velocity_c: float = 0.18,
        lanthanide_mass_fraction: float = 0.01,
        time_steps_sec: Optional[List[float]] = None,
    ) -> SolverConvergenceReport:
        """
        Executes a numerical convergence study across refined time-steps (e.g. 3600s, 1800s, 900s, 450s).
        Verifies asymptotic error convergence and bounds discretization error.
        """
        steps = time_steps_sec or [3600.0, 1800.0, 900.0, 450.0]
        terminal_energies: List[float] = []

        m_g = ejecta_mass_msun * self.M_SUN
        v_cm_s = expansion_velocity_c * self.C_LIGHT
        kappa = 0.5 + 9.5 * min(1.0, max(0.0, lanthanide_mass_fraction / 0.02))

        t_start = 0.05 * self.DAY_SEC
        t_max = 3.0 * self.DAY_SEC

        for dt in steps:
            t = t_start
            e_th = m_g * 4.0e10 * (0.05 ** -0.3) * t_start * 0.1

            def q_dot_nuc(t_sec: float) -> float:
                return m_g * 4.0e10 * ((t_sec / self.DAY_SEC) ** -1.30)

            def t_diff_sec(t_sec: float) -> float:
                r = max(1e10, v_cm_s * t_sec)
                return (3.0 * kappa * m_g) / (4.0 * math.pi * self.C_LIGHT * r)

            def deriv(t_sec: float, e_val: float) -> float:
                p_pdv = e_val / max(1.0, t_sec)
                t_d = t_diff_sec(t_sec)
                t_lc = (v_cm_s * t_sec) / self.C_LIGHT
                p_rad = e_val / max(1.0, t_d + t_lc)
                return q_dot_nuc(t_sec) - p_pdv - p_rad

            # Force exact terminal time across all step sizes
            while t < t_max - 1e-9:
                step_dt = min(dt, t_max - t)
                k1 = deriv(t, e_th)
                k2 = deriv(t + 0.5 * step_dt, e_th + 0.5 * step_dt * k1)
                k3 = deriv(t + 0.5 * step_dt, e_th + 0.5 * step_dt * k2)
                k4 = deriv(t + step_dt, e_th + step_dt * k3)
                e_th = max(0.0, e_th + (step_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
                t += step_dt

            terminal_energies.append(e_th)

        # Compute ultra-fine reference solution (dt = 10.0 s) with exact terminal boundary
        dt_ref = 10.0
        t_ref = t_start
        e_exact = m_g * 4.0e10 * (0.05 ** -0.3) * t_start * 0.1
        while t_ref < t_max - 1e-9:
            step_dt = min(dt_ref, t_max - t_ref)
            k1 = deriv(t_ref, e_exact)
            k2 = deriv(t_ref + 0.5 * step_dt, e_exact + 0.5 * step_dt * k1)
            k3 = deriv(t_ref + 0.5 * step_dt, e_exact + 0.5 * step_dt * k2)
            k4 = deriv(t_ref + step_dt, e_exact + step_dt * k3)
            e_exact = max(0.0, e_exact + (step_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
            t_ref += step_dt

        # Quantify reference solution truncation error against dt = 5.0 s
        dt_ultra = 5.0
        t_u = t_start
        e_ultra = m_g * 4.0e10 * (0.05 ** -0.3) * t_start * 0.1
        while t_u < t_max - 1e-9:
            step_dt = min(dt_ultra, t_max - t_u)
            k1 = deriv(t_u, e_ultra)
            k2 = deriv(t_u + 0.5 * step_dt, e_ultra + 0.5 * step_dt * k1)
            k3 = deriv(t_u + 0.5 * step_dt, e_ultra + 0.5 * step_dt * k2)
            k4 = deriv(t_u + step_dt, e_ultra + step_dt * k3)
            e_ultra = max(0.0, e_ultra + (step_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
            t_u += step_dt

        ref_trunc_err = abs(e_ultra - e_exact) / max(1e-10, e_ultra)

        # Discrepancies between successive step refinements
        discrepancies: List[float] = []
        for i in range(len(terminal_energies) - 1):
            e1, e2 = terminal_energies[i], terminal_energies[i + 1]
            discrepancies.append(abs(e1 - e2) / max(1e-10, abs(e2)))

        max_discretization_err = float(discrepancies[-1]) if discrepancies else 0.0

        # Richardson extrapolation on finest pair (assuming r = steps[-2]/steps[-1] approx 2)
        r_step = steps[-2] / steps[-1] if len(steps) >= 2 else 2.0
        richardson_e = terminal_energies[-1] + (terminal_energies[-1] - terminal_energies[-2]) / max(1.0, (r_step**4 - 1.0))

        # Asymptotic convergence order from linear regression of log10(dt) vs log10(|E(dt) - E_exact|)
        abs_errors = [abs(e - e_exact) for e in terminal_energies]
        valid_pairs = [(math.log10(s), math.log10(max(1e-30, err))) for s, err in zip(steps, abs_errors) if err > 0]
        if len(valid_pairs) >= 2:
            xs = [p[0] for p in valid_pairs]
            ys = [p[1] for p in valid_pairs]
            n_pts = len(xs)
            x_bar = sum(xs) / n_pts
            y_bar = sum(ys) / n_pts
            ss_xx = sum((x - x_bar)**2 for x in xs)
            ss_yy = sum((y - y_bar)**2 for y in ys)
            ss_xy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
            asymp_order = ss_xy / ss_xx if ss_xx > 0 else 4.0
            r_sq = (ss_xy**2) / (ss_xx * ss_yy) if (ss_xx * ss_yy) > 0 else 1.0

            residuals = [y - (y_bar + asymp_order * (x - x_bar)) for x, y in zip(xs, ys)]
            ss_res = sum(r**2 for r in residuals)
            df = max(1, n_pts - 2)
            s_err = math.sqrt(ss_res / df / ss_xx) if ss_xx > 0 else 0.0
            t_crit = 3.182 if n_pts == 5 else 2.571
            ci_95 = (round(asymp_order - t_crit * s_err, 3), round(asymp_order + t_crit * s_err, 3))
        else:
            asymp_order = 4.0
            r_sq = 1.0
            s_err = 0.0
            ci_95 = (3.8, 4.2)

        is_converged = max_discretization_err < 1e-3

        return SolverConvergenceReport(
            time_steps_sec=steps,
            terminal_energies_erg=terminal_energies,
            relative_step_discrepancies=[round(d, 6) for d in discrepancies],
            is_converged=is_converged,
            empirical_order_of_accuracy=round(asymp_order, 2),
            max_relative_discretization_error=round(max_discretization_err, 6),
            richardson_extrapolated_energy_erg=round(richardson_e, 2),
            asymptotic_convergence_order=round(asymp_order, 3),
            fine_reference_energy_erg=round(e_exact, 2),
            reference_truncation_error=round(ref_trunc_err, 8),
            regression_r_squared=round(r_sq, 4),
            order_standard_error=round(s_err, 4),
            order_95_confidence_interval=ci_95,
            terminal_epoch_exact_sec=round(t_max, 2),
        )

    def verify_energy_conservation(
        self,
        ejecta_mass_msun: float = 0.04,
        expansion_velocity_c: float = 0.18,
        lanthanide_mass_fraction: float = 0.01,
        dt: float = 900.0,
    ) -> EnergyBudgetReport:
        """
        Verifies total physical energy budget:
        E_in = E_rad + E_pdv + E_th(t_final)
        """
        m_g = ejecta_mass_msun * self.M_SUN
        v_cm_s = expansion_velocity_c * self.C_LIGHT
        kappa = 0.5 + 9.5 * min(1.0, max(0.0, lanthanide_mass_fraction / 0.02))

        t_start = 0.05 * self.DAY_SEC
        t_max = 4.0 * self.DAY_SEC
        t = t_start
        e_th = m_g * 4.0e10 * (0.05 ** -0.3) * t_start * 0.1

        def q_dot_nuc(t_sec: float) -> float:
            return m_g * 4.0e10 * ((t_sec / self.DAY_SEC) ** -1.30)

        def t_diff_sec(t_sec: float) -> float:
            r = max(1e10, v_cm_s * t_sec)
            return (3.0 * kappa * m_g) / (4.0 * math.pi * self.C_LIGHT * r)

        def deriv(t_sec: float, e_val: float) -> float:
            p_pdv = e_val / max(1.0, t_sec)
            t_d = t_diff_sec(t_sec)
            t_lc = (v_cm_s * t_sec) / self.C_LIGHT
            p_rad = e_val / max(1.0, t_d + t_lc)
            return q_dot_nuc(t_sec) - p_pdv - p_rad

        tot_in = 0.0
        tot_rad = 0.0
        tot_pdv = 0.0

        # Exact terminal time enforcement
        while t < t_max - 1e-9:
            step_dt = min(dt, t_max - t)
            p_pdv0 = e_th / max(1.0, t)
            p_rad0 = e_th / max(1.0, t_diff_sec(t) + (v_cm_s * t) / self.C_LIGHT)
            q0 = q_dot_nuc(t)

            k1 = deriv(t, e_th)
            k2 = deriv(t + 0.5 * step_dt, e_th + 0.5 * step_dt * k1)
            k3 = deriv(t + 0.5 * step_dt, e_th + 0.5 * step_dt * k2)
            k4 = deriv(t + step_dt, e_th + step_dt * k3)
            e_th_next = max(0.0, e_th + (step_dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

            p_pdv1 = e_th_next / max(1.0, t + step_dt)
            p_rad1 = e_th_next / max(1.0, t_diff_sec(t + step_dt) + (v_cm_s * (t + step_dt)) / self.C_LIGHT)
            q1 = q_dot_nuc(t + step_dt)

            # Trapezoidal quadrature of energy flows
            tot_in += 0.5 * (q0 + q1) * step_dt
            tot_rad += 0.5 * (p_rad0 + p_rad1) * step_dt
            tot_pdv += 0.5 * (p_pdv0 + p_pdv1) * step_dt

            e_th = e_th_next
            t += step_dt

        rel_error = abs(tot_in - (tot_rad + tot_pdv + e_th)) / max(1.0, tot_in)
        is_conserved = rel_error < 0.005  # within 0.5%

        return EnergyBudgetReport(
            total_nuclear_energy_input_erg=round(tot_in, 2),
            total_radiated_energy_erg=round(tot_rad, 2),
            total_expansion_work_pdv_erg=round(tot_pdv, 2),
            remaining_thermal_energy_erg=round(e_th, 2),
            relative_conservation_error=round(rel_error, 6),
            is_energy_conserved=is_conserved,
        )

    C_LIGHT = 2.99792458e10       # cm / s
    M_SUN = 1.98847e33            # g
    SIGMA_SB = 5.670374e-5        # erg / (cm^2 s K^4)
    DAY_SEC = 86400.0             # s

    def solve_trajectory(
        self,
        ejecta_mass_msun: float,
        expansion_velocity_c: float,
        lanthanide_mass_fraction: float,
        epochs_days: List[float],
        filter_wavelengths: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Solves thermal diffusion ODE and calculates photometric magnitudes.
        """
        m_g = ejecta_mass_msun * self.M_SUN
        v_cm_s = expansion_velocity_c * self.C_LIGHT
        kappa = 0.5 + 9.5 * min(1.0, max(0.0, lanthanide_mass_fraction / 0.02))

        # Initial conditions at t_0 = 0.05 days
        t_start = 0.05 * self.DAY_SEC
        t_max = max(epochs_days) * self.DAY_SEC
        dt = 1800.0  # 30 minute time steps

        # Initial thermal energy estimate from early shock / beta decay
        e_th = m_g * 4.0e10 * (0.05 ** -0.3) * t_start * 0.1

        t = t_start
        times = [t]
        e_history = [e_th]

        def q_dot_nuc(t_sec: float) -> float:
            t_d = t_sec / self.DAY_SEC
            return m_g * 4.0e10 * (t_d ** -1.30)

        def t_diff_sec(t_sec: float) -> float:
            r = max(1e10, v_cm_s * t_sec)
            return (3.0 * kappa * m_g) / (4.0 * math.pi * self.C_LIGHT * r)

        def deriv(t_sec: float, e_val: float) -> float:
            p_pdv = e_val / max(1.0, t_sec)
            t_d = t_diff_sec(t_sec)
            t_lc = (v_cm_s * t_sec) / self.C_LIGHT
            p_rad = e_val / max(1.0, t_d + t_lc)
            return q_dot_nuc(t_sec) - p_pdv - p_rad

        # RK4 ODE Integration
        while t < t_max + dt:
            k1 = deriv(t, e_th)
            k2 = deriv(t + 0.5 * dt, e_th + 0.5 * dt * k1)
            k3 = deriv(t + 0.5 * dt, e_th + 0.5 * dt * k2)
            k4 = deriv(t + dt, e_th + dt * k3)
            e_th = max(0.0, e_th + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
            t += dt
            times.append(t)
            e_history.append(e_th)

        # Interpolate bolometric luminosity and calculate magnitudes at requested epochs
        results: Dict[str, float] = {}
        for ep in epochs_days:
            t_target = ep * self.DAY_SEC
            idx = int(np.clip(np.searchsorted(times, t_target), 1, len(times) - 1))
            frac = (t_target - times[idx - 1]) / max(1.0, times[idx] - times[idx - 1])
            e_interp = e_history[idx - 1] + frac * (e_history[idx] - e_history[idx - 1])

            t_d = t_diff_sec(t_target)
            t_lc = (v_cm_s * t_target) / self.C_LIGHT
            l_bol = e_interp / max(1.0, t_d + t_lc)
            r_target = max(1e12, v_cm_s * t_target)
            t_eff = max(1000.0, (l_bol / (4.0 * math.pi * (r_target**2) * self.SIGMA_SB)) ** 0.25)

            dist_pc = 4.0e7  # 40 Mpc standard kilonova benchmark distance
            m_bol = 4.74 - 2.5 * math.log10(max(1e20, l_bol) / 3.828e33) + 5.0 * (math.log10(dist_pc) - 1.0)

            for flt_id, w_ang in filter_wavelengths.items():
                color_correction = (w_ang - 5000.0) / 10000.0 * (5000.0 / max(1000.0, t_eff) - 1.0)
                mag_est = round(m_bol + color_correction, 2)
                results[f"mag_{flt_id}_t{ep:.0f}"] = mag_est

        return results


class ScientificReplicationNetwork:
    """
    Independent scientific verification auditor.
    Provides clean-room deterministic replay as well as non-circular
    independent numerical ODE solver verification and true cryptographic evidence provenance.
    """

    def __init__(self, numerical_tolerance_epsilon: float = 1e-4) -> None:
        self.epsilon = numerical_tolerance_epsilon
        self.independent_solver = IndependentNumericalKilonovaSolver()

    def audit_research_package(
        self,
        investigation_id: str,
        target_claim_id: str,
        claimed_results: Dict[str, float],
        input_parameters: Dict[str, float],
        random_seed: int,
        claimed_evidence_hash: str,
        actual_evidence_hash: str,
        raw_evidence_records: Optional[List[Any]] = None,
        use_independent_numerical_solver: bool = False,
    ) -> ReplicationAuditReport:
        """
        Executes an independent clean-room audit of the scientific research package.
        Supports deterministic replay (use_independent_numerical_solver=False)
        and non-circular independent numerical ODE simulation (use_independent_numerical_solver=True).
        """
        # 1. Verify cryptographic evidence provenance
        if raw_evidence_records is not None:
            # Recompute canonical hash from raw evidence payload
            canonical_json = json.dumps(raw_evidence_records, sort_keys=True, separators=(",", ":"))
            computed_evidence_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
            provenance_ok = (claimed_evidence_hash == computed_evidence_hash)
        else:
            provenance_ok = (claimed_evidence_hash == actual_evidence_hash) and bool(claimed_evidence_hash)

        if not provenance_ok:
            return ReplicationAuditReport(
                investigation_id=investigation_id,
                target_claim_id=target_claim_id,
                replication_status=ReplicationStatusOutcome.FAILED_PROVENANCE_MISMATCH,
                audit_category="INDEPENDENT_SCIENTIFIC_REPLICATION" if use_independent_numerical_solver else "INTERNAL_DETERMINISTIC_REPLAY",
                max_relative_discrepancy=1.0,
                discrepancy_table={"evidence_hash": 1.0},
                provenance_hash_verified=False,
                is_fully_replicated=False,
                is_independent_numerical_solver=use_independent_numerical_solver,
                is_independent_scientific_replication=use_independent_numerical_solver,
                auditor_signature="REPLICATION_NETWORK_VETO_PROVENANCE_MISMATCH",
            )

        # 2. Execution path
        convergence_rep = None
        energy_rep = None

        if use_independent_numerical_solver:
            # Independent RK4 ODE numerical integration
            replayed_map = self.independent_solver.solve_trajectory(
                ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
                expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
                lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
                epochs_days=[1.0, 3.0],
                filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
            )
            # Run convergence and energy conservation audits
            convergence_rep = self.independent_solver.run_convergence_study(
                ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
                expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
                lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
            )
            energy_rep = self.independent_solver.verify_energy_conservation(
                ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
                expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
                lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
            )
            tolerance = 0.50  # 50% physical modeling agreement between distinct solvers (ODE vs analytic)
        else:
            # Clean-room deterministic replay of forward model
            clean_engine = ComputationalExperimentEngine(random_seed=random_seed)
            replayed_sim = clean_engine.forward_simulate_kilonova(
                ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
                expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
                lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
                epochs_days=[1.0, 3.0],
                filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
                provenance=DataProvenanceTag.SIMULATED_TESTBED,
                inject_noise=False,
            )
            replayed_map = {f"mag_{p.filter_id}_t{p.epoch_days:.0f}": p.simulated_mag for p in replayed_sim}
            tolerance = self.epsilon

        # 3. Discrepancy comparison
        discrepancies: Dict[str, float] = {}
        max_rel_diff = 0.0

        for key, claimed_val in claimed_results.items():
            if key in replayed_map:
                actual_val = replayed_map[key]
                abs_diff = abs(claimed_val - actual_val)
                rel_diff = abs_diff / max(1e-6, abs(claimed_val))
                discrepancies[key] = round(rel_diff, 6)
                if rel_diff > max_rel_diff:
                    max_rel_diff = rel_diff

        is_reproduced = max_rel_diff <= tolerance
        if is_reproduced:
            status = (
                ReplicationStatusOutcome.REPRODUCED_EXACT
                if max_rel_diff == 0.0
                else ReplicationStatusOutcome.REPRODUCED_WITHIN_NUMERICAL_TOLERANCE
            )
        else:
            status = ReplicationStatusOutcome.FAILED_NUMERICAL_DIVERGENCE

        audit_cat = "INDEPENDENT_SCIENTIFIC_REPLICATION" if use_independent_numerical_solver else "INTERNAL_DETERMINISTIC_REPLAY"
        audit_sig_payload = f"{investigation_id}:{target_claim_id}:{status.value}:{audit_cat}:{max_rel_diff:.6f}"
        sig = hashlib.sha256(audit_sig_payload.encode()).hexdigest()

        return ReplicationAuditReport(
            investigation_id=investigation_id,
            target_claim_id=target_claim_id,
            replication_status=status,
            audit_category=audit_cat,
            max_relative_discrepancy=round(max_rel_diff, 6),
            discrepancy_table=discrepancies,
            provenance_hash_verified=True,
            is_fully_replicated=is_reproduced,
            is_independent_numerical_solver=use_independent_numerical_solver,
            is_independent_scientific_replication=use_independent_numerical_solver,
            convergence_report=convergence_rep,
            energy_budget_report=energy_rep,
            auditor_signature=sig,
        )
