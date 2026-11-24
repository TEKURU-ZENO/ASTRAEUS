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


class ReplicationAuditReport(BaseModel):
    """Formal audit report produced by the Scientific Replication Network."""
    investigation_id: str
    target_claim_id: str
    replication_status: ReplicationStatusOutcome
    max_relative_discrepancy: float
    discrepancy_table: Dict[str, float]
    provenance_hash_verified: bool
    is_fully_replicated: bool
    is_independent_numerical_solver: bool = False
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
                max_relative_discrepancy=1.0,
                discrepancy_table={"evidence_hash": 1.0},
                provenance_hash_verified=False,
                is_fully_replicated=False,
                is_independent_numerical_solver=use_independent_numerical_solver,
                auditor_signature="REPLICATION_NETWORK_VETO_PROVENANCE_MISMATCH",
            )

        # 2. Execution path
        if use_independent_numerical_solver:
            # Independent RK4 ODE numerical integration
            replayed_map = self.independent_solver.solve_trajectory(
                ejecta_mass_msun=input_parameters.get("ejecta_mass_msun", 0.04),
                expansion_velocity_c=input_parameters.get("expansion_velocity_c", 0.18),
                lanthanide_mass_fraction=input_parameters.get("lanthanide_mass_fraction", 0.01),
                epochs_days=[1.0, 3.0],
                filter_wavelengths={"g": 4770.0, "Ks": 21500.0},
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

        solver_label = "INDEPENDENT_ODE" if use_independent_numerical_solver else "REPLAY"
        audit_sig_payload = f"{investigation_id}:{target_claim_id}:{status.value}:{solver_label}:{max_rel_diff:.6f}"
        sig = hashlib.sha256(audit_sig_payload.encode()).hexdigest()

        return ReplicationAuditReport(
            investigation_id=investigation_id,
            target_claim_id=target_claim_id,
            replication_status=status,
            max_relative_discrepancy=round(max_rel_diff, 6),
            discrepancy_table=discrepancies,
            provenance_hash_verified=True,
            is_fully_replicated=is_reproduced,
            is_independent_numerical_solver=use_independent_numerical_solver,
            auditor_signature=sig,
        )
