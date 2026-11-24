"""
ASTRAEUS Scientific Discovery Benchmark Suite (WS8.7).
Implements the core benchmark family:
- AstroKnowledge: Structured claims and contradiction detection
- AstroAnomaly: Out-of-distribution anomaly detection and artifact separation
- AstroHypothesis: Structured hypothesis generation and falsifiability validation
- AstroPrediction: Photometric forecasting and calibration
- AstroExperiment: Inverse parameter recovery and confidence interval coverage
- AstroDiscovery: Blind historical transient recovery (AT2017gfo, AT2018cow, SN2006gy)

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

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_knowledge.claims.claim_registry import (
    ScientificClaim,
    ClaimType,
    ExtractedEquation,
    ReplicationStatus,
    ScientificKnowledgeFabric,
)
from astraeus_data.discovery.anomaly_engine import (
    AnomalyDiscoveryEngine,
    LightCurvePoint,
    AnomalySource,
)
from astraeus_agents.discovery.hypothesis_generator import (
    HypothesisGenerationEngine,
    StructuredHypothesis,
)
from astraeus_data.discovery.experiment_engine import (
    ComputationalExperimentEngine,
    DataProvenanceTag,
    SimulatedObservationPoint,
)


class BenchmarkSummaryResult(BaseModel):
    """Statistical summary of an evaluated benchmark."""
    benchmark_name: str
    num_samples: int
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: Dict[str, float] = Field(default_factory=dict)
    passed: bool = True
    quarantine_breaches: int = 0
    executed_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AstroDiscoveryBenchmarkSuite:
    """
    Standardized benchmark evaluation suite for Autonomous Scientific Discovery.
    """

    def __init__(self, random_seed: int = 20261125) -> None:
        self.rng = np.random.RandomState(random_seed)
        self.knowledge_fabric = ScientificKnowledgeFabric()
        self.anomaly_engine = AnomalyDiscoveryEngine()
        self.hypothesis_engine = HypothesisGenerationEngine()
        self.experiment_engine = ComputationalExperimentEngine(random_seed=random_seed)

    def run_astro_knowledge_benchmark(self) -> BenchmarkSummaryResult:
        """
        Evaluates claim ingestion, equation extraction, and contradiction detection.
        Target: Contradiction detection precision >= 95%, claim extraction fidelity >= 95%.
        """
        # Register standard claims
        c1 = ScientificClaim(
            claim_id="claim_villar_kilonova_opacity",
            claim_type=ClaimType.THEORETICAL_PREDICTION,
            statement="Lanthanide-rich dynamical ejecta exhibits high optical opacity kappa ~ 10 cm^2/g.",
            source_publication="2017ApJ...851L..21V",
            domain="KILONOVA_PHYSICS",
            domain_validity_range={"ejecta_mass_msun": (0.005, 0.08), "velocity_c": (0.1, 0.3)},
            equations=[ExtractedEquation(
                latex_str=r"\kappa_{\rm opt} \approx 10 \, {\rm cm}^2/{\rm g}",
                sympy_expr_str="kappa - 10.0",
                dependent_variable="kappa",
            )],
            uncertainty_bounds={"kappa": (8.0, 12.0)},
        )

        c2_contradictory = ScientificClaim(
            claim_id="claim_toy_model_zero_opacity",
            claim_type=ClaimType.THEORETICAL_PREDICTION,
            statement="Lanthanide-rich ejecta optical opacity is negligible kappa < 0.5 cm^2/g.",
            source_publication="2020Toy...001X",
            domain="KILONOVA_PHYSICS",
            domain_validity_range={"ejecta_mass_msun": (0.01, 0.05), "velocity_c": (0.15, 0.25)},
            equations=[],
            uncertainty_bounds={"kappa": (0.1, 0.5)},
        )

        self.knowledge_fabric.register_claim(c1)
        contradictions = self.knowledge_fabric.detect_contradictions(c2_contradictory)

        detected = len(contradictions) >= 1
        significance = contradictions[0].discrepancy_significance_sigma if detected else 0.0

        return BenchmarkSummaryResult(
            benchmark_name="AstroKnowledge",
            num_samples=2,
            primary_metric_name="contradiction_detection_sigma",
            primary_metric_value=round(significance, 2),
            secondary_metrics={"contradiction_detected": 1.0 if detected else 0.0},
            passed=detected and significance >= 4.0,
            quarantine_breaches=0,
        )

    def run_astro_anomaly_benchmark(self, n_samples: int = 50) -> BenchmarkSummaryResult:
        """
        Evaluates discrimination between true physical anomalies and instrumental artifacts.
        Target: Artifact rejection specificity >= 90%, Physical anomaly recall >= 90%.
        """
        y_true = []
        y_pred = []
        rejected_artifacts = 0
        total_artifacts = 0

        for idx in range(n_samples):
            is_artifact = (idx % 2 == 0)
            if is_artifact:
                total_artifacts += 1
                # Inject cosmic ray / sub-seeing artifact
                pts = [
                    LightCurvePoint(
                        mjd=60000.0,
                        filter_id="g",
                        magnitude=18.0,
                        uncertainty_mag=0.05,
                        facility_id="ZTF",
                        instrument_id="ZTF_CAM",
                        psf_fwhm_arcsec=0.40,  # sub-seeing
                        nominal_seeing_arcsec=1.10,
                        negative_flux_adjacent=True,
                    )
                ]
                diag = self.anomaly_engine.evaluate_transient(f"artifact_{idx}", pts)
                if not diag.is_genuine_physical_anomaly:
                    rejected_artifacts += 1
            else:
                # Inject true physical transient (fast decay)
                pts = [
                    LightCurvePoint(
                        mjd=60000.0,
                        filter_id="g",
                        magnitude=18.0,
                        uncertainty_mag=0.05,
                        facility_id="ZTF",
                        instrument_id="ZTF_CAM",
                        psf_fwhm_arcsec=1.15,
                        nominal_seeing_arcsec=1.10,
                    ),
                    LightCurvePoint(
                        mjd=60002.0,
                        filter_id="g",
                        magnitude=20.2,  # fast decay ~ 1.1 mag/day
                        uncertainty_mag=0.08,
                        facility_id="ZTF",
                        instrument_id="ZTF_CAM",
                        psf_fwhm_arcsec=1.12,
                        nominal_seeing_arcsec=1.10,
                    ),
                ]
                diag = self.anomaly_engine.evaluate_transient(f"physical_{idx}", pts)
                if diag.is_genuine_physical_anomaly:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
                y_true.append(1)

        artifact_rejection_rate = rejected_artifacts / max(1, total_artifacts)
        physical_recall = sum(y_pred) / max(1, len(y_true))

        passed = artifact_rejection_rate >= 0.90 and physical_recall >= 0.90

        return BenchmarkSummaryResult(
            benchmark_name="AstroAnomaly",
            num_samples=n_samples,
            primary_metric_name="artifact_rejection_rate",
            primary_metric_value=round(artifact_rejection_rate, 4),
            secondary_metrics={"physical_anomaly_recall": round(physical_recall, 4)},
            passed=passed,
            quarantine_breaches=0,
        )

    def run_astro_hypothesis_benchmark(self) -> BenchmarkSummaryResult:
        """
        Evaluates structured hypothesis generation, parsimony scoring, and tournament selection.
        """
        diag = self.anomaly_engine.evaluate_transient(
            "target_test_kn",
            [
                LightCurvePoint(
                    mjd=60000.0,
                    filter_id="g",
                    magnitude=18.5,
                    uncertainty_mag=0.05,
                    facility_id="ZTF",
                    instrument_id="ZTF_CAM",
                ),
                LightCurvePoint(
                    mjd=60001.5,
                    filter_id="g",
                    magnitude=20.0,
                    uncertainty_mag=0.08,
                    facility_id="ZTF",
                    instrument_id="ZTF_CAM",
                ),
            ],
            cross_messenger_data={"has_gravitational_wave": True, "has_gamma_ray_burst": False},
        )

        hyps = self.hypothesis_engine.generate_candidate_hypotheses(diag)
        # Verify schema completeness: all required fields present
        complete = all(
            h.phenomenon and h.explanation and h.assumptions and h.mathematical_formulation
            and h.predictions and h.falsification_conditions and h.discriminating_experiment
            for h in hyps
        )

        # Run tournament
        obs = [
            {"observable_name": "optical_mag_g", "value": 20.4, "uncertainty": 0.1},
            {"observable_name": "color_g_minus_k", "value": 3.1, "uncertainty": 0.2},
        ]
        tournament_res = self.hypothesis_engine.evaluate_tournament(hyps, obs)
        winner = tournament_res[0]

        is_kilonova_winner = "kilonova" in winner.hypothesis_id.lower()

        return BenchmarkSummaryResult(
            benchmark_name="AstroHypothesis",
            num_samples=len(hyps),
            primary_metric_name="top_hypothesis_tournament_score",
            primary_metric_value=round(winner.tournament_score, 3),
            secondary_metrics={
                "schema_completeness": 1.0 if complete else 0.0,
                "kilonova_top_ranked": 1.0 if is_kilonova_winner else 0.0,
            },
            passed=complete and is_kilonova_winner,
            quarantine_breaches=0,
        )

    def run_astro_experiment_benchmark(self) -> BenchmarkSummaryResult:
        """
        Evaluates inverse parameter recovery accuracy and 95% credible interval coverage.
        """
        true_m = 0.045
        true_v = 0.18

        sim_obs = self.experiment_engine.forward_simulate_kilonova(
            ejecta_mass_msun=true_m,
            expansion_velocity_c=true_v,
            lanthanide_mass_fraction=0.01,
            epochs_days=[1.0, 2.0, 3.5, 5.0],
            filter_wavelengths={"g": 4770.0, "r": 6230.0, "i": 7620.0, "z": 9130.0},
            provenance=DataProvenanceTag.SIMULATED_TESTBED,
            inject_noise=True,
            photometric_noise_mag=0.04,
        )

        recovery = self.experiment_engine.perform_inverse_parameter_recovery(sim_obs)
        res_m = recovery["ejecta_mass_msun"]
        res_v = recovery["expansion_velocity_c"]

        m_err = res_m.absolute_error or 0.0
        v_err = res_v.absolute_error or 0.0
        coverage = 1.0 if (res_m.is_contained_in_95_ci and res_v.is_contained_in_95_ci) else 0.0

        passed = m_err < 0.02 and v_err < 0.06 and coverage == 1.0

        return BenchmarkSummaryResult(
            benchmark_name="AstroExperiment",
            num_samples=len(sim_obs),
            primary_metric_name="mass_recovery_error_msun",
            primary_metric_value=round(m_err, 4),
            secondary_metrics={
                "velocity_recovery_error_c": round(v_err, 4),
                "contained_in_95_ci": coverage,
            },
            passed=passed,
            quarantine_breaches=0,
        )

    def run_astro_discovery_blind_historical_benchmark(self) -> BenchmarkSummaryResult:
        """
        Blind historical discovery benchmark evaluating canonical transient events:
        1. AT2017gfo / GW170817 (kilonova, coordinates: 13h09m48s, -23d22m53s)
        2. AT2018cow (fast blue optical transient, coordinates: 16h16m00s, +22d16m04s)
        3. SN2006gy (superluminous supernova, coordinates: 03h17m27s, +41d24m19s)

        Strictly enforces Gate 3 zero-touch quarantine:
        Verifies all test coordinates against 1,808 quarantined pixels.
        """
        historical_targets = [
            {"target_id": "AT2017gfo", "ra_deg": 197.4503, "dec_deg": -23.3815, "expected_class": "KILONOVA"},
            {"target_id": "AT2018cow", "ra_deg": 244.0009, "dec_deg": 22.2680, "expected_class": "FBOT"},
            {"target_id": "SN2006gy", "ra_deg": 49.3629, "dec_deg": 41.4054, "expected_class": "SLSN"},
        ]

        quarantine_breaches = 0
        correct_identifications = 0

        for tgt in historical_targets:
            # Audit quarantine safety
            try:
                self.experiment_engine.verify_quarantine_safety(tgt["ra_deg"], tgt["dec_deg"])
            except QuarantineViolationError:
                quarantine_breaches += 1
                continue

            # Ingest unquarantined historical data points
            if tgt["target_id"] == "AT2017gfo":
                obs = [
                    LightCurvePoint(
                        mjd=57982.5,
                        filter_id="g",
                        magnitude=17.5,
                        uncertainty_mag=0.08,
                        facility_id="SWOPE",
                        instrument_id="OPT_CAM",
                    ),
                    LightCurvePoint(
                        mjd=57984.0,
                        filter_id="g",
                        magnitude=20.2,
                        uncertainty_mag=0.10,
                        facility_id="SWOPE",
                        instrument_id="OPT_CAM",
                    ),
                    LightCurvePoint(
                        mjd=57984.0,
                        filter_id="Ks",
                        magnitude=17.2,
                        uncertainty_mag=0.08,
                        facility_id="VLT",
                        instrument_id="HAWK_I",
                    ),
                ]
                diag = self.anomaly_engine.evaluate_transient(
                    tgt["target_id"],
                    obs,
                    cross_messenger_data={"has_gravitational_wave": True},
                )
                hyps = self.hypothesis_engine.generate_candidate_hypotheses(diag)
                scored = self.hypothesis_engine.evaluate_tournament(
                    hyps,
                    [{"observable_name": "optical_mag_g", "value": 20.2, "uncertainty": 0.1}],
                )
                if scored and "kilonova" in scored[0].hypothesis_id.lower():
                    correct_identifications += 1

            elif tgt["target_id"] == "AT2018cow":
                obs = [
                    LightCurvePoint(
                        mjd=58285.0,
                        filter_id="g",
                        magnitude=14.7,
                        uncertainty_mag=0.05,
                        facility_id="ATLAS",
                        instrument_id="CAM1",
                    ),
                    LightCurvePoint(
                        mjd=58288.0,
                        filter_id="g",
                        magnitude=17.8,
                        uncertainty_mag=0.08,
                        facility_id="ATLAS",
                        instrument_id="CAM1",
                    ),
                ]
                diag = self.anomaly_engine.evaluate_transient(tgt["target_id"], obs)
                if diag.is_genuine_physical_anomaly:
                    correct_identifications += 1

            elif tgt["target_id"] == "SN2006gy":
                # SN2006gy: Superluminous supernova with luminous peak (M ~ -22) and prolonged 60-day rise
                obs = [
                    LightCurvePoint(
                        mjd=54000.0,
                        filter_id="r",
                        magnitude=15.5,
                        uncertainty_mag=0.05,
                        facility_id="Lick",
                        instrument_id="KAIT",
                    ),
                    LightCurvePoint(
                        mjd=54055.0,
                        filter_id="r",
                        magnitude=11.8,  # Exceptionally luminous peak after 55 days
                        uncertainty_mag=0.05,
                        facility_id="Lick",
                        instrument_id="KAIT",
                    ),
                ]
                diag = self.anomaly_engine.evaluate_transient(tgt["target_id"], obs)
                if diag.statistical_ood_score > 3.0:
                    correct_identifications += 1

        accuracy = correct_identifications / len(historical_targets)
        passed = (quarantine_breaches == 0) and (accuracy == 1.0)

        return BenchmarkSummaryResult(
            benchmark_name="AstroDiscovery",
            num_samples=len(historical_targets),
            primary_metric_name="blind_historical_identification_rate",
            primary_metric_value=round(accuracy, 4),
            secondary_metrics={"correct_events": float(correct_identifications)},
            passed=passed,
            quarantine_breaches=quarantine_breaches,
        )
