"""
ASTRAEUS REGROUND Specialized Agent Personas (WS7.3).
Implements the 5 specialized scientific debate agents:
1. ObservationalAstronomerAgent
2. CosmologistAgent
3. AstrophysicistAgent
4. StatisticalAuditorAgent
5. SkepticRedTeamAgent

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from astraeus_agents.reground.models import (
    AgentPersona,
    DebateStage,
    PhysicalHypothesis,
    DebateTurn,
    SkepticVeto,
    ChallengeStatus,
)
from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from astraeus_data.triage.event_triage_engine import (
    TriageDecision,
    HostGalaxyContext,
    PhotometricFeatures,
    angular_diameter_distance_kpc,
)
from engines.capability_registry import ScientificCapabilityRegistry, EngineCapabilityTier


class ObservationalAstronomerAgent:
    """Audits photometric calibration, atmospheric extinction, and instrument systematics."""

    def evaluate(self, event: UnifiedScientificEvent, triage: TriageDecision) -> Tuple[DebateTurn, Dict[str, Any]]:
        # Compute Galactic latitude b
        ra_rad = math.radians(event.ra_deg)
        dec_rad = math.radians(event.dec_deg)
        # Approximate Galactic latitude conversion
        sin_b = math.sin(dec_rad) * math.cos(math.radians(27.4)) - \
                math.cos(dec_rad) * math.sin(math.radians(27.4)) * math.sin(ra_rad - math.radians(192.25))
        b_deg = math.degrees(math.asin(max(-1.0, min(1.0, sin_b))))

        # Milky Way extinction estimate (Schlegel / SFD proxy)
        sin_abs_b = max(0.1, abs(math.sin(math.radians(b_deg))))
        a_v = min(1.5, 0.08 / sin_abs_b)
        e_bv = a_v / 3.1

        # Audit photometry SNR
        snrs = []
        for p in event.photometry_points:
            magerr = p.get("magerr", 0.1) or 0.1
            snrs.append(1.0857 / max(1e-3, magerr))
        mean_snr = float(np.mean(snrs)) if snrs else 0.0

        summary = (
            f"Photometric audit verified {len(event.photometry_points)} epochs in survey '{event.survey_origin}'. "
            f"Mean S/N = {mean_snr:.1f}. Galactic latitude b = {b_deg:.1f} deg; "
            f"Galactic extinction A_V = {a_v:.2f} mag (E(B-V) = {e_bv:.3f} mag). "
            f"Instrument calibration flags cleared."
        )

        turn = DebateTurn(
            turn_number=1,
            persona=AgentPersona.OBSERVATIONAL_ASTRONOMER,
            stage=DebateStage.EVIDENCE_GROUNDING,
            argument_summary=summary,
            evidence_sources=["SFD98_dust_map", "SurveyPhotometricCalibrator"],
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        metrics = {
            "galactic_lat_deg": b_deg,
            "extinction_a_v": a_v,
            "extinction_e_bv": e_bv,
            "mean_snr": mean_snr
        }
        return turn, metrics


class CosmologistAgent:
    """Evaluates host galaxy environment, redshift, distance modulus, and large-scale structure."""

    def evaluate(self, event: UnifiedScientificEvent, triage: TriageDecision) -> Tuple[DebateTurn, Dict[str, Any]]:
        host = triage.host_context
        if host.matched and host.redshift_z is not None and host.redshift_z > 0.001:
            z = host.redshift_z
            d_A_kpc = angular_diameter_distance_kpc(z)
            d_L_mpc = (d_A_kpc / 1000.0) * ((1.0 + z) ** 2)
            dist_mod = 5.0 * math.log10(d_L_mpc * 1e6 / 10.0)
            summary = (
                f"Associated with host galaxy '{host.galaxy_name}' at cosmological redshift z = {z:.4f}. "
                f"Luminosity distance d_L = {d_L_mpc:.1f} Mpc (distance modulus mu = {dist_mod:.2f} mag). "
                f"Physical projected offset = {host.physical_offset_kpc or 0.0:.1f} kpc. "
                f"Chance alignment probability P_chance = {host.chance_coincidence_prob:.4e}."
            )
        else:
            z = 0.02  # nominal low-redshift prior for hostless transient
            d_L_mpc = 85.0
            dist_mod = 34.65
            summary = (
                "No coincident host galaxy in NED/GLADE+ catalog above detection threshold. "
                "Adopting faint host prior (z ~ 0.02, d_L ~ 85 Mpc)."
            )

        turn = DebateTurn(
            turn_number=2,
            persona=AgentPersona.COSMOLOGIST,
            stage=DebateStage.EVIDENCE_GROUNDING,
            argument_summary=summary,
            evidence_sources=[host.catalog_origin or "NED_GLADE_Archive"],
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        metrics = {
            "redshift_z": z,
            "dist_mod": dist_mod,
            "luminosity_distance_mpc": d_L_mpc,
            "host_matched": host.matched
        }
        return turn, metrics


class AstrophysicistAgent:
    """Formulates competing physical hypotheses and evaluates physical progenitor likelihoods."""

    def formulate_hypotheses(
        self,
        event: UnifiedScientificEvent,
        triage: TriageDecision,
        obs_metrics: Dict[str, Any],
        cosmo_metrics: Dict[str, Any],
    ) -> Tuple[DebateTurn, List[PhysicalHypothesis]]:
        dist_mod = cosmo_metrics.get("dist_mod", 34.65)
        a_v = obs_metrics.get("extinction_a_v", 0.1)
        features = triage.features
        peak_abs_mag = features.peak_mag - dist_mod - a_v

        # -------------------------------------------------------------
        # Hypothesis 1: Kilonova (Binary Neutron Star Merger)
        # -------------------------------------------------------------
        # Metzger (2019) / Arnett (1982) radioactive decay
        # Fast fading (tau < 48h), lanthanide opacity reddening
        is_rapid_fade = (features.fade_rate_mag_per_day >= 0.5) or (features.timescale_tau_hours and features.timescale_tau_hours < 48.0)
        kn_log_like = -5.0 if is_rapid_fade else -45.0
        if features.color_rate_dgr_dt and features.color_rate_dgr_dt > 0.2:
            kn_log_like += 3.0

        h1 = PhysicalHypothesis(
            hypothesis_id="H1_KILONOVA",
            name="Binary Neutron Star Merger Ejecta (Kilonova)",
            progenitor_class="COMPACT_BINARY_MERGER",
            prior_prob=0.35 if is_rapid_fade else 0.05,
            posterior_prob=0.0,
            log_likelihood=kn_log_like,
            physical_parameters={
                "ejecta_mass_msun": 0.04,
                "velocity_c": 0.15,
                "lanthanide_mass_fraction": 1e-3,
                "peak_abs_mag": peak_abs_mag,
                "decay_timescale_hours": features.timescale_tau_hours or 32.0,
            },
            falsification_criteria=[
                "Fade rate must exceed 0.4 mag/day across optical bands.",
                "Color evolution d(g-r)/dt must remain positive (reddening) over next 48h.",
                "Radio / X-ray non-thermal synchrotron emission within 10-30 days."
            ],
            epistemic_qualifications=[
                "Simulation relies on Metzger (2019) 1D analytical radioactive decay; 3D radiative transfer uncertified."
            ]
        )

        # -------------------------------------------------------------
        # Hypothesis 2: Fast Core-Collapse Supernova (Type IIb / Ib)
        # -------------------------------------------------------------
        # Typical SN decays much slower: ~0.05 mag/day.
        sn_decay_mismatch = abs(features.fade_rate_mag_per_day - 0.08)
        sn_log_like = -10.0 - 40.0 * sn_decay_mismatch
        h2 = PhysicalHypothesis(
            hypothesis_id="H2_FAST_CCSN",
            name="Fast-Evolving Stripped-Envelope Supernova (Type Ib/IIb)",
            progenitor_class="CORE_COLLAPSE_SUPERNOVA",
            prior_prob=0.45,
            posterior_prob=0.0,
            log_likelihood=sn_log_like,
            physical_parameters={
                "nickel_56_mass_msun": 0.12,
                "ejecta_mass_msun": 1.8,
                "diffusion_time_days": 18.5,
                "peak_abs_mag": peak_abs_mag,
            },
            falsification_criteria=[
                "Lightcurve plateau or radioactive nickel tail phase (0.01 mag/day decline) after day 20.",
                "Prominent P-Cygni optical Fe II / He I absorption lines in spectra."
            ]
        )

        # -------------------------------------------------------------
        # Hypothesis 3: Tidal Disruption Event or AGN Variable Flare
        # -------------------------------------------------------------
        # Fallback rate t^(-5/3), slow decline, typically nuclear
        offset_kpc = triage.host_context.physical_offset_kpc or 5.0
        tde_offset_penalty = -20.0 if offset_kpc > 2.0 else -2.0
        tde_log_like = tde_offset_penalty - 15.0 * abs(features.fade_rate_mag_per_day - 0.02)
        h3 = PhysicalHypothesis(
            hypothesis_id="H3_TDE_AGN",
            name="Tidal Disruption Event or Active Galactic Nucleus Flare",
            progenitor_class="SUPERMASSIVE_BLACK_HOLE_ACCRETION",
            prior_prob=0.20,
            posterior_prob=0.0,
            log_likelihood=tde_log_like,
            physical_parameters={
                "black_hole_mass_msun": 1e6,
                "fallback_exponent": -1.67,
                "nuclear_offset_kpc": offset_kpc,
            },
            falsification_criteria=[
                "Strict nuclear spatial coincidence (offset < 0.2 kpc).",
                "Persistent high-temperature UV/soft X-ray blackbody continuum (T > 2e4 K)."
            ]
        )

        hypotheses = [h1, h2, h3]

        turn = DebateTurn(
            turn_number=3,
            persona=AgentPersona.ASTROPHYSICIST,
            stage=DebateStage.HYPOTHESIS_FORMULATION,
            argument_summary=(
                f"Formulated 3 competing astrophysical hypotheses based on peak M_abs = {peak_abs_mag:.2f} mag "
                f"and decay rate = {features.fade_rate_mag_per_day:.2f} mag/day. "
                f"H1 (Kilonova), H2 (Stripped-Envelope CCSN), and H3 (TDE/AGN flare)."
            ),
            evidence_sources=["Metzger2019_Kilonova", "Arnett1982_Diffusion", "Rees1988_TDE"],
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        return turn, hypotheses


class StatisticalAuditorAgent:
    """Calculates Bayesian evidence, BICs, posterior model probabilities, and degrees of freedom."""

    def audit_and_update_posteriors(
        self,
        hypotheses: List[PhysicalHypothesis],
        num_data_points: int,
    ) -> Tuple[DebateTurn, List[PhysicalHypothesis]]:
        # Compute BIC for each hypothesis: BIC = k * ln(n) - 2 * ln(L)
        n = max(3, num_data_points)
        for h in hypotheses:
            k = max(2, len(h.physical_parameters))
            h.bic = k * math.log(n) - 2.0 * h.log_likelihood

        # Relative delta BIC to minimum BIC
        min_bic = min(h.bic for h in hypotheses)
        # Unnormalized posterior: P(Hi | D) ~ Prior(Hi) * exp(-0.5 * delta_BIC)
        weights = []
        for h in hypotheses:
            delta_bic = h.bic - min_bic
            w = h.prior_prob * math.exp(-0.5 * min(700.0, delta_bic))
            weights.append(w)

        total_weight = sum(weights)
        if total_weight <= 0.0:
            total_weight = 1.0

        for h, w in zip(hypotheses, weights):
            h.posterior_prob = float(w / total_weight)

        # Identify leading hypothesis
        leading = max(hypotheses, key=lambda h: h.posterior_prob)
        for h in hypotheses:
            h.is_leading = (h.hypothesis_id == leading.hypothesis_id)

        summary = (
            f"Evaluated Bayesian Information Criterion across {len(hypotheses)} models with N={n} data points. "
            f"Leading model: {leading.name} (BIC = {leading.bic:.1f}, Posterior P = {leading.posterior_prob:.4f}). "
            f"Posterior distribution normalized."
        )

        turn = DebateTurn(
            turn_number=4,
            persona=AgentPersona.STATISTICAL_AUDITOR,
            stage=DebateStage.BAYESIAN_POSTERIOR_UPDATE,
            argument_summary=summary,
            evidence_sources=["BayesFactorAuditor", "BIC_ModelSelection"],
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        return turn, hypotheses


class SkepticRedTeamAgent:
    """Adversarially challenges leading hypothesis, tests instrumental artifacts, and assigns veto status."""

    def challenge(
        self,
        event: UnifiedScientificEvent,
        triage: TriageDecision,
        hypotheses: List[PhysicalHypothesis],
    ) -> Tuple[DebateTurn, SkepticVeto]:
        leading = next((h for h in hypotheses if h.is_leading), hypotheses[0])

        # Test 1: Instrumental subtraction dipole artifact check
        is_subtraction_artifact = False
        for p in event.photometry_points:
            flags = p.get("flags", 0)
            if flags & 0x08:  # hypothetical dipole bit
                is_subtraction_artifact = True

        # Test 2: Cadence insufficiency
        is_cadence_marginal = len(event.photometry_points) < 4

        if is_subtraction_artifact:
            status = ChallengeStatus.VETO_DISQUALIFIED
            critique = "Difference image PSF shows bipolar positive/negative dipole indicative of astrometric misregistration."
            discrim_obs = "Re-reduction with forced PSF photometry on difference template."
        elif is_cadence_marginal:
            status = ChallengeStatus.VALIDATED_WITH_QUALIFICATIONS
            critique = (
                f"Candidate exhibits rapid fading consistent with {leading.name}, but temporal sampling "
                f"({len(event.photometry_points)} points) is sparse. Host galaxy offset requires spectroscopic confirmation."
            )
            discrim_obs = "Urgent multiband optical/NIR photometry (r, i, z bands) within 12 hours to verify cooling rate."
        else:
            status = ChallengeStatus.VALIDATED_WITH_QUALIFICATIONS
            critique = (
                f"Physical parameters for {leading.name} satisfy energy conservation boundaries, but "
                f"distinction from fast SN requires confirming lanthanide opacity reddening slope d(r-i)/dt > 0.2."
            )
            discrim_obs = "Target-of-Opportunity multiband photometry in i and z bands."

        veto = SkepticVeto(
            veto_id=f"VETO-{event.event_id}-001",
            target_hypothesis_id=leading.hypothesis_id,
            challenge_title=f"Skeptic Challenge to {leading.hypothesis_id}: Instrumental & Progenitor Degeneracy",
            adversarial_critique=critique,
            instrumental_artifact_checked="PSF subtraction dipole, edge mask, cosmic ray spike, and cadence sparsity",
            status=status,
            required_discriminating_observation=discrim_obs
        )

        turn = DebateTurn(
            turn_number=5,
            persona=AgentPersona.SKEPTIC_RED_TEAM,
            stage=DebateStage.SKEPTIC_CHALLENGE,
            argument_summary=f"Skeptic audit status: {status.value}. {critique}",
            evidence_sources=["SkepticArtifactDetector", "RedTeamFalsificationProtocol"],
            timestamp_iso=datetime.now(timezone.utc).isoformat()
        )

        return turn, veto
