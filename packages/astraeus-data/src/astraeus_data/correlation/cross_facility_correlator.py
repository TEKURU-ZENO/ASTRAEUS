"""
ASTRAEUS Cross-Facility Spatial-Temporal Correlation Engine (WS7.7.1 Hardened).
Correlates streaming alerts across heterogeneous multi-messenger and multi-facility networks:
- Optical/NIR: Vera C. Rubin Observatory (LSST), Zwicky Transient Facility (ZTF).
- High-Energy: Swift/BAT, Fermi/GBM prompt gamma-ray burst triggers.
- Gravitational Waves: LIGO/Virgo/KAGRA GraceDB notices with 3D localization volumes.
- Neutrinos: IceCube Gold/Bronze neutrino candidate alerts.

Features:
- Rigorous Vincenty great-circle angular separation and instrument-specific error profiles.
- Calibrated background transient rates per instrument (Optical, GRB, GW, Neutrino).
- Exact continuous Gaussian equivalent significance (norm.isf) replacing discrete heuristics.
- Support for 2D skymap 90% credible region area and non-Gaussian core-tail instruments.
- Clear delineation between preliminary candidate, statistically significant, and high-confidence physical association.
- Strict Gate 3 Quarantine Firewall enforcement routing quarantined alerts to dead-letter storage.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from scipy.stats import norm

from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from engines.quarantine_firewall import QuarantineFirewall


class MessengerType(str, Enum):
    OPTICAL = "OPTICAL"
    NIR = "NIR"
    GAMMA_RAY = "GAMMA_RAY"
    XRAY = "XRAY"
    GRAVITATIONAL_WAVE = "GRAVITATIONAL_WAVE"
    NEUTRINO = "NEUTRINO"
    RADIO = "RADIO"


class AssociationGrade(str, Enum):
    PRELIMINARY_CANDIDATE = "PRELIMINARY_CANDIDATE"
    STATISTICALLY_SIGNIFICANT_COINCIDENCE = "STATISTICALLY_SIGNIFICANT_COINCIDENCE"
    HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION = "HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION"


@dataclass
class CorrelatedEventCandidate:
    """Standardized event packet for cross-facility correlation."""
    event_id: str
    facility: str                       # e.g. "RUBIN", "ZTF", "SWIFT_BAT", "FERMI_GBM", "GRACEDB", "ICECUBE"
    messenger_type: MessengerType
    trigger_mjd: float
    ra_deg: float
    dec_deg: float
    localization_error_deg: float       # 1-sigma positional uncertainty radius (or core sigma)
    skymap_90_cr_area_deg2: Optional[float] = None  # 90% credible region area for GW or broad alerts
    tail_error_deg: Optional[float] = None          # Non-Gaussian tail sigma (e.g. Fermi/GBM)
    luminosity_distance_mpc: Optional[float] = None
    distance_err_mpc: Optional[float] = None
    snr: float = 10.0
    classification_hint: str = "UNKNOWN"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiMessengerAssociation:
    """Confirmed cross-facility spatial-temporal coincidence record."""
    association_id: str
    primary_event: CorrelatedEventCandidate
    associated_events: List[CorrelatedEventCandidate]
    messenger_types: List[MessengerType]
    angular_separation_arcsec: float
    temporal_delta_seconds: float
    chance_coincidence_probability: float
    association_significance_sigma: float
    log_bayes_factor_association: float
    suggested_transient_class: str
    association_grade: AssociationGrade = AssociationGrade.PRELIMINARY_CANDIDATE
    significance_type: str = "MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT"
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None
    audit_notes: str = ""


class CrossFacilityCorrelationEngine:
    """
    Coordinates streaming alerts from space and ground facilities,
    quantifying spatial-temporal associations with calibrated background rates
    and continuous Gaussian significance metrics.
    """

    ALL_SKY_AREA_DEG2 = 41252.96  # 4 * pi * (180 / pi)^2

    # Calibrated background rates per facility regime
    BACKGROUND_RATES = {
        # Optical deep transients (~0.05 / deg^2 / day)
        MessengerType.OPTICAL: 0.05 / 86400.0,
        MessengerType.NIR: 0.02 / 86400.0,
        # Fermi/GBM prompt GRB triggers (~240 / yr over all sky)
        MessengerType.GAMMA_RAY: (240.0 / 365.25 / 86400.0) / ALL_SKY_AREA_DEG2,
        # LVK Gravitational Wave candidates (~80 / yr over all sky in O4)
        MessengerType.GRAVITATIONAL_WAVE: (80.0 / 365.25 / 86400.0) / ALL_SKY_AREA_DEG2,
        # IceCube high-energy gold neutrino tracks (~15 / yr over all sky)
        MessengerType.NEUTRINO: (15.0 / 365.25 / 86400.0) / ALL_SKY_AREA_DEG2,
        # Space X-ray transients
        MessengerType.XRAY: (100.0 / 365.25 / 86400.0) / ALL_SKY_AREA_DEG2,
        MessengerType.RADIO: 0.001 / 86400.0,
    }

    @classmethod
    def angular_separation_deg(cls, ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
        """
        Calculates exact great-circle angular distance between two sky coordinates
        using the numerically stable Vincenty formula.
        """
        ra1, dec1 = math.radians(ra1_deg), math.radians(dec1_deg)
        ra2, dec2 = math.radians(ra2_deg), math.radians(dec2_deg)
        d_ra = abs(ra1 - ra2)

        numerator = math.sqrt(
            (math.cos(dec2) * math.sin(d_ra)) ** 2
            + (math.cos(dec1) * math.sin(dec2) - math.sin(dec1) * math.cos(dec2) * math.cos(d_ra)) ** 2
        )
        denominator = math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(d_ra)
        return math.degrees(math.atan2(numerator, denominator))

    @classmethod
    def get_effective_sigma_deg(cls, event: CorrelatedEventCandidate) -> float:
        """
        Computes effective 1-sigma positional uncertainty.
        If a 90% credible region area is given (e.g. GW skymap), converts via:
            sigma_eff = sqrt( Area_90 / (2 * pi * (-ln(0.10))) )
        """
        if event.skymap_90_cr_area_deg2 is not None and event.skymap_90_cr_area_deg2 > 0:
            # 2 * pi * ln(10) ~= 14.4674
            return math.sqrt(event.skymap_90_cr_area_deg2 / (2.0 * math.pi * 2.302585))
        return max(0.0001, event.localization_error_deg)

    @classmethod
    def calculate_calibrated_chance_probability(
        cls,
        search_radius_deg: float,
        time_window_seconds: float,
        candidate_messenger: MessengerType,
    ) -> float:
        """
        Poisson chance coincidence probability evaluated against instrument-specific background rate:
            P_chance = 1 - exp(-lambda), where lambda = Rate * Area * delta_t
        """
        rate_per_deg2_sec = cls.BACKGROUND_RATES.get(candidate_messenger, 1.0e-10)
        area_deg2 = math.pi * (search_radius_deg ** 2)
        expected_poisson_lambda = rate_per_deg2_sec * area_deg2 * max(1.0, abs(time_window_seconds))

        if expected_poisson_lambda < 1e-15:
            return float(expected_poisson_lambda)
        return float(1.0 - math.exp(-expected_poisson_lambda))

    @classmethod
    def evaluate_association(
        cls,
        primary_event: CorrelatedEventCandidate,
        candidate_event: CorrelatedEventCandidate,
        max_prompt_window_sec: float = 60.0,
        max_afterglow_window_sec: float = 259200.0,  # 3 days
    ) -> Optional[MultiMessengerAssociation]:
        """
        Evaluates physical association between primary event and candidate counterpart:
        1. Checks Gate 3 Quarantine Firewall.
        2. Applies instrument-specific spatial containment and temporal windowing.
        3. Calculates continuous Gaussian significance using norm.isf(p_chance).
        4. Calculates spatial and temporal likelihood ratios.
        5. Assigns rigorous association grade.
        """
        # 1. Gate 3 Quarantine Firewall Check
        quar1 = QuarantineFirewall.is_quarantined_coordinate(primary_event.ra_deg, primary_event.dec_deg)
        quar2 = QuarantineFirewall.is_quarantined_coordinate(candidate_event.ra_deg, candidate_event.dec_deg)

        if quar1 or quar2:
            reason = "Gate 3 Quarantine Firewall: Restricted coordinate in Gate 3 test set."
            assoc_id = f"quar_{primary_event.event_id}_{candidate_event.event_id}"
            return MultiMessengerAssociation(
                association_id=assoc_id,
                primary_event=primary_event,
                associated_events=[candidate_event],
                messenger_types=[primary_event.messenger_type, candidate_event.messenger_type],
                angular_separation_arcsec=0.0,
                temporal_delta_seconds=0.0,
                chance_coincidence_probability=1.0,
                association_significance_sigma=0.0,
                log_bayes_factor_association=-999.0,
                suggested_transient_class="QUARANTINED_RESTRICTED",
                association_grade=AssociationGrade.PRELIMINARY_CANDIDATE,
                is_quarantined=True,
                quarantine_reason=reason,
                audit_notes="Association dropped by Gate 3 quarantine covenant.",
            )

        # 2. Compute spatial separation
        sep_deg = cls.angular_separation_deg(
            primary_event.ra_deg,
            primary_event.dec_deg,
            candidate_event.ra_deg,
            candidate_event.dec_deg,
        )
        sep_arcsec = sep_deg * 3600.0

        # Effective localization uncertainties
        sigma1 = cls.get_effective_sigma_deg(primary_event)
        sigma2 = cls.get_effective_sigma_deg(candidate_event)
        combined_sigma_deg = math.sqrt(sigma1 ** 2 + sigma2 ** 2)

        # 3. Compute temporal delta
        delta_t_days = candidate_event.trigger_mjd - primary_event.trigger_mjd
        delta_t_sec = delta_t_days * 86400.0

        is_prompt_pair = (
            {primary_event.messenger_type, candidate_event.messenger_type}
            <= {MessengerType.GRAVITATIONAL_WAVE, MessengerType.GAMMA_RAY, MessengerType.NEUTRINO}
        )

        if is_prompt_pair:
            if not (-10.0 <= delta_t_sec <= max_prompt_window_sec):
                return None
        else:
            if not (-60.0 <= delta_t_sec <= max_afterglow_window_sec):
                return None

        # Max allowed separation: 3.5-sigma combined radius
        max_allowed_separation_deg = max(0.005, combined_sigma_deg * 3.5)
        if sep_deg > max_allowed_separation_deg:
            return None

        # 4. Calibrated statistical significance
        p_chance = cls.calculate_calibrated_chance_probability(
            search_radius_deg=max(sep_deg, combined_sigma_deg),
            time_window_seconds=max(abs(delta_t_sec), 1.0),
            candidate_messenger=candidate_event.messenger_type,
        )

        # Continuous Gaussian equivalent significance: Z = norm.isf(p_chance)
        # Cap between 0.0 and 8.5 sigma to preserve numerical stability
        p_clamped = max(1e-16, min(0.5, p_chance))
        sigma_equiv = float(norm.isf(p_clamped))
        sigma_equiv = round(max(0.0, min(8.5, sigma_equiv)), 2)

        # Likelihood ratio: spatial * temporal
        spatial_p_s = (1.0 / (2.0 * math.pi * (combined_sigma_deg ** 2))) * math.exp(-0.5 * ((sep_deg / combined_sigma_deg) ** 2))
        spatial_p_b = 1.0 / cls.ALL_SKY_AREA_DEG2
        ln_spatial_lr = math.log(max(1e-10, spatial_p_s / spatial_p_b))

        tau_phys_sec = 2.0 if is_prompt_pair else 86400.0
        temporal_p_s = (1.0 / tau_phys_sec) * math.exp(-abs(delta_t_sec) / tau_phys_sec)
        temporal_window_sec = max_prompt_window_sec if is_prompt_pair else max_afterglow_window_sec
        temporal_p_b = 1.0 / temporal_window_sec
        ln_temporal_lr = math.log(max(1e-10, temporal_p_s / temporal_p_b))

        ln_bf = float(ln_spatial_lr + ln_temporal_lr)

        # 5. Determine association grade
        if sigma_equiv >= 5.0 and ln_bf >= 10.0:
            grade = AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION
        elif sigma_equiv >= 3.0 and ln_bf >= 3.0:
            grade = AssociationGrade.STATISTICALLY_SIGNIFICANT_COINCIDENCE
        else:
            grade = AssociationGrade.PRELIMINARY_CANDIDATE

        # 6. Suggested classification
        transient_class = "MULTI_MESSENGER_CANDIDATE"
        m_set = {primary_event.messenger_type, candidate_event.messenger_type}
        if MessengerType.GRAVITATIONAL_WAVE in m_set and MessengerType.GAMMA_RAY in m_set:
            transient_class = "BINARY_NEUTRON_STAR_PROMPT_GRB"
        elif MessengerType.GRAVITATIONAL_WAVE in m_set and (MessengerType.OPTICAL in m_set or MessengerType.NIR in m_set):
            transient_class = "KILONOVA_OPTICAL_NIR_COUNTERPART"
        elif MessengerType.GAMMA_RAY in m_set and (MessengerType.OPTICAL in m_set or MessengerType.XRAY in m_set):
            transient_class = "SHORT_GRB_AFTERGLOW"
        elif MessengerType.NEUTRINO in m_set and MessengerType.GAMMA_RAY in m_set:
            transient_class = "TXZ_0506_LIKE_BLAZAR_FLARE"

        assoc_id = f"mma_{primary_event.event_id}_{candidate_event.event_id}_{int(abs(delta_t_sec))}"

        return MultiMessengerAssociation(
            association_id=assoc_id,
            primary_event=primary_event,
            associated_events=[candidate_event],
            messenger_types=[primary_event.messenger_type, candidate_event.messenger_type],
            angular_separation_arcsec=round(sep_arcsec, 2),
            temporal_delta_seconds=round(delta_t_sec, 2),
            chance_coincidence_probability=float(p_chance),
            association_significance_sigma=float(sigma_equiv),
            log_bayes_factor_association=round(ln_bf, 2),
            suggested_transient_class=transient_class,
            association_grade=grade,
            significance_type="MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT",
            is_quarantined=False,
            audit_notes=(
                f"Multi-messenger association evaluated [{grade.value}]. "
                f"Sep: {sep_arcsec:.1f}\", dt: {delta_t_sec:.1f}s, "
                f"Model-dependent Gaussian-equivalent Significance: {sigma_equiv:.2f} sigma (ln BF: {ln_bf:.1f}, "
                f"P_chance: {p_chance:.2e}). Qualified as model-dependent pending real alert background scans."
            ),
        )

    @classmethod
    def perform_synthetic_time_slide_sanity_test(
        cls,
        primary_event: CorrelatedEventCandidate,
        candidate_event: CorrelatedEventCandidate,
        time_slide_offsets_days: Optional[List[float]] = None,
        coincidence_window_days: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Performs synthetic time-slide perturbation sanity test by offsetting candidate trigger time
        by artificial lags (e.g. +/- 10, 20, 30, 50, 100 days) to break physical simultaneity.
        
        NOTE ON STATISTICAL QUALIFICATION (WS7.8-D):
        This is a software sanity check, NOT a calibrated empirical false-alarm study.
        A zero count in finite synthetic slides does not prove a zero false-alarm rate.
        We report the empirical count along with the 95% Poisson confidence upper bound.
        """
        offsets = time_slide_offsets_days or [
            -100.0, -50.0, -30.0, -20.0, -10.0, 10.0, 20.0, 30.0, 50.0, 100.0
        ]
        slide_results = []
        false_alarms = 0
        for dt_days in offsets:
            slid_cand = CorrelatedEventCandidate(
                event_id=f"{candidate_event.event_id}_slid_{dt_days:+.1f}d",
                facility=candidate_event.facility,
                messenger_type=candidate_event.messenger_type,
                trigger_mjd=candidate_event.trigger_mjd + dt_days,
                ra_deg=candidate_event.ra_deg,
                dec_deg=candidate_event.dec_deg,
                localization_error_deg=candidate_event.localization_error_deg,
                skymap_90_cr_area_deg2=candidate_event.skymap_90_cr_area_deg2,
                luminosity_distance_mpc=candidate_event.luminosity_distance_mpc,
                distance_err_mpc=candidate_event.distance_err_mpc,
                classification_hint=candidate_event.classification_hint,
            )
            assoc = cls.evaluate_association(primary_event, slid_cand)
            is_fa = (assoc is not None and assoc.association_grade == AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION)
            if is_fa:
                false_alarms += 1
            slide_results.append({
                "time_slide_days": dt_days,
                "is_associated": assoc is not None,
                "association_grade": assoc.association_grade.value if assoc else "REJECTED_BY_COINCIDENCE_WINDOW",
            })

        n_trials = len(offsets)
        # Poisson 95% CL upper limit for k false alarms:
        # For k=0: mu_upper = -ln(0.05) ~= 2.99573
        poisson_upper_mu = 2.99573 if false_alarms == 0 else (false_alarms + 1.96 * math.sqrt(false_alarms))
        effective_exposure_days = n_trials * coincidence_window_days
        far_upper_bound_per_year = (poisson_upper_mu / effective_exposure_days) * 365.25

        return {
            "test_type": "SYNTHETIC_PERTURBATION_SANITY_CHECK",
            "is_calibrated_empirical_rate": False,
            "total_slides": n_trials,
            "false_alarms": false_alarms,
            "false_alarms_observed": false_alarms,
            "false_alarm_rate": false_alarms / n_trials,
            "sample_false_alarm_rate": false_alarms / n_trials,
            "poisson_95cl_count_upper_bound": round(poisson_upper_mu, 3),
            "far_95cl_upper_bound_per_year": round(far_upper_bound_per_year, 4),
            "rejection_confirmed": (false_alarms == 0),
            "statistical_warning": (
                "Finite sample zero-count does not establish zero physical false-alarm rate. "
                "Archival background survey calibration required for operational qualification."
            ),
            "slide_details": slide_results,
        }

    # Backward-compatible alias
    perform_time_slide_analysis = perform_synthetic_time_slide_sanity_test

    @classmethod
    def perform_synthetic_spatial_scramble_sanity_test(
        cls,
        primary_event: CorrelatedEventCandidate,
        candidate_event: CorrelatedEventCandidate,
        ra_offsets_deg: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Performs synthetic spatial scrambling perturbation test by displacing candidate sky coordinates
        outside the localization envelope to verify basic software spatial rejection.
        
        NOTE ON STATISTICAL QUALIFICATION (WS7.8-D):
        This is a software sanity check, NOT a calibrated empirical spatial background study.
        """
        offsets = ra_offsets_deg or [
            15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 270.0
        ]
        scramble_results = []
        false_alarms = 0
        for d_ra in offsets:
            scrambled_ra = (candidate_event.ra_deg + d_ra) % 360.0
            scrambled_cand = CorrelatedEventCandidate(
                event_id=f"{candidate_event.event_id}_scrambled_{d_ra:+.1f}deg",
                facility=candidate_event.facility,
                messenger_type=candidate_event.messenger_type,
                trigger_mjd=candidate_event.trigger_mjd,
                ra_deg=scrambled_ra,
                dec_deg=candidate_event.dec_deg,
                localization_error_deg=candidate_event.localization_error_deg,
                skymap_90_cr_area_deg2=candidate_event.skymap_90_cr_area_deg2,
                luminosity_distance_mpc=candidate_event.luminosity_distance_mpc,
                distance_err_mpc=candidate_event.distance_err_mpc,
                classification_hint=candidate_event.classification_hint,
            )
            assoc = cls.evaluate_association(primary_event, scrambled_cand)
            is_fa = (assoc is not None and assoc.association_grade == AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION)
            if is_fa:
                false_alarms += 1
            scramble_results.append({
                "ra_offset_deg": d_ra,
                "is_associated": assoc is not None,
                "association_grade": assoc.association_grade.value if assoc else "REJECTED_BY_ANGULAR_SEPARATION",
            })

        n_trials = len(offsets)
        poisson_upper_mu = 2.99573 if false_alarms == 0 else (false_alarms + 1.96 * math.sqrt(false_alarms))

        return {
            "test_type": "SYNTHETIC_PERTURBATION_SANITY_CHECK",
            "is_calibrated_empirical_rate": False,
            "total_scrambles": n_trials,
            "false_alarms": false_alarms,
            "false_alarms_observed": false_alarms,
            "false_alarm_rate": false_alarms / n_trials,
            "sample_false_alarm_rate": false_alarms / n_trials,
            "poisson_95cl_count_upper_bound": round(poisson_upper_mu, 3),
            "rejection_confirmed": (false_alarms == 0),
            "statistical_warning": (
                "Deterministic coordinate perturbations do not reproduce real survey spatial inhomogeneities, "
                "masking, or galactic plane alert densities."
            ),
            "scramble_details": scramble_results,
        }

    # Backward-compatible alias
    perform_spatial_scrambling_analysis = perform_synthetic_spatial_scramble_sanity_test

    def correlate_stream(
        self,
        new_event: CorrelatedEventCandidate,
        recent_buffer: List[CorrelatedEventCandidate],
    ) -> List[MultiMessengerAssociation]:
        """
        Correlates a newly arrived event against a rolling buffer of recent alerts
        from other facilities, returning all confirmed physical associations.
        """
        associations: List[MultiMessengerAssociation] = []
        for past_event in recent_buffer:
            if past_event.event_id == new_event.event_id:
                continue
            if past_event.facility == new_event.facility:
                continue
            assoc = self.evaluate_association(primary_event=past_event, candidate_event=new_event)
            if assoc is not None:
                associations.append(assoc)

        return associations


@dataclass
class ArchivalBackgroundValidationResult:
    """Scientific verification ledger of archival background false-alarm benchmark."""
    total_trials: int
    exposure_time_years: float
    simulated_optical_alerts: int
    simulated_high_energy_alerts: int
    false_alarms_observed: int
    empirical_far_per_year: float
    poisson_95cl_upper_far_per_year: float
    confidence_interval_95: Tuple[float, float]
    known_associations_tested: int
    known_associations_recovered: int
    recovery_rate: float
    missed_associations_count: int
    trials_factor: float
    significance_label: str
    qualification_status: str
    audit_notes: str


class ArchivalBackgroundBenchmark:
    """
    Executes reproducible archival-background validation tests (WS7.8-D).
    Accounts for representative alert populations, exposure effects, trials factors,
    and reports Poisson confidence intervals rather than claiming zero false-alarm probability.
    """

    @classmethod
    def run_benchmark(
        cls,
        total_trials: int = 100,
        exposure_years: float = 1.0,
        optical_alert_density_per_deg2_day: float = 0.05,
        gw_grb_rate_per_year: float = 320.0,
        random_seed: int = 42,
    ) -> ArchivalBackgroundValidationResult:
        import random
        rng = random.Random(random_seed)

        trials_factor = float(total_trials)
        false_alarms = 0
        known_tested = 10
        known_recovered = 10

        # Run synthetic background coincidence trials over distributed sky coordinates and epochs
        for i in range(total_trials):
            ra_gw = rng.uniform(0.0, 360.0)
            dec_gw = rng.uniform(-90.0, 90.0)
            mjd_gw = 60000.0 + rng.uniform(0.0, exposure_years * 365.25)

            gw_cand = CorrelatedEventCandidate(
                event_id=f"sim_bg_gw_{i}",
                facility="LIGO_SIM",
                messenger_type=MessengerType.GRAVITATIONAL_WAVE,
                trigger_mjd=mjd_gw,
                ra_deg=ra_gw,
                dec_deg=dec_gw,
                localization_error_deg=3.5,
                skymap_90_cr_area_deg2=150.0,
                classification_hint="BNS_BACKGROUND_NOISE",
            )

            # Random background optical transient (drawn independently across the sky and year)
            ra_opt = rng.uniform(0.0, 360.0)
            dec_opt = rng.uniform(-90.0, 90.0)
            mjd_opt = 60000.0 + rng.uniform(0.0, exposure_years * 365.25)

            opt_cand = CorrelatedEventCandidate(
                event_id=f"sim_bg_opt_{i}",
                facility="RUBIN_SIM",
                messenger_type=MessengerType.OPTICAL,
                trigger_mjd=mjd_opt,
                ra_deg=ra_opt,
                dec_deg=dec_opt,
                localization_error_deg=0.001,
                classification_hint="UNRELATED_SN",
            )

            assoc = CrossFacilityCorrelationEngine.evaluate_association(gw_cand, opt_cand)
            if assoc is not None and assoc.association_grade == AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION:
                false_alarms += 1

        # Test recovery of true physically co-located events (e.g. injected GW-GRB prompt counterparts)
        for j in range(known_tested):
            t_inj = 60100.0 + j * 10.0
            ra_inj = 197.45 + (j * 0.5)
            dec_inj = -23.38 + (j * 0.2)

            gw_inj = CorrelatedEventCandidate(
                event_id=f"inj_gw_{j}",
                facility="GRACEDB",
                messenger_type=MessengerType.GRAVITATIONAL_WAVE,
                trigger_mjd=t_inj,
                ra_deg=ra_inj,
                dec_deg=dec_inj,
                localization_error_deg=0.2,
                classification_hint="BNS",
            )
            grb_inj = CorrelatedEventCandidate(
                event_id=f"inj_grb_{j}",
                facility="FERMI_GBM",
                messenger_type=MessengerType.GAMMA_RAY,
                trigger_mjd=t_inj + (1.7 / 86400.0),  # dt = 1.7s prompt offset
                ra_deg=ra_inj + 0.001,
                dec_deg=dec_inj + 0.001,
                localization_error_deg=0.5,
                classification_hint="SHORT_GRB",
            )
            inj_assoc = CrossFacilityCorrelationEngine.evaluate_association(gw_inj, grb_inj)
            if inj_assoc is None or inj_assoc.association_grade != AssociationGrade.HIGH_CONFIDENCE_PHYSICAL_ASSOCIATION:
                known_recovered -= 1

        poisson_mu_upper = 2.99573 if false_alarms == 0 else (false_alarms + 1.96 * math.sqrt(false_alarms))
        far_upper_yr = poisson_mu_upper / exposure_years
        ci_lower = max(0.0, false_alarms - 1.96 * math.sqrt(false_alarms)) / exposure_years
        ci_upper = far_upper_yr

        return ArchivalBackgroundValidationResult(
            total_trials=total_trials,
            exposure_time_years=exposure_years,
            simulated_optical_alerts=total_trials,
            simulated_high_energy_alerts=total_trials,
            false_alarms_observed=false_alarms,
            empirical_far_per_year=false_alarms / exposure_years,
            poisson_95cl_upper_far_per_year=round(far_upper_yr, 4),
            confidence_interval_95=(round(ci_lower, 4), round(ci_upper, 4)),
            known_associations_tested=known_tested,
            known_associations_recovered=known_recovered,
            recovery_rate=known_recovered / known_tested,
            missed_associations_count=known_tested - known_recovered,
            trials_factor=trials_factor,
            significance_label="MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT",
            qualification_status="QUALIFIED_AS_SYNTHETIC_ARCHIVAL_BENCHMARK_ONLY; REAL_SURVEY_CALIBRATION_DEFERRED",
            audit_notes=(
                f"Archival benchmark completed ({total_trials} background trials, {exposure_years:.1f} yr exposure). "
                f"False alarms: {false_alarms} (FAR 95% CL upper bound: {far_upper_yr:.4f}/yr). "
                f"Injected recovery: {known_recovered}/{known_tested} ({known_recovered/known_tested*100:.1f}%). "
                "Retaining MODEL_DEPENDENT_GAUSSIAN_EQUIVALENT; real survey empirical calibration deferred to Phase 8."
            ),
        )
