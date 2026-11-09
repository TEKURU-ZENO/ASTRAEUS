"""
ASTRAEUS Stage 2 Temporal Sequence & Observation Record Engine.
Implements the v1.1.2 observation contract, polymorphic censoring, causal sequence views,
and science-preserving hierarchical sequence decimation.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime, timezone
import hashlib
import math
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator


# Physical and Photometric Conversion Constants
SI_FLUX_DENSITY_PER_NANOMAGGY = 3.631e-32  # W m^-2 Hz^-1
MICROJANSKYS_PER_NANOMAGGY = 3.631          # microJansky (uJy)
CANONICAL_AB_ZERO_POINT = 22.500            # AB magnitude zero point corresponding to 1 nanomaggy
CORPUS_V10_ZERO_POINT = 27.500              # Astronomical Reference Corpus v1.0 internal zero point
CORPUS_V10_TO_NMGY_SCALE = 100.0            # F_nmgy = F_27.5 / 100.0


class ObservationType(str, Enum):
    FORCED_PHOTOMETRY = "FORCED_PHOTOMETRY"
    CATALOG_DETECTION = "CATALOG_DETECTION"
    CATALOG_UPPER_LIMIT = "CATALOG_UPPER_LIMIT"


class QualitySeverity(str, Enum):
    FATAL = "FATAL"                  # Bad pixels, saturation, detector edge defect -> Quarantined
    DEGRADED = "DEGRADED"            # High moon background, seeing > 2.5", cloud -> Inflated sigma
    INFORMATIONAL = "INFORMATIONAL"  # Minor cosmetic flag outside PSF core -> Retained unmodified


class NoiseScaleSource(str, Enum):
    DECLARED_NOISE_FLOOR = "DECLARED_NOISE_FLOOR"
    INFERRED_FROM_THRESHOLD = "INFERRED_FROM_THRESHOLD"
    DERIVED_FROM_ZERO_POINT = "DERIVED_FROM_ZERO_POINT"
    UNKNOWN = "UNKNOWN"


class SequenceViewType(str, Enum):
    FULL_HISTORY = "FULL_HISTORY"                  # Retrospective physical analysis (all landmarks allowed)
    CAUSAL_FORECAST = "CAUSAL_FORECAST"            # Future forecasting (strictly t_pub <= T_cutoff, peak-so-far)
    EARLY_CLASSIFICATION = "EARLY_CLASSIFICATION"  # Early classification (strictly t_pub <= T_cutoff)
    PRIORITIZATION = "PRIORITIZATION"              # Queue follow-up ranking (strictly t_pub <= T_cutoff)


class ObservationRecord(BaseModel):
    """
    Polymorphic observation record enforcing the v1.1.2 scientific contract.
    Decouples physical BMJD(TDB) from operational UTC timestamps, and provides explicit
    effective measurement noise scale for both forced photometry and catalog upper limits.
    """
    model_config = ConfigDict(frozen=True)

    object_id: str
    observation_id: str
    t_obs_bmjd: float                              # Barycentric Modified Julian Date (TDB standard)
    t_pub_utc: Optional[str] = None                # Operational UTC alert publication timestamp (ISO 8601)
    t_ingest_utc: Optional[str] = None             # ASTRAEUS operational ingestion timestamp (ISO 8601)
    has_operational_pub_time: bool = True          # False for historical archival lacking alert broker
    passband: str                                  # Standardized optical filter (u, g, r, i, z, Y, c, o)
    survey_id: str                                 # "ZTF", "ATLAS", "PS1", "DECam", "SYNTHETIC"
    obs_type: ObservationType

    # Flux and measurement uncertainty (nanomaggies, Z0 = 22.500)
    measured_flux_nmgy: Optional[float] = None
    measurement_sigma_nmgy: Optional[float] = None

    # Censored upper limit fields
    upper_limit_flux_nmgy: Optional[float] = None
    upper_limit_sigma_threshold: Optional[float] = None
    noise_scale_source: Optional[NoiseScaleSource] = None
    effective_measurement_sigma_nmgy: float        # Explicit noise scale used for likelihood evaluation

    # Quality and Lineage
    quality_flag: int = 0
    quality_severity: QualitySeverity = QualitySeverity.INFORMATIONAL
    provenance_hash: str

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> "ObservationRecord":
        """Validates that observation type matches populated flux and uncertainty fields."""
        if self.obs_type in (ObservationType.FORCED_PHOTOMETRY, ObservationType.CATALOG_DETECTION):
            if self.measured_flux_nmgy is None:
                raise ValueError(f"Observation type {self.obs_type} must supply measured_flux_nmgy")
            if self.measurement_sigma_nmgy is None or self.measurement_sigma_nmgy <= 0:
                raise ValueError(f"Observation type {self.obs_type} must supply positive measurement_sigma_nmgy")
        elif self.obs_type == ObservationType.CATALOG_UPPER_LIMIT:
            if self.upper_limit_flux_nmgy is None or self.upper_limit_flux_nmgy <= 0:
                raise ValueError("CATALOG_UPPER_LIMIT must supply positive upper_limit_flux_nmgy")
            if self.effective_measurement_sigma_nmgy <= 0:
                raise ValueError("CATALOG_UPPER_LIMIT must supply positive effective_measurement_sigma_nmgy")
        return self

    # --- Unit Conversion Static Helpers ---

    @staticmethod
    def counts_to_nanomaggies(counts: float, count_sigma: float, zero_point: float) -> Tuple[float, float]:
        """
        Converts raw instrumental difference counts and uncertainty to canonical nanomaggies (Z0 = 22.500).
        Preserves signed negative flux values unconditionally.
        """
        scale_factor = 10.0 ** (-0.4 * (zero_point - CANONICAL_AB_ZERO_POINT))
        flux_nmgy = counts * scale_factor
        sigma_nmgy = count_sigma * scale_factor
        return flux_nmgy, sigma_nmgy

    @staticmethod
    def ab_magnitude_to_nanomaggies(mag: float, mag_err: float) -> Tuple[float, float]:
        """
        Converts standard AB magnitude to canonical nanomaggies (Z0 = 22.500).
        F_nmgy = 10^(-0.4 * (m_AB - 22.500))
        """
        flux_nmgy = 10.0 ** (-0.4 * (mag - CANONICAL_AB_ZERO_POINT))
        sigma_nmgy = (math.log(10.0) / 2.5) * flux_nmgy * mag_err
        return flux_nmgy, sigma_nmgy

    @staticmethod
    def corpus_v10_to_nanomaggies(flux_27_5: float, sigma_27_5: float) -> Tuple[float, float]:
        """
        Exact invariant migration from Astronomical Reference Corpus v1.0 (Z0 = 27.500)
        to Stage 2 Canonical Model Input (Z0 = 22.500):
        F_nmgy = F_27.5 / 100.0
        """
        return flux_27_5 / CORPUS_V10_TO_NMGY_SCALE, sigma_27_5 / CORPUS_V10_TO_NMGY_SCALE

    @staticmethod
    def nanomaggies_to_si_flux_density(flux_nmgy: float) -> float:
        """
        Converts nanomaggies to physical SI spectral flux density:
        1 nmgy = 3.631e-32 W m^-2 Hz^-1
        """
        return flux_nmgy * SI_FLUX_DENSITY_PER_NANOMAGGY

    @staticmethod
    def nanomaggies_to_microjanskys(flux_nmgy: float) -> float:
        """
        Converts nanomaggies to microJanskys (uJy):
        1 nmgy = 3.631 uJy
        """
        return flux_nmgy * MICROJANSKYS_PER_NANOMAGGY

    # --- Training Likelihood Evaluation ---

    def compute_training_loss(self, mu_pred: float, sigma_aleatoric: float) -> float:
        """
        Evaluates per-model training loss without epistemic double-counting:
        Total training variance = sigma_aleatoric^2 + effective_measurement_sigma^2.
        - Forced Photometry & Detections: Exact Gaussian log-likelihood on measured flux.
        - Catalog Upper Limits: Censored survival log-likelihood on explicit upper limit threshold.
        """
        total_var = (sigma_aleatoric ** 2) + (self.effective_measurement_sigma_nmgy ** 2)
        total_sigma = math.sqrt(max(total_var, 1e-12))

        if self.obs_type in (ObservationType.FORCED_PHOTOMETRY, ObservationType.CATALOG_DETECTION):
            # Signed Gaussian loss on measured flux
            diff = self.measured_flux_nmgy - mu_pred
            nll = 0.5 * math.log(2.0 * math.pi * total_var) + (diff ** 2) / (2.0 * total_var)
            return nll
        else:
            # Censored Gaussian survival loss: -ln Phi((L_i - mu) / sigma_total)
            z = (self.upper_limit_flux_nmgy - mu_pred) / total_sigma
            # Analytical standard normal CDF
            phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            # Clamp to prevent log(0)
            phi_z = max(phi_z, 1e-15)
            return -math.log(phi_z)


class TemporalSequence(BaseModel):
    """
    A sequence of multi-band photometric observations for an astronomical transient.
    Supports causal operational filtering, science-preserving hierarchical FPS decimation,
    and landmark protection.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_id: str
    observations: List[ObservationRecord]
    view_type: SequenceViewType = SequenceViewType.FULL_HISTORY
    t_cutoff_utc: Optional[str] = None
    is_decimated: bool = False
    raw_token_count: int = 0
    preserved_token_count: int = 0
    unmerged_records_audit: List[Dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_records(
        cls,
        object_id: str,
        records: List[ObservationRecord],
        view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
        t_cutoff_utc: Optional[str] = None
    ) -> "TemporalSequence":
        # Sort chronologically by physical BMJD(TDB)
        sorted_records = sorted(records, key=lambda r: r.t_obs_bmjd)
        seq = cls(
            object_id=object_id,
            observations=sorted_records,
            view_type=view_type,
            t_cutoff_utc=t_cutoff_utc,
            raw_token_count=len(sorted_records),
            preserved_token_count=len(sorted_records)
        )
        if t_cutoff_utc is not None and view_type != SequenceViewType.FULL_HISTORY:
            return seq.filter_by_operational_cutoff(t_cutoff_utc)
        return seq

    def filter_by_operational_cutoff(self, t_cutoff_utc: str) -> "TemporalSequence":
        """
        Enforces operational causal visibility:
        An observation is visible at decision cutoff T_cutoff if and only if:
        t_pub_utc <= T_cutoff_utc.
        """
        from astraeus_data.corpus.time_conversion import is_observation_visible_at_cutoff

        visible_records = [
            r for r in self.observations
            if r.has_operational_pub_time and is_observation_visible_at_cutoff(r.t_pub_utc, t_cutoff_utc)
        ]
        return TemporalSequence(
            object_id=self.object_id,
            observations=visible_records,
            view_type=self.view_type,
            t_cutoff_utc=t_cutoff_utc,
            raw_token_count=self.raw_token_count,
            preserved_token_count=len(visible_records),
            is_decimated=self.is_decimated
        )

    def extract_landmark_indices(self, is_causal: bool = False) -> List[int]:
        """
        Extracts indices of critical astrophysical landmarks:
        - In causal view: Peak-so-far, earliest 5-sigma alert, deepest upper limit, rise inflection.
        - In retrospective view: Global peak, global rise, limits, last point.
        """
        if not self.observations:
            return []

        landmark_indices = set()
        by_passband: Dict[str, List[Tuple[int, ObservationRecord]]] = {}
        for idx, obs in enumerate(self.observations):
            by_passband.setdefault(obs.passband, []).append((idx, obs))

        for pb, obs_list in by_passband.items():
            # 1. Earliest significant detection crossing SNR >= 5
            det_indices = [
                idx for idx, obs in obs_list
                if obs.obs_type in (ObservationType.FORCED_PHOTOMETRY, ObservationType.CATALOG_DETECTION)
                and obs.measured_flux_nmgy is not None
                and obs.effective_measurement_sigma_nmgy > 0
                and (obs.measured_flux_nmgy / obs.effective_measurement_sigma_nmgy) >= 5.0
            ]
            if det_indices:
                first_alert_idx = det_indices[0]
                landmark_indices.add(first_alert_idx)

                # 2. Deepest upper limit immediately prior to first alert
                pre_alert_limits = [
                    (idx, obs) for idx, obs in obs_list
                    if idx < first_alert_idx and obs.upper_limit_flux_nmgy is not None
                ]
                if pre_alert_limits:
                    deepest_limit = min(pre_alert_limits, key=lambda item: item[1].upper_limit_flux_nmgy)
                    landmark_indices.add(deepest_limit[0])

            # 3. Peak observation (peak-so-far in causal view, global peak in retrospective)
            flux_candidates = [
                (idx, obs) for idx, obs in obs_list
                if obs.measured_flux_nmgy is not None
            ]
            if flux_candidates:
                peak_candidate = max(flux_candidates, key=lambda item: item[1].measured_flux_nmgy)
                landmark_indices.add(peak_candidate[0])

            # 4. Final recorded observation in this passband
            landmark_indices.add(obs_list[-1][0])

        return sorted(list(landmark_indices))

    def science_preserving_hierarchical_fps(self, max_tokens: int = 256) -> "TemporalSequence":
        """
        Enforces sequence length <= max_tokens while protecting:
        1. Landmark observations (peaks, early alert, pre-alert limit, last point).
        2. Minimum passband quota (guarantee min 5-10 tokens per active passband).
        3. Temporal Farthest Point Sampling (FPS) with deterministic tie-breaking.
        Archives all dropped observations to sidecar audit log.
        """
        n_total = len(self.observations)
        if n_total <= max_tokens:
            return self

        is_causal = (self.view_type != SequenceViewType.FULL_HISTORY)

        # Stage 1: Landmark tier protection
        landmark_indices = set(self.extract_landmark_indices(is_causal=is_causal))

        # Stage 2: Passband Quota Guarantee (ensure every active passband has at least 5 tokens)
        by_passband: Dict[str, List[int]] = {}
        for idx, obs in enumerate(self.observations):
            by_passband.setdefault(obs.passband, []).append(idx)

        quota_indices = set()
        for pb, indices in by_passband.items():
            if len(indices) <= 5:
                quota_indices.update(indices)
            else:
                # Add evenly spaced tokens
                step = max(1, len(indices) // 5)
                quota_indices.update(indices[::step][:5])

        protected_indices = landmark_indices.union(quota_indices)

        # Stage 3: Passband-Stratified Farthest Point Sampling on Remaining Tokens
        remaining_budget = max_tokens - len(protected_indices)
        if remaining_budget <= 0:
            # If protected landmarks exceed budget, take the most critical landmarks
            selected_indices = sorted(list(protected_indices))[:max_tokens]
        else:
            unselected_indices = [idx for idx in range(n_total) if idx not in protected_indices]
            selected_indices = set(protected_indices)

            # Distribute remaining budget across passbands
            fps_candidates = unselected_indices.copy()
            while len(selected_indices) < max_tokens and fps_candidates:
                # Find candidate maximizing minimum temporal distance to already selected points
                best_cand = None
                max_min_dist = -1.0
                best_snr = -1.0

                current_times = [self.observations[idx].t_obs_bmjd for idx in selected_indices]
                for cand in fps_candidates:
                    t_cand = self.observations[cand].t_obs_bmjd
                    min_dist = min(abs(t_cand - t_curr) for t_curr in current_times)
                    cand_flux = self.observations[cand].measured_flux_nmgy or 0.0
                    cand_sigma = self.observations[cand].effective_measurement_sigma_nmgy
                    cand_snr = abs(cand_flux) / cand_sigma

                    # Deterministic tie-breaking: max temporal distance, then max SNR, then min index
                    if min_dist > max_min_dist or (math.isclose(min_dist, max_min_dist, abs_tol=1e-6) and cand_snr > best_snr):
                        max_min_dist = min_dist
                        best_snr = cand_snr
                        best_cand = cand

                if best_cand is not None:
                    selected_indices.add(best_cand)
                    fps_candidates.remove(best_cand)
                else:
                    break

        sorted_final_indices = sorted(list(selected_indices))
        final_records = [self.observations[i] for i in sorted_final_indices]

        # Audit unmerged / decimated records
        discarded_indices = set(range(n_total)) - set(sorted_final_indices)
        audit_records = [
            {
                "observation_id": self.observations[i].observation_id,
                "t_obs_bmjd": self.observations[i].t_obs_bmjd,
                "passband": self.observations[i].passband,
                "flux_nmgy": self.observations[i].measured_flux_nmgy,
                "sigma_nmgy": self.observations[i].effective_measurement_sigma_nmgy,
                "decimation_reason": "FPS_CAPACITY_EXCEEDED"
            }
            for i in sorted(list(discarded_indices))
        ]

        return TemporalSequence(
            object_id=self.object_id,
            observations=final_records,
            view_type=self.view_type,
            t_cutoff_utc=self.t_cutoff_utc,
            is_decimated=True,
            raw_token_count=n_total,
            preserved_token_count=len(final_records),
            unmerged_records_audit=audit_records
        )


class AkimaPredictiveModel:
    """
    Baseline 1: Akima spline interpolator equipped with an independently calibrated
    Gaussian predictive distribution.
    - Interpolation variance: Leave-One-Out (LOO) cross-validation residual scatter on knots.
    - Extrapolation variance: Exponential variance expansion into the unobserved future.
    """
    def __init__(
        self,
        tau_decay: float = 15.0,
        s_extrap_prior: float = 2.0,
        quiescent_flux: float = 0.0,
        kernel_bandwidth: float = 5.0
    ):
        self.tau_decay = tau_decay
        self.s_extrap_prior = s_extrap_prior
        self.quiescent_flux = quiescent_flux
        self.kernel_bandwidth = kernel_bandwidth
        self.knots_t: np.ndarray = np.array([])
        self.knots_y: np.ndarray = np.array([])
        self.knots_sigma: np.ndarray = np.array([])
        self.loo_residuals: np.ndarray = np.array([])

    def fit(self, times: np.ndarray, fluxes: np.ndarray, sigmas: np.ndarray) -> "AkimaPredictiveModel":
        """
        Fits Akima spline and evaluates Leave-One-Out (LOO) cross-validation error on knots.
        """
        from scipy.interpolate import Akima1DInterpolator

        sort_idx = np.argsort(times)
        self.knots_t = np.asarray(times[sort_idx], dtype=np.float64)
        self.knots_y = np.asarray(fluxes[sort_idx], dtype=np.float64)
        self.knots_sigma = np.asarray(sigmas[sort_idx], dtype=np.float64)

        n = len(self.knots_t)
        if n < 4:
            # Akima requires at least 4 points
            self.loo_residuals = np.abs(self.knots_sigma)
            return self

        loo_res = []
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            t_sub = self.knots_t[mask]
            y_sub = self.knots_y[mask]
            try:
                sub_spline = Akima1DInterpolator(t_sub, y_sub)
                pred_i = float(sub_spline(self.knots_t[i]))
                if math.isnan(pred_i):
                    # Boundary knots outside interpolation domain use linear extrapolation
                    if i == 0 and len(t_sub) >= 2:
                        slope = (y_sub[1] - y_sub[0]) / (t_sub[1] - t_sub[0])
                        pred_i = y_sub[0] + slope * (self.knots_t[0] - t_sub[0])
                    elif len(t_sub) >= 2:
                        slope = (y_sub[-1] - y_sub[-2]) / (t_sub[-1] - t_sub[-2])
                        pred_i = y_sub[-1] + slope * (self.knots_t[-1] - t_sub[-1])
                    else:
                        pred_i = self.knots_y[i]
                loo_res.append(float(self.knots_y[i] - pred_i))
            except Exception:
                loo_res.append(float(self.knots_sigma[i]))

        self.loo_residuals = np.array(loo_res, dtype=np.float64)
        return self

    def predict(self, query_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns predicted mean mu and total standard deviation sigma at query_times.
        """
        from scipy.interpolate import Akima1DInterpolator

        q_t = np.asarray(query_times, dtype=np.float64)
        n_knots = len(self.knots_t)
        if n_knots < 4:
            # Fallback to mean and prior uncertainty
            mean_val = np.mean(self.knots_y) if n_knots > 0 else self.quiescent_flux
            return np.full_like(q_t, mean_val), np.full_like(q_t, self.s_extrap_prior)

        akima_spline = Akima1DInterpolator(self.knots_t, self.knots_y)
        t_min = self.knots_t[0]
        t_max = self.knots_t[-1]
        last_flux = self.knots_y[-1]

        means = []
        sigmas = []

        for t in q_t:
            if t <= t_max:
                # Interpolation or early extrapolation
                if t >= t_min:
                    mu = float(akima_spline(t))
                else:
                    mu = self.quiescent_flux + (self.knots_y[0] - self.quiescent_flux) * math.exp(-(t_min - t) / self.tau_decay)

                # Local interpolation variance via Nadaraya-Watson kernel regression over LOO residuals
                dists = t - self.knots_t
                weights = np.exp(-(dists ** 2) / (2.0 * (self.kernel_bandwidth ** 2)))
                denom = np.sum(weights)
                if denom > 1e-12:
                    s_local_sq = np.sum(weights * (self.loo_residuals ** 2)) / denom
                else:
                    s_local_sq = float(np.mean(self.loo_residuals ** 2))
                var_total = s_local_sq
            else:
                # Extrapolation into the future
                dt = t - t_max
                decay_factor = math.exp(-dt / self.tau_decay)
                mu = self.quiescent_flux + (last_flux - self.quiescent_flux) * decay_factor

                # Extrapolation variance expansion
                dists_edge = t_max - self.knots_t
                weights_edge = np.exp(-(dists_edge ** 2) / (2.0 * (self.kernel_bandwidth ** 2)))
                s_edge_sq = np.sum(weights_edge * (self.loo_residuals ** 2)) / max(np.sum(weights_edge), 1e-12)
                var_total = s_edge_sq + (self.s_extrap_prior ** 2) * (1.0 - decay_factor)

            means.append(mu)
            sigmas.append(math.sqrt(max(var_total, 1e-6)))

        return np.array(means, dtype=np.float64), np.array(sigmas, dtype=np.float64)

    @staticmethod
    def evaluate_crps(y_true: float, mu_pred: float, sigma_pred: float) -> float:
        """
        Exact analytical Continuous Ranked Probability Score (CRPS) under Gaussian predictive distribution.
        """
        if sigma_pred <= 0:
            return abs(y_true - mu_pred)
        z = (y_true - mu_pred) / sigma_pred
        phi_z = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        Phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        crps = sigma_pred * (z * (2.0 * Phi_z - 1.0) + 2.0 * phi_z - (1.0 / math.sqrt(math.pi)))
        return float(crps)
