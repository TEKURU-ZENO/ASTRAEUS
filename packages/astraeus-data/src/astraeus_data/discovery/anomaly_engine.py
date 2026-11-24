"""
ASTRAEUS Anomaly and Discovery Intelligence Engine (WS8.2).
Provides statistical, temporal, and multimodal out-of-distribution (OOD) detection,
with physics-informed discrimination between instrumental artifacts and physical anomalies.

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


class AnomalySource(str, Enum):
    """Categorical source determination for detected anomalies."""
    PHYSICAL_NEW_PHENOMENON = "PHYSICAL_NEW_PHENOMENON"
    PHYSICAL_UNMODELED_EVOLUTION = "PHYSICAL_UNMODELED_EVOLUTION"
    PHYSICAL_MULTIMODAL_INCONSISTENCY = "PHYSICAL_MULTIMODAL_INCONSISTENCY"
    INSTRUMENTAL_PSF_MISMATCH = "INSTRUMENTAL_PSF_MISMATCH"
    INSTRUMENTAL_COSMIC_RAY = "INSTRUMENTAL_COSMIC_RAY"
    INSTRUMENTAL_SUBTRACTION_RESIDUAL = "INSTRUMENTAL_SUBTRACTION_RESIDUAL"
    ATMOSPHERIC_EXTREME_VARIATION = "ATMOSPHERIC_EXTREME_VARIATION"
    BENIGN_CONCORDANT = "BENIGN_CONCORDANT"


class AnomalyDiagnostic(BaseModel):
    """Detailed anomaly diagnostic output for an observed event or light curve."""
    target_id: str
    combined_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized 0-1 anomaly score")
    statistical_ood_score: float = Field(..., ge=0.0, description="Statistical Mahalanobis/z-score distance")
    temporal_residual_score: float = Field(..., ge=0.0, description="Deviation from expected light curve decay")
    multimodal_inconsistency_score: float = Field(..., ge=0.0, description="Cross-band / cross-messenger tension")
    is_genuine_physical_anomaly: bool = Field(..., description="True if anomaly passes artifact rejection criteria")
    source_classification: AnomalySource = Field(..., description="Diagnosed source")
    rejection_reasons: List[str] = Field(default_factory=list, description="Reasons for flagging instrumental artifact")
    physical_indicators: List[str] = Field(default_factory=list, description="Evident physical anomaly signatures")
    psf_fwhm_ratio: float = Field(default=1.0, description="Ratio of measured FWHM to nominal facility seeing")
    subtraction_dipole_score: float = Field(default=0.0, description="Negative-dip residual metric (0 = pure point source)")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    timestamp_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LightCurvePoint(BaseModel):
    """Single photometric data point in an observed transient sequence."""
    mjd: float
    filter_id: str
    magnitude: float
    uncertainty_mag: float
    facility_id: str
    instrument_id: str
    psf_fwhm_arcsec: float = 1.0
    nominal_seeing_arcsec: float = 1.0
    is_subtracted_image: bool = True
    subtraction_residual_chi2: float = 1.0
    negative_flux_adjacent: bool = False


class AnomalyDiscoveryEngine:
    """
    Multimodal anomaly detector combining statistical density models,
    temporal template residuals, and physical artifact discriminators.
    """

    def __init__(
        self,
        ood_threshold_sigma: float = 3.5,
        fwhm_tolerance_ratio: Tuple[float, float] = (0.70, 1.45),
        max_subtraction_chi2: float = 3.0,
    ) -> None:
        self.ood_threshold_sigma = ood_threshold_sigma
        self.fwhm_min_ratio, self.fwhm_max_ratio = fwhm_tolerance_ratio
        self.max_subtraction_chi2 = max_subtraction_chi2

        # Empirical baseline feature distribution (mean, std) for canonical transients:
        # Features: [peak_abs_mag, rise_time_days, decay_rate_mag_per_day, color_g_minus_r]
        self._canonical_feature_means = np.array([-19.0, 15.0, 0.08, 0.20])
        self._canonical_feature_stds = np.array([1.2, 8.0, 0.05, 0.35])

    def evaluate_transient(
        self,
        target_id: str,
        observations: List[LightCurvePoint],
        cross_messenger_data: Optional[Dict[str, Any]] = None,
    ) -> AnomalyDiagnostic:
        """
        Conducts full diagnostic evaluation on a transient observation series.
        """
        if not observations:
            return AnomalyDiagnostic(
                target_id=target_id,
                combined_anomaly_score=0.0,
                statistical_ood_score=0.0,
                temporal_residual_score=0.0,
                multimodal_inconsistency_score=0.0,
                is_genuine_physical_anomaly=False,
                source_classification=AnomalySource.BENIGN_CONCORDANT,
                confidence=1.0,
            )

        # 1. Evaluate Instrumental Artifact Checks
        artifact_reasons = []
        is_instrumental = False
        source = AnomalySource.BENIGN_CONCORDANT

        avg_fwhm_ratio = float(np.mean([p.psf_fwhm_arcsec / max(0.1, p.nominal_seeing_arcsec) for p in observations]))
        max_chi2 = max(p.subtraction_residual_chi2 for p in observations)
        any_negative_dip = any(p.negative_flux_adjacent for p in observations)

        # A: Cosmic ray or hot pixel check (FWHM drastically narrower than atmospheric seeing)
        if avg_fwhm_ratio < self.fwhm_min_ratio and len(observations) <= 2:
            artifact_reasons.append(
                f"Sub-seeing profile detected (FWHM ratio={avg_fwhm_ratio:.2f} < {self.fwhm_min_ratio}), indicative of cosmic ray or bad pixel."
            )
            is_instrumental = True
            source = AnomalySource.INSTRUMENTAL_COSMIC_RAY

        # B: Subtraction dipole / negative residual check
        if any_negative_dip:
            artifact_reasons.append(
                "Adjacent negative flux dipole detected in difference imaging, characteristic of PSF mis-subtraction or astrometric shift."
            )
            is_instrumental = True
            source = AnomalySource.INSTRUMENTAL_SUBTRACTION_RESIDUAL

        # C: Subtraction fit quality
        if max_chi2 > self.max_subtraction_chi2:
            artifact_reasons.append(
                f"Difference image residual chi2 ({max_chi2:.2f}) exceeds threshold ({self.max_subtraction_chi2}), indicating poor reference matching."
            )
            is_instrumental = True
            source = AnomalySource.INSTRUMENTAL_PSF_MISMATCH

        # 2. Evaluate Temporal Residuals
        temporal_score = 0.0
        physical_indicators = []

        if len(observations) >= 2:
            sorted_obs = sorted(observations, key=lambda x: x.mjd)
            dt_days = sorted_obs[-1].mjd - sorted_obs[0].mjd
            dmag = sorted_obs[-1].magnitude - sorted_obs[0].magnitude

            if dt_days > 0.01:
                decay_rate = dmag / dt_days
                # Ultra-fast decay: > 0.8 mag/day (e.g. fast blue optical transient or rapid kilonova)
                if decay_rate > 0.75:
                    temporal_score += 4.0
                    physical_indicators.append(
                        f"Ultra-rapid fading rate ({decay_rate:.2f} mag/day) anomalous for standard supernovae (expected < 0.15 mag/day)."
                    )
                # Unexpected rapid rebrightening (e.g. shock breakout or secondary injection)
                elif decay_rate < -0.50 and dt_days > 0.5:
                    temporal_score += 4.5
                    physical_indicators.append(
                        f"Significant secondary rebrightening ({abs(decay_rate):.2f} mag/day) detected at t={dt_days:.2f}d."
                    )

        # 3. Evaluate Cross-Messenger / Multimodal Inconsistency
        multimodal_score = 0.0
        if cross_messenger_data:
            has_gw = cross_messenger_data.get("has_gravitational_wave", False)
            has_high_energy_prompt = cross_messenger_data.get("has_gamma_ray_burst", False)
            optical_detected = any(p.magnitude < 23.0 for p in observations)

            # High-energy prompt without optical afterglow despite deep early limit
            if has_high_energy_prompt and not optical_detected:
                multimodal_score += 3.5
                physical_indicators.append(
                    "Dark burst or heavy dust obscured regime: High-energy prompt emission present with extreme optical deficit."
                )
            # Gravitational wave with rapid blue optical counterpart
            elif has_gw and optical_detected:
                multimodal_score += 5.0
                physical_indicators.append(
                    "Multi-messenger association: GW strain trigger with coincident blue fast optical counterpart."
                )

        # 4. Statistical OOD Metric (Standardized Euclidean distance from canonical distribution)
        # Approximate feature vector
        measured_features = np.array([
            -18.0 if not observations else min(p.magnitude for p in observations) - 34.0,  # approximate absolute mag
            max(0.5, sorted_obs[-1].mjd - sorted_obs[0].mjd) if len(observations) >= 2 else 5.0,
            abs(dmag / dt_days) if len(observations) >= 2 and dt_days > 0.01 else 0.08,
            0.20,
        ])
        z_scores = np.abs((measured_features - self._canonical_feature_means) / self._canonical_feature_stds)
        ood_score = float(np.linalg.norm(z_scores))

        if ood_score > self.ood_threshold_sigma and not is_instrumental:
            physical_indicators.append(
                f"Multi-feature statistical anomaly (distance={ood_score:.2f} sigma > threshold {self.ood_threshold_sigma})."
            )

        # 5. Synthesis & Combined Anomaly Scoring
        if is_instrumental:
            combined_score = 0.85
            is_genuine = False
        else:
            raw_signal = (0.4 * ood_score) + (0.35 * temporal_score) + (0.25 * multimodal_score)
            combined_score = float(1.0 / (1.0 + math.exp(-0.7 * (raw_signal - 3.0))))
            is_genuine = combined_score >= 0.50

            if is_genuine:
                if multimodal_score > 3.0:
                    source = AnomalySource.PHYSICAL_MULTIMODAL_INCONSISTENCY
                elif temporal_score > 3.0:
                    source = AnomalySource.PHYSICAL_UNMODELED_EVOLUTION
                else:
                    source = AnomalySource.PHYSICAL_NEW_PHENOMENON
            else:
                source = AnomalySource.BENIGN_CONCORDANT

        return AnomalyDiagnostic(
            target_id=target_id,
            combined_anomaly_score=round(combined_score, 4),
            statistical_ood_score=round(ood_score, 2),
            temporal_residual_score=round(temporal_score, 2),
            multimodal_inconsistency_score=round(multimodal_score, 2),
            is_genuine_physical_anomaly=is_genuine,
            source_classification=source,
            rejection_reasons=artifact_reasons,
            physical_indicators=physical_indicators,
            psf_fwhm_ratio=round(avg_fwhm_ratio, 2),
            subtraction_dipole_score=1.0 if any_negative_dip else 0.0,
            confidence=0.96 if is_instrumental else 0.92,
        )
