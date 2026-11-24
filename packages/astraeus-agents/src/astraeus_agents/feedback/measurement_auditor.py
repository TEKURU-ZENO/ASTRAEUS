"""
ASTRAEUS Sequential Scientific Feedback - Measurement Quality & Provenance Auditor (WS7.6-B).
Audits incoming observational payloads for zero-point drift, seeing/aperture mismatch,
detector saturation, telemetry inconsistencies, and Gate 3 quarantine compliance.
Prevents uncalibrated, corrupted, or deliberately misleading measurements from contaminating Bayesian posteriors.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_agents.feedback.observation_models import (
    CalibratedObservation,
    CalibrationStatus,
    MeasurementType,
)


@dataclass
class MeasurementAuditResult:
    """Outcome of automated observational quality and calibration verification."""
    observation_id: str
    is_valid_for_inference: bool
    confidence_score: float  # 0.0 (corrupted) to 1.0 (pristine)
    anomalies_detected: List[str] = field(default_factory=list)
    recommended_action: str = "ACCEPT_FOR_INFERENCE"  # ACCEPT_FOR_INFERENCE, REJECT_CORRUPTED, DOWNWEIGHT
    audit_notes: str = ""


class MeasurementQualityAuditor:
    """
    Independent observational data auditor.
    Protects scientific downstream inference against corrupted or miscalibrated measurements.
    """

    MAX_PERMISSIBLE_ZP_DRIFT_MAG = 0.35  # Maximum tolerated absolute deviation from nominal zero-point
    FATAL_INSTRUMENT_FLAGS = {
        "SATURATED",
        "BAD_PIXEL_IN_CORE",
        "CORRUPTED_FLATFIELD",
        "GUIDING_FAILURE",
        "COSMIC_RAY_IN_APERTURE",
        "READOUT_ERROR",
    }

    @classmethod
    def audit_observation(cls, obs: CalibratedObservation) -> MeasurementAuditResult:
        """
        Executes multi-point quality, calibration, and provenance verification on `obs`.
        """
        anomalies: List[str] = []

        # 1. Gate 3 Zero-Touch Quarantine Verification
        try:
            QuarantineFirewall.assert_not_quarantined(
                ra_deg=obs.target_ra_deg,
                dec_deg=obs.target_dec_deg,
                identifier=f"audit_{obs.observation_id}",
                context="observation_quality_audit",
            )
        except QuarantineViolationError as e:
            return MeasurementAuditResult(
                observation_id=obs.observation_id,
                is_valid_for_inference=False,
                confidence_score=0.0,
                anomalies_detected=["GATE_3_QUARANTINE_VIOLATION"],
                recommended_action="REJECT_CORRUPTED",
                audit_notes=f"Target coordinates violate Gate 3 Quarantine Covenant: {str(e)}",
            )

        # 2. Instrument Flag Verification
        fatal_flags_present = [f for f in obs.instrument_flags if f.upper() in cls.FATAL_INSTRUMENT_FLAGS]
        if fatal_flags_present:
            anomalies.append(f"Fatal instrument telemetry flags detected: {', '.join(fatal_flags_present)}")

        # 3. Explicit Calibration Status
        if obs.calibration_status == CalibrationStatus.CALIBRATION_FAILED:
            anomalies.append("Upstream data reduction pipeline flagged calibration as FAILED.")
        elif obs.calibration_status == CalibrationStatus.UNCALIBRATED:
            anomalies.append("Payload is uncalibrated raw counts; physical flux scale absent.")

        # 4. Zero-Point Drift Check (Detects uncalibrated zero-point shifts / cloudy aperture loss)
        if obs.calibration_zero_point is not None and obs.nominal_zero_point is not None:
            zp_drift = abs(obs.calibration_zero_point - obs.nominal_zero_point)
            if zp_drift > cls.MAX_PERMISSIBLE_ZP_DRIFT_MAG:
                anomalies.append(
                    f"Severe zero-point drift: delta_ZP = {zp_drift:.3f} mag exceeds "
                    f"threshold {cls.MAX_PERMISSIBLE_ZP_DRIFT_MAG:.2f} mag (nominal={obs.nominal_zero_point:.2f}, "
                    f"measured={obs.calibration_zero_point:.2f}). Calibration suspect."
                )

        # 5. Seeing vs. Photometric Uncertainty Inconsistency Check
        # If seeing is very poor (e.g. >2.5 arcsec) but reported magnitude error is suspiciously small (<0.015 mag)
        if obs.is_detection() and obs.apparent_mag_err is not None:
            if obs.seeing_fwhm_arcsec > 2.5 and obs.apparent_mag_err < 0.015:
                anomalies.append(
                    f"Unphysical uncertainty: seeing={obs.seeing_fwhm_arcsec:.2f}'' is degraded but reported "
                    f"photometric error {obs.apparent_mag_err:.4f} mag is implausibly tiny. Possible unmodeled aperture loss."
                )
            if obs.apparent_mag_err <= 0.0:
                anomalies.append(f"Invalid non-positive photometric uncertainty: sigma_m = {obs.apparent_mag_err}")

        # 6. Upper-limit consistency check
        if obs.is_upper_limit():
            lim = obs.limiting_mag_5sigma
            if lim is not None and (lim < 10.0 or lim > 32.0):
                anomalies.append(f"Unphysical 5-sigma limiting magnitude reported: m_lim = {lim:.2f}")

        # Determine validity and confidence
        has_fatal_anomaly = any(
            "Fatal" in a or "FAILED" in a or "Severe zero-point drift" in a or "GATE_3" in a or "Invalid non-positive" in a
            for a in anomalies
        )

        if has_fatal_anomaly:
            is_valid = False
            rec_action = "REJECT_CORRUPTED"
            conf = 0.0
        elif anomalies:
            # Minor non-fatal anomalies: downweight
            is_valid = True
            rec_action = "DOWNWEIGHT"
            conf = 0.50
        else:
            is_valid = True
            rec_action = "ACCEPT_FOR_INFERENCE"
            conf = 1.0

        return MeasurementAuditResult(
            observation_id=obs.observation_id,
            is_valid_for_inference=is_valid,
            confidence_score=conf,
            anomalies_detected=anomalies,
            recommended_action=rec_action,
            audit_notes=f"Audit completed: {len(anomalies)} anomalies identified.",
        )
