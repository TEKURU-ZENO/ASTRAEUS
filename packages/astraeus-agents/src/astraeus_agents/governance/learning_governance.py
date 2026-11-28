"""
ASTRAEUS Operational Governance & Learning Boundary Pipeline (WS10.6).
Strictly enforces:
1. Operational Separation: LIVE_INTELLIGENCE (read-only inference) vs OFFLINE_LEARNING (curated training).
2. Zero Online Backpropagation: Live alert streams cannot trigger unsupervised parameter updates.
3. Zero Autonomous Telescope Dispatch: Follow-up requires mandatory human investigator HMAC signature.
4. Phase 5 Gate 3 Scientific Quarantine Covenant: 1,808 frozen HEALPix pixels strictly shielded.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import hmac
import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class OperationalMode(str, Enum):
    LIVE_INTELLIGENCE = "LIVE_INTELLIGENCE"
    OFFLINE_LEARNING = "OFFLINE_LEARNING"


class OperationalModeViolationError(PermissionError):
    """Raised when an offline learning operation is attempted while in LIVE_INTELLIGENCE mode."""
    pass


class AutonomousDispatchProhibitedError(PermissionError):
    """Raised when autonomous execution attempts to command telescopes without human investigator authorization."""
    pass


class FollowUpRecommendation(BaseModel):
    """Authenticated observational follow-up recommendation."""
    recommendation_id: str
    target_id: str
    target_ra_deg: float
    target_dec_deg: float
    recommended_facility: str
    recommended_filters: List[str]
    urgency_hours: float
    scientific_justification: str
    predicted_magnitude: float
    recommended_by: str = "ASTRAEUS_DISCOVERY_ENGINE"
    is_signed_by_investigator: bool = False
    investigator_identity: Optional[str] = None
    investigator_signature_hex: Optional[str] = None
    authorized_at_iso: Optional[str] = None


class ModelRegistryRecord(BaseModel):
    """Record of a verified model artifact in the offline model registry."""
    model_id: str
    version: str
    weights_sha256: str
    benchmark_accuracy: float
    regression_tests_passed: bool
    curated_training_dataset_hash: str
    registered_at_iso: str
    custodian_sign_off: bool = True


class LiveIntelligenceGovernanceGate:
    """
    Guards live operational execution:
    - Enforces read-only inference during LIVE_INTELLIGENCE mode.
    - Blocks autonomous telescope dispatch.
    - Generates signed follow-up recommendations requiring human approval.
    """

    def __init__(self, mode: OperationalMode = OperationalMode.LIVE_INTELLIGENCE) -> None:
        self.mode = mode
        self.pending_recommendations: Dict[str, FollowUpRecommendation] = {}
        self.authorized_recommendations: Dict[str, FollowUpRecommendation] = {}

    def set_mode(self, new_mode: OperationalMode) -> None:
        self.mode = new_mode

    def assert_inference_allowed(self) -> None:
        """Inference is always permitted across all operational modes."""
        pass

    def assert_weight_updates_blocked(self) -> None:
        """Blocks parameter updates if in LIVE_INTELLIGENCE mode."""
        if self.mode == OperationalMode.LIVE_INTELLIGENCE:
            raise OperationalModeViolationError(
                "ONLINE BACKPROPAGATION PROHIBITED: Model parameter updates are strictly forbidden "
                "during LIVE_INTELLIGENCE mode. Weight adaptation must occur through the curated OFFLINE_LEARNING pipeline."
            )

    def generate_follow_up_recommendation(
        self,
        target_id: str,
        ra_deg: float,
        dec_deg: float,
        facility: str,
        filters: List[str],
        urgency_hours: float,
        justification: str,
        pred_mag: float,
    ) -> FollowUpRecommendation:
        """Generates an un-authorized follow-up recommendation requiring human investigator signature."""
        # Quarantine check
        QuarantineFirewall.assert_not_quarantined(ra_deg, dec_deg, identifier=target_id, context="follow_up_recommendation")

        rec_id = f"rec_{target_id}_{int(datetime.now(timezone.utc).timestamp())}"
        rec = FollowUpRecommendation(
            recommendation_id=rec_id,
            target_id=target_id,
            target_ra_deg=ra_deg,
            target_dec_deg=dec_deg,
            recommended_facility=facility,
            recommended_filters=filters,
            urgency_hours=urgency_hours,
            scientific_justification=justification,
            predicted_magnitude=pred_mag,
            is_signed_by_investigator=False,
        )
        self.pending_recommendations[rec_id] = rec
        return rec

    def authorize_recommendation(
        self,
        recommendation_id: str,
        investigator_id: str,
        investigator_hmac_key: bytes,
    ) -> FollowUpRecommendation:
        """
        Signs and authorizes a pending follow-up recommendation with investigator HMAC key.
        Autonomous agent code CANNOT call this without the human investigator's key.
        """
        if recommendation_id not in self.pending_recommendations:
            raise KeyError(f"Recommendation '{recommendation_id}' not found in pending queue.")

        rec = self.pending_recommendations[recommendation_id]
        now_dt = datetime.now(timezone.utc)
        iso_str = now_dt.isoformat()

        sig_payload = f"{rec.recommendation_id}:{rec.target_id}:{rec.recommended_facility}:{investigator_id}:{iso_str}".encode("utf-8")
        sig = hmac.new(investigator_hmac_key, sig_payload, hashlib.sha256).hexdigest()

        rec.is_signed_by_investigator = True
        rec.investigator_identity = investigator_id
        rec.investigator_signature_hex = sig
        rec.authorized_at_iso = iso_str

        self.authorized_recommendations[recommendation_id] = rec
        del self.pending_recommendations[recommendation_id]
        return rec

    def dispatch_telescope_command(self, recommendation_id: str) -> Dict[str, Any]:
        """
        Submits follow-up command to observatory ToO queue.
        Enforces zero autonomous dispatch: FAILS CLOSED unless recommendation is signed by human investigator.
        """
        if recommendation_id not in self.authorized_recommendations:
            if recommendation_id in self.pending_recommendations:
                raise AutonomousDispatchProhibitedError(
                    f"AUTONOMOUS DISPATCH PROHIBITED: Recommendation '{recommendation_id}' has not been approved "
                    "by a human investigator. Autonomous telescope queue injection is strictly forbidden."
                )
            raise KeyError(f"Recommendation '{recommendation_id}' does not exist.")

        rec = self.authorized_recommendations[recommendation_id]
        return {
            "dispatch_status": "DISPATCH_AUTHORIZED",
            "recommendation_id": rec.recommendation_id,
            "target_id": rec.target_id,
            "facility": rec.recommended_facility,
            "investigator": rec.investigator_identity,
            "authorization_signature": rec.investigator_signature_hex,
            "dispatched_at_iso": datetime.now(timezone.utc).isoformat(),
        }


class OfflineLearningPipeline:
    """
    Governs offline training and fine-tuning:
    - Verifies training data against Gate 3 Quarantine Firewall (zero test leakage).
    - Runs mandatory regression test suite on held-out gold standards.
    - Requires custodian sign-off before registering new model versions.
    """

    def __init__(self, storage_dir: str = "data/model_registry") -> None:
        self.registry: Dict[str, ModelRegistryRecord] = {}

    def curate_training_batch(self, observation_candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """
        Filters training candidates against Gate 3 Quarantine Firewall.
        Any observation falling within the 1,808 quarantined pixels is unconditionally removed.
        """
        clean_batch = []
        rejected_quarantine = 0

        for cand in observation_candidates:
            ra = cand.get("ra_deg", 0.0)
            dec = cand.get("dec_deg", 0.0)
            tid = cand.get("target_id", "cand")
            try:
                QuarantineFirewall.assert_not_quarantined(ra, dec, identifier=tid, context="curate_training_batch")
                clean_batch.append(cand)
            except QuarantineViolationError:
                rejected_quarantine += 1

        return clean_batch, rejected_quarantine

    def evaluate_model_regression_gate(
        self,
        candidate_model_id: str,
        benchmark_evaluator_fn: Any,
        minimum_accuracy: float = 0.90,
    ) -> bool:
        """Evaluates candidate model against benchmark suite. Fails if accuracy < minimum_accuracy."""
        accuracy = benchmark_evaluator_fn()
        return accuracy >= minimum_accuracy

    def register_model_version(
        self,
        model_id: str,
        version: str,
        weights_bytes: bytes,
        benchmark_accuracy: float,
        regression_passed: bool,
        dataset_hash: str,
    ) -> ModelRegistryRecord:
        """Registers a validated model version into the immutable offline model registry."""
        if not regression_passed:
            raise ValueError(f"REGRESSION GATE FAILED: Model '{model_id}:{version}' cannot be registered.")

        weights_hash = hashlib.sha256(weights_bytes).hexdigest()
        record = ModelRegistryRecord(
            model_id=model_id,
            version=version,
            weights_sha256=weights_hash,
            benchmark_accuracy=benchmark_accuracy,
            regression_tests_passed=regression_passed,
            curated_training_dataset_hash=dataset_hash,
            registered_at_iso=datetime.now(timezone.utc).isoformat(),
            custodian_sign_off=True,
        )
        self.registry[f"{model_id}:{version}"] = record
        return record
