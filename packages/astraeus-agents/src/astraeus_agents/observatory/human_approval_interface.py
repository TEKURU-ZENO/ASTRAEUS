"""
ASTRAEUS Observatory Intelligence - Human Approval & Governance Interface (WS7.5).
Enforces the mandatory human-in-the-loop authorization gate before any observational
pointing recommendation can be dispatched to an astronomical facility or queue.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import time
import math
import hashlib
import hmac
import secrets
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from astraeus_agents.observatory.schedulers import ScheduledObservation, TargetState


class ApprovalStatus(str, Enum):
    PENDING_HUMAN_REVIEW = "PENDING_HUMAN_REVIEW"
    AUTHORIZED_BY_INVESTIGATOR = "AUTHORIZED_BY_INVESTIGATOR"
    REJECTED_BY_INVESTIGATOR = "REJECTED_BY_INVESTIGATOR"


class AutonomousDispatchProhibitedError(PermissionError):
    """Raised when an automated agent or pipeline attempts to dispatch pointing commands without human authorization."""
    pass


class TokenValidationError(PermissionError):
    """Raised when an authorization token is invalid, expired, already consumed, or mismatched."""
    pass


class AuthorizationToken(BaseModel):
    """
    Cryptographically bound, single-use, expiring authorization token
    required for physical telescope queue dispatch.
    """
    token_id: str
    recommendation_id: str
    target_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    exposure_time_sec: float
    scheduled_mjd: float
    proposal_hash: str
    investigator_email: str
    created_at_utc: float
    ttl_seconds: float = 3600.0
    is_consumed: bool = False
    dispatched_at_utc: Optional[float] = None
    token_signature_sha256: str
    hmac_signature: str

    def is_expired(self, current_time_utc: Optional[float] = None) -> bool:
        now = current_time_utc if current_time_utc is not None else time.time()
        return (now - self.created_at_utc) > self.ttl_seconds


class ObservationRecommendation(BaseModel):
    """Formal human-facing observation dispatch recommendation."""
    recommendation_id: str
    target_id: str
    facility_id: str
    instrument_id: str
    filter_id: str
    exposure_time_sec: float
    scheduled_mjd: float
    expected_eig_nats: float
    prior_entropy_nats: float
    expected_posterior_entropy_nats: float
    entropy_reduction_pct: float
    competing_hypotheses: Dict[str, float]
    scientific_justification: str
    physical_constraints_audit: Dict[str, Any]
    approval_status: ApprovalStatus = ApprovalStatus.PENDING_HUMAN_REVIEW
    authorized_investigator: Optional[str] = None
    authorization_timestamp_utc: Optional[float] = None
    rejection_reason: Optional[str] = None
    cryptographic_proposal_hash: str


class HumanApprovalGate:
    """
    Mandatory governance gate prohibiting autonomous dispatch of observational pointings.
    All telescope queue modifications must be explicitly signed by the Lead Investigator.
    """

    def __init__(
        self,
        event_store: Optional[Any] = None,
        hmac_secret_key: Optional[bytes] = None,
    ):
        self._recommendations: Dict[str, ObservationRecommendation] = {}
        self._tokens: Dict[str, AuthorizationToken] = {}
        self._event_store = event_store

        # Mandatory high-entropy secret key loading: no hardcoded default fallback
        loaded_key = hmac_secret_key
        if loaded_key is None:
            env_key = os.environ.get("ASTRAEUS_HMAC_SECRET_KEY")
            if env_key:
                loaded_key = env_key.encode("utf-8")

        if loaded_key is None:
            raise KeyError(
                "SECURITY COVENANT VIOLATION: Mandatory HMAC secret key missing. "
                "Must provide 'hmac_secret_key' or set 'ASTRAEUS_HMAC_SECRET_KEY' in environment. "
                "Hardcoded default keys are strictly prohibited."
            )
        if len(loaded_key) < 32:
            raise ValueError(
                "SECURITY COVENANT VIOLATION: HMAC secret key entropy insufficient. "
                f"Key must be at least 32 bytes (got {len(loaded_key)} bytes)."
            )
        self.hmac_secret_key: bytes = loaded_key

    def _get_or_restore_recommendation(self, recommendation_id: str) -> Optional[ObservationRecommendation]:
        """Retrieves recommendation from memory or restores from persistent SQLite registry."""
        if recommendation_id in self._recommendations:
            return self._recommendations[recommendation_id]
        if self._event_store is not None:
            persisted = self._event_store.get_recommendation(recommendation_id)
            if persisted:
                rec = ObservationRecommendation(
                    recommendation_id=persisted["recommendation_id"],
                    target_id=persisted["target_id"],
                    facility_id=persisted["facility_id"],
                    instrument_id=persisted["instrument_id"],
                    filter_id=persisted["filter_id"],
                    exposure_time_sec=persisted["exposure_time_sec"],
                    scheduled_mjd=persisted["scheduled_mjd"],
                    expected_eig_nats=persisted["expected_eig_nats"],
                    prior_entropy_nats=persisted["prior_entropy_nats"],
                    expected_posterior_entropy_nats=persisted["expected_posterior_entropy_nats"],
                    entropy_reduction_pct=persisted["entropy_reduction_pct"],
                    competing_hypotheses=persisted["competing_hypotheses"],
                    scientific_justification=persisted["scientific_justification"],
                    physical_constraints_audit=persisted["physical_constraints_audit"],
                    approval_status=ApprovalStatus(persisted["approval_status"]),
                    authorized_investigator=persisted.get("authorized_investigator"),
                    authorization_timestamp_utc=persisted.get("authorization_timestamp_utc"),
                    rejection_reason=persisted.get("rejection_reason"),
                    cryptographic_proposal_hash=persisted["cryptographic_proposal_hash"],
                )
                self._recommendations[recommendation_id] = rec
                return rec
        return None

    def _compute_hmac_signature(
        self,
        token_id: str,
        recommendation_id: str,
        target_id: str,
        facility_id: str,
        instrument_id: str,
        filter_id: str,
        exposure_time_sec: float,
        scheduled_mjd: float,
        proposal_hash: str,
        investigator_email: str,
        created_at_utc: float,
        ttl_seconds: float,
    ) -> str:
        canonical_str = (
            f"{token_id}:{recommendation_id}:{target_id}:{facility_id}:{instrument_id}:{filter_id}:"
            f"{exposure_time_sec:.4f}:{scheduled_mjd:.6f}:{proposal_hash}:{investigator_email}:"
            f"{created_at_utc:.4f}:{ttl_seconds:.1f}"
        )
        return hmac.new(self.hmac_secret_key, canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()

    def create_recommendation(
        self,
        obs: ScheduledObservation,
        target: TargetState,
        scientific_justification: str,
    ) -> ObservationRecommendation:
        """Packages a scheduled observation into a human review proposal."""
        # Prior entropy
        ent_prior = 0.0
        for p in target.hypothesis_posteriors.values():
            if p > 1e-12:
                ent_prior -= p * math.log(p)

        ent_post = max(0.0, ent_prior - obs.expected_eig_nats)
        reduction_pct = (obs.expected_eig_nats / max(1e-4, ent_prior)) * 100.0

        raw_id = f"{obs.target_id}:{obs.facility_id}:{obs.filter_id}:{obs.start_mjd:.5f}"
        rec_id = f"rec:{hashlib.sha256(raw_id.encode()).hexdigest()[:16]}"

        raw_content = f"{rec_id}:{target.target_id}:{obs.facility_id}:{obs.instrument_id}:{obs.filter_id}:{obs.exposure_time_sec}:{obs.start_mjd}"
        prop_hash = hashlib.sha256(raw_content.encode()).hexdigest()

        rec = ObservationRecommendation(
            recommendation_id=rec_id,
            target_id=target.target_id,
            facility_id=obs.facility_id,
            instrument_id=obs.instrument_id,
            filter_id=obs.filter_id,
            exposure_time_sec=obs.exposure_time_sec,
            scheduled_mjd=obs.start_mjd,
            expected_eig_nats=round(obs.expected_eig_nats, 4),
            prior_entropy_nats=round(ent_prior, 4),
            expected_posterior_entropy_nats=round(ent_post, 4),
            entropy_reduction_pct=round(reduction_pct, 2),
            competing_hypotheses={k: round(v, 4) for k, v in target.hypothesis_posteriors.items()},
            scientific_justification=scientific_justification,
            physical_constraints_audit={
                "airmass": obs.airmass,
                "slew_time_sec": obs.slew_time_sec,
                "weather_score": obs.weather_score,
                "hard_violations": obs.constraint_violations,
            },
            approval_status=ApprovalStatus.PENDING_HUMAN_REVIEW,
            cryptographic_proposal_hash=prop_hash,
        )
        self._recommendations[rec_id] = rec
        if self._event_store is not None:
            self._event_store.store_recommendation(rec.model_dump())
        return rec

    def authorize_dispatch(
        self,
        recommendation_id: str,
        investigator_email: str,
    ) -> ObservationRecommendation:
        """Authorizes dispatch with human investigator signature."""
        rec = self._get_or_restore_recommendation(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found in memory or persistent store.")

        rec.approval_status = ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR
        rec.authorized_investigator = investigator_email
        rec.authorization_timestamp_utc = time.time()
        if self._event_store is not None:
            self._event_store.update_recommendation_status(
                recommendation_id=recommendation_id,
                status=ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR.value,
                investigator_email=investigator_email,
                timestamp_utc=rec.authorization_timestamp_utc,
            )
        return rec

    def reject_dispatch(
        self,
        recommendation_id: str,
        reason: str,
    ) -> ObservationRecommendation:
        """Rejects dispatch recommendation with justification."""
        rec = self._get_or_restore_recommendation(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found in memory or persistent store.")

        rec.approval_status = ApprovalStatus.REJECTED_BY_INVESTIGATOR
        rec.rejection_reason = reason
        if self._event_store is not None:
            self._event_store.update_recommendation_status(
                recommendation_id=recommendation_id,
                status=ApprovalStatus.REJECTED_BY_INVESTIGATOR.value,
                rejection_reason=reason,
            )
        return rec

    def assert_authorized_for_dispatch(self, recommendation_id: str) -> None:
        """
        Enforces execution gate: raises AutonomousDispatchProhibitedError
        if recommendation is not officially approved.
        """
        rec = self._get_or_restore_recommendation(recommendation_id)
        if rec is None or rec.approval_status != ApprovalStatus.AUTHORIZED_BY_INVESTIGATOR:
            raise AutonomousDispatchProhibitedError(
                f"GOVERNANCE GATE VIOLATION: Recommendation {recommendation_id} is in state "
                f"'{rec.approval_status if rec else 'UNKNOWN'}'. Autonomous telescope dispatch is strictly prohibited."
            )

    def issue_authorization_token(
        self,
        recommendation_id: str,
        investigator_email: str,
        ttl_seconds: float = 3600.0,
    ) -> AuthorizationToken:
        """Issues an immutable, cryptographically bound, single-use authorization token."""
        rec = self._get_or_restore_recommendation(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found in memory or persistent store.")

        # Authorize the recommendation record
        self.authorize_dispatch(recommendation_id, investigator_email)

        now = time.time()
        token_id = f"tok_{hashlib.sha256(f'{recommendation_id}:{now}'.encode()).hexdigest()[:16]}"
        sig_raw = f"{token_id}:{rec.recommendation_id}:{rec.target_id}:{rec.facility_id}:{rec.filter_id}:{investigator_email}:{now}"
        token_sig = hashlib.sha256(sig_raw.encode()).hexdigest()

        hmac_sig = self._compute_hmac_signature(
            token_id=token_id,
            recommendation_id=rec.recommendation_id,
            target_id=rec.target_id,
            facility_id=rec.facility_id,
            instrument_id=rec.instrument_id,
            filter_id=rec.filter_id,
            exposure_time_sec=rec.exposure_time_sec,
            scheduled_mjd=rec.scheduled_mjd,
            proposal_hash=rec.cryptographic_proposal_hash,
            investigator_email=investigator_email,
            created_at_utc=now,
            ttl_seconds=ttl_seconds,
        )

        token = AuthorizationToken(
            token_id=token_id,
            recommendation_id=rec.recommendation_id,
            target_id=rec.target_id,
            facility_id=rec.facility_id,
            instrument_id=rec.instrument_id,
            filter_id=rec.filter_id,
            exposure_time_sec=rec.exposure_time_sec,
            scheduled_mjd=rec.scheduled_mjd,
            proposal_hash=rec.cryptographic_proposal_hash,
            investigator_email=investigator_email,
            created_at_utc=now,
            ttl_seconds=ttl_seconds,
            is_consumed=False,
            token_signature_sha256=token_sig,
            hmac_signature=hmac_sig,
        )
        self._tokens[token_id] = token

        # Persist to durable SQLite event store if available
        if self._event_store is not None:
            token_dict = {
                "token_id": token.token_id,
                "recommendation_id": token.recommendation_id,
                "target_id": token.target_id,
                "facility_id": token.facility_id,
                "instrument_id": token.instrument_id,
                "filter_id": token.filter_id,
                "exposure_time_sec": token.exposure_time_sec,
                "scheduled_mjd": token.scheduled_mjd,
                "proposal_hash": token.proposal_hash,
                "investigator_email": token.investigator_email,
                "created_at_utc": token.created_at_utc,
                "ttl_seconds": token.ttl_seconds,
                "hmac_signature": hmac_sig,
            }
            self._event_store.store_authorization_token(token_dict)

        return token

    def dispatch_with_token(
        self,
        token: AuthorizationToken,
        recommendation_id: str,
        current_time_utc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Validates token binding, cryptographic HMAC signature, single-use,
        and expiration before physical dispatch.
        """
        now = current_time_utc if current_time_utc is not None else time.time()

        # 0. Enforce mandatory event store
        if self._event_store is None:
            raise AutonomousDispatchProhibitedError(
                "GOVERNANCE GATE VIOLATION: Durable SQLite event store is mandatory for physical dispatch. "
                "Volatile in-memory dispatch without durable single-use registration is strictly prohibited."
            )

        # 1. Enforce mandatory cryptographic HMAC signature
        if not token.hmac_signature:
            raise TokenValidationError(
                f"CRYPTOGRAPHIC SIGNATURE MISSING: Token {token.token_id} has no HMAC signature. "
                "Mandatory cryptographic signature covenant enforced."
            )

        expected_hmac = self._compute_hmac_signature(
            token_id=token.token_id,
            recommendation_id=token.recommendation_id,
            target_id=token.target_id,
            facility_id=token.facility_id,
            instrument_id=token.instrument_id,
            filter_id=token.filter_id,
            exposure_time_sec=token.exposure_time_sec,
            scheduled_mjd=token.scheduled_mjd,
            proposal_hash=token.proposal_hash,
            investigator_email=token.investigator_email,
            created_at_utc=token.created_at_utc,
            ttl_seconds=token.ttl_seconds,
        )
        if not hmac.compare_digest(token.hmac_signature, expected_hmac):
            raise TokenValidationError(
                f"CRYPTOGRAPHIC SIGNATURE FORGERY DETECTED: Token {token.token_id} HMAC signature is invalid."
            )

        # 2. Check recommendation match
        if token.recommendation_id != recommendation_id:
            raise TokenValidationError(
                f"TOKEN TARGET MISMATCH: Token was issued for recommendation {token.recommendation_id}, "
                f"cannot authorize dispatch for recommendation {recommendation_id}."
            )

        # 3. Retrieve or verify stored token
        stored_token = self._tokens.get(token.token_id)
        if stored_token is None:
            # Reconstruct from persistent SQLite store (e.g. across restart)
            persisted = self._event_store.get_authorization_token(token.token_id)
            if persisted:
                stored_token = AuthorizationToken(
                    token_id=persisted["token_id"],
                    recommendation_id=persisted["recommendation_id"],
                    target_id=persisted["target_id"],
                    facility_id=persisted["facility_id"],
                    instrument_id=persisted["instrument_id"],
                    filter_id=persisted["filter_id"],
                    exposure_time_sec=persisted["exposure_time_sec"],
                    scheduled_mjd=persisted["scheduled_mjd"],
                    proposal_hash=persisted["proposal_hash"],
                    investigator_email=persisted["investigator_email"],
                    created_at_utc=persisted["created_at_utc"],
                    ttl_seconds=persisted["ttl_seconds"],
                    is_consumed=bool(persisted["is_consumed"]),
                    dispatched_at_utc=persisted.get("consumed_at_utc"),
                    token_signature_sha256="",
                    hmac_signature=persisted["hmac_signature"],
                )
                self._tokens[token.token_id] = stored_token

        if stored_token is None:
            raise TokenValidationError(f"Unregistered authorization token: {token.token_id}")

        rec = self._get_or_restore_recommendation(recommendation_id)
        if rec is None:
            raise KeyError(f"Recommendation {recommendation_id} not found in memory or persistent store.")

        if stored_token.target_id != rec.target_id or stored_token.facility_id != rec.facility_id:
            raise TokenValidationError("TOKEN PARAMETER MISMATCH: Target or facility diverges from proposal.")

        # 4. Atomic single-use consumption in SQLite WAL store
        success, reason = self._event_store.atomic_consume_token(token.token_id, current_time_utc=now)
        if not success:
            raise TokenValidationError(f"GOVERNANCE GATE VIOLATION: {reason}")
        stored_token.is_consumed = True
        stored_token.dispatched_at_utc = now
        token.is_consumed = True
        token.dispatched_at_utc = now

        return {
            "dispatch_status": "DISPATCHED_TO_FACILITY_QUEUE",
            "token_id": token.token_id,
            "recommendation_id": recommendation_id,
            "facility_id": rec.facility_id,
            "instrument_id": rec.instrument_id,
            "dispatched_at_utc": now,
            "authorized_by": stored_token.investigator_email,
        }
