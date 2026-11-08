"""
ASTRAEUS Epistemic Stratification Module.
Defines the 5-tier scientific knowledge taxonomy and validation rules.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from astraeus_core.exceptions import EpistemicViolationError


class EpistemicTier(str, Enum):
    """
    Five-tier epistemic ontological taxonomy enforcing rigorous scientific boundaries.
    """
    OBSERVED = "directly_observed"
    INFERRED = "statistically_inferred"
    PREDICTED = "physically_predicted"
    SIMULATED = "numerically_simulated"
    HYPOTHESIZED = "hypothesized"


class ClaimType(str, Enum):
    """
    Taxonomy of scientific claims requiring distinct evidentiary verification standards.
    """
    MEASURED_SOURCE = "measured_source"             # Individual astronomical object observation
    PHYSICAL_MECHANISM = "physical_mechanism"       # Specific physical process (e.g. shock interaction)
    POPULATION_RELATION = "population_relationship" # Statistical relationship across a population (e.g. period-luminosity)
    THEORETICAL_MODEL = "theoretical_model"         # Analytical theory or equation of state


class EpistemicMetadata(BaseModel):
    """
    Metadata describing the epistemic justification, limits, and gate verification
    of a scientific data entity or claim.
    """
    tier: EpistemicTier = Field(
        ...,
        description="Epistemic tier of the associated entity or claim."
    )
    claim_type: Optional[ClaimType] = Field(
        default=None,
        description="Type of scientific claim, dictating evidentiary verification standards."
    )
    verification_gate_passed: bool = Field(
        default=False,
        description="Whether this entity has passed its tier-specific invalidation gate."
    )
    evidentiary_basis: List[str] = Field(
        default_factory=list,
        description="IDs of underlying primary entities or observations used as evidence."
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit physical or statistical assumptions underlying this entity."
    )
    validity_envelope: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Domain of validity (e.g., parameter range, redshift limits, spatial scale)."
    )
    falsification_criteria: Optional[List[str]] = Field(
        default=None,
        description="Specific observational or mathematical tests that would invalidate this entity."
    )

    def assert_valid_promotion(self, target_tier: EpistemicTier) -> None:
        """
        Guards against epistemic conflation. Prevents direct promotion from
        speculative or simulated tiers to directly observed without physical measurement.
        A verified hypothesis does not become an observation; it transitions to a corroborated hypothesis.
        """
        if target_tier == EpistemicTier.OBSERVED and self.tier != EpistemicTier.OBSERVED:
            raise EpistemicViolationError(
                f"Illegal Epistemic Promotion: Cannot promote an entity of tier '{self.tier.value}' "
                f"to '{EpistemicTier.OBSERVED.value}'. Only direct certified instrument measurements qualify. "
                f"A verified hypothesis transitions to corroborated status, never directly observed."
            )
        if target_tier == EpistemicTier.PREDICTED and self.tier == EpistemicTier.HYPOTHESIZED:
            if not self.verification_gate_passed:
                raise EpistemicViolationError(
                    "Cannot promote an unverified hypothesis to physically predicted without mathematical/conservation proofs."
                )
