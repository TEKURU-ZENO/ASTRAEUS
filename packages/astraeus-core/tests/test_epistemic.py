import pytest
from astraeus_core.epistemic import EpistemicTier, EpistemicMetadata, ClaimType
from astraeus_core.exceptions import EpistemicViolationError


def test_epistemic_tier_enumeration():
    assert EpistemicTier.OBSERVED.value == "directly_observed"
    assert EpistemicTier.INFERRED.value == "statistically_inferred"
    assert EpistemicTier.PREDICTED.value == "physically_predicted"
    assert EpistemicTier.SIMULATED.value == "numerically_simulated"
    assert EpistemicTier.HYPOTHESIZED.value == "hypothesized"


def test_claim_types():
    assert ClaimType.MEASURED_SOURCE.value == "measured_source"
    assert ClaimType.PHYSICAL_MECHANISM.value == "physical_mechanism"
    assert ClaimType.POPULATION_RELATION.value == "population_relationship"
    assert ClaimType.THEORETICAL_MODEL.value == "theoretical_model"


def test_illegal_promotion_to_observed():
    metadata = EpistemicMetadata(
        tier=EpistemicTier.HYPOTHESIZED,
        verification_gate_passed=False,
        assumptions=["Hypothetical tidal disruption event"],
    )
    with pytest.raises(EpistemicViolationError, match="Illegal Epistemic Promotion"):
        metadata.assert_valid_promotion(EpistemicTier.OBSERVED)


def test_illegal_promotion_unverified_hypothesis():
    metadata = EpistemicMetadata(
        tier=EpistemicTier.HYPOTHESIZED,
        verification_gate_passed=False,
    )
    with pytest.raises(EpistemicViolationError, match="Cannot promote an unverified hypothesis"):
        metadata.assert_valid_promotion(EpistemicTier.PREDICTED)


def test_valid_promotion_verified_hypothesis():
    metadata = EpistemicMetadata(
        tier=EpistemicTier.HYPOTHESIZED,
        verification_gate_passed=True,
    )
    # Should not raise
    metadata.assert_valid_promotion(EpistemicTier.PREDICTED)
