"""
ASTRAEUS Astronomical Entity Module.
The master scientific object schema enforcing epistemic stratification and IVOA provenance contracts.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator

from astraeus_core.epistemic import EpistemicTier
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.observations import ObservationRecord, CalibratedMeasurement
from astraeus_core.provenance import ProvEntity
from astraeus_core.conformal import ConformalPredictionInterval
from astraeus_core.scoring import CompositeAnomalyScore
from astraeus_core.exceptions import EpistemicViolationError, ProvenanceMissingError


class AstronomicalEntity(BaseModel):
    """
    The unified scientific data entity schema.
    Represents celestial sources, transients, candidate discoveries, and theoretical models.
    """
    schema_version: str = Field(
        default="https://astraeus.org/schemas/v1/astronomical_entity.json",
        description="JSON Schema specification URI."
    )
    entity_id: str = Field(
        ...,
        description="Canonical persistent identifier (e.g. 'astraeus:entity:ztf:2026_0192')."
    )
    entity_type: str = Field(
        ...,
        description="Object category: 'transient_candidate', 'variable_star', 'galaxy', 'supernova', etc."
    )
    epistemic_tier: EpistemicTier = Field(
        ...,
        description="Ontological status: observed, inferred, predicted, simulated, hypothesized."
    )
    coordinates: CelestialCoordinates = Field(
        ...,
        description="Astrometrically calibrated sky position and HEALPix partitioning."
    )
    observations: List[ObservationRecord] = Field(
        default_factory=list,
        description="Direct detector observations linked to certified instruments."
    )
    measurements: List[CalibratedMeasurement] = Field(
        default_factory=list,
        description="Physical quantities and extracted features."
    )
    available_modalities: List[str] = Field(
        default_factory=list,
        description="List of available data modalities: 'imaging_cutout', 'photometric_time_series', 'spectrum', 'catalog'."
    )
    foundation_embedding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Multimodal latent vector, backbone version, and local latent density."
    )
    conformal_inferences: Dict[str, ConformalPredictionInterval] = Field(
        default_factory=dict,
        description="Calibrated statistical inferences with finite-sample prediction intervals."
    )
    anomaly_score: Optional[CompositeAnomalyScore] = Field(
        default=None,
        description="Composite anomaly score evaluated by the triage engine."
    )
    provenance: ProvEntity = Field(
        ...,
        description="IVOA Provenance record tracking activity, agent, and entity inputs."
    )

    @model_validator(mode="after")
    def validate_epistemic_invariants(self) -> "AstronomicalEntity":
        """
        Enforces strict epistemological gates across all created entities.
        """
        # Gate 1: Directly Observed entities must originate from physical observations
        if self.epistemic_tier == EpistemicTier.OBSERVED:
            if not self.observations:
                raise EpistemicViolationError(
                    f"Entity '{self.entity_id}' has tier 'directly_observed' but contains no observation records."
                )
            for obs in self.observations:
                if not obs.ivoa_provenance_id:
                    raise ProvenanceMissingError(
                        f"Observation '{obs.observation_id}' lacks mandatory 'ivoa_provenance_id'."
                    )

        # Gate 2: Statistically Inferred entities must have an associated generating activity
        elif self.epistemic_tier == EpistemicTier.INFERRED:
            if not self.provenance.was_generated_by:
                raise ProvenanceMissingError(
                    f"Inferred entity '{self.entity_id}' must record 'was_generated_by' activity in provenance."
                )

        # Gate 3: Hypothesized entities cannot claim directly observed observations without linking via used
        elif self.epistemic_tier == EpistemicTier.HYPOTHESIZED:
            if self.observations and not self.provenance.used_entities:
                raise EpistemicViolationError(
                    f"Hypothesized entity '{self.entity_id}' contains direct observations but lacks 'used_entities' lineage."
                )

        return self

    def promote_to_inferred(
        self,
        new_activity_id: str,
        agent_id: str,
        inferences: Dict[str, ConformalPredictionInterval]
    ) -> "AstronomicalEntity":
        """
        Derives an inferred entity from an observed entity with explicit provenance lineage.
        """
        if self.epistemic_tier != EpistemicTier.OBSERVED:
            raise EpistemicViolationError("Can only derive inferred properties from directly observed entities.")

        derived_prov = ProvEntity(
            entity_id=f"{self.entity_id}:inferred",
            entity_type="inferred_properties",
            was_generated_by=new_activity_id,
            was_associated_with=agent_id,
            used_entities=[self.entity_id],
        )

        return AstronomicalEntity(
            entity_id=f"{self.entity_id}:inferred",
            entity_type=self.entity_type,
            epistemic_tier=EpistemicTier.INFERRED,
            coordinates=self.coordinates,
            observations=self.observations,
            measurements=self.measurements,
            available_modalities=self.available_modalities,
            foundation_embedding=self.foundation_embedding,
            conformal_inferences=inferences,
            provenance=derived_prov,
        )
