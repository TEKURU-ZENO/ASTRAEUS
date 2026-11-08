"""
ASTRAEUS Provenance Module.
Implements IVOA Provenance Data Model v1.0 and W3C PROV-DM entity-activity-agent bindings.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ProvAgent(BaseModel):
    """
    An agent responsible for an activity (human researcher, ML model, software pipeline).
    """
    agent_id: str = Field(..., description="Unique persistent identifier for the agent.")
    agent_type: str = Field(
        ...,
        description="Type of agent: 'software_agent', 'human_researcher', 'autonomous_subagent'."
    )
    name: str = Field(..., description="Human-readable name of the agent or pipeline.")
    affiliation: Optional[str] = "ASTRAEUS Research Consortium"


class ProvActivity(BaseModel):
    """
    A computational or observational activity that generates or transforms scientific entities.
    """
    activity_id: str = Field(..., description="Unique activity identifier (e.g. 'act:ztf_ingest_001').")
    activity_name: str = Field(..., description="Name of the activity or routine.")
    software_name: str = Field(default="ASTRAEUS-Engine")
    software_version: str = Field(default="0.1.0")
    git_hash: str = Field(..., description="Git commit hash of the executing code.")
    execution_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="ISO 8601 UTC timestamp of execution."
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution hyperparameters, seed values, configuration parameters."
    )


class ProvEntity(BaseModel):
    """
    IVOA Provenance Data Model v1.0 Entity record.
    Tracks lineage relationships: wasGeneratedBy, wasAssociatedWith, used.
    """
    entity_id: str = Field(..., description="Canonical URI or ID of the data entity.")
    entity_type: str = Field(..., description="Type of entity: observation, measurement, model_weight, etc.")
    was_generated_by: Optional[str] = Field(
        default=None,
        description="Activity ID that generated this entity (IVOA wasGeneratedBy)."
    )
    was_associated_with: Optional[str] = Field(
        default=None,
        description="Agent ID associated with generation (IVOA wasAssociatedWith)."
    )
    used_entities: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite entities consumed to produce this entity (IVOA used)."
    )
    generation_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


def create_provenance_record(
    entity_id: str,
    entity_type: str,
    activity_id: str,
    agent_id: str,
    used_entities: Optional[List[str]] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ProvEntity:
    """Helper factory for standard IVOA provenance records."""
    return ProvEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        was_generated_by=activity_id,
        was_associated_with=agent_id,
        used_entities=used_entities or [],
        metadata=extra_metadata or {},
    )
