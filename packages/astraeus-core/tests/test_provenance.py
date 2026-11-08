from astraeus_core.provenance import (
    ProvAgent,
    ProvActivity,
    ProvEntity,
    create_provenance_record,
)


def test_provenance_creation():
    agent = ProvAgent(
        agent_id="agent:pipeline:ingest",
        agent_type="software_agent",
        name="ZTF-Ingestion-Pipeline",
    )
    activity = ProvActivity(
        activity_id="act:ztf_ingest_001",
        activity_name="ingest_alert_packet",
        git_hash="c3a4f59e01",
        parameters={"band": "g", "zero_point": 26.3},
    )
    entity_prov = create_provenance_record(
        entity_id="astraeus:obs:ztf:1001",
        entity_type="observation",
        activity_id=activity.activity_id,
        agent_id=agent.agent_id,
        used_entities=["raw:packet:998877"],
    )

    assert entity_prov.entity_id == "astraeus:obs:ztf:1001"
    assert entity_prov.was_generated_by == "act:ztf_ingest_001"
    assert entity_prov.was_associated_with == "agent:pipeline:ingest"
    assert "raw:packet:998877" in entity_prov.used_entities
