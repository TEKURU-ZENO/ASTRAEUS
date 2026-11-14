"""
Tests for ASTRAEUS Relational Storage, Schemas, Constraints, Batch Import,
Multi-Table Atomicity, and Strict Live Mode Enforcement.
"""

import pytest
from sqlalchemy import select, func
from astraeus_core.exceptions import LiveBackendUnavailableError
from astraeus_data.storage import BackendMode
from astraeus_data.storage.relational_engine import (
    create_database_engine,
    init_schema,
    drop_schema,
    generate_postgresql_ddl,
    transactional_session,
)
from astraeus_data.storage.schema import (
    PersistentSkyLoci,
    PhysicalTransientEpisodes,
    CatalogReports,
    HostCounterparts,
    Candidates10k,
    PhotometryRecords,
    RawSnapshots,
)
from astraeus_data.storage.importer import CohortBatchImporter


@pytest.fixture
def db_engine():
    """Provides an isolated in-memory relational database engine with foreign keys enabled."""
    engine = create_database_engine("sqlite:///:memory:", backend_mode=BackendMode.LOCAL_EMBEDDED)
    init_schema(engine)
    yield engine
    drop_schema(engine)


def test_strict_live_mode_prohibits_silent_fallback():
    """Verifies that requesting STRICT_LIVE with SQLite or unreachable Postgres raises LiveBackendUnavailableError."""
    with pytest.raises(LiveBackendUnavailableError, match="STRICT_LIVE mode requires a production PostgreSQL URL"):
        create_database_engine("sqlite:///:memory:", backend_mode=BackendMode.STRICT_LIVE)

    with pytest.raises(LiveBackendUnavailableError, match="Failed to connect to PostgreSQL"):
        create_database_engine("postgresql+psycopg2://user:pass@127.0.0.1:54321/nonexistent", backend_mode=BackendMode.STRICT_LIVE)


def test_schema_constraints_and_foreign_keys(db_engine):
    """Verifies table creation, check constraints, and foreign key cascade behavior."""
    with transactional_session(db_engine) as session:
        locus = PersistentSkyLoci(
            sky_locus_id="LOC_TEST_001",
            ra_deg=180.5,
            dec_deg=45.0,
            healpix_nside32=100,
            assigned_base_split="train",
            episode_count=1,
        )
        session.add(locus)

    # Check constraint violation
    with pytest.raises(Exception):
        with transactional_session(db_engine) as session:
            bad_locus = PersistentSkyLoci(
                sky_locus_id="LOC_BAD",
                ra_deg=400.0,
                dec_deg=0.0,
                healpix_nside32=100,
                assigned_base_split="train",
                episode_count=1,
            )
            session.add(bad_locus)

    # Foreign key cascade delete
    with transactional_session(db_engine) as session:
        ep = PhysicalTransientEpisodes(
            physical_event_id="EP_TEST_001",
            sky_locus_id="LOC_TEST_001",
            ra_deg=180.5,
            dec_deg=45.0,
            healpix_nside32=100,
            t_start_mjd=59000.0,
            t_end_mjd=59050.0,
            assigned_split="train",
            temporal_split="archival_observation_date_holdout",
            coarse_class="SN_Ia",
            fine_class="SN_Ia",
            evidence_tier="L1",
            taxonomy_version="1.0",
        )
        session.add(ep)

    with transactional_session(db_engine) as session:
        ep_count = session.scalar(select(func.count(PhysicalTransientEpisodes.physical_event_id)))
        assert ep_count == 1

    with transactional_session(db_engine) as session:
        loc = session.get(PersistentSkyLoci, "LOC_TEST_001")
        session.delete(loc)

    with transactional_session(db_engine) as session:
        ep_count = session.scalar(select(func.count(PhysicalTransientEpisodes.physical_event_id)))
        assert ep_count == 0


def test_postgresql_ddl_export():
    """Verifies generation of production PostgreSQL DDL statements."""
    ddl = generate_postgresql_ddl()
    assert "CREATE TABLE persistent_sky_loci" in ddl
    assert "CREATE TABLE physical_transient_episodes" in ddl
    assert "CREATE TABLE catalog_reports" in ddl
    assert "CREATE TABLE host_counterparts" in ddl
    assert "CREATE TABLE candidates_10k" in ddl
    assert "CREATE TABLE photometry_records" in ddl
    assert "CREATE TABLE raw_snapshots" in ddl
    assert "chk_episode_evidence_tier" in ddl
    assert "chk_counterpart_type" in ddl


def test_cohort_10k_batch_ingestion_and_counts(db_engine):
    """
    Ingests the real 10,000-candidate cohort, 4 entities, snapshots, and photometry.
    Verifies exact row counts and multi-table atomic transaction.
    """
    importer = CohortBatchImporter(db_engine)
    metrics = importer.import_cohort(atomic=True)

    counts = importer.get_row_counts()
    assert counts["persistent_sky_loci"] == 9928
    assert counts["physical_transient_episodes"] == 9968
    assert counts["catalog_reports"] == 4479
    assert counts["host_counterparts"] == 1061
    assert counts["candidates_10k"] == 10000
    assert counts["raw_snapshots"] == 33
    assert counts["photometry_records"] == 101077

    # Accounting reconciliation
    assert metrics["manifest_entity_rows"] == 35469
    assert metrics["photometry_observation_rows"] == 101077
    assert metrics["total_rows_imported"] == 136546
    assert metrics["is_single_atomic_transaction"] is True
    assert metrics["throughput_rows_per_sec"] > 10000.0


def test_true_idempotent_upsert_replay(db_engine):
    """Confirms true idempotent replay (running upsert twice leaves row counts unchanged)."""
    importer = CohortBatchImporter(db_engine)
    importer.import_cohort(atomic=True)

    # Replay upsert
    res = importer.upsert_cohort()
    assert res["status"] == "PASS"
    assert res["idempotent_replay_success"] is True
    assert res["new_loci_inserted"] == 0

    counts = importer.get_row_counts()
    assert counts["persistent_sky_loci"] == 9928


def test_multi_table_atomic_rollback(db_engine):
    """Verifies that an injected failure during multi-table import rolls back cleanly across all tables."""
    importer = CohortBatchImporter(db_engine)
    res = importer.test_multi_table_atomic_rollback()
    assert res["status"] == "PASS"
    assert res["atomic_rollback_verified"] is True
    assert res["final_counts"]["persistent_sky_loci"] == 0
