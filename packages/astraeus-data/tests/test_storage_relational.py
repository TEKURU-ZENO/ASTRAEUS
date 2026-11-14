"""
Tests for ASTRAEUS Relational Storage, Schemas, Constraints, and Batch Import.
"""

import pytest
from sqlalchemy import select, func
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
    engine = create_database_engine("sqlite:///:memory:")
    init_schema(engine)
    yield engine
    drop_schema(engine)


def test_schema_constraints_and_foreign_keys(db_engine):
    """Verifies table creation, check constraints, and foreign key cascade behavior."""
    # 1. Test locus creation with valid coordinates
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

    # 2. Test check constraint on negative episode count or invalid RA
    with pytest.raises(Exception):
        with transactional_session(db_engine) as session:
            bad_locus = PersistentSkyLoci(
                sky_locus_id="LOC_BAD",
                ra_deg=400.0,  # Invalid RA
                dec_deg=0.0,
                healpix_nside32=100,
                assigned_base_split="train",
                episode_count=1,
            )
            session.add(bad_locus)

    # 3. Test foreign key cascade delete
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
            evidence_tier="L1_TRACEABLE_SPECTROSCOPIC",
            taxonomy_version="1.0",
        )
        session.add(ep)

    with transactional_session(db_engine) as session:
        ep_count = session.scalar(select(func.count(PhysicalTransientEpisodes.physical_event_id)))
        assert ep_count == 1

    # Delete locus -> cascade should delete episode
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
    Verifies exact row counts against the ingestion manifests.
    """
    importer = CohortBatchImporter(db_engine)
    metrics = importer.import_cohort()

    counts = importer.get_row_counts()
    assert counts["persistent_sky_loci"] == 9928
    assert counts["physical_transient_episodes"] == 9968
    assert counts["catalog_reports"] == 4479
    assert counts["host_counterparts"] == 1061
    assert counts["candidates_10k"] == 10000
    assert counts["raw_snapshots"] == 33
    assert counts["photometry_records"] > 0

    assert metrics["throughput_rows_per_sec"] > 100.0


def test_idempotency_and_duplicate_prevention(db_engine):
    """Confirms duplicate insertions are rejected."""
    importer = CohortBatchImporter(db_engine)
    importer.import_cohort()

    res = importer.test_idempotent_reingestion()
    assert res["status"] == "PASS"
    assert res["duplicate_prevented"] is True


def test_transaction_rollback_on_injected_failure(db_engine):
    """Verifies that an injected integrity failure mid-transaction cleanly rolls back."""
    importer = CohortBatchImporter(db_engine)
    importer.import_cohort()

    res = importer.test_rollback_on_injected_failure()
    assert res["status"] == "PASS"
    assert res["rollback_verified"] is True
    assert res["leaked_rows"] == 0
