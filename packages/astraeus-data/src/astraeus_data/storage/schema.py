"""
ASTRAEUS Relational Database Schema
Implements the 4-Entity Astronomical Identity Model and Observational Persistence
using SQLAlchemy 2.0 with PostgreSQL and SQLite compatibility.
"""

from __future__ import annotations

import datetime
from typing import Optional, List
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    BigInteger,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
    Index,
    func,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PersistentSkyLoci(Base):
    """
    Entity 1: Persistent Sky Locus.
    Represents an invariant astronomical coordinate on the celestial sphere,
    persisting across transient outbursts, flares, and recurring episodes.
    """
    __tablename__ = "persistent_sky_loci"

    sky_locus_id = Column(String(64), primary_key=True, index=True)
    ra_deg = Column(Float, nullable=False)
    dec_deg = Column(Float, nullable=False)
    healpix_nside32 = Column(Integer, nullable=False, index=True)
    assigned_base_split = Column(String(32), nullable=False)
    member_oids = Column(Text, nullable=True)  # JSON-encoded list of survey object IDs
    linked_tns_names = Column(Text, nullable=True)  # JSON-encoded list of TNS names
    linked_external_ids = Column(Text, nullable=True)  # JSON-encoded list of external IDs
    episode_count = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    episodes = relationship("PhysicalTransientEpisodes", back_populates="locus", cascade="all, delete-orphan")
    counterparts = relationship("HostCounterparts", back_populates="locus", cascade="all, delete-orphan")
    reports = relationship("CatalogReports", back_populates="locus", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("ra_deg >= 0.0 AND ra_deg < 360.0", name="chk_locus_ra_bounds"),
        CheckConstraint("dec_deg >= -90.0 AND dec_deg <= 90.0", name="chk_locus_dec_bounds"),
        CheckConstraint("episode_count >= 1", name="chk_locus_episode_count_positive"),
        Index("ix_locus_coords", "ra_deg", "dec_deg"),
    )


class PhysicalTransientEpisodes(Base):
    """
    Entity 2: Physical Transient Episode.
    Represents a time-bounded physical astronomical event (e.g. SN, nova outburst, TDE flare).
    Multiple episodes may share a single Persistent Sky Locus (e.g., nuclear flares, recurrent novae).
    """
    __tablename__ = "physical_transient_episodes"

    physical_event_id = Column(String(64), primary_key=True, index=True)
    sky_locus_id = Column(String(64), ForeignKey("persistent_sky_loci.sky_locus_id", ondelete="CASCADE"), nullable=False, index=True)
    ra_deg = Column(Float, nullable=False)
    dec_deg = Column(Float, nullable=False)
    healpix_nside32 = Column(Integer, nullable=False, index=True)
    t_start_mjd = Column(Float, nullable=False)
    t_end_mjd = Column(Float, nullable=False)
    assigned_split = Column(String(32), nullable=False, index=True)
    temporal_split = Column(String(64), nullable=False)
    member_oids = Column(Text, nullable=True)
    matched_tns_name = Column(String(64), nullable=True)
    matched_external_id = Column(String(64), nullable=True)
    coarse_class = Column(String(64), nullable=False, index=True)
    fine_class = Column(String(64), nullable=False)
    evidence_tier = Column(String(64), nullable=False, index=True)
    taxonomy_version = Column(String(32), nullable=False)
    is_supernova = Column(String(16), nullable=True)
    is_quarantined = Column(Boolean, nullable=False, default=False)
    quarantine_reasons = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    locus = relationship("PersistentSkyLoci", back_populates="episodes")
    reports = relationship("CatalogReports", back_populates="episode", cascade="all, delete-orphan")
    photometry = relationship("PhotometryRecords", back_populates="episode", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("t_start_mjd <= t_end_mjd", name="chk_episode_time_order"),
        CheckConstraint(
            "evidence_tier IN ('L1', 'L2', 'L3', 'L4', 'L5', "
            "'L1_TRACEABLE_SPECTROSCOPIC', 'L2_PUBLIC_SURVEY_SPECTROSCOPIC', "
            "'L3_HOST_MATCHED_EXTRAGALACTIC', 'L4_HIGH_CONFIDENCE_MACHINE_LABEL', 'L5_ARCHIVAL_UNLABELED')",
            name="chk_episode_evidence_tier"
        ),
        Index("ix_episode_locus_temporal", "sky_locus_id", "t_start_mjd"),
    )


class CatalogReports(Base):
    """
    Entity 3: Catalog Report Record.
    Captures individual transient announcements, broker classifications, and reporting alerts
    (e.g., TNS discovery report, ALeRCE classifier alert) supporting a physical episode.
    """
    __tablename__ = "catalog_reports"

    report_id = Column(String(64), primary_key=True, index=True)
    source_system = Column(String(64), nullable=False, index=True)
    source_record_id = Column(String(128), nullable=False)
    sky_locus_id = Column(String(64), ForeignKey("persistent_sky_loci.sky_locus_id", ondelete="CASCADE"), nullable=False, index=True)
    linked_episode_id = Column(String(64), ForeignKey("physical_transient_episodes.physical_event_id", ondelete="CASCADE"), nullable=False, index=True)
    ra_deg = Column(Float, nullable=False)
    dec_deg = Column(Float, nullable=False)
    evidence_tier = Column(String(64), nullable=False)
    raw_class = Column(String(64), nullable=True)
    discovery_mjd_utc = Column(Float, nullable=True)
    class_bibcodes = Column(Text, nullable=True)
    has_host_redshift_only = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    locus = relationship("PersistentSkyLoci", back_populates="reports")
    episode = relationship("PhysicalTransientEpisodes", back_populates="reports")

    __table_args__ = (
        UniqueConstraint("source_system", "source_record_id", name="uq_report_source_record"),
    )


class HostCounterparts(Base):
    """
    Entity 4: Catalog Counterpart Record (formerly HostCounterpartRecord).
    Represents astrometrically associated cross-match counterparts:
    host galaxies, Gaia DR3 quasars (QSO), or Gaia DR3 variable stars (Varstar).
    """
    __tablename__ = "host_counterparts"

    counterpart_id = Column(String(64), primary_key=True, index=True)
    sky_locus_id = Column(String(64), ForeignKey("persistent_sky_loci.sky_locus_id", ondelete="CASCADE"), nullable=False, index=True)
    source_catalog = Column(String(64), nullable=False)
    ra_deg = Column(Float, nullable=False)
    dec_deg = Column(Float, nullable=False)
    angular_offset_arcsec = Column(Float, nullable=False)
    host_redshift = Column(Float, nullable=True)
    gaia_source_id = Column(String(64), nullable=True, index=True)
    is_nuclear_locus = Column(Boolean, nullable=False, default=False)
    counterpart_type = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    locus = relationship("PersistentSkyLoci", back_populates="counterparts")

    __table_args__ = (
        CheckConstraint(
            "counterpart_type IN ('CATALOG_CROSSMATCH', 'GAIA_DR3_QSO', 'GAIA_DR3_VARSTAR', 'HOST_GALAXY')",
            name="chk_counterpart_type"
        ),
    )


class Candidates10k(Base):
    """
    Unified Candidate View / Materialized Table.
    Represents the operational 10,000-candidate cohort tracking multi-survey availability,
    reconciliation status, assigned splits, and discovery metrics.
    """
    __tablename__ = "candidates_10k"

    candidate_id = Column(String(64), primary_key=True, index=True)
    physical_event_id = Column(String(64), ForeignKey("physical_transient_episodes.physical_event_id", ondelete="CASCADE"), nullable=False, index=True)
    sky_locus_id = Column(String(64), ForeignKey("persistent_sky_loci.sky_locus_id", ondelete="CASCADE"), nullable=False, index=True)
    host_counterpart_id = Column(String(64), ForeignKey("host_counterparts.counterpart_id", ondelete="SET NULL"), nullable=True)
    cohort_stratum = Column(String(64), nullable=False, index=True)
    ra_deg = Column(Float, nullable=False)
    dec_deg = Column(Float, nullable=False)
    healpix_nside32 = Column(Integer, nullable=False, index=True)
    assigned_split = Column(String(32), nullable=False, index=True)
    temporal_split = Column(String(64), nullable=False)
    reconciliation_status = Column(String(64), nullable=False)
    matched_tns_name = Column(String(64), nullable=True)
    matched_external_id = Column(String(64), nullable=True)
    coarse_class = Column(String(64), nullable=False, index=True)
    fine_class = Column(String(64), nullable=False)
    evidence_tier = Column(String(64), nullable=False, index=True)
    taxonomy_version = Column(String(32), nullable=False)
    is_supernova = Column(String(16), nullable=True)
    availability_status = Column(String(64), nullable=False)
    has_operational_pub_time = Column(Boolean, nullable=False, default=False)
    has_msip_alert = Column(Boolean, nullable=False, default=False)
    has_dr24_lc = Column(Boolean, nullable=False, default=False)
    has_forced_photometry = Column(Boolean, nullable=False, default=False)
    has_alerce_access = Column(Boolean, nullable=False, default=False)
    has_independent_tns = Column(Boolean, nullable=False, default=False)
    alerce_checked = Column(Boolean, nullable=False, default=False)
    alerce_http_status = Column(String(32), nullable=True)
    dr24_checked = Column(Boolean, nullable=False, default=False)
    n_alerce_det = Column(Integer, nullable=False, default=0)
    n_alerce_ul = Column(Integer, nullable=False, default=0)
    n_dr24_clean = Column(Integer, nullable=False, default=0)
    first_mjd = Column(Float, nullable=True)
    last_mjd = Column(Float, nullable=True)
    is_quarantined = Column(Boolean, nullable=False, default=False)
    quarantine_reasons = Column(Text, nullable=True)
    reference_corpus_overlap_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())


class PhotometryRecords(Base):
    """
    Photometry Records Table.
    Stores multi-epoch observations preserving negative flux values, upper limits,
    flux errors, and photometric processing quality flags.
    """
    __tablename__ = "photometry_records"

    observation_id = Column(String(64), primary_key=True, index=True)
    physical_event_id = Column(String(64), ForeignKey("physical_transient_episodes.physical_event_id", ondelete="CASCADE"), nullable=False, index=True)
    sky_locus_id = Column(String(64), ForeignKey("persistent_sky_loci.sky_locus_id", ondelete="CASCADE"), nullable=False, index=True)
    survey = Column(String(32), nullable=False, index=True)
    mjd_tai = Column(Float, nullable=False, index=True)
    filter_band = Column(String(16), nullable=False)
    flux_mjy = Column(Float, nullable=True)
    flux_error_mjy = Column(Float, nullable=True)
    mag = Column(Float, nullable=True)
    mag_err = Column(Float, nullable=True)
    is_upper_limit = Column(Boolean, nullable=False, default=False, index=True)
    limiting_mag = Column(Float, nullable=True)
    catflags = Column(Integer, nullable=False, default=0)
    is_negative_flux = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    episode = relationship("PhysicalTransientEpisodes", back_populates="photometry")

    __table_args__ = (
        Index("ix_photo_episode_mjd", "physical_event_id", "mjd_tai"),
    )


class RawSnapshots(Base):
    """
    Raw Survey Snapshots Metadata Table.
    Tracks cryptographically hashed raw alert and catalog files stored in MinIO/Object storage.
    """
    __tablename__ = "raw_snapshots"

    snapshot_id = Column(String(64), primary_key=True, index=True)
    artifact_id = Column(String(128), nullable=False, index=True)
    source_system = Column(String(64), nullable=False)
    source_uri = Column(Text, nullable=True)
    local_relative_path = Column(String(255), nullable=False)
    retrieved_at_utc = Column(String(64), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    size_bytes = Column(BigInteger, nullable=False)
    row_count = Column(Integer, nullable=False)
    schema_columns = Column(Text, nullable=True)
    http_status = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
