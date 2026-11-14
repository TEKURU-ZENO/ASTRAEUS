"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-11-13 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. persistent_sky_loci
    op.create_table(
        'persistent_sky_loci',
        sa.Column('sky_locus_id', sa.String(64), primary_key=True),
        sa.Column('ra_deg', sa.Float(), nullable=False),
        sa.Column('dec_deg', sa.Float(), nullable=False),
        sa.Column('healpix_nside32', sa.Integer(), nullable=False),
        sa.Column('assigned_base_split', sa.String(32), nullable=False),
        sa.Column('member_oids', sa.Text(), nullable=True),
        sa.Column('linked_tns_names', sa.Text(), nullable=True),
        sa.Column('linked_external_ids', sa.Text(), nullable=True),
        sa.Column('episode_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.CheckConstraint('ra_deg >= 0.0 AND ra_deg < 360.0', name='chk_locus_ra_bounds'),
        sa.CheckConstraint('dec_deg >= -90.0 AND dec_deg <= 90.0', name='chk_locus_dec_bounds'),
        sa.CheckConstraint('episode_count >= 1', name='chk_locus_episode_count_positive'),
    )
    op.create_index('ix_persistent_sky_loci_sky_locus_id', 'persistent_sky_loci', ['sky_locus_id'])
    op.create_index('ix_persistent_sky_loci_healpix_nside32', 'persistent_sky_loci', ['healpix_nside32'])
    op.create_index('ix_locus_coords', 'persistent_sky_loci', ['ra_deg', 'dec_deg'])

    # 2. physical_transient_episodes
    op.create_table(
        'physical_transient_episodes',
        sa.Column('physical_event_id', sa.String(64), primary_key=True),
        sa.Column('sky_locus_id', sa.String(64), sa.ForeignKey('persistent_sky_loci.sky_locus_id', ondelete='CASCADE'), nullable=False),
        sa.Column('ra_deg', sa.Float(), nullable=False),
        sa.Column('dec_deg', sa.Float(), nullable=False),
        sa.Column('healpix_nside32', sa.Integer(), nullable=False),
        sa.Column('t_start_mjd', sa.Float(), nullable=False),
        sa.Column('t_end_mjd', sa.Float(), nullable=False),
        sa.Column('assigned_split', sa.String(32), nullable=False),
        sa.Column('temporal_split', sa.String(64), nullable=False),
        sa.Column('member_oids', sa.Text(), nullable=True),
        sa.Column('matched_tns_name', sa.String(64), nullable=True),
        sa.Column('matched_external_id', sa.String(64), nullable=True),
        sa.Column('coarse_class', sa.String(64), nullable=False),
        sa.Column('fine_class', sa.String(64), nullable=False),
        sa.Column('evidence_tier', sa.String(64), nullable=False),
        sa.Column('taxonomy_version', sa.String(32), nullable=False),
        sa.Column('is_supernova', sa.String(16), nullable=True),
        sa.Column('is_quarantined', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('quarantine_reasons', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.CheckConstraint('t_start_mjd <= t_end_mjd', name='chk_episode_time_order'),
        sa.CheckConstraint(
            "evidence_tier IN ('L1', 'L2', 'L3', 'L4', 'L5', "
            "'L1_TRACEABLE_SPECTROSCOPIC', 'L2_PUBLIC_SURVEY_SPECTROSCOPIC', "
            "'L3_HOST_MATCHED_EXTRAGALACTIC', 'L4_HIGH_CONFIDENCE_MACHINE_LABEL', 'L5_ARCHIVAL_UNLABELED')",
            name='chk_episode_evidence_tier'
        ),
    )
    op.create_index('ix_physical_transient_episodes_physical_event_id', 'physical_transient_episodes', ['physical_event_id'])
    op.create_index('ix_physical_transient_episodes_sky_locus_id', 'physical_transient_episodes', ['sky_locus_id'])
    op.create_index('ix_physical_transient_episodes_assigned_split', 'physical_transient_episodes', ['assigned_split'])
    op.create_index('ix_physical_transient_episodes_coarse_class', 'physical_transient_episodes', ['coarse_class'])
    op.create_index('ix_physical_transient_episodes_evidence_tier', 'physical_transient_episodes', ['evidence_tier'])
    op.create_index('ix_episode_locus_temporal', 'physical_transient_episodes', ['sky_locus_id', 't_start_mjd'])

    # 3. catalog_reports
    op.create_table(
        'catalog_reports',
        sa.Column('report_id', sa.String(64), primary_key=True),
        sa.Column('source_system', sa.String(64), nullable=False),
        sa.Column('source_record_id', sa.String(128), nullable=False),
        sa.Column('sky_locus_id', sa.String(64), sa.ForeignKey('persistent_sky_loci.sky_locus_id', ondelete='CASCADE'), nullable=False),
        sa.Column('linked_episode_id', sa.String(64), sa.ForeignKey('physical_transient_episodes.physical_event_id', ondelete='CASCADE'), nullable=False),
        sa.Column('ra_deg', sa.Float(), nullable=False),
        sa.Column('dec_deg', sa.Float(), nullable=False),
        sa.Column('evidence_tier', sa.String(64), nullable=False),
        sa.Column('raw_class', sa.String(64), nullable=True),
        sa.Column('discovery_mjd_utc', sa.Float(), nullable=True),
        sa.Column('class_bibcodes', sa.Text(), nullable=True),
        sa.Column('has_host_redshift_only', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.UniqueConstraint('source_system', 'source_record_id', name='uq_report_source_record'),
    )
    op.create_index('ix_catalog_reports_report_id', 'catalog_reports', ['report_id'])
    op.create_index('ix_catalog_reports_source_system', 'catalog_reports', ['source_system'])
    op.create_index('ix_catalog_reports_sky_locus_id', 'catalog_reports', ['sky_locus_id'])
    op.create_index('ix_catalog_reports_linked_episode_id', 'catalog_reports', ['linked_episode_id'])

    # 4. host_counterparts
    op.create_table(
        'host_counterparts',
        sa.Column('counterpart_id', sa.String(64), primary_key=True),
        sa.Column('sky_locus_id', sa.String(64), sa.ForeignKey('persistent_sky_loci.sky_locus_id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_catalog', sa.String(64), nullable=False),
        sa.Column('ra_deg', sa.Float(), nullable=False),
        sa.Column('dec_deg', sa.Float(), nullable=False),
        sa.Column('angular_offset_arcsec', sa.Float(), nullable=False),
        sa.Column('host_redshift', sa.Float(), nullable=True),
        sa.Column('gaia_source_id', sa.String(64), nullable=True),
        sa.Column('is_nuclear_locus', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('counterpart_type', sa.String(64), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.CheckConstraint(
            "counterpart_type IN ('CATALOG_CROSSMATCH', 'GAIA_DR3_QSO', 'GAIA_DR3_VARSTAR', 'HOST_GALAXY')",
            name='chk_counterpart_type'
        ),
    )
    op.create_index('ix_host_counterparts_counterpart_id', 'host_counterparts', ['counterpart_id'])
    op.create_index('ix_host_counterparts_sky_locus_id', 'host_counterparts', ['sky_locus_id'])
    op.create_index('ix_host_counterparts_gaia_source_id', 'host_counterparts', ['gaia_source_id'])
    op.create_index('ix_host_counterparts_counterpart_type', 'host_counterparts', ['counterpart_type'])

    # 5. candidates_10k
    op.create_table(
        'candidates_10k',
        sa.Column('candidate_id', sa.String(64), primary_key=True),
        sa.Column('physical_event_id', sa.String(64), sa.ForeignKey('physical_transient_episodes.physical_event_id', ondelete='CASCADE'), nullable=False),
        sa.Column('sky_locus_id', sa.String(64), sa.ForeignKey('persistent_sky_loci.sky_locus_id', ondelete='CASCADE'), nullable=False),
        sa.Column('host_counterpart_id', sa.String(64), sa.ForeignKey('host_counterparts.counterpart_id', ondelete='SET NULL'), nullable=True),
        sa.Column('cohort_stratum', sa.String(64), nullable=False),
        sa.Column('ra_deg', sa.Float(), nullable=False),
        sa.Column('dec_deg', sa.Float(), nullable=False),
        sa.Column('healpix_nside32', sa.Integer(), nullable=False),
        sa.Column('assigned_split', sa.String(32), nullable=False),
        sa.Column('temporal_split', sa.String(64), nullable=False),
        sa.Column('reconciliation_status', sa.String(64), nullable=False),
        sa.Column('matched_tns_name', sa.String(64), nullable=True),
        sa.Column('matched_external_id', sa.String(64), nullable=True),
        sa.Column('coarse_class', sa.String(64), nullable=False),
        sa.Column('fine_class', sa.String(64), nullable=False),
        sa.Column('evidence_tier', sa.String(64), nullable=False),
        sa.Column('taxonomy_version', sa.String(32), nullable=False),
        sa.Column('is_supernova', sa.String(16), nullable=True),
        sa.Column('availability_status', sa.String(64), nullable=False),
        sa.Column('has_operational_pub_time', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('has_msip_alert', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('has_dr24_lc', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('has_forced_photometry', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('has_alerce_access', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('has_independent_tns', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('alerce_checked', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('alerce_http_status', sa.String(32), nullable=True),
        sa.Column('dr24_checked', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('n_alerce_det', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_alerce_ul', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_dr24_clean', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('first_mjd', sa.Float(), nullable=True),
        sa.Column('last_mjd', sa.Float(), nullable=True),
        sa.Column('is_quarantined', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('quarantine_reasons', sa.Text(), nullable=True),
        sa.Column('reference_corpus_overlap_id', sa.String(64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_candidates_10k_candidate_id', 'candidates_10k', ['candidate_id'])
    op.create_index('ix_candidates_10k_physical_event_id', 'candidates_10k', ['physical_event_id'])
    op.create_index('ix_candidates_10k_sky_locus_id', 'candidates_10k', ['sky_locus_id'])
    op.create_index('ix_candidates_10k_cohort_stratum', 'candidates_10k', ['cohort_stratum'])
    op.create_index('ix_candidates_10k_healpix_nside32', 'candidates_10k', ['healpix_nside32'])
    op.create_index('ix_candidates_10k_assigned_split', 'candidates_10k', ['assigned_split'])
    op.create_index('ix_candidates_10k_coarse_class', 'candidates_10k', ['coarse_class'])
    op.create_index('ix_candidates_10k_evidence_tier', 'candidates_10k', ['evidence_tier'])

    # 6. photometry_records
    op.create_table(
        'photometry_records',
        sa.Column('observation_id', sa.String(64), primary_key=True),
        sa.Column('physical_event_id', sa.String(64), sa.ForeignKey('physical_transient_episodes.physical_event_id', ondelete='CASCADE'), nullable=False),
        sa.Column('sky_locus_id', sa.String(64), sa.ForeignKey('persistent_sky_loci.sky_locus_id', ondelete='CASCADE'), nullable=False),
        sa.Column('survey', sa.String(32), nullable=False),
        sa.Column('mjd_tai', sa.Float(), nullable=False),
        sa.Column('filter_band', sa.String(16), nullable=False),
        sa.Column('flux_mjy', sa.Float(), nullable=True),
        sa.Column('flux_error_mjy', sa.Float(), nullable=True),
        sa.Column('mag', sa.Float(), nullable=True),
        sa.Column('mag_err', sa.Float(), nullable=True),
        sa.Column('is_upper_limit', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('limiting_mag', sa.Float(), nullable=True),
        sa.Column('catflags', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('is_negative_flux', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_photometry_records_observation_id', 'photometry_records', ['observation_id'])
    op.create_index('ix_photometry_records_physical_event_id', 'photometry_records', ['physical_event_id'])
    op.create_index('ix_photometry_records_sky_locus_id', 'photometry_records', ['sky_locus_id'])
    op.create_index('ix_photometry_records_survey', 'photometry_records', ['survey'])
    op.create_index('ix_photometry_records_mjd_tai', 'photometry_records', ['mjd_tai'])
    op.create_index('ix_photometry_records_is_upper_limit', 'photometry_records', ['is_upper_limit'])
    op.create_index('ix_photometry_records_is_negative_flux', 'photometry_records', ['is_negative_flux'])
    op.create_index('ix_photo_episode_mjd', 'photometry_records', ['physical_event_id', 'mjd_tai'])

    # 7. raw_snapshots
    op.create_table(
        'raw_snapshots',
        sa.Column('snapshot_id', sa.String(64), primary_key=True),
        sa.Column('artifact_id', sa.String(128), nullable=False),
        sa.Column('source_system', sa.String(64), nullable=False),
        sa.Column('source_uri', sa.Text(), nullable=True),
        sa.Column('local_relative_path', sa.String(255), nullable=False),
        sa.Column('retrieved_at_utc', sa.String(64), nullable=False),
        sa.Column('sha256', sa.String(64), nullable=False, unique=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=False),
        sa.Column('schema_columns', sa.Text(), nullable=True),
        sa.Column('http_status', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
    )
    op.create_index('ix_raw_snapshots_snapshot_id', 'raw_snapshots', ['snapshot_id'])
    op.create_index('ix_raw_snapshots_artifact_id', 'raw_snapshots', ['artifact_id'])
    op.create_index('ix_raw_snapshots_sha256', 'raw_snapshots', ['sha256'])


def downgrade() -> None:
    op.drop_table('raw_snapshots')
    op.drop_table('photometry_records')
    op.drop_table('candidates_10k')
    op.drop_table('host_counterparts')
    op.drop_table('catalog_reports')
    op.drop_table('physical_transient_episodes')
    op.drop_table('persistent_sky_loci')
