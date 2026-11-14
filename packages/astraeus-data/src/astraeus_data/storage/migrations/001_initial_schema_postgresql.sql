-- ASTRAEUS Production PostgreSQL Schema Migration (001_initial_schema)
-- Dialect: PostgreSQL 14+ / 15+ / 16+

CREATE TABLE persistent_sky_loci (
	sky_locus_id VARCHAR(64) NOT NULL, 
	ra_deg FLOAT NOT NULL, 
	dec_deg FLOAT NOT NULL, 
	healpix_nside32 INTEGER NOT NULL, 
	assigned_base_split VARCHAR(32) NOT NULL, 
	member_oids TEXT, 
	linked_tns_names TEXT, 
	linked_external_ids TEXT, 
	episode_count INTEGER NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE, 
	updated_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (sky_locus_id), 
	CONSTRAINT chk_locus_ra_bounds CHECK (ra_deg >= 0.0 AND ra_deg < 360.0), 
	CONSTRAINT chk_locus_dec_bounds CHECK (dec_deg >= -90.0 AND dec_deg <= 90.0), 
	CONSTRAINT chk_locus_episode_count_positive CHECK (episode_count >= 1)
);

CREATE INDEX ix_locus_coords ON persistent_sky_loci (ra_deg, dec_deg);

CREATE INDEX ix_persistent_sky_loci_sky_locus_id ON persistent_sky_loci (sky_locus_id);

CREATE INDEX ix_persistent_sky_loci_healpix_nside32 ON persistent_sky_loci (healpix_nside32);

CREATE TABLE physical_transient_episodes (
	physical_event_id VARCHAR(64) NOT NULL, 
	sky_locus_id VARCHAR(64) NOT NULL, 
	ra_deg FLOAT NOT NULL, 
	dec_deg FLOAT NOT NULL, 
	healpix_nside32 INTEGER NOT NULL, 
	t_start_mjd FLOAT NOT NULL, 
	t_end_mjd FLOAT NOT NULL, 
	assigned_split VARCHAR(32) NOT NULL, 
	temporal_split VARCHAR(64) NOT NULL, 
	member_oids TEXT, 
	matched_tns_name VARCHAR(64), 
	matched_external_id VARCHAR(64), 
	coarse_class VARCHAR(64) NOT NULL, 
	fine_class VARCHAR(64) NOT NULL, 
	evidence_tier VARCHAR(64) NOT NULL, 
	taxonomy_version VARCHAR(32) NOT NULL, 
	is_supernova VARCHAR(16), 
	is_quarantined BOOLEAN NOT NULL, 
	quarantine_reasons TEXT, 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (physical_event_id), 
	CONSTRAINT chk_episode_time_order CHECK (t_start_mjd <= t_end_mjd), 
	CONSTRAINT chk_episode_evidence_tier CHECK (evidence_tier IN ('L1', 'L2', 'L3', 'L4', 'L5', 'L1_TRACEABLE_SPECTROSCOPIC', 'L2_PUBLIC_SURVEY_SPECTROSCOPIC', 'L3_HOST_MATCHED_EXTRAGALACTIC', 'L4_HIGH_CONFIDENCE_MACHINE_LABEL', 'L5_ARCHIVAL_UNLABELED')), 
	FOREIGN KEY(sky_locus_id) REFERENCES persistent_sky_loci (sky_locus_id) ON DELETE CASCADE
);

CREATE INDEX ix_physical_transient_episodes_evidence_tier ON physical_transient_episodes (evidence_tier);

CREATE INDEX ix_physical_transient_episodes_healpix_nside32 ON physical_transient_episodes (healpix_nside32);

CREATE INDEX ix_physical_transient_episodes_physical_event_id ON physical_transient_episodes (physical_event_id);

CREATE INDEX ix_physical_transient_episodes_coarse_class ON physical_transient_episodes (coarse_class);

CREATE INDEX ix_physical_transient_episodes_sky_locus_id ON physical_transient_episodes (sky_locus_id);

CREATE INDEX ix_episode_locus_temporal ON physical_transient_episodes (sky_locus_id, t_start_mjd);

CREATE INDEX ix_physical_transient_episodes_assigned_split ON physical_transient_episodes (assigned_split);

CREATE TABLE catalog_reports (
	report_id VARCHAR(64) NOT NULL, 
	source_system VARCHAR(64) NOT NULL, 
	source_record_id VARCHAR(128) NOT NULL, 
	sky_locus_id VARCHAR(64) NOT NULL, 
	linked_episode_id VARCHAR(64) NOT NULL, 
	ra_deg FLOAT NOT NULL, 
	dec_deg FLOAT NOT NULL, 
	evidence_tier VARCHAR(64) NOT NULL, 
	raw_class VARCHAR(64), 
	discovery_mjd_utc FLOAT, 
	class_bibcodes TEXT, 
	has_host_redshift_only BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (report_id), 
	CONSTRAINT uq_report_source_record UNIQUE (source_system, source_record_id), 
	FOREIGN KEY(sky_locus_id) REFERENCES persistent_sky_loci (sky_locus_id) ON DELETE CASCADE, 
	FOREIGN KEY(linked_episode_id) REFERENCES physical_transient_episodes (physical_event_id) ON DELETE CASCADE
);

CREATE INDEX ix_catalog_reports_report_id ON catalog_reports (report_id);

CREATE INDEX ix_catalog_reports_sky_locus_id ON catalog_reports (sky_locus_id);

CREATE INDEX ix_catalog_reports_linked_episode_id ON catalog_reports (linked_episode_id);

CREATE INDEX ix_catalog_reports_source_system ON catalog_reports (source_system);

CREATE TABLE host_counterparts (
	counterpart_id VARCHAR(64) NOT NULL, 
	sky_locus_id VARCHAR(64) NOT NULL, 
	source_catalog VARCHAR(64) NOT NULL, 
	ra_deg FLOAT NOT NULL, 
	dec_deg FLOAT NOT NULL, 
	angular_offset_arcsec FLOAT NOT NULL, 
	host_redshift FLOAT, 
	gaia_source_id VARCHAR(64), 
	is_nuclear_locus BOOLEAN NOT NULL, 
	counterpart_type VARCHAR(64) NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (counterpart_id), 
	CONSTRAINT chk_counterpart_type CHECK (counterpart_type IN ('CATALOG_CROSSMATCH', 'GAIA_DR3_QSO', 'GAIA_DR3_VARSTAR', 'HOST_GALAXY')), 
	FOREIGN KEY(sky_locus_id) REFERENCES persistent_sky_loci (sky_locus_id) ON DELETE CASCADE
);

CREATE INDEX ix_host_counterparts_gaia_source_id ON host_counterparts (gaia_source_id);

CREATE INDEX ix_host_counterparts_counterpart_id ON host_counterparts (counterpart_id);

CREATE INDEX ix_host_counterparts_sky_locus_id ON host_counterparts (sky_locus_id);

CREATE INDEX ix_host_counterparts_counterpart_type ON host_counterparts (counterpart_type);

CREATE TABLE candidates_10k (
	candidate_id VARCHAR(64) NOT NULL, 
	physical_event_id VARCHAR(64) NOT NULL, 
	sky_locus_id VARCHAR(64) NOT NULL, 
	host_counterpart_id VARCHAR(64), 
	cohort_stratum VARCHAR(64) NOT NULL, 
	ra_deg FLOAT NOT NULL, 
	dec_deg FLOAT NOT NULL, 
	healpix_nside32 INTEGER NOT NULL, 
	assigned_split VARCHAR(32) NOT NULL, 
	temporal_split VARCHAR(64) NOT NULL, 
	reconciliation_status VARCHAR(64) NOT NULL, 
	matched_tns_name VARCHAR(64), 
	matched_external_id VARCHAR(64), 
	coarse_class VARCHAR(64) NOT NULL, 
	fine_class VARCHAR(64) NOT NULL, 
	evidence_tier VARCHAR(64) NOT NULL, 
	taxonomy_version VARCHAR(32) NOT NULL, 
	is_supernova VARCHAR(16), 
	availability_status VARCHAR(64) NOT NULL, 
	has_operational_pub_time BOOLEAN NOT NULL, 
	has_msip_alert BOOLEAN NOT NULL, 
	has_dr24_lc BOOLEAN NOT NULL, 
	has_forced_photometry BOOLEAN NOT NULL, 
	has_alerce_access BOOLEAN NOT NULL, 
	has_independent_tns BOOLEAN NOT NULL, 
	alerce_checked BOOLEAN NOT NULL, 
	alerce_http_status VARCHAR(32), 
	dr24_checked BOOLEAN NOT NULL, 
	n_alerce_det INTEGER NOT NULL, 
	n_alerce_ul INTEGER NOT NULL, 
	n_dr24_clean INTEGER NOT NULL, 
	first_mjd FLOAT, 
	last_mjd FLOAT, 
	is_quarantined BOOLEAN NOT NULL, 
	quarantine_reasons TEXT, 
	reference_corpus_overlap_id VARCHAR(64), 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (candidate_id), 
	FOREIGN KEY(physical_event_id) REFERENCES physical_transient_episodes (physical_event_id) ON DELETE CASCADE, 
	FOREIGN KEY(sky_locus_id) REFERENCES persistent_sky_loci (sky_locus_id) ON DELETE CASCADE, 
	FOREIGN KEY(host_counterpart_id) REFERENCES host_counterparts (counterpart_id) ON DELETE SET NULL
);

CREATE INDEX ix_candidates_10k_cohort_stratum ON candidates_10k (cohort_stratum);

CREATE INDEX ix_candidates_10k_sky_locus_id ON candidates_10k (sky_locus_id);

CREATE INDEX ix_candidates_10k_evidence_tier ON candidates_10k (evidence_tier);

CREATE INDEX ix_candidates_10k_physical_event_id ON candidates_10k (physical_event_id);

CREATE INDEX ix_candidates_10k_assigned_split ON candidates_10k (assigned_split);

CREATE INDEX ix_candidates_10k_candidate_id ON candidates_10k (candidate_id);

CREATE INDEX ix_candidates_10k_healpix_nside32 ON candidates_10k (healpix_nside32);

CREATE INDEX ix_candidates_10k_coarse_class ON candidates_10k (coarse_class);

CREATE TABLE photometry_records (
	observation_id VARCHAR(64) NOT NULL, 
	physical_event_id VARCHAR(64) NOT NULL, 
	sky_locus_id VARCHAR(64) NOT NULL, 
	survey VARCHAR(32) NOT NULL, 
	mjd_tai FLOAT NOT NULL, 
	filter_band VARCHAR(16) NOT NULL, 
	flux_mjy FLOAT, 
	flux_error_mjy FLOAT, 
	mag FLOAT, 
	mag_err FLOAT, 
	is_upper_limit BOOLEAN NOT NULL, 
	limiting_mag FLOAT, 
	catflags INTEGER NOT NULL, 
	is_negative_flux BOOLEAN NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (observation_id), 
	FOREIGN KEY(physical_event_id) REFERENCES physical_transient_episodes (physical_event_id) ON DELETE CASCADE, 
	FOREIGN KEY(sky_locus_id) REFERENCES persistent_sky_loci (sky_locus_id) ON DELETE CASCADE
);

CREATE INDEX ix_photo_episode_mjd ON photometry_records (physical_event_id, mjd_tai);

CREATE INDEX ix_photometry_records_observation_id ON photometry_records (observation_id);

CREATE INDEX ix_photometry_records_physical_event_id ON photometry_records (physical_event_id);

CREATE INDEX ix_photometry_records_is_upper_limit ON photometry_records (is_upper_limit);

CREATE INDEX ix_photometry_records_survey ON photometry_records (survey);

CREATE INDEX ix_photometry_records_is_negative_flux ON photometry_records (is_negative_flux);

CREATE INDEX ix_photometry_records_mjd_tai ON photometry_records (mjd_tai);

CREATE INDEX ix_photometry_records_sky_locus_id ON photometry_records (sky_locus_id);

CREATE TABLE raw_snapshots (
	snapshot_id VARCHAR(64) NOT NULL, 
	artifact_id VARCHAR(128) NOT NULL, 
	source_system VARCHAR(64) NOT NULL, 
	source_uri TEXT, 
	local_relative_path VARCHAR(255) NOT NULL, 
	retrieved_at_utc VARCHAR(64) NOT NULL, 
	sha256 VARCHAR(64) NOT NULL, 
	size_bytes BIGINT NOT NULL, 
	row_count INTEGER NOT NULL, 
	schema_columns TEXT, 
	http_status INTEGER NOT NULL, 
	created_at TIMESTAMP WITH TIME ZONE, 
	PRIMARY KEY (snapshot_id)
);

CREATE UNIQUE INDEX ix_raw_snapshots_sha256 ON raw_snapshots (sha256);

CREATE INDEX ix_raw_snapshots_snapshot_id ON raw_snapshots (snapshot_id);

CREATE INDEX ix_raw_snapshots_artifact_id ON raw_snapshots (artifact_id);
