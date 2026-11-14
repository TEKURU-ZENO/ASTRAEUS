"""
Unit tests for ASTRAEUS Real-Survey Sprint 1 & Sprint 2 Data Integrity Infrastructure:
- Snapshot SHA-256 manifest & tamper detection
- TNS public catalog banner handling, label provenance, & AT/non-SN separation
- ALeRCE signed difference flux (`isdiffpos`), upper limits, & causal cutoffs
- ZTF DR24 HATS partition resolution & catflags bit-15 FATAL filtering
- Physical-event deduplication, ambiguity quarantine, & Gate A leakage checks
"""

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from astraeus_data.corpus.sequence import (
    ObservationType,
    QualitySeverity,
    SequenceViewType,
)
from astraeus_data.real_survey.alerce_connector import (
    alerce_lightcurve_to_records,
    alerce_lightcurve_to_sequence,
)
from astraeus_data.real_survey.external_catalog_connector import (
    ExternalCatalogRecord,
)
from astraeus_data.real_survey.manifest import (
    RetrievalManifestEntry,
    SnapshotManifest,
    compute_file_sha256,
)
from astraeus_data.real_survey.reconciliation import (
    ReconciliationStatus,
    ZtfObjectSummary,
    build_identity_graph_and_reconcile,
    reconcile_ztf_and_tns,
)
from astraeus_data.real_survey.timestamps import (
    AvailabilityStatus,
    TimestampBundle,
    summarize_availability,
)
from astraeus_data.real_survey.tns_connector import (
    CoarseTransientClass,
    EvidenceTier,
    LabelProvenanceTier,
    TnsObjectRecord,
    extract_ztf_ids,
    load_tns_dataframe,
    map_tns_type_with_provenance,
    row_to_tns_record,
)
from astraeus_data.real_survey.ztf_dr24_connector import (
    CATFLAG_CLOUD_OR_MOON_BIT,
    classify_ztf_dr24_catflag,
    dr24_row_to_observation_records,
    hats_partition_relative_url,
    resolve_partition_from_healpix9,
    skycoord_to_healpix9,
)


def test_snapshot_manifest_integrity_and_tamper_detection(tmp_path: Path):
    raw_file = tmp_path / "sample.json"
    raw_file.write_text('{"ok": true}', encoding="utf-8")

    entry = RetrievalManifestEntry(
        artifact_id="test_artifact",
        source_system="ALERCE_ZTF_V1",
        source_uri="https://example.org/sample.json",
        local_relative_path="sample.json",
        retrieved_at_utc="2026-11-12T00:00:00+00:00",
        sha256=compute_file_sha256(raw_file),
        size_bytes=raw_file.stat().st_size,
        row_count=1,
    )
    manifest = SnapshotManifest()
    manifest.add_or_update(entry)
    manifest_path = manifest.save(tmp_path / "manifest.json")

    loaded = SnapshotManifest.load(manifest_path)
    ok = loaded.verify_all(tmp_path)
    assert ok["passed"] is True
    assert ok["verified_count"] == 1

    # Tamper same-length byte and verify SHA-256 check rejects it
    raw_file.write_text('{"ok": fals}', encoding="utf-8")
    tampered = loaded.verify_all(tmp_path)
    assert tampered["passed"] is False
    assert "sha256_mismatch" in tampered["failures"][0]["reason"]


def test_tns_taxonomy_and_at_unclassified_never_negative():
    # L1 traceable spectroscopic with Class_ADS_bibcodes
    c, f, p, is_sn = map_tns_type_with_provenance(
        "SN Ia-91T-like", "SN", class_bibcodes="2024TNSCR1000....1A"
    )
    assert c == CoarseTransientClass.SN_IA
    assert f == "SN Ia-91T-like"
    assert p == EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC
    assert is_sn is True

    # L2 classified WITHOUT traceable Class_ADS_bibcodes
    c_l2, f_l2, p_l2, is_sn_l2 = map_tns_type_with_provenance(
        "SN Ia", "SN", class_bibcodes=""
    )
    assert c_l2 == CoarseTransientClass.SN_IA
    assert p_l2 == EvidenceTier.L2_REPORTED_CLASSIFICATION_UNVERIFIED
    assert is_sn_l2 is True

    # Non-SN contaminant CV -> CV_NOVA + L1 when bibcode present
    c_cv, f_cv, p_cv, is_sn_cv = map_tns_type_with_provenance(
        "CV", "AT", class_bibcodes="2024TNSCR2000....1B"
    )
    assert c_cv == CoarseTransientClass.CV_NOVA
    assert p_cv == EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC
    assert is_sn_cv is False

    # Unclassified AT / empty MUST remain L5 and is_supernova=None, even if redshift present
    for blank in (None, "", "AT", "Other"):
        _, _, p_u, is_sn_u = map_tns_type_with_provenance(
            blank, "AT", class_bibcodes="2024TNSCR9999....1Z", has_redshift=True
        )
        assert p_u == EvidenceTier.L5_UNCLASSIFIED
        assert is_sn_u is None

    assert extract_ztf_ids("ZTF21aaabbbb, ATLAS21xyz, ZTF21aaabbbb, ZTF22ccdddeee") == [
        "ZTF21aaabbbb",
        "ZTF22ccdddeee",
    ]


def test_load_tns_zip_with_banner_line(tmp_path: Path):
    csv_text = "\n".join(
        [
            "2026-11-12 00:00:00",
            '"objid","name_prefix","name","ra","declination","redshift","typeid","type","reporting_groupid","reporting_group","source_groupid","source_group","discoverydate","discoverymag","discmagfilter","filter","reporters","time_received","internal_names","Discovery_ADS_bibcode","Class_ADS_bibcodes","creationdate","lastmodified"',
            '"101","SN","2024abc","150.123","-2.456","0.042","3","SN Ia","48","ZTF","48","ZTF","2024-03-01 05:00:00.000","18.5","110","g","Team","2024-03-01","ZTF24aaabbbb","","2024TNSCR...","2024-03-01","2024-03-02"',
            '"102","AT","2024def","210.500","15.250","","","","48","ZTF","48","ZTF","2024-04-01 06:00:00.000","19.8","111","r","Team","2024-04-01","ATLAS24xyz","","","2024-04-01","2024-04-01"',
        ]
    )
    zip_path = tmp_path / "tns.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("tns_public_objects.csv", csv_text)

    df, banner = load_tns_dataframe(zip_path, ztf_only=False)
    assert banner == "2026-11-12 00:00:00"
    assert len(df) == 2

    df_ztf, _ = load_tns_dataframe(zip_path, ztf_only=True)
    assert len(df_ztf) == 1
    rec = row_to_tns_record(df_ztf.iloc[0].to_dict(), tns_snapshot_utc=banner)
    assert rec.tns_name == "SN2024abc"
    assert rec.ztf_ids == ["ZTF24aaabbbb"]
    assert rec.coarse_class == CoarseTransientClass.SN_IA
    assert rec.is_supernova is True


def test_alerce_signed_flux_upper_limits_and_causal_cutoff():
    detections = [
        {
            "candid": "1",
            "mjd": 60310.2,  # 2024-01-01T04:48:00Z
            "fid": 1,
            "magpsf": 17.5,  # F = 100 nmgy
            "sigmapsf": 0.05,
            "isdiffpos": 1,
            "rb": 0.92,
        },
        {
            "candid": "2",
            "mjd": 60315.2,  # 2024-01-06T04:48:00Z
            "fid": 2,
            "magpsf": 20.0,  # |F| = 10 nmgy, negative subtraction
            "sigmapsf": 0.1,
            "isdiffpos": -1,
            "rb": 0.40,
        },
    ]
    non_detections = [{"mjd": 60308.0, "fid": 1, "diffmaglim": 20.5}]

    # Without independently observed broker pub timestamps, must emit AVAILABILITY_TIME_UNKNOWN
    recs_unknown = alerce_lightcurve_to_records(
        oid="ZTF24testobj",
        ra_deg=150.0,
        dec_deg=2.0,
        detections=detections,
        non_detections=non_detections,
        min_rb_score=0.5,
    )
    assert len(recs_unknown) == 3
    assert all(not r.has_operational_pub_time and r.t_pub_utc is None for r in recs_unknown)
    avail_summary = summarize_availability(recs_unknown)
    assert avail_summary["availability_time_unknown_count"] == 3
    assert avail_summary["known_operational_pub_time_count"] == 0
    assert recs_unknown[0].obs_type == ObservationType.CATALOG_UPPER_LIMIT
    assert recs_unknown[0].upper_limit_sigma_threshold == 5.0
    assert recs_unknown[0].effective_measurement_sigma_nmgy == pytest.approx(
        recs_unknown[0].upper_limit_flux_nmgy / 5.0
    )
    assert recs_unknown[1].measured_flux_nmgy == pytest.approx(100.0, rel=1e-4)
    assert recs_unknown[2].measured_flux_nmgy < 0.0  # Preserves isdiffpos=-1 sign!
    assert recs_unknown[2].measured_flux_nmgy == pytest.approx(-10.0, rel=1e-4)
    assert recs_unknown[2].quality_severity == QualitySeverity.DEGRADED

    # Operational causal early-classification cutoff MUST exclude unknown-pub-time alerts
    seq_unknown_early = alerce_lightcurve_to_sequence(
        oid="ZTF24testobj",
        ra_deg=150.0,
        dec_deg=2.0,
        payload={"detections": detections, "non_detections": non_detections},
        view_type=SequenceViewType.EARLY_CLASSIFICATION,
        t_cutoff_utc="2024-01-03T00:00:00+00:00",
    )
    assert len(seq_unknown_early.observations) == 0

    # With explicitly known broker publication timestamps, causal cutoff enforces t_pub <= cutoff
    known_pubs = {
        "1": "2024-01-01T05:00:00+00:00",
        "2": "2024-01-06T05:00:00+00:00",
        "ul_1_60308.000000": "2023-12-30T05:00:00+00:00",
    }
    seq_known_early = alerce_lightcurve_to_sequence(
        oid="ZTF24testobj",
        ra_deg=150.0,
        dec_deg=2.0,
        payload={"detections": detections, "non_detections": non_detections},
        view_type=SequenceViewType.EARLY_CLASSIFICATION,
        t_cutoff_utc="2024-01-03T00:00:00+00:00",
        known_pub_times_by_candid=known_pubs,
    )
    assert len(seq_known_early.observations) == 2
    assert all(r.measured_flux_nmgy is None or r.measured_flux_nmgy > 0 for r in seq_known_early.observations)

    # TimestampBundle invariant check
    with pytest.raises(ValueError, match="KNOWN_OPERATIONAL"):
        TimestampBundle(
            t_obs_mjd_utc=60310.0,
            t_pub_utc=None,
            availability_status=AvailabilityStatus.KNOWN_OPERATIONAL,
        )


def test_ztf_dr24_partition_resolution_and_catflags():
    hp9 = skycoord_to_healpix9(180.0, 30.0)
    norder3_pix = hp9 >> (2 * (9 - 3))
    part_df = pd.DataFrame(
        [{"Norder": 3, "Dir": 0, "Npix": norder3_pix, "num_rows": 1000}]
    )
    assert resolve_partition_from_healpix9(hp9, part_df) == (3, 0, norder3_pix)
    assert (
        hats_partition_relative_url(3, 0, norder3_pix)
        == f"dataset/Norder=3/Dir=0/Npix={norder3_pix}/part0.snappy.parquet"
    )
    assert classify_ztf_dr24_catflag(0) == QualitySeverity.INFORMATIONAL
    assert classify_ztf_dr24_catflag(1) == QualitySeverity.DEGRADED
    assert classify_ztf_dr24_catflag(CATFLAG_CLOUD_OR_MOON_BIT) == QualitySeverity.FATAL

    row = {
        "objectid": 999888777,
        "filterid": 2,
        "lightcurve": {
            "hmjd": [59000.1, 59001.1, 59002.1],
            "mag": [18.0, 18.1, 18.2],
            "magerr": [0.03, 0.04, 0.05],
            "catflags": [0, 1, CATFLAG_CLOUD_OR_MOON_BIT],
        },
    }
    recs = dr24_row_to_observation_records(row, exclude_fatal=True)
    assert len(recs) == 2
    assert all(r.has_operational_pub_time is False for r in recs)
    assert recs[0].passband == "r"
    assert recs[1].quality_severity == QualitySeverity.DEGRADED


def test_reconciliation_dedup_ambiguity_and_gate_a():
    tns_records = [
        TnsObjectRecord(
            objid=1,
            tns_name="SN2024one",
            name_prefix="SN",
            ra_deg=150.0000,
            dec_deg=20.0000,
            healpix_nside32=0,
            discovery_mjd_utc=60350.0,
            raw_type="SN Ia",
            coarse_class=CoarseTransientClass.SN_IA,
            fine_class="SN Ia",
            evidence_tier=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            label_provenance=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            is_supernova=True,
            has_class_bibcode=True,
            ztf_ids=["ZTF24aaaaaaa"],
        ),
        # Distinct older SN at the SAME nuclear sky locus 6 years earlier (must NOT false-merge!)
        TnsObjectRecord(
            objid=11,
            tns_name="SN2018older",
            name_prefix="SN",
            ra_deg=150.0001,
            dec_deg=20.0001,
            healpix_nside32=0,
            discovery_mjd_utc=58220.0,
            raw_type="SN II",
            coarse_class=CoarseTransientClass.SN_CC,
            fine_class="SN II",
            evidence_tier=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            label_provenance=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            is_supernova=True,
            has_class_bibcode=True,
            ztf_ids=["ZTF18olderab"],
        ),
        # Close concurrent pair separated by ~3.0" -> must trigger AMBIGUOUS_MULTI_COUNTERPART
        TnsObjectRecord(
            objid=2,
            tns_name="AT2024amb1",
            name_prefix="AT",
            ra_deg=200.0000,
            dec_deg=10.0000,
            healpix_nside32=0,
            discovery_mjd_utc=60200.0,
            raw_type=None,
            coarse_class=CoarseTransientClass.UNCLASSIFIED,
            fine_class="UNCLASSIFIED",
            evidence_tier=EvidenceTier.L5_UNCLASSIFIED,
            label_provenance=EvidenceTier.L5_UNCLASSIFIED,
            is_supernova=None,
        ),
        TnsObjectRecord(
            objid=3,
            tns_name="AT2024amb2",
            name_prefix="AT",
            ra_deg=200.0008,
            dec_deg=10.0000,
            healpix_nside32=0,
            discovery_mjd_utc=60210.0,
            raw_type="CV",
            coarse_class=CoarseTransientClass.CV_NOVA,
            fine_class="CV",
            evidence_tier=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            label_provenance=EvidenceTier.L1_TRACEABLE_SPECTROSCOPIC,
            is_supernova=False,
            has_class_bibcode=True,
        ),
    ]

    external_records = [
        ExternalCatalogRecord(
            external_id="gaia_dr3:123456789",
            source_catalog="GAIA_DR3_QSO",
            ra_deg=85.0000,
            dec_deg=12.0000,
            healpix_nside32=0,
            coarse_class=CoarseTransientClass.AGN_QSO,
            fine_class="GaiaDR3_QSO_AGN",
            evidence_tier=EvidenceTier.L3_EXTERNAL_CATALOG,
            is_supernova=False,
            is_broker_derived=False,
            catalog_redshift=1.24,
        )
    ]

    ztf_objects = [
        # Two ZTF OIDs for the SAME physical SN2024one event (0.4" apart, same 2024 epoch)
        ZtfObjectSummary(
            oid="ZTF24aaaaaaa", ra_deg=150.0000, dec_deg=20.0000, firstmjd=60350.2, lastmjd=60390.0
        ),
        ZtfObjectSummary(
            oid="ZTF24dupbbbb", ra_deg=150.0001, dec_deg=20.0000, firstmjd=60351.0, lastmjd=60380.0
        ),
        # Older SN2018older at the SAME sky locus (MJD 58220) -> shared locus, SEPARATE episode!
        ZtfObjectSummary(
            oid="ZTF18olderab", ra_deg=150.0001, dec_deg=20.0001, firstmjd=58220.0, lastmjd=58290.0
        ),
        # Ambiguous match near AT2024amb1 & AT2024amb2
        ZtfObjectSummary(
            oid="ZTF24ambcccc", ra_deg=200.0003, dec_deg=10.0000, firstmjd=60205.0, lastmjd=60220.0
        ),
        # Independent Gaia DR3 QSO L3 contaminant
        ZtfObjectSummary(
            oid="ZTF_GAIA_QSO_1",
            ra_deg=85.0001,
            dec_deg=12.0000,
            firstmjd=59000.0,
            lastmjd=60400.0,
            gaia_source_id="123456789",
        ),
        # Completely unmatched ZTF object -> must remain L5 / is_supernova=None
        ZtfObjectSummary(
            oid="ZTF24unmatch", ra_deg=45.0000, dec_deg=-15.0000, firstmjd=60100.0
        ),
    ]

    bundle, report = build_identity_graph_and_reconcile(
        ztf_objects=ztf_objects,
        tns_records=tns_records,
        external_records=external_records,
        temporal_cutoff_mjd=60310.0,
    )

    assert report.passed_gate_a is True
    assert report.physical_event_split_leakage_count == 0
    assert report.sky_locus_split_leakage_count == 0
    assert report.spatial_cluster_split_leakage_count == 0
    assert report.unclassified_as_negative_count == 0
    assert report.broker_truth_in_primary_count == 0
    assert report.multi_episode_loci_count == 1
    assert len(bundle.host_counterparts) == 1

    by_oid = {e.ztf_oid: e for e in bundle.reconciled_candidates}
    # Same-epoch multi-OID detections share SAME physical_event_id
    assert by_oid["ZTF24aaaaaaa"].physical_event_id == "phys:tns:SN2024one"
    assert by_oid["ZTF24dupbbbb"].physical_event_id == "phys:tns:SN2024one"
    assert by_oid["ZTF24aaaaaaa"].temporal_split == "archival_observation_date_holdout"

    # Older 2018 transient shares PersistentSkyLocus AND spatial split, NOT physical episode!
    assert by_oid["ZTF18olderab"].sky_locus_id == by_oid["ZTF24aaaaaaa"].sky_locus_id
    assert by_oid["ZTF18olderab"].assigned_split == by_oid["ZTF24aaaaaaa"].assigned_split
    assert by_oid["ZTF18olderab"].physical_event_id == "phys:tns:SN2018older"
    assert by_oid["ZTF18olderab"].temporal_split == "historical_dev"

    # Independent Gaia QSO L3 contaminant
    assert by_oid["ZTF_GAIA_QSO_1"].reconciliation_status == ReconciliationStatus.EXTERNAL_CATALOG_MATCH
    assert by_oid["ZTF_GAIA_QSO_1"].evidence_tier == EvidenceTier.L3_EXTERNAL_CATALOG
    assert by_oid["ZTF_GAIA_QSO_1"].coarse_class == CoarseTransientClass.AGN_QSO
    assert by_oid["ZTF_GAIA_QSO_1"].is_supernova is False

    assert (
        by_oid["ZTF24ambcccc"].reconciliation_status
        == ReconciliationStatus.AMBIGUOUS_MULTI_COUNTERPART
    )
    assert by_oid["ZTF24ambcccc"].is_quarantined is True

    assert by_oid["ZTF24unmatch"].reconciliation_status == ReconciliationStatus.UNMATCHED
    assert by_oid["ZTF24unmatch"].evidence_tier == EvidenceTier.L5_UNCLASSIFIED
    assert by_oid["ZTF24unmatch"].is_supernova is None

    # P0.4 invariant: broker classifiers cannot be masqueraded as L3_EXTERNAL_CATALOG
    with pytest.raises(ValueError, match="P0.4 violation"):
        ExternalCatalogRecord(
            external_id="alerce:1",
            source_catalog="ALeRCE_lc_classifier",
            ra_deg=10.0,
            dec_deg=10.0,
            healpix_nside32=0,
            coarse_class=CoarseTransientClass.AGN_QSO,
            fine_class="AGN",
            evidence_tier=EvidenceTier.L3_EXTERNAL_CATALOG,
            is_broker_derived=True,
        )
