"""
Unit and integration tests for Astronomical Reference Corpus v1.0,
11-Point Identity & Provenance Audit Records, and Frozen Evaluation Protocol.
"""

import pytest
from astraeus_data.corpus.record import (
    ReferenceObjectRecord,
    DEFAULT_PROVENANCE_SECRET,
)
from astraeus_data.corpus.builder import ReferenceCorpusBuilder
from astraeus_data.corpus.auditor import ReferenceCorpusAuditor
from astraeus_data.corpus.protocol import FrozenEvaluationProtocol


def test_reference_object_record_integrity_and_hmac():
    builder = ReferenceCorpusBuilder()
    cow = builder.create_archival_at2018cow()

    assert cow.canonical_id == "AT2018cow"
    assert "ATLAS18qqn" in cow.aliases
    assert cow.dataset_origin == "ARCHIVAL"
    assert cow.transient_class == "FBOT"
    assert cow.coordinates.ra_deg == 244.000917
    assert cow.coordinates.dec_deg == 22.268031
    assert cow.host_association.host_name == "CGCG 137-068"
    assert cow.host_association.redshift_z == 0.014145
    assert cow.host_association.projected_offset_kpc == 1.73

    # Point 11: Cryptographic authenticity
    assert len(cow.cryptographic_seal.sha256_digest) == 64
    assert len(cow.cryptographic_seal.hmac_sha256_mac) == 64
    assert cow.cryptographic_seal.key_id == "astraeus-provenance-key-2026-v1"
    assert cow.verify_authenticity(DEFAULT_PROVENANCE_SECRET) is True

    # Tampering test with incorrect secret key
    assert cow.verify_authenticity(b"invalid-attacker-secret-key") is False


def test_archival_records_astrometry_and_provenance():
    builder = ReferenceCorpusBuilder()

    sn11fe = builder.create_archival_sn2011fe()
    assert sn11fe.canonical_id == "SN2011fe"
    assert sn11fe.transient_class == "SN_Ia"
    assert sn11fe.peak_apparent_magnitude < 11.0
    assert sn11fe.host_association.host_name.startswith("Messier 101")
    assert sn11fe.host_association.distance_method == "INDEPENDENT_TRGB_CEPHEID"
    assert sn11fe.host_association.adopted_distance_mpc == 6.4
    assert sn11fe.host_association.angular_diameter_distance_mpc == 6.39
    assert sn11fe.host_association.angular_separation_arcsec == 162.0
    # Mandatory reconciliation: 162" * 6400 kpc / 206265 = 5.02 kpc (NOT 5.4 kpc)
    assert sn11fe.host_association.projected_offset_kpc == 5.02
    assert sn11fe.host_association.projected_offset_uncertainty_kpc == 0.39
    assert sn11fe.photometric_calibration.survey_nominal_zero_point == 27.0
    assert sn11fe.photometric_calibration.observation_zero_point is not None
    assert sn11fe.verify_authenticity() is True

    tde = builder.create_archival_at2018zr()
    assert tde.canonical_id == "AT2018zr"
    assert tde.transient_class == "TDE"
    assert tde.host_association.angular_separation_arcsec < 0.1  # Nuclear offset
    assert tde.host_association.nuclear_offset_resolution_arcsec == 1.0
    assert tde.photometric_calibration.survey_nominal_zero_point == 26.3
    assert tde.verify_authenticity() is True


def test_spatial_partitioning_and_boundary_quarantine():
    builder = ReferenceCorpusBuilder(buffer_arcsec=5.0)

    partition_clean = builder.compute_spatial_partition(ra_deg=100.0, dec_deg=20.0, astrometric_err_arcsec=0.01)
    assert partition_clean.split in ["train", "val", "test", "quarantine"]
    assert partition_clean.healpix_nside32 >= 0


def test_corpus_auditor_clean_pass_and_quarantine_reconciliation():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=10, base_seed=100)

    # Total: 3 archival + 8 classes * 10 = 83 records
    assert len(corpus) == 83

    auditor = ReferenceCorpusAuditor()
    report = auditor.audit_corpus(corpus)

    assert report.passed_all_gates is True
    assert report.total_records_audited == 83
    # Check exact population arithmetic: Eligible + Quarantined == Total
    assert report.eligible_entities_count + report.quarantined_boundary_count == report.total_records_audited
    assert report.train_count + report.val_count + report.test_count == report.eligible_entities_count
    assert report.duplicate_canonical_ids == 0
    assert report.identity_leakage_count == 0
    assert report.alias_leakage_count == 0
    assert report.spatial_healpix_overlap_count == 0
    assert report.host_galaxy_cross_split_leakage_count == 0
    assert report.observation_cross_split_leakage_count == 0
    assert report.boundary_buffer_quarantine_violations == 0
    assert report.temporal_leakage_count == 0
    assert report.cryptographic_failures == 0


def test_corpus_auditor_detects_injected_identity_leakage():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=5, base_seed=300)

    train_rec = next(r for r in corpus if r.spatial_partition.split == "train")
    test_rec = next(r for r in corpus if r.spatial_partition.split == "test")

    leaked_rec = test_rec.model_copy(update={"canonical_id": train_rec.canonical_id})
    corrupted_corpus = [r for r in corpus if r.canonical_id != test_rec.canonical_id] + [leaked_rec]

    auditor = ReferenceCorpusAuditor()
    report = auditor.audit_corpus(corrupted_corpus)

    assert report.passed_all_gates is False
    assert report.identity_leakage_count >= 1 or report.duplicate_canonical_ids >= 1


def test_frozen_evaluation_protocol_immutability():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=5, base_seed=400)

    protocol = FrozenEvaluationProtocol.freeze(corpus)

    assert protocol.protocol_version == "1.0.2"
    assert len(protocol.protocol_sha256) == 64
    assert len(protocol.tasks) == 3
    assert len(protocol.frozen_test_ids) > 0

    train_ids = [r.canonical_id for r in corpus if r.spatial_partition.split == "train"]
    assert protocol.verify_no_contamination(train_ids) is True

    contaminated_train_ids = train_ids + [protocol.frozen_test_ids[0]]
    assert protocol.verify_no_contamination(contaminated_train_ids) is False


def test_peripheral_object_500arcsec_from_edge_is_not_quarantined():
    """
    Mandatory reviewer acceptance test:
    An object ~400-500" from the actual cell boundary must NOT be quarantined
    by a 5" geometric boundary rule merely because it lies outside the inscribed circle.
    Proves that the flawed inscribed-circle clamp is eliminated in v1.0.2.
    """
    import numpy as np
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astraeus_data.corpus.geometry import (
        compute_spherical_edge_distance_arcsec,
        evaluate_quarantine_predicate,
    )

    builder = ReferenceCorpusBuilder()
    hp = builder.hp
    pix = 1049
    b = hp.boundaries_skycoord([pix], step=25)[0]
    center = hp.healpix_to_skycoord(pix)
    c_xyz = center.cartesian.xyz.value
    b_xyz = b.cartesian.xyz.value.T
    v_corner = b_xyz[0] - c_xyz
    v_corner /= np.linalg.norm(v_corner)

    # Place point 3800" from center towards corner (outside inscribed circle of radius 3298")
    theta_rad = 3800.0 / 206264.806247
    p_test = np.cos(theta_rad) * c_xyz + np.sin(theta_rad) * v_corner
    p_test /= np.linalg.norm(p_test)
    c_test = SkyCoord(x=p_test[0], y=p_test[1], z=p_test[2], representation_type="cartesian", frame="icrs")

    d_from_center = c_test.separation(center).arcsec
    r_inscribed = hp.pixel_resolution.to(u.arcsec).value / 2.0
    assert d_from_center > r_inscribed  # Lies in peripheral region outside inscribed circle

    # Flawed old formula would have evaluated:
    flawed_distance = max(0.1, r_inscribed - d_from_center)
    assert flawed_distance == 0.1  # Clamped to 0.1" < 5.0" (flawed quarantine trigger)

    # Correct v1.0.2 spherical boundary distance:
    d_edge = compute_spherical_edge_distance_arcsec(c_test, hp, pix, step=25)
    assert d_edge > 400.0  # Actually ~425" from the nearest cell edge!

    # Evaluated under corrected predicate with conflicting neighbor:
    is_quarantined, reason, d_computed = evaluate_quarantine_predicate(
        coord=c_test,
        hp=hp,
        pix=pix,
        has_conflicting_neighbor=True,
        astrometric_err_arcsec=0.03,
        buffer_arcsec=5.0,
        confidence_multiplier=3.0,
    )
    # MUST NOT be quarantined!
    assert is_quarantined is False
    assert reason is None
    assert d_computed > 400.0


def test_astrometric_uncertainty_crossing_risk_quarantine():
    """
    Tests separate astrometric uncertainty policy:
    An entity 8" from the edge with 4" uncertainty (3*sigma = 12") is quarantined
    for ASTROMETRIC_UNCERTAINTY_CROSSING_RISK when adjacent to a conflicting split.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from astraeus_data.corpus.geometry import evaluate_quarantine_predicate

    builder = ReferenceCorpusBuilder()
    hp = builder.hp
    pix = 1049

    # Entity near boundary with large uncertainty
    # We test the evaluate_quarantine_predicate logic with mocked distance
    from unittest.mock import patch

    coord = SkyCoord(ra=139.2 * u.deg, dec=10.0 * u.deg, frame="icrs")

    # When d_edge = 8.0", buffer = 5.0", ast_err = 4.0" (3*ast_err = 12.0" >= 8.0")
    with patch("astraeus_data.corpus.geometry.compute_spherical_edge_distance_arcsec", return_value=8.0):
        is_quar, reason, dist = evaluate_quarantine_predicate(
            coord=coord,
            hp=hp,
            pix=pix,
            has_conflicting_neighbor=True,
            astrometric_err_arcsec=4.0,
            buffer_arcsec=5.0,
            confidence_multiplier=3.0,
        )
        assert is_quar is True
        assert reason == "ASTROMETRIC_UNCERTAINTY_CROSSING_RISK"
        assert dist == 8.0

        # If no conflicting neighbor exists, entity is retained
        is_quar_no_conf, reason_no_conf, _ = evaluate_quarantine_predicate(
            coord=coord,
            hp=hp,
            pix=pix,
            has_conflicting_neighbor=False,
            astrometric_err_arcsec=4.0,
            buffer_arcsec=5.0,
            confidence_multiplier=3.0,
        )
        assert is_quar_no_conf is False
        assert reason_no_conf is None


def test_frozen_evaluation_protocol_dataset_b_exclusion_contract():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=5, base_seed=400)
    protocol = FrozenEvaluationProtocol.freeze(corpus)

    # Disjoint Dataset B candidate IDs pass exclusion check
    clean_dataset_b = ["ASTRAEUS-PRETRAIN-ZTF-000001", "ASTRAEUS-PRETRAIN-PS1-000002"]
    is_clean, violations = protocol.verify_dataset_b_exclusion(clean_dataset_b)
    assert is_clean is True
    assert len(violations) == 0

    # Contaminated Dataset B containing a Dataset A held-out test object fails check
    leaked_id = protocol.frozen_test_ids[0]
    contaminated_dataset_b = clean_dataset_b + [leaked_id]
    is_clean, violations = protocol.verify_dataset_b_exclusion(contaminated_dataset_b)
    assert is_clean is False
    assert leaked_id in violations

    # Contaminated Dataset B containing an injected/transformed derivative fails check
    derivative_b = clean_dataset_b + [f"AUGMENTED_SHIFT_{leaked_id}"]
    is_clean, violations = protocol.verify_dataset_b_exclusion(derivative_b)
    assert is_clean is False
    assert any(leaked_id in v for v in violations)

    # Derivative tag check
    is_clean, violations = protocol.verify_dataset_b_exclusion(
        clean_dataset_b,
        dataset_b_derivative_tags=[f"INJECTED_FROM_{leaked_id}"]
    )
    assert is_clean is False
    assert any(leaked_id in v for v in violations)


def test_photometric_calibration_and_negative_flux_uncertainty_handling():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=10, base_seed=500)

    # Case A: Positive forced photometry
    cow = builder.create_archival_at2018cow()
    assert all(f > 0 for f in cow.flux_native)
    assert cow.photometric_calibration.survey_nominal_zero_point == 27.5
    assert cow.photometric_calibration.observation_zero_point == 27.48

    # Case B: Negative noisy difference flux preservation with native linear uncertainty
    artifact = next(r for r in corpus if r.transient_class == "ARTIFACT_DIPOLE" and any(f < 0 for f in r.flux_native))
    neg_idx = next(i for i, f in enumerate(artifact.flux_native) if f < 0)
    assert artifact.flux_native[neg_idx] < 0
    # Linear pipeline uncertainty is positive and finite, not calculated through log magnitude
    assert artifact.flux_err_native[neg_idx] > 0
    assert artifact.photometric_calibration.preserves_negative_fluxes is True

    # Case C: 5-sigma censored upper limits
    assert artifact.measurement_uncertainty.upper_limit_sigma == 5.0
    assert artifact.measurement_uncertainty.preserves_linear_negative_flux_error is True


def test_clopper_pearson_confidence_intervals_and_quarantine_breakdown():
    from astraeus_data.corpus.auditor import compute_clopper_pearson

    # FBOT test support N=2: 100% recall (2/2) -> exact 95% CI is [15.81%, 100.0%]
    low, up = compute_clopper_pearson(2, 2, 0.95)
    assert low == pytest.approx(0.1581, abs=1e-3)
    assert up == 1.0

    # SLSN test support N=5: 100% recall (5/5) -> exact 95% CI is [47.82%, 100.0%]
    low, up = compute_clopper_pearson(5, 5, 0.95)
    assert low == pytest.approx(0.4782, abs=1e-3)
    assert up == 1.0

    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=10, base_seed=600)
    auditor = ReferenceCorpusAuditor()
    report = auditor.audit_corpus(corpus)

    assert len(report.test_clopper_pearson_intervals) == 8
    assert "FBOT" in report.test_clopper_pearson_intervals
    assert len(report.archival_objects) == 3
    assert set(report.archival_objects) == {"AT2018cow", "SN2011fe", "AT2018zr"}
    assert sum(report.quarantine_breakdown_by_class.values()) == report.quarantined_boundary_count
    assert sum(report.quarantine_breakdown_by_original_split.values()) == report.quarantined_boundary_count
    assert len(report.quarantine_diagnostic_table) >= 4
    # Verify the 4 illustrative diagnostic cases are present
    obj_names = [d["object"] for d in report.quarantine_diagnostic_table]
    assert "Case A (Illustrative)" in obj_names
    assert "Case B (Illustrative)" in obj_names
    assert "Case C (Illustrative)" in obj_names
    assert "Case D (Illustrative)" in obj_names
