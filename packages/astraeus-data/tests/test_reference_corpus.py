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
    assert len(cow.cryptographic_seal.hmac_sha256_signature) == 64
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
    assert sn11fe.verify_authenticity() is True

    tde = builder.create_archival_at2018zr()
    assert tde.canonical_id == "AT2018zr"
    assert tde.transient_class == "TDE"
    assert tde.host_association.angular_separation_arcsec < 0.1  # Nuclear offset
    assert tde.verify_authenticity() is True


def test_spatial_partitioning_and_boundary_quarantine():
    builder = ReferenceCorpusBuilder(buffer_arcsec=5.0)

    # Object with large astrometric uncertainty should trigger 3-sigma quarantine near boundary
    partition_clean = builder.compute_spatial_partition(ra_deg=100.0, dec_deg=20.0, astrometric_err_arcsec=0.01)
    assert partition_clean.split in ["train", "val", "test"]
    assert partition_clean.healpix_nside32 >= 0


def test_corpus_auditor_clean_pass():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=10, base_seed=100)

    # Total: 3 archival + 8 classes * 10 = 83 records
    assert len(corpus) == 83

    auditor = ReferenceCorpusAuditor()
    report = auditor.audit_corpus(corpus)

    assert report.passed_all_gates is True
    assert report.total_records_audited == 83
    assert report.duplicate_canonical_ids == 0
    assert report.identity_leakage_count == 0
    assert report.alias_leakage_count == 0
    assert report.spatial_healpix_overlap_count == 0
    assert report.temporal_leakage_count == 0
    assert report.cryptographic_failures == 0
    assert "FBOT" in report.class_distribution_per_split["train"] or "FBOT" in report.class_distribution_per_split["test"] or "FBOT" in report.class_distribution_per_split["val"]


def test_corpus_auditor_detects_injected_identity_leakage():
    builder = ReferenceCorpusBuilder()
    corpus = builder.build_reference_corpus_v1(n_synthetic_per_class=5, base_seed=300)

    # Force a canonical ID collision between train and test
    train_rec = next(r for r in corpus if r.spatial_partition.split == "train")
    test_rec = next(r for r in corpus if r.spatial_partition.split == "test")

    # Construct a leaked duplicate test record with identical canonical ID
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

    assert protocol.protocol_version == "1.0.0"
    assert len(protocol.protocol_sha256) == 64
    assert len(protocol.tasks) == 3
    assert len(protocol.frozen_test_ids) > 0

    train_ids = [r.canonical_id for r in corpus if r.spatial_partition.split == "train"]
    assert protocol.verify_no_contamination(train_ids) is True

    # Test contamination detection
    contaminated_train_ids = train_ids + [protocol.frozen_test_ids[0]]
    assert protocol.verify_no_contamination(contaminated_train_ids) is False
