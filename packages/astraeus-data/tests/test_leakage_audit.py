import pytest
from astraeus_data.synthetic import generate_synthetic_entity
from astraeus_data.leakage_audit import (
    run_full_leakage_audit,
    audit_healpix_assignment_overlap,
    audit_source_identity_leakage,
    audit_alias_leakage,
    audit_temporal_leakage,
)


def test_healpix_assignment_overlap():
    train_pixels = [10, 20, 30]
    test_pixels = [40, 50, 60]
    assert audit_healpix_assignment_overlap(train_pixels, test_pixels) == 0

    # Test overlap detection
    assert audit_healpix_assignment_overlap([10, 20], [20, 30]) == 1


def test_source_identity_leakage():
    e1 = generate_synthetic_entity("supernova_ia", seed=1)
    e2 = generate_synthetic_entity("supernova_ia", seed=2)
    assert audit_source_identity_leakage([e1], [e2]) == 0

    # Same entity in both
    assert audit_source_identity_leakage([e1], [e1]) == 1


def test_alias_leakage():
    train_aliases = {"e1": {"ZTF20aaa", "AT2020xnd"}}
    test_clean = {"e2": {"ZTF20bbb", "AT2020abc"}}
    assert audit_alias_leakage(train_aliases, test_clean) == 0

    test_leaked = {"e3": {"ZTF20ccc", "AT2020xnd"}}  # Overlap on AT2020xnd!
    assert audit_alias_leakage(train_aliases, test_leaked) == 1


def test_temporal_leakage():
    e = generate_synthetic_entity("supernova_ia", seed=10)
    # Synthetic entity has MJDs around 59000-59100
    assert audit_temporal_leakage([e], t_split_mjd=60000.0) == 0

    # If t_split is 59050.0, some observations should violate
    assert audit_temporal_leakage([e], t_split_mjd=59050.0) > 0


def test_run_full_leakage_audit():
    train = [generate_synthetic_entity("supernova_ia", seed=i) for i in range(5)]
    test = [generate_synthetic_entity("supernova_ia", seed=i + 100) for i in range(5)]

    report = run_full_leakage_audit(
        train_entities=train,
        test_entities=test,
        split_manifest_path="configs/data_splits_healpix.json",
        t_split_mjd=60000.0
    )
    assert report.healpix_assignment_overlap_count == 0
    assert report.source_identity_overlap_count == 0
    assert report.temporal_leakage_count == 0
