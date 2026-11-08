import pytest
from pathlib import Path
import tempfile
import pyarrow.parquet as pq

from astraeus_data.synthetic import generate_synthetic_entity, generate_bazin_lightcurve
from astraeus_data.partitioning import (
    load_split_manifest,
    entities_to_arrow_table,
    save_partitioned_parquet,
    audit_zero_leakage,
)
from astraeus_core.exceptions import DataLeakageError


def test_synthetic_entity_generation():
    entity = generate_synthetic_entity(entity_type="supernova_ia", seed=101)
    assert entity.entity_type == "supernova_ia"
    assert len(entity.observations) == 20
    assert entity.coordinates.healpix_index_nside32 is not None
    assert entity.provenance.was_generated_by.startswith("act:sim_generator:")


def test_synthetic_fbot_anomaly():
    fbot = generate_synthetic_entity(entity_type="fbot_anomaly", seed=202)
    assert fbot.entity_type == "fbot_anomaly"
    assert len(fbot.observations) == 20


def test_zero_leakage_audit_clean():
    train_pixels = [1, 2, 3, 4]
    test_pixels = [5, 6, 7, 8]
    assert audit_zero_leakage(train_pixels, test_pixels) is True


def test_zero_leakage_audit_detects_leakage():
    train_pixels = [1, 2, 3, 4]
    test_pixels = [4, 5, 6]  # Overlap on pixel 4!
    with pytest.raises(DataLeakageError, match="Spatial data leakage detected"):
        audit_zero_leakage(train_pixels, test_pixels)


def test_partitioned_parquet_roundtrip():
    entities = [
        generate_synthetic_entity("supernova_ia", seed=i)
        for i in range(10)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = save_partitioned_parquet(entities, tmpdir)
        assert out_dir.exists()

        # Read back using pyarrow dataset
        dataset = pq.read_table(str(out_dir))
        assert len(dataset) == 10
        assert "split" in dataset.column_names
        assert "healpix_nside32" in dataset.column_names
        assert "latest_flux" in dataset.column_names
