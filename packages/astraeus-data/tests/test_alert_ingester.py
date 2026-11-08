import pytest
import tempfile
from pathlib import Path

from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate
from astraeus_data.ingestion.alert_ingester import AlertIngestionEngine, IngestionStatus


@pytest.fixture
def sample_alert():
    cand = ZtfCandidate(
        jd=2459125.5,
        fid=1,
        pid=9988,
        diffmaglim=20.8,
        isdiffpos=1,
        ra=200.5,
        dec=15.2,
        magpsf=18.1,
        sigmapsf=0.05,
        rb=0.88,
    )
    prv = ZtfCandidate(
        jd=2459123.5,
        fid=2,
        pid=9987,
        diffmaglim=20.5,
        isdiffpos=0,
        ra=200.5,
        dec=15.2,
        magpsf=None,
        sigmapsf=None,
    )
    return ZtfAlertPacket(
        candid=500600700,
        objectId="ZTF21abcxyz",
        candidate=cand,
        prv_candidates=[prv],
    )


def test_alert_ingestion_success(sample_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AlertIngestionEngine(storage_root=tmpdir)
        res = engine.ingest_packet(sample_alert)

        assert res.status == IngestionStatus.INGESTED
        assert res.candid == 500600700
        assert (Path(tmpdir) / "raw" / "500600700.json").exists()


def test_idempotent_replay_no_duplicates(sample_alert):
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AlertIngestionEngine(storage_root=tmpdir)

        # First ingestion
        res1 = engine.ingest_packet(sample_alert)
        assert res1.status == IngestionStatus.INGESTED

        # Second ingestion (replay)
        res2 = engine.ingest_packet(sample_alert)
        assert res2.status == IngestionStatus.DUPLICATE_SKIPPED
        assert res2.raw_hash == res1.raw_hash


def test_quarantine_invalid_coordinates():
    raw_invalid = {
        "candid": 999000111,
        "objectId": "ZTF_Invalid",
        "candidate": {
            "jd": 2459125.5,
            "fid": 1,
            "pid": 9988,
            "diffmaglim": 20.8,
            "isdiffpos": 1,
            "ra": 380.0,  # Invalid RA > 360
            "dec": 15.2,
        }
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = AlertIngestionEngine(storage_root=tmpdir)
        res = engine.ingest_raw_dict(raw_invalid)

        assert res.status == IngestionStatus.QUARANTINED
        assert "less than 360" in res.error_message or "validation error" in res.error_message.lower()
        assert (Path(tmpdir) / "quarantine" / "999000111_schema_error.json").exists()
