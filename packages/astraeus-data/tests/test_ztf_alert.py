import pytest
from astraeus_data.schemas.ztf_alert import ZtfAlertPacket, ZtfCandidate


def test_ztf_candidate_fields():
    cand = ZtfCandidate(
        jd=2459123.5,
        fid=1,
        pid=12345,
        diffmaglim=20.5,
        isdiffpos="t",
        ra=184.2,
        dec=-5.8,
        magpsf=17.5,
        sigmapsf=0.04,
        chipsf=1.1,
        fwhm=2.2,
        rb=0.92,
        drb=0.98,
        scorr=8.5,
    )
    assert cand.mjd == pytest.approx(59123.0)
    assert cand.filter_name == "ZTF_g"
    assert cand.is_detection is True


def test_ztf_candidate_upper_limit():
    upper_lim = ZtfCandidate(
        jd=2459120.5,
        fid=2,
        pid=12344,
        diffmaglim=21.0,
        isdiffpos=0,
        ra=184.2,
        dec=-5.8,
        magpsf=None,
        sigmapsf=None,
    )
    assert upper_lim.filter_name == "ZTF_r"
    assert upper_lim.is_detection is False


def test_ztf_alert_packet_hashing():
    cand = ZtfCandidate(
        jd=2459123.5,
        fid=1,
        pid=12345,
        diffmaglim=20.5,
        isdiffpos=1,
        ra=184.2,
        dec=-5.8,
        magpsf=17.5,
        sigmapsf=0.04,
    )
    packet = ZtfAlertPacket(
        candid=100200300,
        objectId="ZTF20acvhrgm",
        candidate=cand,
    )
    h1 = packet.compute_payload_hash()
    h2 = packet.compute_payload_hash()
    assert h1 == h2
    assert len(h1) == 64
