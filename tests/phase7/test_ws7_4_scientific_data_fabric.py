"""
ASTRAEUS Phase 7 - Workstream 7.4 Scientific Data Fabric Verification Tests.
Verifies:
1. Multi-broker connectivity (Fink, Lasair) with offline replay buffers and rate limiting.
2. Delayed ground-truth linkage in TNS without lookahead bias.
3. Strict offline learning gate: never backpropagate on unconfirmed alerts.
4. Gaia DR3 astrometric star vetting (RUWE, parallax, proper motion).
5. GraceDB GW notice matching and AstroSat/HEASARC high-energy upper limits.
6. Gate 3 Zero-Touch Quarantine Firewall enforcement across all fabric archives.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import time
import math
from pathlib import Path

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_data.fabric.broker_connector import (
    BrokerType,
    AlertQueryFilter,
    FinkBrokerClient,
    LasairBrokerClient,
)
from astraeus_data.fabric.tns_connector import (
    TnsClient,
    TnsRecord,
    TnsClassificationCertainty,
    OfflineLearningGate,
    UnconfirmedAlertTrainingProhibitedError,
)
from astraeus_data.fabric.archive_connectors import (
    GaiaTapClient,
    GaiaSourceRecord,
    GraceDbClient,
    GraceDbSuperevent,
    AstroSatHeasarcClient,
    MultiWavelengthRecord,
)
from astraeus_data.fabric.data_fabric import (
    ScientificDataFabric,
    OperatingMode,
    TransientEnvironmentContext,
)


def get_quarantined_coordinate() -> tuple[float, float]:
    """Helper to get a known coordinate inside the Gate 3 frozen HEALPix pixels."""
    QuarantineFirewall._initialize()
    q_pix = next(iter(QuarantineFirewall.get_quarantined_pixels()))
    coord = QuarantineFirewall._HP.healpix_to_skycoord(q_pix)
    return float(coord.ra.deg), float(coord.dec.deg)


def test_fink_and_lasair_broker_clients():
    """Verifies broker querying, cone search, filtering, and query ledger logging."""
    fink = FinkBrokerClient(offline_mode=True)
    lasair = LasairBrokerClient(offline_mode=True)

    # Load replay alerts
    mock_alerts = [
        {
            "objectId": "ZTF20acvhrgm",
            "candidate": {
                "candid": 123456789,
                "ra": 180.0,
                "dec": 30.0,
                "jd": 2459000.5,
                "fink_class": "Early SN candidate",
            },
        },
        {
            "objectId": "ZTF21aaaknls",
            "candidate": {
                "candid": 987654321,
                "ra": 180.001,
                "dec": 30.001,
                "jd": 2459002.5,
                "fink_class": "Kilonova candidate",
            },
        },
    ]

    fink.load_replay_buffer(mock_alerts)
    lasair.load_replay_buffer(mock_alerts)

    # 1. Query object
    obj_res = fink.query_object("ZTF20acvhrgm")
    assert obj_res is not None
    assert obj_res["objectId"] == "ZTF20acvhrgm"

    # 2. Cone search
    q_filter = AlertQueryFilter(ra_deg=180.0, dec_deg=30.0, radius_arcsec=10.0)
    cone_res = fink.query_cone(q_filter)
    assert len(cone_res) == 2

    # 3. Class-filtered alerts
    kn_res = fink.query_latest_alerts_by_class("Kilonova", n_alerts=10)
    assert len(kn_res) == 1
    assert kn_res[0]["objectId"] == "ZTF21aaaknls"

    # 4. Query ledger audit
    ledger = fink.get_query_ledger()
    assert len(ledger) >= 3
    for entry in ledger:
        assert entry.broker == BrokerType.FINK
        assert entry.quarantine_checked is True
        assert len(entry.sha256_hash) == 64


def test_broker_quarantine_firewall_enforcement():
    """Verifies that broker cone searches in quarantined pixels raise QuarantineViolationError."""
    fink = FinkBrokerClient(offline_mode=True)
    q_ra, q_dec = get_quarantined_coordinate()

    q_filter = AlertQueryFilter(ra_deg=q_ra, dec_deg=q_dec, radius_arcsec=5.0)
    with pytest.raises(QuarantineViolationError):
        fink.query_cone(q_filter)


def test_tns_delayed_ground_truth_linkage_no_lookahead():
    """
    Verifies delayed ground-truth linkage:
    Before classification release epoch, candidate must appear unclassified.
    After release epoch, spectroscopic confirmation becomes available.
    """
    tns = TnsClient(offline_mode=True)

    rec = TnsRecord(
        iau_name="SN 2024test",
        internal_name="ZTF24aaaabbb",
        ra_deg=120.0,
        dec_deg=45.0,
        discovery_date_mjd=60000.0,
        discovery_mag=18.5,
        redshift=0.035,
        redshift_err=0.001,
        classification="SN Ia",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=60005.0,  # Spectroscopic confirmation 5 days later
        spectroscopy_instrument="SEDM",
        spectroscopy_telescope="P60",
    )
    tns.add_record(rec)

    # 1. Query at t = 60002.0 (2 days after trigger): classification must be masked!
    early_link = tns.link_ground_truth("SN 2024test", as_of_mjd=60002.0)
    assert early_link is not None
    assert early_link.certainty == TnsClassificationCertainty.UNCLASSIFIED
    assert early_link.classification is None
    assert early_link.spectroscopy_instrument is None
    assert early_link.is_spectroscopically_confirmed() is False

    # 2. Query at t = 60005.5 (after classification registered): full ground truth available
    late_link = tns.link_ground_truth("SN 2024test", as_of_mjd=60005.5)
    assert late_link is not None
    assert late_link.certainty == TnsClassificationCertainty.SPECTROSCOPIC
    assert late_link.classification == "SN Ia"
    assert late_link.spectroscopy_instrument == "SEDM"
    assert late_link.is_spectroscopically_confirmed() is True


def test_offline_learning_gate_enforcement():
    """
    Verifies that online backpropagation on unconfirmed alerts is strictly prohibited,
    and only verified spectroscopic labels in offline learning mode are authorized.
    """
    gate = OfflineLearningGate()

    unconfirmed_cand = TnsRecord(
        iau_name="AT 2024unconf",
        ra_deg=100.0,
        dec_deg=20.0,
        discovery_date_mjd=60100.0,
        certainty=TnsClassificationCertainty.UNCLASSIFIED,
    )

    # 1. Attempting to train on unconfirmed alert raises exception
    with pytest.raises(UnconfirmedAlertTrainingProhibitedError, match="Online backpropagation on unconfirmed alerts is strictly prohibited"):
        gate.verify_training_candidate(unconfirmed_cand)

    # 2. Photometric candidate without explicit ablation raises exception
    photometric_cand = TnsRecord(
        iau_name="AT 2024phot",
        ra_deg=100.0,
        dec_deg=20.0,
        discovery_date_mjd=60100.0,
        classification="SN Ia-like",
        certainty=TnsClassificationCertainty.PHOTOMETRIC,
    )
    with pytest.raises(UnconfirmedAlertTrainingProhibitedError, match="Spectroscopic ground-truth is required"):
        gate.verify_training_candidate(photometric_cand)

    # 3. Verified spectroscopic candidate succeeds
    spectro_cand = TnsRecord(
        iau_name="SN 2024spec",
        ra_deg=100.0,
        dec_deg=20.0,
        discovery_date_mjd=60100.0,
        classification="SN Ia",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=60103.0,
        spectroscopy_instrument="LRIS",
    )
    authorized = gate.verify_training_candidate(spectro_cand)
    assert authorized is True
    assert gate.get_authorized_training_count() == 1


def test_gaia_tap_client_astrometric_star_vetting():
    """
    Verifies Gaia DR3 astrometric star vetting logic (RUWE, parallax, proper motion).
    """
    gaia = GaiaTapClient(offline_mode=True)

    # 1. Galactic foreground star (high parallax SNR and proper motion)
    star = GaiaSourceRecord(
        source_id=1234567890,
        ra_deg=50.0,
        dec_deg=10.0,
        parallax_mas=12.5,
        parallax_error_mas=0.5,  # SNR = 25.0
        pmra_mas_yr=35.0,
        pmdec_mas_yr=-20.0,
        ruwe=1.02,
        phot_g_mean_mag=14.2,
    )
    gaia.add_source(star)

    is_star, match = gaia.check_galactic_star_contamination(50.0, 10.0, search_radius_arcsec=1.5)
    assert is_star is True
    assert match is not None
    assert match.is_confirmed_galactic_star() is True

    # 2. Extragalactic coordinate without Gaia star
    is_star_ext, match_ext = gaia.check_galactic_star_contamination(150.0, 40.0, search_radius_arcsec=1.5)
    assert is_star_ext is False
    assert match_ext is None


def test_gracedb_and_heasarc_archive_connectors():
    """Verifies GraceDB GW superevents and AstroSat/HEASARC multi-wavelength records."""
    gracedb = GraceDbClient(offline_mode=True)
    heasarc = AstroSatHeasarcClient(offline_mode=True)

    # GraceDB event
    gw_ev = GraceDbSuperevent(
        superevent_id="S240101a",
        event_time_mjd=60000.10,
        far_hz=1e-12,
        prob_bns=0.85,
        prob_nsbh=0.10,
        prob_bbh=0.01,
        prob_terrestrial=0.04,
        prob_has_remnant=0.92,
        area_90_deg2=45.0,
        distance_mean_mpc=120.0,
        distance_std_mpc=25.0,
        central_ra_deg=180.0,
        central_dec_deg=0.0,
    )
    gracedb.add_superevent(gw_ev)

    assert gw_ev.is_electromagnetic_promising is True
    matches = gracedb.query_events_in_window(59999.0, 60001.0)
    assert len(matches) == 1

    # AstroSat / HEASARC observation
    obs = MultiWavelengthRecord(
        observation_id="astrosat_czti_001",
        telescope="AstroSat",
        instrument="CZTI",
        ra_deg=180.0,
        dec_deg=0.0,
        epoch_mjd=60000.15,
        exposure_sec=5000.0,
        is_detection=False,
        flux_upper_limit_3sigma=1.2e-13,
        energy_band_kev="20-150 keV",
    )
    heasarc.add_observation(obs)

    he_matches = heasarc.query_counterparts(180.0, 0.0, epoch_mjd=60000.12, time_window_days=1.0)
    assert len(he_matches) == 1
    assert he_matches[0].is_detection is False
    assert he_matches[0].flux_upper_limit_3sigma == 1.2e-13


def test_scientific_data_fabric_end_to_end_context_assembly():
    """
    Verifies end-to-end multi-archive context assembly via ScientificDataFabric,
    operating mode switching, and Gate 3 Quarantine Firewall defense-in-depth.
    """
    fabric = ScientificDataFabric(mode=OperatingMode.LIVE_INTELLIGENCE, offline_mode=True)

    # Populate mock archives
    star = GaiaSourceRecord(
        source_id=999888,
        ra_deg=80.0,
        dec_deg=15.0,
        parallax_mas=8.0,
        parallax_error_mas=0.2,
        pmra_mas_yr=18.0,
        pmdec_mas_yr=5.0,
        ruwe=1.04,
        phot_g_mean_mag=15.5,
    )
    fabric.gaia.add_source(star)

    # 1. Query transient environment coincident with Galactic star
    env = fabric.crossmatch_transient_environment(
        target_id="ZTF24starflare",
        ra_deg=80.0,
        dec_deg=15.0,
        epoch_mjd=60010.0,
    )
    assert env.is_galactic_star is True
    assert env.quarantine_cleared is True
    assert env.healpix_pixel_nside32 is not None

    # 2. In LIVE_INTELLIGENCE mode, training authorization is blocked
    spec_cand = TnsRecord(
        iau_name="SN 2024live",
        ra_deg=80.0,
        dec_deg=15.0,
        discovery_date_mjd=60000.0,
        classification="SN Ia",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
    )
    with pytest.raises(UnconfirmedAlertTrainingProhibitedError, match="System is in LIVE_INTELLIGENCE mode"):
        fabric.verify_and_authorize_training_event(spec_cand)

    # 3. Switch to OFFLINE_LEARNING mode: training authorized
    fabric.set_operating_mode(OperatingMode.OFFLINE_LEARNING)
    assert fabric.verify_and_authorize_training_event(spec_cand) is True

    # 4. Quarantine violation check: fabric query halts before contacting archives
    q_ra, q_dec = get_quarantined_coordinate()
    with pytest.raises(QuarantineViolationError):
        fabric.crossmatch_transient_environment(
            target_id="ZTF24quarantine",
            ra_deg=q_ra,
            dec_deg=q_dec,
            epoch_mjd=60010.0,
        )
