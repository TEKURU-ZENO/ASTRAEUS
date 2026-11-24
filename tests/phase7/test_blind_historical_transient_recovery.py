"""
ASTRAEUS Phase 7 - Blind Historical Transient Recovery Benchmark (WS7.4).
Evaluates autonomous triage accuracy, macro-F1 score, decision lead time,
and calibration against independent spectroscopic ground-truth from IAU TNS and ZTF BTS.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import pytest
import math
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError
from astraeus_data.fabric.data_fabric import (
    ScientificDataFabric,
    OperatingMode,
)
from astraeus_data.fabric.tns_connector import (
    TnsRecord,
    TnsClassificationCertainty,
)
from astraeus_data.fabric.archive_connectors import (
    GaiaSourceRecord,
)
from astraeus_data.fabric.replay_engine import (
    ReplayAlertEpoch,
    HistoricalStreamingReplayEngine,
)
from astraeus_data.triage.event_triage_engine import (
    EventTriageEngine,
    AnomalyClass,
    PriorityTier,
)


def build_curated_historical_cohort() -> tuple[list[ReplayAlertEpoch], list[TnsRecord], list[GaiaSourceRecord], list[dict]]:
    """
    Constructs a safe, unquarantined historical transient cohort with independent
    spectroscopic TNS ground truth and multi-epoch light curves.
    """
    QuarantineFirewall._initialize()
    
    # 1. Supernova Ia (ZTF17aabtvsy / SN 2022yei)
    # RA = 158.8837, Dec = 37.6497 (Pixel unquarantined)
    sn_ia_target = "SN2022yei"
    sn_ia_tns = TnsRecord(
        iau_name="SN 2022yei",
        internal_name="ZTF17aabtvsy",
        ra_deg=158.8837,
        dec_deg=37.6497,
        discovery_date_mjd=59850.0,
        discovery_mag=18.03,
        redshift=0.0692,
        classification="SN Ia",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=59858.0,  # Spectroscopic confirmation 8 days after trigger
        spectroscopy_instrument="SEDM",
    )

    # 2. Core-Collapse Supernova (SN II: ZTF18aaaibml / SN 2020buc)
    # RA = 152.1303, Dec = 9.2397
    sn_ii_target = "SN2020buc"
    sn_ii_tns = TnsRecord(
        iau_name="SN 2020buc",
        internal_name="ZTF18aaaibml",
        ra_deg=152.1303,
        dec_deg=9.2397,
        discovery_date_mjd=58900.0,
        discovery_mag=18.26,
        redshift=0.0348,
        classification="SN II",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=58906.0,  # 6 days later
        spectroscopy_instrument="DBSP",
    )

    # 3. Tidal Disruption Event (TDE: ZTF17aaazdba / AT 2019azh)
    # RA = 123.3206, Dec = 22.6483
    tde_target = "AT2019azh"
    tde_tns = TnsRecord(
        iau_name="AT 2019azh",
        internal_name="ZTF17aaazdba",
        ra_deg=123.3206,
        dec_deg=22.6483,
        discovery_date_mjd=58550.0,
        discovery_mag=15.28,
        redshift=0.0220,
        classification="TDE",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=58562.0,  # 12 days later
        spectroscopy_instrument="LRIS",
    )

    # 4. Fast-Fading Kilonova Candidate (GW170817-like rapid radioactive cooling)
    # RA = 197.4503, Dec = -23.3815 (unquarantined)
    kn_target = "AT2017gfo_proxy"
    kn_tns = TnsRecord(
        iau_name="AT 2017gfo",
        internal_name="AT2017gfo_proxy",
        ra_deg=197.4503,
        dec_deg=-23.3815,
        discovery_date_mjd=57982.52,
        discovery_mag=17.47,
        redshift=0.0097,
        classification="Kilonova",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=57983.80,  # Confirmed within 1.3 days
        spectroscopy_instrument="X-shooter",
    )

    # 5. Galactic Flare / Variable Star (coincident with bright Gaia star)
    # RA = 140.0, Dec = 20.0
    star_target = "ZTF24starflare"
    star_tns = TnsRecord(
        iau_name="AT 2024star",
        internal_name="ZTF24starflare",
        ra_deg=140.0,
        dec_deg=20.0,
        discovery_date_mjd=60100.0,
        discovery_mag=15.10,
        classification="Variable Star",
        certainty=TnsClassificationCertainty.SPECTROSCOPIC,
        classification_date_mjd=60101.0,
    )
    star_gaia = GaiaSourceRecord(
        source_id=888777666,
        ra_deg=140.0,
        dec_deg=20.0,
        parallax_mas=15.0,
        parallax_error_mas=0.3,
        pmra_mas_yr=42.0,
        pmdec_mas_yr=-12.0,
        ruwe=1.01,
        phot_g_mean_mag=13.8,
    )

    tns_records = [sn_ia_tns, sn_ii_tns, tde_tns, kn_tns, star_tns]
    gaia_records = [star_gaia]

    host_galaxies = [
        {"name": "WISEA J103532.09+373859.0", "ra": 158.8837 + 0.0003, "dec": 37.6497 + 0.0003, "redshift": 0.0692},
        {"name": "WISEA J100831.27+091423.0", "ra": 152.1303 + 0.0003, "dec": 9.2397 + 0.0003, "redshift": 0.0348},
        {"name": "SDSS J081316.95+223853.9", "ra": 123.3206 + 0.00003, "dec": 22.6483 + 0.00003, "redshift": 0.0220},
        {"name": "NGC 4993", "ra": 197.4503 + 0.0020, "dec": -23.3815 + 0.0020, "redshift": 0.0097},
    ]

    # Verify all cohort coordinates are outside Gate 3 quarantine
    for rec in tns_records:
        pix = QuarantineFirewall.compute_healpix_nside32(rec.ra_deg, rec.dec_deg)
        assert not QuarantineFirewall.is_quarantined_pixel(pix), f"Target {rec.iau_name} is in quarantined pixel {pix}!"

    # Generate chronological alert epochs
    alert_stream: list[ReplayAlertEpoch] = []

    # Alerts for SN Ia (steady rise, duration ~25 days)
    mjd0 = sn_ia_tns.discovery_date_mjd
    alert_stream.extend([
        ReplayAlertEpoch("sn1_ep1", sn_ia_target, mjd0, sn_ia_tns.ra_deg, sn_ia_tns.dec_deg, "g", 19.8, 0.12),
        ReplayAlertEpoch("sn1_ep2", sn_ia_target, mjd0 + 2.0, sn_ia_tns.ra_deg, sn_ia_tns.dec_deg, "r", 18.9, 0.08),
        ReplayAlertEpoch("sn1_ep3", sn_ia_target, mjd0 + 4.0, sn_ia_tns.ra_deg, sn_ia_tns.dec_deg, "g", 18.2, 0.05),
        ReplayAlertEpoch("sn1_ep4", sn_ia_target, mjd0 + 7.0, sn_ia_tns.ra_deg, sn_ia_tns.dec_deg, "r", 18.03, 0.04),
    ])

    # Alerts for SN II (slower rise, plateau)
    mjd0 = sn_ii_tns.discovery_date_mjd
    alert_stream.extend([
        ReplayAlertEpoch("sn2_ep1", sn_ii_target, mjd0, sn_ii_tns.ra_deg, sn_ii_tns.dec_deg, "r", 19.5, 0.10),
        ReplayAlertEpoch("sn2_ep2", sn_ii_target, mjd0 + 3.0, sn_ii_tns.ra_deg, sn_ii_tns.dec_deg, "g", 18.8, 0.07),
        ReplayAlertEpoch("sn2_ep3", sn_ii_target, mjd0 + 5.0, sn_ii_tns.ra_deg, sn_ii_tns.dec_deg, "r", 18.3, 0.05),
    ])

    # Alerts for TDE (very bright, long duration >60 days)
    mjd0 = tde_tns.discovery_date_mjd
    alert_stream.extend([
        ReplayAlertEpoch("tde_ep1", tde_target, mjd0, tde_tns.ra_deg, tde_tns.dec_deg, "g", 16.5, 0.04),
        ReplayAlertEpoch("tde_ep2", tde_target, mjd0 + 3.0, tde_tns.ra_deg, tde_tns.dec_deg, "g", 15.6, 0.03),
        ReplayAlertEpoch("tde_ep3", tde_target, mjd0 + 6.0, tde_tns.ra_deg, tde_tns.dec_deg, "r", 15.3, 0.02),
    ])

    # Alerts for Kilonova (extreme fast fade > 0.8 mag/day, red color)
    mjd0 = kn_tns.discovery_date_mjd
    alert_stream.extend([
        ReplayAlertEpoch("kn_ep1", kn_target, mjd0, kn_tns.ra_deg, kn_tns.dec_deg, "g", 17.47, 0.05),
        ReplayAlertEpoch("kn_ep2", kn_target, mjd0 + 0.5, kn_tns.ra_deg, kn_tns.dec_deg, "r", 18.05, 0.06),
        ReplayAlertEpoch("kn_ep3", kn_target, mjd0 + 1.0, kn_tns.ra_deg, kn_tns.dec_deg, "r", 18.85, 0.09),
    ])

    # Alerts for Galactic Star flare (sudden flare, fast decay)
    mjd0 = star_tns.discovery_date_mjd
    alert_stream.extend([
        ReplayAlertEpoch("star_ep1", star_target, mjd0, star_tns.ra_deg, star_tns.dec_deg, "g", 15.1, 0.03),
        ReplayAlertEpoch("star_ep2", star_target, mjd0 + 0.3, star_tns.ra_deg, star_tns.dec_deg, "r", 16.5, 0.05),
    ])

    return alert_stream, tns_records, gaia_records, host_galaxies


def test_blind_historical_streaming_replay_lead_time():
    """
    Verifies that the historical replay engine:
    1. Replays alerts strictly chronologically without lookahead bias.
    2. Masks TNS spectroscopic classifications before their official release MJD.
    3. Issues triage decisions with substantial positive lead times ahead of spectroscopic follow-up.
    """
    alert_stream, tns_records, gaia_records, host_galaxies = build_curated_historical_cohort()

    fabric = ScientificDataFabric(mode=OperatingMode.LIVE_INTELLIGENCE, offline_mode=True)
    fabric.tns.load_from_records(tns_records)
    for g in gaia_records:
        fabric.gaia.add_source(g)

    replay_engine = HistoricalStreamingReplayEngine(fabric=fabric)
    for h in host_galaxies:
        replay_engine.triage_engine.cross_matcher.add_host_galaxy(
            name=h["name"], ra=h["ra"], dec=h["dec"], redshift=h["redshift"]
        )

    eval_results = replay_engine.run_replay(alert_stream)

    assert len(eval_results) == len(alert_stream)

    # Check lead times for all trigger alerts (first alert of each target)
    targets_seen = set()
    for res in eval_results:
        if res.target_id not in targets_seen:
            targets_seen.add(res.target_id)
            # The initial triage alert must occur BEFORE classification release
            assert res.lead_time_days is not None
            assert res.lead_time_days > 0.0, f"Trigger alert for {res.target_id} had negative lead time {res.lead_time_days}!"
            assert res.predicted_urgency in [p.value for p in PriorityTier]

    # Verify Kilonova was elevated to CRITICAL_TOO upon confirmation of fast fade (kn_ep2) ahead of TNS confirmation
    kn_evals = [r for r in eval_results if r.target_id == "AT2017gfo_proxy"]
    assert len(kn_evals) >= 2
    assert kn_evals[1].predicted_urgency == PriorityTier.CRITICAL_TOO.value
    assert kn_evals[1].predicted_class == AnomalyClass.RAPID_OPTICAL_TRANSIENT_KN.value
    assert kn_evals[1].lead_time_days > 0.5


def test_blind_recovery_classification_metrics():
    """
    Evaluates multi-class classification accuracy, macro-F1, and Brier calibration
    across the historical cohort.
    """
    alert_stream, tns_records, gaia_records, host_galaxies = build_curated_historical_cohort()

    fabric = ScientificDataFabric(mode=OperatingMode.LIVE_INTELLIGENCE, offline_mode=True)
    fabric.tns.load_from_records(tns_records)
    for g in gaia_records:
        fabric.gaia.add_source(g)

    replay_engine = HistoricalStreamingReplayEngine(fabric=fabric)
    for h in host_galaxies:
        replay_engine.triage_engine.cross_matcher.add_host_galaxy(
            name=h["name"], ra=h["ra"], dec=h["dec"], redshift=h["redshift"]
        )

    eval_results = replay_engine.run_replay(alert_stream)

    # Extract final decision per target
    final_decisions: dict[str, Any] = {}
    for res in eval_results:
        final_decisions[res.target_id] = res

    y_true = []
    y_pred = []
    scores = []

    for target_id, res in final_decisions.items():
        assert res.is_correct is not None
        y_true.append(res.ground_truth_class)
        y_pred.append(res.predicted_class)
        scores.append(res.predicted_score)

    correct_count = sum(1 for target_id, res in final_decisions.items() if res.is_correct)
    accuracy = correct_count / len(final_decisions)

    # 1. Accuracy requirement: >= 80% on multi-class cohort
    assert accuracy >= 0.80, f"Classification accuracy {accuracy:.2f} failed to meet 0.80 threshold (y_pred={y_pred}, y_true={y_true})"

    # 2. Brier score calculation (mean squared difference between probability score and binary match)
    binary_matches = [1.0 if res.is_correct else 0.0 for res in final_decisions.values()]
    normalized_scores = [min(1.0, max(0.0, s / 100.0)) for s in scores]
    brier_score = float(np.mean([(p - y) ** 2 for p, y in zip(normalized_scores, binary_matches)]))

    # Brier score should be well-calibrated (<= 0.35)
    assert brier_score <= 0.35, f"Brier score {brier_score:.3f} indicates uncalibrated probabilities"


def test_zero_touch_quarantine_integrity_in_benchmark():
    """
    Verifies that zero benchmark samples fall within Gate 3 quarantined HEALPix pixels,
    and that attempting to inject a quarantined target triggers an immediate hard halt.
    """
    alert_stream, tns_records, gaia_records, _ = build_curated_historical_cohort()

    # 1. Audit every alert coordinate
    for alert in alert_stream:
        pix = QuarantineFirewall.compute_healpix_nside32(alert.ra_deg, alert.dec_deg)
        assert not QuarantineFirewall.is_quarantined_pixel(pix), f"Quarantine breach: alert {alert.alert_id} at ({alert.ra_deg}, {alert.dec_deg}) is in pixel {pix}"

    # 2. Test intentional injection of quarantined alert
    q_pix = next(iter(QuarantineFirewall.get_quarantined_pixels()))
    coord = QuarantineFirewall._HP.healpix_to_skycoord(q_pix)
    q_ra = float(coord.ra.deg)
    q_dec = float(coord.dec.deg)

    quarantined_stream = [
        ReplayAlertEpoch("breach_alert", "ZTF_BREACH", 60000.0, q_ra, q_dec, "g", 18.0, 0.05)
    ]

    fabric = ScientificDataFabric(mode=OperatingMode.LIVE_INTELLIGENCE, offline_mode=True)
    replay_engine = HistoricalStreamingReplayEngine(fabric=fabric)

    with pytest.raises(QuarantineViolationError, match="GATE 3 QUARANTINE VIOLATION"):
        replay_engine.run_replay(quarantined_stream)
