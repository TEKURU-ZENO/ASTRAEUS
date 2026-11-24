"""
ASTRAEUS Phase 7 WS7.2 Intelligent Triage and Prioritization Test Suite.
Tests:
1. Astrometric separation & angular diameter distance cosmology calculations
2. Contextual catalogue cross-matching (NED/GLADE+ host galaxies, Gaia reference stars)
3. Photometric feature extraction (rise/fade rates, colors, decay timescale tau)
4. Rapid optical transient (Kilonova) classification and ToO urgency
5. Superluminous supernova / TDE classification and host offset
6. Multi-messenger GW counterpart triage
7. Stellar flare / variable star deprioritization
8. Thread-safe priority queue heap ordering
9. End-to-end ingestion-to-triage pipeline integration
10. Quarantine firewall enforcement in triage engine
"""

import pytest
import math
from datetime import datetime, timezone

from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent
from astraeus_data.triage.event_triage_engine import (
    PriorityTier,
    AnomalyClass,
    HostGalaxyContext,
    ReferenceStarContext,
    PhotometricFeatures,
    TriageDecision,
    CatalogCrossMatcher,
    PhotometricFeatureExtractor,
    HeuristicAnomalyDetector,
    PrioritizationQueue,
    EventTriageEngine,
    angular_separation_arcsec,
    angular_diameter_distance_kpc,
)


def test_angular_separation_and_cosmological_distance():
    # 1 arcmin separation along RA at Dec = 0
    sep = angular_separation_arcsec(10.0, 0.0, 10.0 + 1.0 / 60.0, 0.0)
    assert pytest.approx(sep, rel=1e-3) == 60.0

    # Angular diameter distance at z = 0.05 (~200 Mpc comoving)
    dA_kpc = angular_diameter_distance_kpc(0.05)
    assert 180000.0 < dA_kpc < 220000.0  # ~200,000 kpc


def test_catalog_cross_matcher_host_and_star():
    matcher = CatalogCrossMatcher()
    matcher.add_host_galaxy(
        name="NGC 4993",
        ra=197.45037,
        dec=-23.63789,
        redshift=0.00973,
        log_mstar=10.6,
        sfr=0.01,
        catalog="GLADE+"
    )
    matcher.add_reference_star(
        source_id="Gaia-DR3-9999",
        ra=150.0,
        dec=20.0,
        pm_mas_yr=15.2,
        parallax_mas=4.1,
        is_variable=True
    )

    # Query near NGC 4993 with 10 arcsec offset
    d_ra = 10.0 / (3600.0 * math.cos(math.radians(-23.63789)))
    host_ctx = matcher.match_host(197.45037 + d_ra, -23.63789)
    assert host_ctx.matched is True
    assert host_ctx.galaxy_name == "NGC 4993"
    assert host_ctx.redshift_z == pytest.approx(0.00973)
    assert host_ctx.angular_separation_arcsec == pytest.approx(10.0, rel=1e-2)
    assert host_ctx.physical_offset_kpc is not None
    assert 1.0 < host_ctx.physical_offset_kpc < 3.0
    assert host_ctx.chance_coincidence_prob < 0.05

    # Query near Gaia star
    star_ctx = matcher.match_star(150.0 + 0.2 / 3600.0, 20.0)
    assert star_ctx.matched_star is True
    assert star_ctx.source_id == "Gaia-DR3-9999"
    assert star_ctx.proper_motion_mas_yr == pytest.approx(15.2)
    assert star_ctx.is_known_variable is True


def test_photometric_feature_extractor():
    # Construct synthetic rapid fading lightcurve with 3 bands
    points = [
        {"mjd": 60000.0, "mag": 19.5, "filter_band": "g"},
        {"mjd": 60000.0, "mag": 19.3, "filter_band": "r"},
        {"mjd": 60000.5, "mag": 18.0, "filter_band": "g"},  # peak g
        {"mjd": 60000.5, "mag": 18.1, "filter_band": "r"},  # peak r
        {"mjd": 60001.5, "mag": 19.2, "filter_band": "g"},  # fading, reddening
        {"mjd": 60001.5, "mag": 18.7, "filter_band": "r"},
        {"mjd": 60002.5, "mag": 20.8, "filter_band": "g"},
        {"mjd": 60002.5, "mag": 19.5, "filter_band": "r"},
    ]

    features = PhotometricFeatureExtractor.extract(points, host_z=0.01)
    assert features.num_points == 8
    assert "g" in features.bands_observed and "r" in features.bands_observed
    assert features.peak_mag == 18.0
    assert features.peak_mjd == 60000.5
    assert features.time_baseline_days == pytest.approx(2.5)
    assert features.rise_rate_mag_per_day > 0.0
    assert features.fade_rate_mag_per_day > 0.5  # rapid fade
    assert features.timescale_tau_hours is not None
    assert features.timescale_tau_hours < 48.0  # tau < 48 hours
    assert features.color_rate_dgr_dt is not None
    assert features.color_rate_dgr_dt > 0.0  # reddening


def test_fast_fading_kilonova_classification_and_urgency():
    event = UnifiedScientificEvent(
        event_id="ASTRAEUS-KN-TEST-001",
        survey_origin="RUBIN",
        ra_deg=197.45,
        dec_deg=-23.64,
        healpix_pixel_nside32=100,
        event_time_mjd=60001.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="abcd1234hash",
        quarantine_cleared=True
    )

    # Fast fading with rapid reddening
    features = PhotometricFeatures(
        num_points=6,
        bands_observed=["g", "r", "i"],
        time_baseline_days=2.0,
        peak_mag=17.5,
        peak_mjd=60000.0,
        latest_mag=19.2,
        fade_rate_mag_per_day=0.85,
        timescale_tau_hours=30.6,
        color_g_minus_r=0.6,
        color_rate_dgr_dt=0.45
    )

    host = HostGalaxyContext(
        matched=True,
        galaxy_name="NGC 4993",
        redshift_z=0.0097,
        angular_separation_arcsec=10.2,
        physical_offset_kpc=2.1,
        chance_coincidence_prob=0.001
    )
    star = ReferenceStarContext(matched_star=False)

    anomaly_class, score, is_too, justification = HeuristicAnomalyDetector.classify(
        event=event,
        features=features,
        host=host,
        star=star
    )

    assert anomaly_class == AnomalyClass.RAPID_OPTICAL_TRANSIENT_KN
    assert score >= 85.0
    assert is_too is True
    assert "Rapid decay timescale" in justification
    assert "lanthanide reddening" in justification


def test_superluminous_sn_classification():
    event = UnifiedScientificEvent(
        event_id="ASTRAEUS-SLSN-TEST-001",
        survey_origin="RUBIN",
        ra_deg=220.0,
        dec_deg=15.0,
        healpix_pixel_nside32=200,
        event_time_mjd=60005.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="slsn1234hash",
        quarantine_cleared=True
    )

    features = PhotometricFeatures(
        num_points=12,
        bands_observed=["g", "r", "i", "z"],
        time_baseline_days=25.0,
        peak_mag=16.2,
        peak_mjd=59995.0,
        latest_mag=16.8,
        fade_rate_mag_per_day=0.06,
        estimated_abs_mag=-21.8  # Superluminous
    )

    host = HostGalaxyContext(
        matched=True,
        galaxy_name="SDSS J1440+1500",
        redshift_z=0.18,
        physical_offset_kpc=0.4,  # Nuclear alignment
        chance_coincidence_prob=0.002
    )
    star = ReferenceStarContext(matched_star=False)

    anomaly_class, score, is_too, justification = HeuristicAnomalyDetector.classify(
        event=event,
        features=features,
        host=host,
        star=star
    )

    assert anomaly_class == AnomalyClass.SUPERLUMINOUS_SN_TDE
    assert score >= 85.0
    assert is_too is True
    assert "Nuclear alignment" in justification


def test_gravitational_wave_multi_messenger_triage():
    event = UnifiedScientificEvent(
        event_id="ASTRAEUS-GW-TEST-001",
        survey_origin="LVK_GW",
        ra_deg=180.0,
        dec_deg=10.0,
        healpix_pixel_nside32=300,
        event_time_mjd=60000.0,
        photometry_points=[],
        event_metadata={"gw_coincidence": True, "prob_bns": 0.95},
        raw_payload_hash="gwhash1234",
        quarantine_cleared=True,
        is_too_promising=True
    )

    features = PhotometricFeatures()
    host = HostGalaxyContext(matched=False)
    star = ReferenceStarContext(matched_star=False)

    anomaly_class, score, is_too, justification = HeuristicAnomalyDetector.classify(
        event=event,
        features=features,
        host=host,
        star=star
    )

    assert anomaly_class == AnomalyClass.MULTI_MESSENGER_GW_COUNTERPART
    assert score >= 95.0
    assert is_too is True


def test_stellar_flare_deprioritization():
    event = UnifiedScientificEvent(
        event_id="ASTRAEUS-STAR-TEST-001",
        survey_origin="RUBIN",
        ra_deg=100.0,
        dec_deg=-10.0,
        healpix_pixel_nside32=400,
        event_time_mjd=60000.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="starhash1234",
        quarantine_cleared=True
    )

    features = PhotometricFeatures(num_points=3, fade_rate_mag_per_day=0.1)
    host = HostGalaxyContext(matched=False)
    star = ReferenceStarContext(
        matched_star=True,
        source_id="Gaia-DR3-FlareStar",
        separation_arcsec=0.15,
        proper_motion_mas_yr=24.5,
        is_known_variable=True
    )

    anomaly_class, score, is_too, justification = HeuristicAnomalyDetector.classify(
        event=event,
        features=features,
        host=host,
        star=star
    )

    assert anomaly_class == AnomalyClass.STELLAR_FLARE_VARIABLE
    assert score <= 20.0
    assert is_too is False
    assert "High proper motion" in justification


def test_prioritization_queue_ordering():
    queue = PrioritizationQueue()

    d_low = TriageDecision(
        event_id="LOW",
        survey_origin="RUBIN",
        priority_score=25.0,
        priority_tier=PriorityTier.LOW_PRIORITY_ARCHIVE,
        primary_anomaly_class=AnomalyClass.STELLAR_FLARE_VARIABLE,
        is_too_urgent=False,
        features=PhotometricFeatures(),
        host_context=HostGalaxyContext(),
        star_context=ReferenceStarContext(),
        reasoning_summary="",
        recommended_action="",
        triage_timestamp_iso=datetime.now(timezone.utc).isoformat(),
        raw_payload_hash="h1"
    )

    d_high = TriageDecision(
        event_id="HIGH",
        survey_origin="RUBIN",
        priority_score=75.0,
        priority_tier=PriorityTier.HIGH_PRIORITY,
        primary_anomaly_class=AnomalyClass.EXTRAGALACTIC_SN,
        is_too_urgent=False,
        features=PhotometricFeatures(),
        host_context=HostGalaxyContext(),
        star_context=ReferenceStarContext(),
        reasoning_summary="",
        recommended_action="",
        triage_timestamp_iso=datetime.now(timezone.utc).isoformat(),
        raw_payload_hash="h2"
    )

    d_critical = TriageDecision(
        event_id="CRITICAL",
        survey_origin="RUBIN",
        priority_score=94.0,
        priority_tier=PriorityTier.CRITICAL_TOO,
        primary_anomaly_class=AnomalyClass.RAPID_OPTICAL_TRANSIENT_KN,
        is_too_urgent=True,
        features=PhotometricFeatures(),
        host_context=HostGalaxyContext(),
        star_context=ReferenceStarContext(),
        reasoning_summary="",
        recommended_action="",
        triage_timestamp_iso=datetime.now(timezone.utc).isoformat(),
        raw_payload_hash="h3"
    )

    queue.push(d_low)
    queue.push(d_critical)
    queue.push(d_high)

    assert len(queue) == 3
    # Peek should be critical
    top = queue.peek()
    assert top is not None
    assert top.event_id == "CRITICAL"

    # Pop order: CRITICAL -> HIGH -> LOW
    assert queue.pop().event_id == "CRITICAL"
    assert queue.pop().event_id == "HIGH"
    assert queue.pop().event_id == "LOW"
    assert len(queue) == 0


def test_event_triage_engine_end_to_end():
    matcher = CatalogCrossMatcher()
    matcher.add_host_galaxy("NGC 4993", 197.45, -23.64, 0.0097)

    engine = EventTriageEngine(cross_matcher=matcher)

    event = UnifiedScientificEvent(
        event_id="RUBIN-LIVE-001",
        survey_origin="RUBIN",
        ra_deg=197.4501,
        dec_deg=-23.6401,
        healpix_pixel_nside32=500,
        event_time_mjd=60002.0,
        photometry_points=[
            {"mjd": 60000.0, "mag": 19.0, "filter_band": "g"},
            {"mjd": 60000.0, "mag": 18.8, "filter_band": "r"},
            {"mjd": 60001.0, "mag": 17.5, "filter_band": "g"},
            {"mjd": 60001.0, "mag": 17.6, "filter_band": "r"},
            {"mjd": 60002.0, "mag": 19.3, "filter_band": "g"},
            {"mjd": 60002.0, "mag": 18.4, "filter_band": "r"},
        ],
        event_metadata={"filter": "r"},
        raw_payload_hash="hash998877",
        quarantine_cleared=True
    )

    decision = engine.triage_event(event)

    assert decision.event_id == "RUBIN-LIVE-001"
    assert decision.priority_tier == PriorityTier.CRITICAL_TOO
    assert decision.primary_anomaly_class == AnomalyClass.RAPID_OPTICAL_TRANSIENT_KN
    assert decision.host_context.matched is True
    assert decision.host_context.galaxy_name == "NGC 4993"
    assert len(engine.queue) == 1


def test_quarantine_enforcement_in_triage():
    engine = EventTriageEngine()

    quarantined_event = UnifiedScientificEvent(
        event_id="QUARANTINED-EV-001",
        survey_origin="RUBIN",
        ra_deg=100.0,
        dec_deg=10.0,
        healpix_pixel_nside32=500,
        event_time_mjd=60000.0,
        photometry_points=[],
        event_metadata={},
        raw_payload_hash="forbidden_hash",
        quarantine_cleared=False  # Gate 3 Quarantine violation
    )

    with pytest.raises(ValueError, match="failed Gate 3 quarantine firewall"):
        engine.triage_event(quarantined_event)
