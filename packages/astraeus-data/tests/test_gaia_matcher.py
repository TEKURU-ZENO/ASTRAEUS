import pytest
from astraeus_data.crossmatch.gaia_matcher import (
    ProbabilisticGaiaMatcher,
    GaiaCandidate,
    MatchClassification,
)


def test_epoch_propagation_proper_motion():
    matcher = ProbabilisticGaiaMatcher()
    # High proper motion: 1000 mas/yr in Dec over 5 years -> 5000 mas = 5 arcsec = 5 / 3600 deg
    prop_ra, prop_dec = matcher.propagate_epoch(
        ra_deg=100.0,
        dec_deg=0.0,
        pmra_mas_yr=0.0,
        pmdec_mas_yr=1000.0,
        t_ref_jyr=2016.0,
        t_target_jyr=2021.0,
    )
    expected_delta_dec = 5.0 / 3600.0
    assert prop_dec == pytest.approx(expected_delta_dec, rel=1e-3)
    assert prop_ra == pytest.approx(100.0)


def test_secure_match_single_counterpart():
    matcher = ProbabilisticGaiaMatcher(prior_match_prob=0.50)
    candidate = GaiaCandidate(
        source_id=1234567890123456,
        ra_deg=150.00002,  # ~0.07 arcsec separation
        dec_deg=20.00001,
        ra_error_mas=1.5,
        dec_error_mas=1.2,
        pmra_mas_yr=0.5,
        pmdec_mas_yr=-0.2,
        phot_g_mean_mag=16.5,
        ref_epoch_jyr=2016.0,
    )
    # Target observation in 2016.1
    result = matcher.match(
        alert_ra=150.00002,
        alert_dec=20.00001,
        alert_ra_err_arcsec=0.05,
        alert_dec_err_arcsec=0.05,
        alert_epoch_jyr=2016.1,
        gaia_candidates=[candidate],
        local_density_sources_per_sq_arcsec=0.001,  # Sparse background
        search_radius_arcsec=2.0,
    )

    assert result.classification == MatchClassification.SECURE_MATCH
    assert result.best_source_id == 1234567890123456
    assert result.posterior_match_prob >= 0.90
    assert result.chance_coincidence_prob < 0.01


def test_ambiguous_match_multiple_counterparts():
    matcher = ProbabilisticGaiaMatcher(prior_match_prob=0.50)
    # Two candidates in a dense field close to alert
    cand1 = GaiaCandidate(
        source_id=111,
        ra_deg=150.00005,
        dec_deg=20.00005,
        ra_error_mas=10.0,
        dec_error_mas=10.0,
    )
    cand2 = GaiaCandidate(
        source_id=222,
        ra_deg=150.00008,
        dec_deg=20.00008,
        ra_error_mas=10.0,
        dec_error_mas=10.0,
    )
    result = matcher.match(
        alert_ra=150.00006,
        alert_dec=20.00006,
        alert_ra_err_arcsec=0.10,
        alert_dec_err_arcsec=0.10,
        alert_epoch_jyr=2016.0,
        gaia_candidates=[cand1, cand2],
        local_density_sources_per_sq_arcsec=0.05,  # Dense field
        search_radius_arcsec=2.0,
    )

    assert result.classification == MatchClassification.AMBIGUOUS_MATCH
    assert len(result.alternative_candidates) > 0


def test_no_reliable_match():
    matcher = ProbabilisticGaiaMatcher()
    result = matcher.match(
        alert_ra=50.0,
        alert_dec=-10.0,
        alert_ra_err_arcsec=0.1,
        alert_dec_err_arcsec=0.1,
        alert_epoch_jyr=2021.0,
        gaia_candidates=[],
        local_density_sources_per_sq_arcsec=0.001,
        search_radius_arcsec=2.0,
    )
    assert result.classification == MatchClassification.NO_RELIABLE_MATCH
    assert result.best_source_id is None
