import pytest
from astraeus_core.epistemic import EpistemicTier
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.observations import ObservationRecord, CalibratedMeasurement
from astraeus_core.provenance import ProvEntity
from astraeus_core.conformal import ConformalPredictionInterval
from astraeus_core.scoring import CompositeAnomalyScore
from astraeus_core.entity import AstronomicalEntity
from astraeus_core.exceptions import EpistemicViolationError, ProvenanceMissingError


@pytest.fixture
def sample_coords():
    return CelestialCoordinates(
        ra_deg=210.825,
        dec_deg=54.348,
        uncertainty_mas={"ra": 0.05, "dec": 0.04},
    )


@pytest.fixture
def sample_observation():
    meas = CalibratedMeasurement(
        quantity_name="flux_density",
        value=18.42,
        uncertainty=0.03,
        unit="mag_AB",
        filter_band="ZTF_g",
    )
    return ObservationRecord(
        observation_id="astraeus:obs:ztf:55001",
        survey="ZTF",
        instrument="ZTF-Camera",
        filter_band="ZTF_g",
        mjd=59123.4,
        calibrated_flux=meas,
        ivoa_provenance_id="act:ztf:pipeline_v2",
    )


def test_observed_entity_valid(sample_coords, sample_observation):
    prov = ProvEntity(
        entity_id="astraeus:entity:sn_candidate_01",
        entity_type="astronomical_object",
        was_generated_by="act:ingest:01",
    )
    entity = AstronomicalEntity(
        entity_id="astraeus:entity:sn_candidate_01",
        entity_type="transient_candidate",
        epistemic_tier=EpistemicTier.OBSERVED,
        coordinates=sample_coords,
        observations=[sample_observation],
        provenance=prov,
    )
    assert entity.entity_id == "astraeus:entity:sn_candidate_01"
    assert entity.epistemic_tier == EpistemicTier.OBSERVED


def test_observed_entity_fails_without_observations(sample_coords):
    prov = ProvEntity(
        entity_id="astraeus:entity:empty_obs",
        entity_type="astronomical_object",
    )
    with pytest.raises(EpistemicViolationError, match="contains no observation records"):
        AstronomicalEntity(
            entity_id="astraeus:entity:empty_obs",
            entity_type="transient_candidate",
            epistemic_tier=EpistemicTier.OBSERVED,
            coordinates=sample_coords,
            observations=[],
            provenance=prov,
        )


def test_inferred_entity_requires_generating_activity(sample_coords):
    prov_no_activity = ProvEntity(
        entity_id="astraeus:entity:inferred_01",
        entity_type="inferred_properties",
    )
    with pytest.raises(ProvenanceMissingError, match="must record 'was_generated_by'"):
        AstronomicalEntity(
            entity_id="astraeus:entity:inferred_01",
            entity_type="galaxy",
            epistemic_tier=EpistemicTier.INFERRED,
            coordinates=sample_coords,
            provenance=prov_no_activity,
        )


def test_conformal_prediction_coverage():
    interval = ConformalPredictionInterval(
        target_property="photometric_redshift",
        point_estimate=0.21,
        lower_bound=0.18,
        upper_bound=0.24,
        coverage_level=0.90,
        latent_density_sigma=0.08,
    )
    assert interval.interval_width == pytest.approx(0.06)
    assert interval.contains(0.20)
    assert not interval.contains(0.35)


def test_composite_anomaly_score():
    score = CompositeAnomalyScore(
        entity_id="candidate:transient:001",
        mahalanobis_distance=3.5,
        epistemic_uncertainty=1.2,
        physical_residual=4.0,
        physical_model_applied="bazin_transient",
        alpha=0.4,
        beta=0.3,
        gamma=0.3,
    )
    # 0.4*3.5 + 0.3*1.2 + 0.3*4.0 = 1.4 + 0.36 + 1.2 = 2.96
    assert score.composite_score == pytest.approx(2.96)
    assert score.is_candidate_for_followup(threshold=2.5)


def test_normalized_anomaly_score_and_uncertainty_decomposition():
    from astraeus_core.conformal import UncertaintyDecomposition

    decomp = UncertaintyDecomposition(
        aleatoric=0.03,
        epistemic=0.08,
        calibration_margin=0.02,
        distribution_shift=0.05,
    )
    interval = ConformalPredictionInterval(
        target_property="peak_magnitude",
        point_estimate=17.2,
        lower_bound=16.8,
        upper_bound=17.6,
        coverage_level=0.95,
        uncertainty_decomposition=decomp,
    )
    assert interval.uncertainty_decomposition.aleatoric == 0.03
    assert interval.uncertainty_decomposition.distribution_shift == 0.05

    score_norm = CompositeAnomalyScore(
        entity_id="candidate:transient:002",
        mahalanobis_distance=4.2,
        epistemic_uncertainty=2.1,
        physical_residual=1.5,
        mahalanobis_normalized=0.85,
        uncertainty_normalized=0.90,
        residual_normalized=0.40,
        physical_model_applied="arnett_sn_ia",
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
    )
    # 0.5*0.85 + 0.3*0.90 + 0.2*0.40 = 0.425 + 0.27 + 0.08 = 0.775
    assert score_norm.composite_score == pytest.approx(0.775)


def test_promote_to_inferred_preserves_lineage(sample_coords, sample_observation):
    prov = ProvEntity(
        entity_id="astraeus:entity:sn_01",
        entity_type="astronomical_object",
        was_generated_by="act:ingest:01",
    )
    obs_entity = AstronomicalEntity(
        entity_id="astraeus:entity:sn_01",
        entity_type="transient_candidate",
        epistemic_tier=EpistemicTier.OBSERVED,
        coordinates=sample_coords,
        observations=[sample_observation],
        provenance=prov,
    )

    interval = ConformalPredictionInterval(
        target_property="photometric_redshift",
        point_estimate=0.15,
        lower_bound=0.12,
        upper_bound=0.18,
    )
    inferred = obs_entity.promote_to_inferred(
        new_activity_id="act:model_inference:astromer_v1",
        agent_id="agent:astrophysics_model",
        inferences={"z_phot": interval},
    )

    assert inferred.epistemic_tier == EpistemicTier.INFERRED
    assert inferred.provenance.used_entities == ["astraeus:entity:sn_01"]
    assert inferred.provenance.was_generated_by == "act:model_inference:astromer_v1"
