"""
ASTRAEUS Synthetic Benchmark Generator.
Generates controlled synthetic astrophysical populations (SNe Ia, FBOTs, RR Lyrae, artifacts)
with ground-truth parameters for leakage and anomaly verification.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import uuid

from astraeus_core.epistemic import EpistemicTier
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.observations import ObservationRecord, CalibratedMeasurement
from astraeus_core.provenance import ProvEntity, create_provenance_record
from astraeus_core.entity import AstronomicalEntity


def generate_bazin_lightcurve(
    t: np.ndarray,
    t0: float,
    a: float = 1.0,
    b: float = 0.0,
    tau_rise: float = 5.0,
    tau_fall: float = 25.0
) -> np.ndarray:
    """Standard Bazin function parameterizing optical transients."""
    return a * (np.exp(-(t - t0) / tau_fall) / (1.0 + np.exp(-(t - t0) / tau_rise))) + b


def generate_synthetic_entity(
    entity_type: str = "supernova_ia",
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    seed: Optional[int] = None
) -> AstronomicalEntity:
    """
    Synthesizes an AstronomicalEntity with realistic multi-epoch observations.
    Supported types: 'supernova_ia', 'fbot_anomaly', 'rr_lyrae', 'detector_artifact'.
    """
    rng = np.random.default_rng(seed)

    if ra_deg is None:
        ra_deg = float(rng.uniform(0.0, 359.99))
    if dec_deg is None:
        dec_deg = float(rng.uniform(-89.9, 89.9))

    coords = CelestialCoordinates(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        uncertainty_mas={"ra": float(rng.uniform(0.01, 0.1)), "dec": float(rng.uniform(0.01, 0.1))},
    )
    coords.compute_healpix_index(nside=32)

    uid = str(uuid.uuid4())[:8]
    entity_id = f"astraeus:synthetic:{entity_type}:{uid}"
    activity_id = f"act:sim_generator:{uid}"

    # Generate multi-epoch photometry
    epochs = np.sort(rng.uniform(59000.0, 59100.0, size=20))
    observations: List[ObservationRecord] = []

    if entity_type == "supernova_ia":
        t0 = 59030.0
        fluxes = 18.0 - 2.5 * np.log10(np.clip(generate_bazin_lightcurve(epochs, t0, a=2.5, tau_rise=4.0, tau_fall=22.0), 1e-4, 10.0))
        fluxes += rng.normal(0, 0.05, size=len(epochs))
    elif entity_type == "fbot_anomaly":
        # Fast Blue Optical Transient: extreme rise (1-2 days) and very steep decay
        t0 = 59020.0
        fluxes = 17.5 - 2.5 * np.log10(np.clip(generate_bazin_lightcurve(epochs, t0, a=4.0, tau_rise=1.2, tau_fall=6.0), 1e-4, 10.0))
        fluxes += rng.normal(0, 0.08, size=len(epochs))
    elif entity_type == "rr_lyrae":
        period = 0.55  # days
        fluxes = 16.5 + 0.4 * np.sin(2 * np.pi * (epochs - 59000.0) / period) + rng.normal(0, 0.03, size=len(epochs))
    elif entity_type == "detector_artifact":
        # Flat baseline with single 1-epoch spike
        fluxes = np.full_like(epochs, 21.0)
        spike_idx = rng.integers(0, len(epochs))
        fluxes[spike_idx] = 14.2  # unphysical single-epoch jump
    else:
        fluxes = rng.normal(19.0, 0.5, size=len(epochs))

    for i, (mjd, mag) in enumerate(zip(epochs, fluxes)):
        band = "ZTF_g" if i % 2 == 0 else "ZTF_r"
        obs_id = f"astraeus:obs:{uid}_{i}"
        flag = 1 if (entity_type == "detector_artifact" and i == spike_idx) else 0

        meas = CalibratedMeasurement(
            quantity_name="apparent_magnitude",
            value=float(mag),
            uncertainty=0.04,
            unit="mag_AB",
            filter_band=band,
            quality_flags=flag,
            epistemic_tier=EpistemicTier.OBSERVED,
        )

        obs = ObservationRecord(
            observation_id=obs_id,
            survey="ZTF_Synthetic",
            instrument="ZTF-Camera-Surrogate",
            filter_band=band,
            mjd=float(mjd),
            exposure_time_sec=30.0,
            calibrated_flux=meas,
            quality_flags=flag,
            ivoa_provenance_id=activity_id,
        )
        observations.append(obs)

    prov = create_provenance_record(
        entity_id=entity_id,
        entity_type=entity_type,
        activity_id=activity_id,
        agent_id="agent:synthetic_generator",
    )

    return AstronomicalEntity(
        entity_id=entity_id,
        entity_type=entity_type,
        epistemic_tier=EpistemicTier.OBSERVED,
        coordinates=coords,
        observations=observations,
        available_modalities=["photometric_time_series"],
        provenance=prov,
    )
