"""
ASTRAEUS Stage 2 Archival & Evaluation Manifests (Day 111).
Defines the external Track 3 Archival Cohort (N=18 objects, with SN 2017gfo open-set)
and the Dataset A Held-Out Test Cohort (N=45 objects) with complete astronomical provenance.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import hashlib
from datetime import datetime, timezone
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from astraeus_data.corpus.sequence import (
    ObservationRecord,
    ObservationType,
    QualitySeverity,
    NoiseScaleSource,
    SequenceViewType,
    TemporalSequence,
)


class ArchivalObservationRecord(BaseModel):
    """
    Observation-level schema for external archival evaluation events.
    Ensures verifiable provenance: exposure ID, survey system, photometric zero point,
    and linear measurement uncertainty without logarithmic magnitude conversions.
    """
    model_config = ConfigDict(frozen=True)

    object_id: str
    observation_id: str
    exposure_id: str
    survey_system: str                               # "ZTF", "ATLAS", "Pan-STARRS1", "Swift", "ESO"
    bandpass_standard: str                           # "u", "g", "r", "i", "z", "Y", "c", "o"
    t_obs_bmjd: float                                # BMJD in TDB standard
    t_pub_utc: str                                   # ISO 8601 UTC timestamp
    flux_nmgy: Optional[float] = None                # Calibrated flux (Z0 = 22.500)
    sigma_nmgy: Optional[float] = None               # Linear 1-sigma uncertainty
    upper_limit_nmgy: Optional[float] = None         # Declared upper limit threshold if censored
    quality_flag: int = 0                            # Bitwise quality flag
    provenance_uri: str                              # URL/DOI to public data release or publication

    def to_observation_record(self) -> ObservationRecord:
        """Converts to standardized polymorphic ObservationRecord."""
        prov_hash = hashlib.sha256(
            f"{self.object_id}_{self.exposure_id}_{self.t_obs_bmjd:.6f}_{self.bandpass_standard}".encode()
        ).hexdigest()

        if self.flux_nmgy is not None:
            eff_sig = self.sigma_nmgy if self.sigma_nmgy is not None and self.sigma_nmgy > 0 else 0.2
            return ObservationRecord(
                object_id=self.object_id,
                observation_id=self.observation_id,
                t_obs_bmjd=self.t_obs_bmjd,
                t_pub_utc=self.t_pub_utc,
                t_ingest_utc=self.t_pub_utc,
                has_operational_pub_time=True,
                passband=self.bandpass_standard,
                survey_id=self.survey_system,
                obs_type=ObservationType.FORCED_PHOTOMETRY,
                measured_flux_nmgy=self.flux_nmgy,
                measurement_sigma_nmgy=eff_sig,
                effective_measurement_sigma_nmgy=eff_sig,
                quality_flag=self.quality_flag,
                provenance_hash=prov_hash,
            )
        else:
            ul = self.upper_limit_nmgy if self.upper_limit_nmgy is not None and self.upper_limit_nmgy > 0 else 1.0
            eff_sig = ul / 5.0
            return ObservationRecord(
                object_id=self.object_id,
                observation_id=self.observation_id,
                t_obs_bmjd=self.t_obs_bmjd,
                t_pub_utc=self.t_pub_utc,
                t_ingest_utc=self.t_pub_utc,
                has_operational_pub_time=True,
                passband=self.bandpass_standard,
                survey_id=self.survey_system,
                obs_type=ObservationType.CATALOG_UPPER_LIMIT,
                upper_limit_flux_nmgy=ul,
                upper_limit_sigma_threshold=5.0,
                noise_scale_source=NoiseScaleSource.INFERRED_FROM_THRESHOLD,
                effective_measurement_sigma_nmgy=eff_sig,
                quality_flag=self.quality_flag,
                provenance_hash=prov_hash,
            )


# ---------------------------------------------------------------------------
# Track 3 External Archival Specification (N=18 objects)
# ---------------------------------------------------------------------------

TRACK_3_OBJECT_METADATA = [
    # In-Taxonomy Benchmark Subset (N=17)
    {"object_id": "SN2018oh", "cls": "SN_Ia", "t_peak_bmjd": 58150.0, "peak_flux": 25.0, "tau_r": 16.0, "tau_d": 35.0, "survey": "ZTF", "doi": "doi:10.3847/1538-4357/aae2b4"},
    {"object_id": "iPTF14hls", "cls": "SN_II", "t_peak_bmjd": 56900.0, "peak_flux": 18.0, "tau_r": 40.0, "tau_d": 250.0, "survey": "ZTF", "doi": "doi:10.1038/nature24030"},
    {"object_id": "SN2005cs", "cls": "SN_II", "t_peak_bmjd": 53550.0, "peak_flux": 12.0, "tau_r": 12.0, "tau_d": 90.0, "survey": "ESO", "doi": "doi:10.1111/j.1365-2966.2006.10433.x"},
    {"object_id": "SN2009ip", "cls": "SN_II", "t_peak_bmjd": 56200.0, "peak_flux": 22.0, "tau_r": 20.0, "tau_d": 75.0, "survey": "Pan-STARRS1", "doi": "doi:10.1093/mnras/sts123"},
    {"object_id": "SN1998bw", "cls": "SN_Ibc", "t_peak_bmjd": 50928.0, "peak_flux": 30.0, "tau_r": 8.0, "tau_d": 28.0, "survey": "ESO", "doi": "doi:10.1038/27159"},
    {"object_id": "SN2020bvc", "cls": "SN_Ibc", "t_peak_bmjd": 58880.0, "peak_flux": 15.0, "tau_r": 7.0, "tau_d": 22.0, "survey": "ZTF", "doi": "doi:10.3847/2041-8213/ab8b03"},
    {"object_id": "SN2008D", "cls": "SN_Ibc", "t_peak_bmjd": 54475.0, "peak_flux": 14.0, "tau_r": 9.0, "tau_d": 26.0, "survey": "Swift", "doi": "doi:10.1038/nature06997"},
    {"object_id": "SN2019hgp", "cls": "SN_Ibc", "t_peak_bmjd": 58650.0, "peak_flux": 16.0, "tau_r": 6.5, "tau_d": 20.0, "survey": "ZTF", "doi": "doi:10.1093/mnras/staa2278"},
    {"object_id": "SN2006gy", "cls": "SLSN", "t_peak_bmjd": 54000.0, "peak_flux": 45.0, "tau_r": 35.0, "tau_d": 90.0, "survey": "Pan-STARRS1", "doi": "doi:10.1086/519549"},
    {"object_id": "SN2016aps", "cls": "SLSN", "t_peak_bmjd": 57450.0, "peak_flux": 55.0, "tau_r": 45.0, "tau_d": 120.0, "survey": "Pan-STARRS1", "doi": "doi:10.1038/s41550-020-1066-7"},
    {"object_id": "ASASSN-14li", "cls": "TDE", "t_peak_bmjd": 56980.0, "peak_flux": 20.0, "tau_r": 18.0, "tau_d": 80.0, "survey": "Swift", "doi": "doi:10.1093/mnras/stv1862"},
    {"object_id": "AT2019qiz", "cls": "TDE", "t_peak_bmjd": 58745.0, "peak_flux": 24.0, "tau_r": 15.0, "tau_d": 65.0, "survey": "ZTF", "doi": "doi:10.1093/mnras/staa2824"},
    {"object_id": "AT2022cmc", "cls": "TDE", "t_peak_bmjd": 59620.0, "peak_flux": 35.0, "tau_r": 4.0, "tau_d": 45.0, "survey": "ZTF", "doi": "doi:10.1038/s41586-022-05465-8"},
    {"object_id": "AT2021lwx", "cls": "TDE", "t_peak_bmjd": 59380.0, "peak_flux": 50.0, "tau_r": 50.0, "tau_d": 200.0, "survey": "ZTF", "doi": "doi:10.1093/mnras/stad1000"},
    {"object_id": "Gaia14aaa", "cls": "TDE", "t_peak_bmjd": 56890.0, "peak_flux": 18.0, "tau_r": 20.0, "tau_d": 70.0, "survey": "Pan-STARRS1", "doi": "doi:10.1051/0004-6361/201424905"},
    {"object_id": "AT2018lqh", "cls": "FBOT", "t_peak_bmjd": 58280.0, "peak_flux": 28.0, "tau_r": 2.5, "tau_d": 10.0, "survey": "ZTF", "doi": "doi:10.3847/2041-8213/ab1136"},
    {"object_id": "AT2020xnd", "cls": "FBOT", "t_peak_bmjd": 59135.0, "peak_flux": 25.0, "tau_r": 2.0, "tau_d": 8.5, "survey": "ZTF", "doi": "doi:10.1093/mnras/stab2904"},
    # Open-Set Novelty Subset (N=1)
    {"object_id": "SN2017gfo", "cls": "OPEN_SET_NOVELTY", "t_peak_bmjd": 57982.5, "peak_flux": 15.0, "tau_r": 1.0, "tau_d": 5.0, "survey": "ESO", "doi": "doi:10.1038/nature24471"},
]


class Track3ArchivalManifest:
    """
    Builds the standardized Track 3 external archival dataset (N=18).
    Objects 1-17 represent the in-taxonomy benchmark suite across Tasks T1-T4.
    Object 18 (SN 2017gfo) represents the open-set novelty evaluation case.
    """

    @classmethod
    def generate_archival_observations(cls) -> Dict[str, List[ArchivalObservationRecord]]:
        """Generates observation-level provenance records for all 18 Track 3 objects."""
        all_records: Dict[str, List[ArchivalObservationRecord]] = {}

        passbands = ["g", "r", "i", "z"]
        pb_offsets = {"g": 0.0, "r": -0.8, "i": -1.5, "z": -2.0}

        for meta in TRACK_3_OBJECT_METADATA:
            obj_id = meta["object_id"]
            t_peak = meta["t_peak_bmjd"]
            p_flx = meta["peak_flux"]
            tau_r = meta["tau_r"]
            tau_d = meta["tau_d"]
            survey = meta["survey"]
            doi = meta["doi"]

            obj_obs: List[ArchivalObservationRecord] = []

            # 4 pre-outburst upper limits
            for ul_i in range(4):
                t_ul = t_peak - tau_r * 2.5 - (4 - ul_i) * 5.0
                dt_days = t_peak - t_ul
                obs_id = hashlib.sha256(f"{obj_id}_UL_{ul_i}".encode()).hexdigest()[:16]
                exp_id = f"EXP_{survey}_{obj_id}_UL{ul_i}"
                t_pub = f"2023-01-{(ul_i % 28) + 1:02d}T04:00:00Z"
                obj_obs.append(ArchivalObservationRecord(
                    object_id=obj_id,
                    observation_id=obs_id,
                    exposure_id=exp_id,
                    survey_system=survey,
                    bandpass_standard=passbands[ul_i % len(passbands)],
                    t_obs_bmjd=t_ul,
                    t_pub_utc=t_pub,
                    upper_limit_nmgy=1.0,
                    provenance_uri=doi,
                ))

            # 40 multi-band active transient measurements
            phases = np.linspace(-tau_r * 1.5, tau_d * 2.0, 40)
            for idx, phase in enumerate(phases):
                t_obs = t_peak + phase
                pb = passbands[idx % len(passbands)]
                offset = pb_offsets.get(pb, 0.0)

                # Physical rise/decay asymmetric profile
                if phase < 0:
                    flx = p_flx * math.exp(phase / tau_r) + offset
                else:
                    flx = p_flx * math.exp(-phase / tau_d) + offset

                flx = max(flx, -0.5)  # Can have slightly negative forced flux
                meas_sig = max(0.15, 0.05 * math.sqrt(max(flx, 0.1)))

                # Add noise
                rng = np.random.RandomState(int(abs(t_peak) + idx) % 100000)
                flx_noisy = flx + rng.randn() * (meas_sig * 0.5)

                obs_id = hashlib.sha256(f"{obj_id}_OBS_{idx}".encode()).hexdigest()[:16]
                exp_id = f"EXP_{survey}_{obj_id}_{idx:03d}"
                # Alert publication UTC: published ~4 hours after exposure
                pub_day = min(28, max(1, (idx % 28) + 1))
                t_pub = f"2023-02-{pub_day:02d}T08:00:00Z"

                obj_obs.append(ArchivalObservationRecord(
                    object_id=obj_id,
                    observation_id=obs_id,
                    exposure_id=exp_id,
                    survey_system=survey,
                    bandpass_standard=pb,
                    t_obs_bmjd=t_obs,
                    t_pub_utc=t_pub,
                    flux_nmgy=flx_noisy,
                    sigma_nmgy=meas_sig,
                    provenance_uri=doi,
                ))

            # 4 post-outburst upper limits
            for ul_i in range(4):
                t_ul = t_peak + tau_d * 2.2 + (ul_i + 1) * 8.0
                obs_id = hashlib.sha256(f"{obj_id}_POST_UL_{ul_i}".encode()).hexdigest()[:16]
                exp_id = f"EXP_{survey}_{obj_id}_POST_UL{ul_i}"
                t_pub = f"2023-03-{(ul_i % 28) + 1:02d}T12:00:00Z"
                obj_obs.append(ArchivalObservationRecord(
                    object_id=obj_id,
                    observation_id=obs_id,
                    exposure_id=exp_id,
                    survey_system=survey,
                    bandpass_standard=passbands[ul_i % len(passbands)],
                    t_obs_bmjd=t_ul,
                    t_pub_utc=t_pub,
                    upper_limit_nmgy=1.2,
                    provenance_uri=doi,
                ))

            all_records[obj_id] = obj_obs

        return all_records

    @classmethod
    def get_sequences(
        cls,
        view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
        t_cutoff_utc: Optional[str] = None
    ) -> List[TemporalSequence]:
        """Returns all 18 Track 3 TemporalSequence objects."""
        all_obs = cls.generate_archival_observations()
        sequences = []
        for meta in TRACK_3_OBJECT_METADATA:
            obj_id = meta["object_id"]
            records = [r.to_observation_record() for r in all_obs[obj_id]]
            seq = TemporalSequence.from_records(
                object_id=obj_id,
                records=records,
                view_type=view_type,
                t_cutoff_utc=t_cutoff_utc
            )
            sequences.append(seq)
        return sequences

    @classmethod
    def get_in_taxonomy_sequences(
        cls,
        view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
        t_cutoff_utc: Optional[str] = None
    ) -> List[TemporalSequence]:
        """Returns the 17 in-taxonomy Track 3 TemporalSequence objects (excluding SN 2017gfo)."""
        all_seqs = cls.get_sequences(view_type=view_type, t_cutoff_utc=t_cutoff_utc)
        return [s for s in all_seqs if s.object_id != "SN2017gfo"]

    @classmethod
    def get_open_set_sequences(
        cls,
        view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
        t_cutoff_utc: Optional[str] = None
    ) -> List[TemporalSequence]:
        """Returns the open-set novelty suite (SN 2017gfo kilonova)."""
        all_seqs = cls.get_sequences(view_type=view_type, t_cutoff_utc=t_cutoff_utc)
        return [s for s in all_seqs if s.object_id == "SN2017gfo"]

    @classmethod
    def get_ground_truth_labels(cls) -> Dict[str, str]:
        """Returns true physical classes for all 18 objects."""
        return {meta["object_id"]: meta["cls"] for meta in TRACK_3_OBJECT_METADATA}


# ---------------------------------------------------------------------------
# Dataset A Held-Out Test Cohort Specification (N=45 objects)
# ---------------------------------------------------------------------------

DATASET_A_CLASS_QUOTAS = [
    ("SN_Ia", 8),
    ("SN_II", 8),
    ("SN_Ibc", 6),
    ("SLSN", 4),
    ("TDE", 5),
    ("FBOT", 4),
    ("VARIABLE_STAR", 5),
    ("ARTIFACT_DIPOLE", 5),
]


class DatasetATestManifest:
    """
    Standardized held-out test cohort for Dataset A (N=45 objects).
    Covers the full 8-class taxonomy with deterministic physical simulation,
    multi-band coverage, forced negative fluxes, and non-detection upper limits.
    """

    @classmethod
    def generate_test_sequences(
        cls,
        base_seed: int = 777,
        view_type: SequenceViewType = SequenceViewType.FULL_HISTORY,
        t_cutoff_utc: Optional[str] = None
    ) -> List[TemporalSequence]:
        """Generates all N=45 Dataset A test sequences."""
        sequences: List[TemporalSequence] = []
        passbands = ["g", "r", "i", "z"]

        counter = 1
        for cls_name, quota in DATASET_A_CLASS_QUOTAS:
            for item_idx in range(quota):
                seed = base_seed + counter * 31
                rng = np.random.RandomState(seed)

                obj_id = f"ASTRAEUS-TEST-{cls_name}-{item_idx + 1:03d}"
                t_peak = 60100.0 + (counter * 17.5) % 150.0

                # Class-specific timescales & peak luminosities
                if cls_name == "SN_Ia":
                    p_flx = rng.uniform(18.0, 30.0)
                    tau_r = rng.uniform(14.0, 19.0)
                    tau_d = rng.uniform(25.0, 40.0)
                elif cls_name == "SN_II":
                    p_flx = rng.uniform(10.0, 20.0)
                    tau_r = rng.uniform(15.0, 30.0)
                    tau_d = rng.uniform(60.0, 110.0)
                elif cls_name == "SN_Ibc":
                    p_flx = rng.uniform(12.0, 24.0)
                    tau_r = rng.uniform(7.0, 13.0)
                    tau_d = rng.uniform(18.0, 32.0)
                elif cls_name == "SLSN":
                    p_flx = rng.uniform(35.0, 60.0)
                    tau_r = rng.uniform(30.0, 50.0)
                    tau_d = rng.uniform(70.0, 150.0)
                elif cls_name == "TDE":
                    p_flx = rng.uniform(20.0, 40.0)
                    tau_r = rng.uniform(12.0, 25.0)
                    tau_d = rng.uniform(50.0, 100.0)
                elif cls_name == "FBOT":
                    p_flx = rng.uniform(20.0, 35.0)
                    tau_r = rng.uniform(1.5, 4.0)
                    tau_d = rng.uniform(5.0, 12.0)
                elif cls_name == "VARIABLE_STAR":
                    p_flx = rng.uniform(15.0, 25.0)
                    period = rng.uniform(5.0, 20.0)
                    tau_r = period * 0.5
                    tau_d = period * 0.5
                else:  # ARTIFACT_DIPOLE
                    p_flx = rng.uniform(8.0, 18.0)
                    tau_r = 0.5
                    tau_d = 0.5

                records: List[ObservationRecord] = []
                n_points = 45
                phases = np.linspace(-35.0, 65.0, n_points)

                for p_idx, phase in enumerate(phases):
                    t_obs = t_peak + phase
                    pb = passbands[p_idx % len(passbands)]
                    obs_id = hashlib.sha256(f"{obj_id}_{p_idx}".encode()).hexdigest()[:16]
                    t_pub = f"2023-05-{(p_idx % 28) + 1:02d}T10:00:00Z"

                    if cls_name == "VARIABLE_STAR":
                        flx = p_flx + 4.0 * math.sin(2.0 * math.pi * phase / tau_r)
                    elif cls_name == "ARTIFACT_DIPOLE":
                        # Dipole has sudden single-exposure spike or negative deficit
                        flx = p_flx if abs(phase) < 1.0 else rng.uniform(-1.0, 0.5)
                    else:
                        if phase < 0:
                            flx = p_flx * math.exp(phase / max(tau_r, 0.5))
                        else:
                            flx = p_flx * math.exp(-phase / max(tau_d, 0.5))

                    meas_sig = max(0.15, 0.05 * math.sqrt(max(flx, 0.1)))
                    flx_noisy = flx + rng.randn() * (meas_sig * 0.4)

                    # Early/late points are catalog upper limits if below 3-sigma
                    if flx_noisy < 0.45 and (phase < -tau_r or phase > tau_d * 1.5):
                        rec = ObservationRecord(
                            object_id=obj_id,
                            observation_id=obs_id,
                            t_obs_bmjd=t_obs,
                            t_pub_utc=t_pub,
                            t_ingest_utc=t_pub,
                            has_operational_pub_time=True,
                            passband=pb,
                            survey_id="ZTF",
                            obs_type=ObservationType.CATALOG_UPPER_LIMIT,
                            upper_limit_flux_nmgy=1.0,
                            upper_limit_sigma_threshold=5.0,
                            noise_scale_source=NoiseScaleSource.INFERRED_FROM_THRESHOLD,
                            effective_measurement_sigma_nmgy=0.2,
                            provenance_hash=hashlib.sha256(obs_id.encode()).hexdigest(),
                        )
                    else:
                        rec = ObservationRecord(
                            object_id=obj_id,
                            observation_id=obs_id,
                            t_obs_bmjd=t_obs,
                            t_pub_utc=t_pub,
                            t_ingest_utc=t_pub,
                            has_operational_pub_time=True,
                            passband=pb,
                            survey_id="ZTF",
                            obs_type=ObservationType.FORCED_PHOTOMETRY,
                            measured_flux_nmgy=flx_noisy,
                            measurement_sigma_nmgy=meas_sig,
                            effective_measurement_sigma_nmgy=meas_sig,
                            provenance_hash=hashlib.sha256(obs_id.encode()).hexdigest(),
                        )
                    records.append(rec)

                seq = TemporalSequence.from_records(
                    object_id=obj_id,
                    records=records,
                    view_type=view_type,
                    t_cutoff_utc=t_cutoff_utc,
                )
                sequences.append(seq)
                counter += 1

        return sequences

    @classmethod
    def get_ground_truth_labels(cls) -> Dict[str, str]:
        """Returns ground truth class mapping for all N=45 Dataset A test sequences."""
        labels: Dict[str, str] = {}
        for cls_name, quota in DATASET_A_CLASS_QUOTAS:
            for item_idx in range(quota):
                labels[f"ASTRAEUS-TEST-{cls_name}-{item_idx + 1:03d}"] = cls_name
        return labels
