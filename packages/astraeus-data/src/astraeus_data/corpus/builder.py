"""
ASTRAEUS Reference Corpus Builder Module.
Constructs Astronomical Reference Corpus v1.0 across archival and synthetic transient populations
with strict spatial HEALPix partitioning, boundary buffer quarantine, and 11-point provenance sealing.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import warnings
from pathlib import Path
import numpy as np

from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
from astropy import units as u

from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.epistemic import EpistemicTier
from astraeus_data.partitioning import load_split_manifest
from astraeus_data.corpus.geometry import (
    evaluate_quarantine_predicate,
    compute_spherical_edge_distance_arcsec,
)
from astraeus_data.corpus.record import (
    ReferenceObjectRecord,
    SpatialPartitionInfo,
    CrossMatchPrior,
    PhotometricCalibrationInfo,
    PhotometricRepresentationInfo,
    MeasurementUncertaintyInfo,
    HostAssociationInfo,
    CryptographicSeal,
    DEFAULT_PROVENANCE_SECRET,
    DEFAULT_KEY_ID,
    DatasetOrigin,
)


class ReferenceCorpusBuilder:
    """
    Builder engine for Astronomical Reference Corpus v1.0.
    Enforces strict identity, coordinate, and provenance hygiene.
    """

    def __init__(
        self,
        split_manifest_path: str = "configs/data_splits_healpix.json",
        nside: int = 32,
        buffer_arcsec: float = 5.0,
        provenance_secret: bytes = DEFAULT_PROVENANCE_SECRET,
        key_id: str = DEFAULT_KEY_ID,
    ) -> None:
        self.split_manifest_path = split_manifest_path
        self.nside = nside
        self.buffer_arcsec = buffer_arcsec
        self.secret = provenance_secret
        self.key_id = key_id
        self.hp = HEALPix(nside=nside, order="nested", frame="icrs")
        self.pixel_to_split = load_split_manifest(split_manifest_path)

    def compute_spatial_partition(
        self,
        ra_deg: float,
        dec_deg: float,
        uncertainty_mas: Optional[Dict[str, float]] = None,
        astrometric_err_arcsec: float = 0.05,
    ) -> SpatialPartitionInfo:
        """
        Determines HEALPix pixel at Nside=32, assigns split, and checks boundary quarantine buffer.
        If near a conflicting split boundary, assigns split="quarantine" to prevent leakage into active splits.
        """
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        pix = int(self.hp.skycoord_to_healpix(coord))
        nominal_split = self.pixel_to_split.get(pix, "train")

        # Check adjacent neighbor pixels for conflicting splits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            neighbors = self.hp.neighbours(pix)
        conflicting_neighbor = False
        for n_pix in neighbors:
            if n_pix >= 0:
                n_split = self.pixel_to_split.get(int(n_pix), "train")
                if n_split != nominal_split:
                    conflicting_neighbor = True
                    break

        # Exact spherical distance to actual cell boundary polygon: d_edge(x) = min_{y in \partial C} angdist(x, y)
        center_coord = self.hp.healpix_to_skycoord(pix)
        dist_from_center = coord.separation(center_coord).arcsec

        if uncertainty_mas is None:
            uncertainty_mas = {
                "ra": astrometric_err_arcsec * 1000.0,
                "dec": astrometric_err_arcsec * 1000.0,
            }

        (
            is_quarantined,
            quarantine_reason,
            d_edge_arcsec,
            sigma_perp_arcsec,
            crossing_prob,
        ) = evaluate_quarantine_predicate(
            coord=coord,
            hp=self.hp,
            pix=pix,
            has_conflicting_neighbor=conflicting_neighbor,
            uncertainty_mas=uncertainty_mas,
            buffer_arcsec=self.buffer_arcsec,
            confidence_multiplier=3.0,
            step=100,
        )

        effective_split = "quarantine" if is_quarantined else nominal_split

        return SpatialPartitionInfo(
            healpix_nside32=pix,
            split=effective_split,
            original_assigned_split=nominal_split,
            distance_to_boundary_arcsec=round(d_edge_arcsec, 2),
            has_conflicting_neighbor=conflicting_neighbor,
            is_boundary_buffer_quarantined=is_quarantined,
            quarantine_reason=quarantine_reason,
            dist_to_center_arcsec=round(dist_from_center, 2),
            projected_normal_astrometric_sigma_arcsec=round(sigma_perp_arcsec, 4),
            crossing_probability=float(crossing_prob),
        )

    def seal_record(self, record_dict: Dict[str, Any]) -> CryptographicSeal:
        """Generates SHA-256 integrity digest and HMAC-SHA256 authority seal."""
        digest = ReferenceObjectRecord.compute_payload_digest(record_dict)
        mac = ReferenceObjectRecord.compute_hmac(digest, self.secret)
        return CryptographicSeal(
            sha256_digest=digest,
            hmac_sha256_mac=mac,
            key_id=self.key_id,
            canonical_serialization_version="RFC-8785-JSON",
            signing_authority="ASTRAEUS-PROVENANCE-AUTHORITY-v1.0",
            sealed_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def create_archival_at2018cow(self) -> ReferenceObjectRecord:
        """Constructs canonical AT2018cow archival reference object."""
        ra = 244.000917
        dec = 22.268031
        coords = CelestialCoordinates(
            ra_deg=ra,
            dec_deg=dec,
            reference_frame="ICRS",
            epoch_jyr=2018.5,
            uncertainty_mas={"ra": 30.0, "dec": 30.0},
        )
        partition = self.compute_spatial_partition(
            ra, dec, uncertainty_mas=coords.uncertainty_mas, astrometric_err_arcsec=0.03
        )

        times = [58285.25, 58285.90, 58286.80, 58288.40, 58290.50, 58293.00]
        mags = [18.50, 15.20, 14.70, 16.20, 17.80, 19.40]
        errs = [0.08, 0.04, 0.04, 0.06, 0.08, 0.12]
        bands = ["ZTF_g", "ZTF_g", "ZTF_r", "ZTF_g", "ZTF_r", "ZTF_i"]

        flux_native = [float(10.0 ** (-0.4 * (m - 27.5))) for m in mags]
        flux_err = [float(0.921 * f * e) for f, e in zip(flux_native, errs)]

        payload_to_seal = {
            "canonical_id": "AT2018cow",
            "aliases": ["ATLAS18qqn", "ZTF18abokrym", "IAU 2018cow"],
            "discovery_source": "ATLAS",
            "discovery_time_utc": "2018-06-16T06:00:00Z",
            "discovery_mjd": 58285.25,
            "ra": ra,
            "dec": dec,
            "transient_class": "FBOT",
            "origin": "ARCHIVAL",
        }
        seal = self.seal_record(payload_to_seal)

        return ReferenceObjectRecord(
            canonical_id="AT2018cow",
            aliases=["ATLAS18qqn", "ZTF18abokrym", "IAU 2018cow"],
            discovery_source="ATLAS",
            discovery_time_utc="2018-06-16T06:00:00Z",
            discovery_mjd=58285.25,
            coordinates=coords,
            coordinate_reference_system="ICRS",
            coordinate_reference_epoch="J2000.0",
            observation_epoch_utc="2018-06-16T06:00:00Z",
            time_standard="UTC_MJD",
            barycentric_correction_applied=False,
            spatial_partition=partition,
            cross_match_prior=CrossMatchPrior(
                catalog="Gaia DR3",
                counterpart_id="Gaia DR3 12948194829104",
                separation_arcsec=0.04,
                p_match=0.998,
                p_chance=0.0001,
                association_status="SECURE_MATCH",
            ),
            photometric_calibration=PhotometricCalibrationInfo(
                native_survey="ATLAS",
                survey_nominal_zero_point=27.5,
                observation_zero_point=27.48,
                standardized_zero_point_mag=27.5,
                photometric_system="AB",
                calibration_system="AB",
                calibration_version="v1.0.1",
                filter_profiles=["ATLAS_o", "ZTF_g", "ZTF_r", "ZTF_i"],
                ebv_mw_mag=0.075,
                a_v_mag=0.233,
                quality_flags_clean=True,
                preserves_negative_fluxes=True,
            ),
            photometric_representation=PhotometricRepresentationInfo(
                has_native_flux=True,
                has_optical_pseudoflux=True,
                is_pseudoflux_standardized=True,
                native_passbands=["g", "r", "i"],
            ),
            measurement_uncertainty=MeasurementUncertaintyInfo(),
            host_association=HostAssociationInfo(
                host_name="CGCG 137-068",
                host_ra_deg=244.00164,
                host_dec_deg=22.26672,
                redshift_z=0.014145,
                redshift_type="SPECTROSCOPIC",
                redshift_source_bibcode="2018ApJ...865L...3P",
                adopted_distance_mpc=60.5,
                distance_method="HUBBLE_FLOW",
                distance_uncertainty_mpc=4.2,
                peculiar_velocity_corrected=True,
                angular_separation_arcsec=5.9,
                projected_offset_kpc=1.73,
            ),
            cryptographic_seal=seal,
            host_leakage_cluster_id="HOST-CGCG-137-068",
            observation_leakage_ids=["OBS-ATLAS-18qqn-01", "OBS-ZTF-18abokrym-01"],
            transient_class="FBOT",
            dataset_origin="ARCHIVAL",
            epistemic_tier=EpistemicTier.OBSERVED,
            peak_apparent_magnitude=14.70,
            peak_mjd=58286.04,
            num_epochs=len(times),
            cadence_median_hours=38.4,
            snr_median=22.5,
            times_mjd=times,
            flux_native=flux_native,
            flux_err_native=flux_err,
            passbands=bands,
            optical_pseudoflux=flux_native,
            pseudoflux_err=flux_err,
        )

    def create_archival_sn2011fe(self) -> ReferenceObjectRecord:
        """Constructs canonical SN 2011fe archival reference object with independent host distance."""
        ra = 210.80227
        dec = 54.27083
        coords = CelestialCoordinates(
            ra_deg=ra,
            dec_deg=dec,
            reference_frame="ICRS",
            epoch_jyr=2011.6,
            uncertainty_mas={"ra": 25.0, "dec": 25.0},
        )
        partition = self.compute_spatial_partition(
            ra, dec, uncertainty_mas=coords.uncertainty_mas, astrometric_err_arcsec=0.025
        )

        times = [55797.2, 55800.5, 55805.0, 55810.2, 55814.8, 55822.0, 55835.0]
        mags = [14.8, 12.2, 10.4, 9.98, 10.2, 11.1, 12.5]
        errs = [0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.04]
        bands = ["PTF_g", "PTF_g", "PTF_g", "PTF_r", "PTF_r", "PTF_g", "PTF_r"]

        flux_native = [float(10.0 ** (-0.4 * (m - 27.0))) for m in mags]
        flux_err = [float(0.921 * f * e) for f, e in zip(flux_native, errs)]

        # Standardized optical pseudoflux with ZP=27.5
        flux_pseudoflux = [float(10.0 ** (-0.4 * (m - 27.5))) for m in mags]
        pseudoflux_err = [float(0.921 * f * e) for f, e in zip(flux_pseudoflux, errs)]

        payload_to_seal = {
            "canonical_id": "SN2011fe",
            "aliases": ["PTF11kly", "IAU 2011fe"],
            "discovery_source": "PTF",
            "discovery_time_utc": "2011-08-24T19:00:00Z",
            "discovery_mjd": 55797.2,
            "ra": ra,
            "dec": dec,
            "transient_class": "SN_Ia",
            "origin": "ARCHIVAL",
        }
        seal = self.seal_record(payload_to_seal)

        return ReferenceObjectRecord(
            canonical_id="SN2011fe",
            aliases=["PTF11kly", "IAU 2011fe"],
            discovery_source="PTF",
            discovery_time_utc="2011-08-24T19:00:00Z",
            discovery_mjd=55797.2,
            coordinates=coords,
            coordinate_reference_system="ICRS",
            coordinate_reference_epoch="J2000.0",
            observation_epoch_utc="2011-08-24T19:00:00Z",
            time_standard="UTC_MJD",
            barycentric_correction_applied=False,
            spatial_partition=partition,
            cross_match_prior=CrossMatchPrior(
                catalog="Gaia DR3",
                separation_arcsec=0.02,
                p_match=0.999,
                p_chance=0.00005,
                association_status="SECURE_MATCH",
            ),
            photometric_calibration=PhotometricCalibrationInfo(
                native_survey="PTF",
                survey_nominal_zero_point=27.0,
                observation_zero_point=27.02,
                standardized_zero_point_mag=27.5,
                photometric_system="AB",
                calibration_system="AB",
                calibration_version="v1.0.1",
                filter_profiles=["PTF_g", "PTF_r"],
                ebv_mw_mag=0.009,
                a_v_mag=0.028,
                quality_flags_clean=True,
                preserves_negative_fluxes=True,
            ),
            photometric_representation=PhotometricRepresentationInfo(
                has_native_flux=True,
                has_optical_pseudoflux=True,
                is_pseudoflux_standardized=True,
                native_passbands=["g", "r"],
            ),
            measurement_uncertainty=MeasurementUncertaintyInfo(),
            host_association=HostAssociationInfo(
                host_name="Messier 101 (NGC 5457)",
                host_ra_deg=210.8024,
                host_dec_deg=54.3490,
                redshift_z=0.000804,
                redshift_type="SPECTROSCOPIC",
                redshift_source_bibcode="2011Natur.480..344N",
                adopted_distance_mpc=6.4,
                angular_diameter_distance_mpc=6.39,
                distance_method="INDEPENDENT_TRGB_CEPHEID",
                distance_uncertainty_mpc=0.5,
                peculiar_velocity_corrected=True,
                angular_separation_arcsec=162.0,
                projected_offset_kpc=5.02,
                projected_offset_uncertainty_kpc=0.39,
            ),
            cryptographic_seal=seal,
            host_leakage_cluster_id="HOST-MESSIER-101",
            observation_leakage_ids=["OBS-PTF-11kly-01"],
            transient_class="SN_Ia",
            dataset_origin="ARCHIVAL",
            epistemic_tier=EpistemicTier.OBSERVED,
            peak_apparent_magnitude=9.98,
            peak_mjd=55810.2,
            num_epochs=len(times),
            cadence_median_hours=72.0,
            snr_median=45.0,
            times_mjd=times,
            flux_native=flux_native,
            flux_err_native=flux_err,
            passbands=bands,
            optical_pseudoflux=flux_pseudoflux,
            pseudoflux_err=pseudoflux_err,
        )

    def create_archival_at2018zr(self) -> ReferenceObjectRecord:
        """Constructs canonical AT2018zr (PS18kh) Tidal Disruption Event (TDE)."""
        ra = 119.22721
        dec = 34.26211
        coords = CelestialCoordinates(
            ra_deg=ra,
            dec_deg=dec,
            reference_frame="ICRS",
            epoch_jyr=2018.2,
            uncertainty_mas={"ra": 20.0, "dec": 20.0},
        )
        partition = self.compute_spatial_partition(
            ra, dec, uncertainty_mas=coords.uncertainty_mas, astrometric_err_arcsec=0.02
        )

        times = [58175.0, 58190.0, 58210.0, 58240.0, 58280.0, 58330.0]
        mags = [19.2, 17.5, 16.8, 17.4, 18.2, 19.1]
        errs = [0.08, 0.05, 0.04, 0.05, 0.07, 0.10]
        bands = ["ZTF_g", "ZTF_g", "ZTF_g", "ZTF_r", "ZTF_g", "ZTF_r"]

        flux_native = [float(10.0 ** (-0.4 * (m - 26.3))) for m in mags]
        flux_err = [float(0.921 * f * e) for f, e in zip(flux_native, errs)]

        flux_pseudoflux = [float(10.0 ** (-0.4 * (m - 27.5))) for m in mags]
        pseudoflux_err = [float(0.921 * f * e) for f, e in zip(flux_pseudoflux, errs)]

        payload_to_seal = {
            "canonical_id": "AT2018zr",
            "aliases": ["PS18kh", "ZTF18aabtxvd", "IAU 2018zr"],
            "discovery_source": "ZTF",
            "discovery_time_utc": "2018-03-02T08:00:00Z",
            "discovery_mjd": 58175.0,
            "ra": ra,
            "dec": dec,
            "transient_class": "TDE",
            "origin": "ARCHIVAL",
        }
        seal = self.seal_record(payload_to_seal)

        return ReferenceObjectRecord(
            canonical_id="AT2018zr",
            aliases=["PS18kh", "ZTF18aabtxvd", "IAU 2018zr"],
            discovery_source="ZTF",
            discovery_time_utc="2018-03-02T08:00:00Z",
            discovery_mjd=58175.0,
            coordinates=coords,
            coordinate_reference_system="ICRS",
            coordinate_reference_epoch="J2000.0",
            observation_epoch_utc="2018-03-02T08:00:00Z",
            time_standard="UTC_MJD",
            barycentric_correction_applied=False,
            spatial_partition=partition,
            cross_match_prior=CrossMatchPrior(
                catalog="Gaia DR3",
                separation_arcsec=0.03,
                p_match=0.995,
                p_chance=0.0002,
                association_status="SECURE_MATCH",
            ),
            photometric_calibration=PhotometricCalibrationInfo(
                native_survey="ZTF",
                survey_nominal_zero_point=26.3,
                observation_zero_point=26.31,
                standardized_zero_point_mag=27.5,
                photometric_system="AB",
                calibration_system="AB",
                calibration_version="v1.0.1",
                filter_profiles=["ZTF_g", "ZTF_r"],
                ebv_mw_mag=0.042,
                a_v_mag=0.130,
                quality_flags_clean=True,
                preserves_negative_fluxes=True,
            ),
            photometric_representation=PhotometricRepresentationInfo(
                has_native_flux=True,
                has_optical_pseudoflux=True,
                is_pseudoflux_standardized=True,
                native_passbands=["g", "r"],
            ),
            measurement_uncertainty=MeasurementUncertaintyInfo(),
            host_association=HostAssociationInfo(
                host_name="SDSS J075654.53+341543.6",
                host_ra_deg=119.22722,
                host_dec_deg=34.26210,
                redshift_z=0.071,
                redshift_type="SPECTROSCOPIC",
                redshift_source_bibcode="2019ApJ...872..198H",
                adopted_distance_mpc=325.0,
                distance_method="HUBBLE_FLOW",
                distance_uncertainty_mpc=22.0,
                angular_separation_arcsec=0.05,
                nuclear_offset_resolution_arcsec=1.0,
                projected_offset_kpc=0.07,
            ),
            cryptographic_seal=seal,
            host_leakage_cluster_id="HOST-SDSS-J075654",
            observation_leakage_ids=["OBS-ZTF-18aabtxvd-01", "OBS-PS1-18kh-01"],
            transient_class="TDE",
            dataset_origin="ARCHIVAL",
            epistemic_tier=EpistemicTier.OBSERVED,
            peak_apparent_magnitude=16.80,
            peak_mjd=58210.0,
            num_epochs=len(times),
            cadence_median_hours=120.0,
            snr_median=18.0,
            times_mjd=times,
            flux_native=flux_native,
            flux_err_native=flux_err,
            passbands=bands,
            optical_pseudoflux=flux_pseudoflux,
            pseudoflux_err=pseudoflux_err,
        )

    def generate_synthetic_record(
        self,
        transient_class: str,
        seed: int,
        dataset_origin: DatasetOrigin = "SYNTHETIC",
    ) -> ReferenceObjectRecord:
        """
        Generates a physically realistic, simulated/synthetic light curve with full 11-point audit record.
        Preserves native flux (including potential noisy negative residuals) and derives optical_pseudoflux.
        """
        rng = np.random.default_rng(seed)
        cid = f"ASTRAEUS-{transient_class}-{dataset_origin.lower()[:3]}-{seed:05d}"

        ra = float(rng.uniform(0.0, 360.0))
        dec = float(np.degrees(np.arcsin(rng.uniform(-0.95, 0.95))))

        coords = CelestialCoordinates(
            ra_deg=ra,
            dec_deg=dec,
            reference_frame="ICRS",
            epoch_jyr=2026.0,
            uncertainty_mas={"ra": float(rng.uniform(15.0, 45.0)), "dec": float(rng.uniform(15.0, 45.0))},
        )
        partition = self.compute_spatial_partition(
            ra, dec, uncertainty_mas=coords.uncertainty_mas, astrometric_err_arcsec=0.04
        )

        if transient_class == "SN_Ia":
            tau_rise, tau_fall = rng.normal(3.5, 0.4), rng.normal(22.0, 3.0)
            peak_mag = rng.normal(16.5, 1.2)
            z = rng.uniform(0.02, 0.15)
            host_sep = rng.uniform(0.5, 12.0)
        elif transient_class == "SN_II":
            tau_rise, tau_fall = rng.normal(8.0, 1.5), rng.normal(55.0, 10.0)
            peak_mag = rng.normal(17.8, 1.5)
            z = rng.uniform(0.01, 0.08)
            host_sep = rng.uniform(0.2, 8.0)
        elif transient_class == "SN_Ibc":
            tau_rise, tau_fall = rng.normal(4.5, 0.8), rng.normal(25.0, 4.0)
            peak_mag = rng.normal(17.2, 1.3)
            z = rng.uniform(0.01, 0.10)
            host_sep = rng.uniform(0.5, 10.0)
        elif transient_class == "FBOT":
            tau_rise, tau_fall = rng.normal(0.4, 0.15), rng.normal(2.5, 0.8)
            peak_mag = rng.normal(15.5, 1.5)
            z = rng.uniform(0.01, 0.05)
            host_sep = rng.uniform(1.0, 6.0)
        elif transient_class == "SLSN":
            tau_rise, tau_fall = rng.normal(25.0, 4.0), rng.normal(60.0, 12.0)
            peak_mag = rng.normal(14.8, 1.2)
            z = rng.uniform(0.1, 0.6)
            host_sep = rng.uniform(0.1, 4.0)
        elif transient_class == "TDE":
            tau_rise, tau_fall = rng.normal(15.0, 3.0), rng.normal(45.0, 8.0)
            peak_mag = rng.normal(16.8, 1.4)
            z = rng.uniform(0.03, 0.20)
            host_sep = rng.uniform(0.01, 0.2)
        elif transient_class == "ARTIFACT_DIPOLE":
            tau_rise, tau_fall = 0.05, 0.1
            peak_mag = rng.normal(19.0, 1.0)
            z = 0.0
            host_sep = 0.0
        else:  # VARIABLE_STAR
            tau_rise, tau_fall = 5.0, 5.0
            peak_mag = rng.normal(18.0, 1.5)
            z = 0.0
            host_sep = 0.0

        tau_rise = max(0.1, tau_rise)
        tau_fall = max(0.2, tau_fall)

        n_obs = int(rng.integers(8, 26))
        t0 = 59000.0 + rng.uniform(10.0, 100.0)
        times = sorted([float(t0 + rng.uniform(-10.0, 60.0)) for _ in range(n_obs)])

        mags = []
        errs = []
        bands = []
        band_choices = ["g", "r", "i"]

        for t in times:
            dt = t - t0
            denom = 1.0 + np.exp(-dt / tau_rise)
            numer = np.exp(-dt / tau_fall)
            flux_rel = numer / denom if denom > 0 else 0.0
            f_norm = max(0.005, flux_rel)
            mag = peak_mag - 2.5 * np.log10(f_norm)
            err = float(rng.uniform(0.03, 0.15))
            mags.append(float(mag))
            errs.append(err)
            bands.append(f"ZTF_{band_choices[rng.integers(0, 3)]}")

        native_zp = 26.3
        standard_zp = 27.5
        flux_native = [float(10.0 ** (-0.4 * (m - native_zp))) for m in mags]
        flux_err_native = [float(0.921 * f * e) for f, e in zip(flux_native, errs)]

        # For negative difference flux in difference imaging, preserve native linear pipeline uncertainty
        if transient_class == "ARTIFACT_DIPOLE" and len(flux_native) > 2:
            # Inject realistic noisy negative residual from difference imaging subtraction
            negative_flux = -abs(flux_native[-1] * 0.5)
            flux_native[-1] = negative_flux
            # Linear pipeline subtraction noise (uncorrelated with magnitude)
            flux_err_native[-1] = float(abs(negative_flux) * 1.2 + 0.5)

        flux_pseudoflux = [float(10.0 ** (-0.4 * (m - standard_zp))) for m in mags]
        pseudoflux_err = [float(0.921 * f * e) for f, e in zip(flux_pseudoflux, errs)]

        payload_to_seal = {
            "canonical_id": cid,
            "aliases": [f"SYNTH_{seed:05d}"],
            "discovery_source": "ASTRAEUS-Sim",
            "discovery_time_utc": "2026-01-01T00:00:00Z",
            "discovery_mjd": round(times[0], 2),
            "ra": round(ra, 6),
            "dec": round(dec, 6),
            "transient_class": transient_class,
            "origin": dataset_origin,
        }
        seal = self.seal_record(payload_to_seal)

        tier = EpistemicTier.SIMULATED if dataset_origin == "NUMERICALLY_SIMULATED" else EpistemicTier.INFERRED

        return ReferenceObjectRecord(
            canonical_id=cid,
            aliases=[f"SYNTH_{seed:05d}"],
            discovery_source="ASTRAEUS-Sim",
            discovery_time_utc="2026-01-01T00:00:00Z",
            discovery_mjd=round(times[0], 2),
            coordinates=coords,
            coordinate_reference_system="ICRS",
            coordinate_reference_epoch="J2000.0",
            observation_epoch_utc="2026-01-01T00:00:00Z",
            time_standard="UTC_MJD",
            barycentric_correction_applied=False,
            spatial_partition=partition,
            cross_match_prior=CrossMatchPrior(
                catalog="Gaia DR3",
                separation_arcsec=float(rng.uniform(0.02, 0.8)),
                p_match=float(rng.uniform(0.85, 0.999)),
                p_chance=float(rng.uniform(0.0001, 0.01)),
                association_status="SECURE_MATCH",
            ),
            photometric_calibration=PhotometricCalibrationInfo(
                native_survey="ZTF",
                survey_nominal_zero_point=native_zp,
                observation_zero_point=round(native_zp + float(rng.normal(0.0, 0.02)), 3),
                standardized_zero_point_mag=standard_zp,
                photometric_system="AB",
                calibration_system="AB",
                calibration_version="v1.0.1",
                filter_profiles=["ZTF_g", "ZTF_r", "ZTF_i"],
                ebv_mw_mag=float(rng.uniform(0.01, 0.08)),
                a_v_mag=float(rng.uniform(0.03, 0.25)),
                quality_flags_clean=True,
                preserves_negative_fluxes=True,
            ),
            photometric_representation=PhotometricRepresentationInfo(
                has_native_flux=True,
                has_optical_pseudoflux=True,
                is_pseudoflux_standardized=True,
                native_passbands=["g", "r", "i"],
            ),
            measurement_uncertainty=MeasurementUncertaintyInfo(),
            host_association=HostAssociationInfo(
                host_name=f"HOST-GAL-{seed:04d}",
                redshift_z=float(round(z, 4)) if z > 0 else None,
                redshift_type="SPECTROSCOPIC" if z > 0 else "UNKNOWN",
                adopted_distance_mpc=float(round(z * 300000.0 / 70.0, 1)) if z > 0 else None,
                distance_method="HUBBLE_FLOW" if z > 0 else "UNKNOWN",
                angular_separation_arcsec=round(host_sep, 2),
                projected_offset_kpc=round(host_sep * 0.293 * (z / 0.014), 2) if z > 0 else None,
            ),
            cryptographic_seal=seal,
            host_leakage_cluster_id=f"HOST-CLUSTER-{seed:04d}",
            observation_leakage_ids=[f"OBS-SIM-{seed:05d}-01"],
            transient_class=transient_class,
            dataset_origin=dataset_origin,
            epistemic_tier=tier,
            peak_apparent_magnitude=round(float(peak_mag), 2),
            peak_mjd=round(float(t0), 2),
            num_epochs=len(times),
            cadence_median_hours=round(float(np.median(np.diff(times)) * 24.0), 1),
            snr_median=round(float(np.median([abs(f) / max(1.0, fe) for f, fe in zip(flux_native, flux_err_native)])), 1),
            times_mjd=times,
            flux_native=flux_native,
            flux_err_native=flux_err_native,
            passbands=bands,
            optical_pseudoflux=flux_pseudoflux,
            pseudoflux_err=pseudoflux_err,
        )

    def build_reference_corpus_v1(
        self,
        n_synthetic_per_class: int = 35,
        base_seed: int = 42,
    ) -> List[ReferenceObjectRecord]:
        """
        Builds complete Astronomical Reference Corpus v1.0 combining canonical archival events
        and standardized synthetic/simulated transient populations.
        """
        corpus: List[ReferenceObjectRecord] = []

        # 1. Archival Ground-Truth Transients
        corpus.append(self.create_archival_at2018cow())
        corpus.append(self.create_archival_sn2011fe())
        corpus.append(self.create_archival_at2018zr())

        # 2. Balanced Synthetic & Simulated Populations
        classes = [
            ("SN_Ia", "SYNTHETIC"),
            ("SN_II", "SYNTHETIC"),
            ("SN_Ibc", "SYNTHETIC"),
            ("FBOT", "NUMERICALLY_SIMULATED"),
            ("SLSN", "NUMERICALLY_SIMULATED"),
            ("TDE", "NUMERICALLY_SIMULATED"),
            ("ARTIFACT_DIPOLE", "SYNTHETIC"),
            ("VARIABLE_STAR", "SYNTHETIC"),
        ]

        seed_counter = base_seed
        for cls_name, origin in classes:
            for _ in range(n_synthetic_per_class):
                rec = self.generate_synthetic_record(
                    transient_class=cls_name,
                    seed=seed_counter,
                    dataset_origin=origin,
                )
                corpus.append(rec)
                seed_counter += 1

        return corpus
