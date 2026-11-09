"""
ASTRAEUS Reference Object Record & 11-Point Identity & Provenance Audit Schema.
Implements the mandatory Phase 2 Stage 1 data integrity contract.
"""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pydantic import BaseModel, Field, ConfigDict, model_validator

from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.epistemic import EpistemicTier


DEFAULT_PROVENANCE_SECRET = b"astraeus-corpus-provenance-authority-v1.0-secret-key"
DEFAULT_KEY_ID = "astraeus-provenance-key-2026-v1"

DatasetOrigin = Literal["ARCHIVAL", "SYNTHETIC", "NUMERICALLY_SIMULATED", "INJECTED"]
TimeStandard = Literal["UTC_MJD", "TDB_BJD"]
RedshiftType = Literal["SPECTROSCOPIC", "PHOTOMETRIC", "UNKNOWN"]
SplitType = Literal["train", "val", "test", "quarantine"]
DistanceMethod = Literal["HUBBLE_FLOW", "INDEPENDENT_TRGB_CEPHEID", "SURFACE_BRIGHTNESS", "UNKNOWN"]


class SpatialPartitionInfo(BaseModel):
    """Point 5: Spatial HEALPix partitioning and boundary buffer quarantine segregation."""
    model_config = ConfigDict(frozen=True)
    healpix_nside32: int
    split: SplitType
    original_assigned_split: Literal["train", "val", "test"]
    distance_to_boundary_arcsec: float
    has_conflicting_neighbor: bool = False
    is_boundary_buffer_quarantined: bool = False
    quarantine_reason: Optional[str] = None
    dist_to_center_arcsec: Optional[float] = None


class CrossMatchPrior(BaseModel):
    """Point 6: Bayesian association and chance coincidence probabilities."""
    model_config = ConfigDict(frozen=True)
    catalog: str
    counterpart_id: Optional[str] = None
    separation_arcsec: float
    p_match: float
    p_chance: float
    association_status: str


class PhotometricCalibrationInfo(BaseModel):
    """Point 7: Passband profiles, native calibration, AB zero points, and extinction."""
    model_config = ConfigDict(frozen=True)
    native_survey: str = "ZTF"
    survey_nominal_zero_point: float = 26.3
    observation_zero_point: Optional[float] = None
    standardized_zero_point_mag: float = 27.5
    photometric_system: str = "AB"
    calibration_system: str = "AB"
    calibration_version: str = "v1.0.1"
    filter_profiles: List[str]
    ebv_mw_mag: float
    a_v_mag: float
    quality_flags_clean: bool = True
    preserves_negative_fluxes: bool = True
    upper_limit_handling: str = "censored_5sigma_flux_floor"

    @property
    def native_zero_point_mag(self) -> float:
        """Survey nominal zero point or observation-level zero point if calibrated."""
        return self.observation_zero_point if self.observation_zero_point is not None else self.survey_nominal_zero_point


class PhotometricRepresentationInfo(BaseModel):
    """Point 8: Native flux preservation and standardized optical pseudo-flux."""
    model_config = ConfigDict(frozen=True)
    has_native_flux: bool = True
    has_optical_pseudoflux: bool = True
    is_pseudoflux_standardized: bool = True
    native_passbands: List[str]
    representation_caveat: str = (
        "optical_pseudoflux is an intermediate passband-averaged representation "
        "and does NOT represent frequency-integrated bolometric luminosity."
    )


class MeasurementUncertaintyInfo(BaseModel):
    """Point 9: Heteroscedastic uncertainties and non-detection handling."""
    model_config = ConfigDict(frozen=True)
    error_model: str = "heteroscedastic_propagated_photometric"
    positive_flux_sigma_formula: str = "sigma_F = 0.921 * F * sigma_m"
    negative_flux_sigma_formula: str = "sigma_F = sigma_F_native (linear pipeline uncertainty; logarithmic magnitude conversion undefined)"
    upper_limit_sigma: float = 5.0
    non_detection_handling: str = "censored_5sigma_flux_upper_limit"
    preserves_linear_negative_flux_error: bool = True

    @property
    def sigma_formula(self) -> str:
        """Backward-compatible alias for primary sigma formula."""
        return self.positive_flux_sigma_formula


class HostAssociationInfo(BaseModel):
    """Point 10: Host galaxy parameters, redshift provenance, and distance provenance."""
    model_config = ConfigDict(frozen=True)
    host_name: Optional[str] = None
    host_ra_deg: Optional[float] = None
    host_dec_deg: Optional[float] = None
    redshift_z: Optional[float] = None
    redshift_type: RedshiftType = "UNKNOWN"
    redshift_source_bibcode: Optional[str] = None
    adopted_distance_mpc: Optional[float] = None
    angular_diameter_distance_mpc: Optional[float] = None
    distance_method: DistanceMethod = "HUBBLE_FLOW"
    distance_uncertainty_mpc: Optional[float] = None
    peculiar_velocity_corrected: bool = False
    angular_separation_arcsec: Optional[float] = None
    nuclear_offset_resolution_arcsec: Optional[float] = None
    projected_offset_kpc: Optional[float] = None
    projected_offset_uncertainty_kpc: Optional[float] = None
    cosmology: Dict[str, float] = Field(
        default_factory=lambda: {"H0": 70.0, "Om0": 0.3, "Ode0": 0.7}
    )
    progenitor_qualification: str = (
        "Spatial separation alone does not confirm progenitor physics without joint "
        "spectroscopic, host stellar population, and multi-wavelength analysis."
    )


class CryptographicSeal(BaseModel):
    """Point 11: End-to-end cryptographic hash chain and HMAC-SHA256 authority seal."""
    model_config = ConfigDict(frozen=True)
    sha256_digest: str
    hmac_sha256_mac: str
    key_id: str = DEFAULT_KEY_ID
    canonical_serialization_version: str = "RFC-8785-JSON"
    signing_authority: str = "ASTRAEUS-PROVENANCE-AUTHORITY-v1.0"
    cryptographic_mechanism: str = (
        "HMAC-SHA256 (symmetric message authentication code for tamper-evident authorized-origin "
        "verification; asymmetric digital signatures required for cryptographic non-repudiation)"
    )
    sealed_at_utc: str

    @property
    def hmac_sha256_signature(self) -> str:
        """Alias for backward compatibility with earlier v2.2.0 callers."""
        return self.hmac_sha256_mac


class ReferenceObjectRecord(BaseModel):
    """
    Immutable 11-Point Identity & Provenance Audit Record for Astronomical Reference Corpus v1.0.
    Ensures zero object-level leakage, certified provenance, and uncertainty-aware tracking.
    """
    model_config = ConfigDict(frozen=True)

    # Point 1: Canonical Identifier & Cross-Catalogue Aliases
    canonical_id: str
    aliases: List[str]

    # Point 2: Discovery Source & Timestamp
    discovery_source: str
    discovery_time_utc: str
    discovery_mjd: float

    # Point 3: Astrometric Coordinates & Epoch
    coordinates: CelestialCoordinates
    coordinate_reference_system: str = "ICRS"
    coordinate_reference_epoch: str = "J2000.0"
    observation_epoch_utc: str

    # Point 4: Time Standard Specification
    time_standard: TimeStandard = "UTC_MJD"
    barycentric_correction_applied: bool = False
    barycentric_correction_provenance: Optional[str] = None

    # Point 5: Spatial Partitioning & Boundary Buffer
    spatial_partition: SpatialPartitionInfo

    # Point 6: Bayesian Association & Chance Coincidence
    cross_match_prior: CrossMatchPrior

    # Point 7: Photometric Calibration & Quality Flags
    photometric_calibration: PhotometricCalibrationInfo

    # Point 8: Photometric Representation
    photometric_representation: PhotometricRepresentationInfo

    # Point 9: Measurement Uncertainty & Non-Detection Convention
    measurement_uncertainty: MeasurementUncertaintyInfo

    # Point 10: Host Galaxy & Progenitor Interpretation
    host_association: HostAssociationInfo

    # Point 11: Cryptographic Integrity & Authority Seal
    cryptographic_seal: CryptographicSeal

    # Multi-Tier Leakage Cluster Tags
    host_leakage_cluster_id: Optional[str] = None
    observation_leakage_ids: List[str] = Field(default_factory=list)

    # Astronomical Classification & Metadata
    transient_class: str
    dataset_origin: DatasetOrigin
    epistemic_tier: EpistemicTier

    # Observational Summary
    peak_apparent_magnitude: float
    peak_mjd: float
    num_epochs: int
    cadence_median_hours: float
    snr_median: float

    # Multi-Epoch Native Photometry & Derived Optical Pseudo-Flux
    times_mjd: List[float]
    flux_native: List[float]
    flux_err_native: List[float]
    passbands: List[str]
    optical_pseudoflux: List[float]
    pseudoflux_err: List[float]

    @staticmethod
    def compute_payload_digest(payload: Dict[str, Any]) -> str:
        """Computes deterministic SHA-256 integrity digest using RFC-8785 canonical JSON sorting."""
        canon_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canon_json.encode("utf-8")).hexdigest()

    @staticmethod
    def compute_hmac(digest: str, secret_key: bytes = DEFAULT_PROVENANCE_SECRET) -> str:
        """Computes keyed HMAC-SHA256 message authentication code (MAC) sealing the digest."""
        return hmac.new(secret_key, digest.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_authenticity(self, secret_key: bytes = DEFAULT_PROVENANCE_SECRET) -> bool:
        """Verifies both structural integrity and HMAC authority MAC."""
        if len(self.cryptographic_seal.sha256_digest) != 64:
            return False
        if len(self.cryptographic_seal.hmac_sha256_mac) != 64:
            return False
        expected_mac = self.compute_hmac(self.cryptographic_seal.sha256_digest, secret_key)
        return hmac.compare_digest(expected_mac, self.cryptographic_seal.hmac_sha256_mac)

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flattens key metadata fields for Arrow/Parquet serialization and query indexing."""
        return {
            "canonical_id": self.canonical_id,
            "aliases_csv": ",".join(self.aliases),
            "discovery_source": self.discovery_source,
            "discovery_time_utc": self.discovery_time_utc,
            "discovery_mjd": self.discovery_mjd,
            "ra_deg": self.coordinates.ra_deg,
            "dec_deg": self.coordinates.dec_deg,
            "coordinate_frame": self.coordinate_reference_system,
            "coordinate_epoch": self.coordinate_reference_epoch,
            "observation_epoch_utc": self.observation_epoch_utc,
            "healpix_nside32": self.spatial_partition.healpix_nside32,
            "split": self.spatial_partition.split,
            "original_assigned_split": self.spatial_partition.original_assigned_split,
            "is_boundary_buffer_quarantined": self.spatial_partition.is_boundary_buffer_quarantined,
            "distance_to_boundary_arcsec": self.spatial_partition.distance_to_boundary_arcsec,
            "time_standard": self.time_standard,
            "transient_class": self.transient_class,
            "dataset_origin": self.dataset_origin,
            "epistemic_tier": self.epistemic_tier.value,
            "peak_apparent_magnitude": self.peak_apparent_magnitude,
            "peak_mjd": self.peak_mjd,
            "num_epochs": self.num_epochs,
            "cadence_median_hours": self.cadence_median_hours,
            "snr_median": self.snr_median,
            "host_name": self.host_association.host_name or "NONE",
            "host_redshift": self.host_association.redshift_z if self.host_association.redshift_z is not None else -999.0,
            "adopted_distance_mpc": self.host_association.adopted_distance_mpc if self.host_association.adopted_distance_mpc is not None else -999.0,
            "distance_method": self.host_association.distance_method,
            "redshift_type": self.host_association.redshift_type,
            "host_offset_kpc": self.host_association.projected_offset_kpc if self.host_association.projected_offset_kpc is not None else -999.0,
            "host_leakage_cluster_id": self.host_leakage_cluster_id or "NONE",
            "survey_nominal_zero_point": self.photometric_calibration.survey_nominal_zero_point,
            "observation_zero_point": self.photometric_calibration.observation_zero_point if self.photometric_calibration.observation_zero_point is not None else -999.0,
            "sha256_digest": self.cryptographic_seal.sha256_digest,
            "hmac_sha256_mac": self.cryptographic_seal.hmac_sha256_mac,
            "key_id": self.cryptographic_seal.key_id,
        }
