"""
ASTRAEUS Temporal Harmonization & Barycentric Correction Module.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-TIME-SPEC-001

Implements:
1. Physical time-scale transformations (UTC <-> TAI <-> TT <-> TDB).
2. Standardized observatory topocentric geodetic coordinates across 7 key facilities.
3. Barycentric light-travel-time corrections via Astropy's library barycentric correction
   engine (pinned ephemeris: 'builtin', IAU/SOFA numerical algorithms).
4. Fail-closed standardization of CanonicalObservationRecord to BMJD(TDB) with
   full provenance tracking, stale-result invalidation, and explicit error states.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import math
import numpy as np
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astraeus_data.adapters.canonical import CanonicalObservationRecord

# Authoritative Leap Seconds and Physical Offsets (2017-2026)
LEAP_SECONDS_TAI_UTC = 37.0
TT_TAI_OFFSET_SECONDS = 32.184
TT_UTC_OFFSET_SECONDS = LEAP_SECONDS_TAI_UTC + TT_TAI_OFFSET_SECONDS  # 69.184s
SECONDS_PER_DAY = 86400.0

# Supported Time Scales
VALID_TIME_SCALES = ("UTC", "TAI", "TT", "TDB")

# Supported Exposure Midpoint Conventions
VALID_EXPOSURE_CONVENTIONS = ("MIDPOINT", "START_PLUS_HALF_EXPTIME", "SHUTTER_OPEN")

# Standardized Observatory Geodetic Coordinates (WGS-84)
OBSERVATORY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "PALOMAR": {
        "name": "Palomar Observatory (Samuel Oschin)",
        "lat_deg": 33.3563,
        "lon_deg": -116.8650,
        "height_m": 1706.0,
        "default_survey": "ZTF",
    },
    "HALEAKALA": {
        "name": "Haleakala Observatory (ATLAS HK / Pan-STARRS1)",
        "lat_deg": 20.7075,
        "lon_deg": -156.2570,
        "height_m": 3055.0,
        "default_survey": "ATLAS",
    },
    "MAUNA_LOA": {
        "name": "Mauna Loa Observatory (ATLAS ML)",
        "lat_deg": 19.5362,
        "lon_deg": -155.5763,
        "height_m": 3394.0,
        "default_survey": "ATLAS",
    },
    "SUTHERLAND": {
        "name": "Sutherland Observatory (ATLAS SA)",
        "lat_deg": -32.3760,
        "lon_deg": 20.8108,
        "height_m": 1798.0,
        "default_survey": "ATLAS",
    },
    "EL_SAUCE": {
        "name": "El Sauce Observatory (ATLAS CH)",
        "lat_deg": -30.4726,
        "lon_deg": -70.7634,
        "height_m": 1560.0,
        "default_survey": "ATLAS",
    },
    "CERRO_TOLOLO": {
        "name": "Cerro Tololo Inter-American Observatory (DECam)",
        "lat_deg": -30.1690,
        "lon_deg": -70.8065,
        "height_m": 2207.0,
        "default_survey": "DECam",
    },
    "CERRO_PACHON": {
        "name": "Vera C. Rubin Observatory (Cerro Pachon)",
        "lat_deg": -30.2444,
        "lon_deg": -70.7494,
        "height_m": 2663.0,
        "default_survey": "Rubin",
    },
}

SURVEY_TO_DEFAULT_OBSERVATORY: Dict[str, str] = {
    "ZTF": "PALOMAR",
    "ATLAS": "HALEAKALA",
    "DECAM": "CERRO_TOLOLO",
    "PS1": "HALEAKALA",
    "PAN-STARRS1": "HALEAKALA",
    "RUBIN": "CERRO_PACHON",
    "LSST": "CERRO_PACHON",
}


class TemporalConversionError(ValueError):
    """Raised when temporal standardization fails under strict fail-closed governance."""
    pass


class TemporalHarmonizer:
    """
    Standardizes heterogeneous survey exposure timestamps to physical barycentric standards (BMJD_TDB).
    Guarantees fail-closed validation, explicit error reporting, and provenance tracking.
    """

    _cached_locations: Dict[str, EarthLocation] = {}
    EPHEMERIS_PINNED: str = "builtin"

    @classmethod
    def get_observatory_location(cls, observatory_key: str) -> EarthLocation:
        """
        Retrieves or creates cached astropy EarthLocation.
        Fails closed with ValueError if observatory is not registered.
        """
        key = observatory_key.upper().strip()
        if key in cls._cached_locations:
            return cls._cached_locations[key]

        meta = OBSERVATORY_REGISTRY.get(key)
        if meta is None:
            raise ValueError(
                f"Unknown observatory '{observatory_key}'. Must be one of: {sorted(OBSERVATORY_REGISTRY.keys())}. "
                "Fail-closed per ASTRAEUS-P5-TIME-SPEC-001 (no silent fallbacks permitted)."
            )

        loc = EarthLocation.from_geodetic(
            lon=meta["lon_deg"] * u.deg,
            lat=meta["lat_deg"] * u.deg,
            height=meta["height_m"] * u.m,
        )
        cls._cached_locations[key] = loc
        return loc

    @classmethod
    def compute_barycentric_delay_days(
        cls,
        mjd: float,
        ra_deg: float,
        dec_deg: float,
        observatory_key: str = "PALOMAR",
        native_scale: str = "UTC",
    ) -> float:
        """
        Computes the library's barycentric light-travel-time correction in days to the Solar System Barycenter (SSB)
        using Astropy's builtin ephemeris (IAU/SOFA algorithms).
        
        Correction corresponds to: dt_bary = t_barycentric - t_geocentric.
        Positive when source light arrives earlier at the SSB than at the geocentric/topocentric observer.
        Fails closed on invalid coordinates, unsupported time scales, or unregistered observatories.
        """
        if not (0.0 <= ra_deg < 360.0):
            raise ValueError(f"Right Ascension {ra_deg} deg out of valid range [0, 360).")
        if not (-90.0 <= dec_deg <= 90.0):
            raise ValueError(f"Declination {dec_deg} deg out of valid range [-90, 90].")

        scale_clean = native_scale.upper().strip()
        if scale_clean not in VALID_TIME_SCALES:
            raise ValueError(
                f"Unsupported time scale '{native_scale}'. Supported scales are {VALID_TIME_SCALES}. "
                "Fail-closed per ASTRAEUS-P5-TIME-SPEC-001 (no silent fallback to UTC permitted)."
            )

        loc = cls.get_observatory_location(observatory_key)
        t = Time(mjd, format="mjd", scale=scale_clean.lower(), location=loc)
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        # Pinned Astropy barycentric light travel time calculation
        dt_bary = t.light_travel_time(coord, "barycentric", ephemeris=cls.EPHEMERIS_PINNED)
        dt_days = float(dt_bary.to(u.day).value)

        # Enforce physical geometric bound: Earth orbit radius / c ~ 0.00578 days (~499s).
        # With lunar/planetary perturbations, maximum bound is 0.00585 days.
        if abs(dt_days) > 0.00585:
            raise ValueError(
                f"Computed barycentric light travel delay {dt_days:.6f} days exceeds physical 1 AU bound (0.00585 days)."
            )

        return dt_days

    @classmethod
    def standardize_record_time(
        cls,
        record: CanonicalObservationRecord,
        observatory_key: Optional[str] = None,
        force_recompute: bool = False,
        fail_on_error: bool = False,
    ) -> CanonicalObservationRecord:
        """
        Standardizes an observation record by computing its Barycentric Modified Julian Date (BMJD_TDB).
        
        Provenance & Idempotency:
        - If already converted, verifies that existing conversion matches current record coordinates,
          native timestamp, scale, and observatory. Recomputes if inputs changed (stale-result invalidation).
        - If conversion fails, explicitly records failure in conversion_status and quarantine_reason;
          does not silently swallow exceptions or leave the record appearing valid.
        - If fail_on_error is True, raises TemporalConversionError directly.
        """
        # Determine target observatory
        obs_key = observatory_key
        if obs_key is None:
            survey_upper = record.survey_id.upper().strip()
            obs_key = SURVEY_TO_DEFAULT_OBSERVATORY.get(survey_upper)
            if obs_key is None:
                err_msg = f"No default observatory mapped for survey '{record.survey_id}'. Explicit observatory_key required."
                if fail_on_error:
                    raise TemporalConversionError(err_msg)
                updated_data = record.model_dump()
                updated_data["conversion_status"] = "FAILED"
                updated_data["t_converted_bmjd"] = None
                updated_data["converted_time_scale"] = None
                updated_data["quarantine_reason"] = f"TEMPORAL_CONVERSION_FAILURE: {err_msg}"
                updated_data["time_conversion_provenance"] = {
                    "status": "FAILED",
                    "error": err_msg,
                    "survey_id": record.survey_id,
                }
                return CanonicalObservationRecord(**updated_data)

        # Stale-result invalidation check
        if record.t_converted_bmjd is not None and not force_recompute:
            prov = record.time_conversion_provenance or {}
            is_stale = (
                prov.get("ra_deg") != record.ra_deg
                or prov.get("dec_deg") != record.dec_deg
                or prov.get("t_native") != record.t_native
                or prov.get("native_time_scale") != record.native_time_scale
                or prov.get("observatory") != obs_key.upper().strip()
            )
            if not is_stale:
                return record

        try:
            # Validate exposure midpoint convention
            conv = record.exposure_midpoint_convention.upper().strip()
            if conv not in VALID_EXPOSURE_CONVENTIONS:
                raise ValueError(
                    f"Unrecognized exposure midpoint convention '{record.exposure_midpoint_convention}'. "
                    f"Must be one of {VALID_EXPOSURE_CONVENTIONS}."
                )

            # Compute effective exposure midpoint timestamp
            t_input_mjd = record.t_native
            if conv == "SHUTTER_OPEN" and record.t_err_days > 0.0:
                # Half-exposure shift to midpoint
                t_input_mjd = record.t_native + (record.t_err_days / 2.0)

            # Compute barycentric light-travel delay
            dt_days = cls.compute_barycentric_delay_days(
                mjd=t_input_mjd,
                ra_deg=record.ra_deg,
                dec_deg=record.dec_deg,
                observatory_key=obs_key,
                native_scale=record.native_time_scale,
            )

            # Compute TDB scale conversion from local topocentric timestamp
            loc = cls.get_observatory_location(obs_key)
            scale_clean = record.native_time_scale.upper().strip().lower()

            t_local = Time(t_input_mjd, format="mjd", scale=scale_clean, location=loc)
            # Add light travel correction and evaluate on TDB scale
            t_bary_tdb = (t_local.tdb + dt_days * u.day).mjd

            updated_data = record.model_dump()
            updated_data["t_converted_bmjd"] = float(t_bary_tdb)
            updated_data["converted_time_scale"] = "BMJD_TDB"
            updated_data["time_conversion_method"] = "ASTROPY_BARYCORR_V1"
            updated_data["t_obs_bmjd"] = float(t_bary_tdb)
            updated_data["conversion_status"] = "SUCCESS"
            updated_data["time_conversion_provenance"] = {
                "status": "SUCCESS",
                "ephemeris": cls.EPHEMERIS_PINNED,
                "astropy_version": astropy.__version__,
                "observatory": obs_key.upper().strip(),
                "ra_deg": float(record.ra_deg),
                "dec_deg": float(record.dec_deg),
                "t_native": float(record.t_native),
                "t_input_midpoint_mjd": float(t_input_mjd),
                "native_time_scale": record.native_time_scale,
                "exposure_midpoint_convention": conv,
                "dt_bary_days": float(dt_days),
                "dt_bary_seconds": float(dt_days * SECONDS_PER_DAY),
            }

            return CanonicalObservationRecord(**updated_data)

        except Exception as e:
            if fail_on_error:
                raise TemporalConversionError(f"Temporal standardization failed: {str(e)}") from e

            # Explicit fail-closed quarantine state: do not masquerade as untouched record
            updated_data = record.model_dump()
            updated_data["t_converted_bmjd"] = None
            updated_data["converted_time_scale"] = None
            updated_data["time_conversion_method"] = "FAILED"
            updated_data["conversion_status"] = "FAILED"
            updated_data["quarantine_reason"] = f"TEMPORAL_CONVERSION_FAILURE: {str(e)}"
            updated_data["time_conversion_provenance"] = {
                "status": "FAILED",
                "error": str(e),
                "ephemeris": cls.EPHEMERIS_PINNED,
                "observatory": str(obs_key),
                "ra_deg": float(record.ra_deg),
                "dec_deg": float(record.dec_deg),
                "t_native": float(record.t_native),
                "native_time_scale": record.native_time_scale,
            }
            return CanonicalObservationRecord(**updated_data)
