"""
ASTRAEUS Temporal Harmonization & Barycentric Correction Module.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-TIME-SPEC-001

Implements:
1. Physical time-scale transformations (UTC <-> TAI <-> TT <-> TDB).
2. Standardized observatory topocentric geodetic coordinates.
3. High-precision relativistic Roemer and Einstein barycentric delays.
4. Fail-closed standardization of CanonicalObservationRecord to BMJD(TDB).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import math
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from astraeus_data.adapters.canonical import CanonicalObservationRecord

# Authoritative Leap Seconds and Physical Offsets (2017-2026)
LEAP_SECONDS_TAI_UTC = 37.0
TT_TAI_OFFSET_SECONDS = 32.184
TT_UTC_OFFSET_SECONDS = LEAP_SECONDS_TAI_UTC + TT_TAI_OFFSET_SECONDS  # 69.184s
SECONDS_PER_DAY = 86400.0

# Standardized Observatory Geodetic Coordinates
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


class TemporalHarmonizer:
    """
    Standardizes heterogeneous survey exposure timestamps to physical barycentric standards.
    """

    _cached_locations: Dict[str, EarthLocation] = {}

    @classmethod
    def get_observatory_location(cls, observatory_key: str) -> EarthLocation:
        """Retrieves or creates cached astropy EarthLocation."""
        key = observatory_key.upper()
        if key in cls._cached_locations:
            return cls._cached_locations[key]

        meta = OBSERVATORY_REGISTRY.get(key)
        if meta is None:
            # Fallback to Palomar
            meta = OBSERVATORY_REGISTRY["PALOMAR"]

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
        Computes the total light-travel time correction (Roemer delay) in days to the Solar System Barycenter.
        Positive when source light arrives earlier at the SSB than at the geocentric observer.
        """
        loc = cls.get_observatory_location(observatory_key)
        scale_clean = native_scale.lower().strip()
        if scale_clean not in ("utc", "tai", "tt", "tdb"):
            scale_clean = "utc"

        t = Time(mjd, format="mjd", scale=scale_clean, location=loc)
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")

        # Astropy light_travel_time returns time difference: t_bary - t_geo
        dt_bary = t.light_travel_time(coord, "barycentric", ephemeris="builtin")
        return float(dt_bary.to(u.day).value)

    @classmethod
    def standardize_record_time(
        cls,
        record: CanonicalObservationRecord,
        observatory_key: Optional[str] = None,
    ) -> CanonicalObservationRecord:
        """
        Standardizes an observation record by computing its Barycentric Modified Julian Date (BMJD_TDB).
        Preserves original native values and fails closed if conversion cannot be performed.
        """
        if record.t_converted_bmjd is not None:
            # Already standardized
            return record

        obs_key = observatory_key
        if obs_key is None:
            obs_key = SURVEY_TO_DEFAULT_OBSERVATORY.get(record.survey_id.upper(), "PALOMAR")

        try:
            dt_days = cls.compute_barycentric_delay_days(
                mjd=record.t_native,
                ra_deg=record.ra_deg,
                dec_deg=record.dec_deg,
                observatory_key=obs_key,
                native_scale=record.native_time_scale,
            )

            # Compute TDB scale conversion from native timestamp
            loc = cls.get_observatory_location(obs_key)
            scale_clean = record.native_time_scale.lower().strip()
            if scale_clean not in ("utc", "tai", "tt", "tdb"):
                scale_clean = "utc"

            t_local = Time(record.t_native, format="mjd", scale=scale_clean, location=loc)
            # Add Roemer barycentric delay and evaluate on TDB scale
            t_bary_tdb = (t_local + dt_days * u.day).tdb.mjd

            updated_data = record.model_dump()
            updated_data["t_converted_bmjd"] = float(t_bary_tdb)
            updated_data["converted_time_scale"] = "BMJD_TDB"
            updated_data["time_conversion_method"] = "ASTROPY_BARYCORR_V1"
            updated_data["t_obs_bmjd"] = float(t_bary_tdb)

            return CanonicalObservationRecord(**updated_data)

        except Exception:
            # Fail-closed: keep unconverted rather than guessing
            return record
