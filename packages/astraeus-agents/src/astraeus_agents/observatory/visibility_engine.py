"""
ASTRAEUS Observatory Intelligence - High-Precision Visibility & Ephemeris Engine (WS7.5).
Calculates local sidereal time, altitude, azimuth, plane-parallel/refraction airmass,
solar/lunar ephemeris, astronomical twilight, and space-telescope field-of-regard limits.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pydantic import BaseModel, Field

from astraeus_agents.observatory.facility_registry import (
    ObservatoryFacility,
    ObservatoryLocation,
    WavelengthRegime,
)


@dataclass
class TargetVisibilityState:
    """Detailed instantaneous visibility assessment for a target from a facility."""
    facility_id: str
    target_ra_deg: float
    target_dec_deg: float
    epoch_mjd: float
    is_observable: bool
    altitude_deg: float
    azimuth_deg: float
    airmass: float
    solar_altitude_deg: float
    solar_elongation_deg: float
    lunar_distance_deg: float
    lunar_phase: float  # 0.0 = new moon, 1.0 = full moon
    is_astronomical_night: bool
    rejection_reasons: List[str]


class VisibilityEngine:
    """Astronomical visibility and orbital field-of-regard calculator."""

    @staticmethod
    def mjd_to_jd(mjd: float) -> float:
        return mjd + 2400000.5

    @classmethod
    def calculate_lst_hours(cls, mjd: float, longitude_deg: float) -> float:
        """
        Computes Local Sidereal Time (LST) in hours.
        longitude_deg is positive East, negative West.
        """
        jd = cls.mjd_to_jd(mjd)
        d = jd - 2451545.0  # Days since J2000.0
        # Greenwich Mean Sidereal Time in degrees (IAU formula)
        gmst_deg = (280.46061837 + 360.98564736629 * d) % 360.0
        if gmst_deg < 0:
            gmst_deg += 360.0
        
        # Local Sidereal Time
        lst_deg = (gmst_deg + longitude_deg) % 360.0
        if lst_deg < 0:
            lst_deg += 360.0
        return lst_deg / 15.0

    @classmethod
    def calculate_solar_ephemeris(cls, mjd: float) -> Tuple[float, float]:
        """
        Computes geocentric equatorial coordinates (RA_deg, Dec_deg) of the Sun.
        Standard Keplerian approximation accurate to ~0.01 deg.
        """
        n = mjd - 51544.5
        L = (280.460 + 0.9856474 * n) % 360.0
        g = math.radians((357.528 + 0.9856003 * n) % 360.0)
        lam = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2.0 * g))
        eps = math.radians(23.439 - 0.0000004 * n)

        sin_dec = math.sin(eps) * math.sin(lam)
        dec_rad = math.asin(min(1.0, max(-1.0, sin_dec)))

        cos_lam = math.cos(lam)
        sin_lam = math.sin(lam)
        ra_rad = math.atan2(math.cos(eps) * sin_lam, cos_lam)
        ra_deg = (math.degrees(ra_rad)) % 360.0

        return ra_deg, math.degrees(dec_rad)

    @classmethod
    def calculate_lunar_ephemeris(cls, mjd: float) -> Tuple[float, float, float]:
        """
        Computes approximate geocentric Moon coordinates (RA_deg, Dec_deg, phase).
        phase: 0.0 = New, 1.0 = Full.
        """
        d = mjd - 51544.5
        # Mean longitude of Moon and elongation
        L = (218.316 + 13.176396 * d) % 360.0
        M = math.radians((134.963 + 13.064993 * d) % 360.0)
        D = math.radians((297.850 + 12.190749 * d) % 360.0)

        # Geocentric ecliptic longitude of Moon
        l_moon = math.radians(L + 6.289 * math.sin(M))
        b_moon = math.radians(5.128 * math.sin(math.radians((93.272 + 13.229350 * d) % 360.0)))
        eps = math.radians(23.439 - 0.0000004 * d)

        # Equatorial conversion
        sin_dec = math.sin(b_moon) * math.cos(eps) + math.cos(b_moon) * math.sin(eps) * math.sin(l_moon)
        dec_rad = math.asin(min(1.0, max(-1.0, sin_dec)))
        cos_dec = math.cos(dec_rad)
        sin_ra = (math.cos(b_moon) * math.cos(eps) * math.sin(l_moon) - math.sin(b_moon) * math.sin(eps)) / cos_dec
        cos_ra = (math.cos(b_moon) * math.cos(l_moon)) / cos_dec
        ra_deg = math.degrees(math.atan2(sin_ra, cos_ra)) % 360.0

        # Phase approximation: (1 - cos(D)) / 2
        phase = (1.0 - math.cos(D)) / 2.0
        return ra_deg, math.degrees(dec_rad), phase

    @classmethod
    def angular_separation_deg(cls, ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Computes great-circle angular distance in degrees via haversine formula."""
        r1, d1 = math.radians(ra1), math.radians(dec1)
        r2, d2 = math.radians(ra2), math.radians(dec2)
        dra = r2 - r1
        ddec = d2 - d1
        a = math.sin(ddec / 2.0) ** 2 + math.cos(d1) * math.cos(d2) * (math.sin(dra / 2.0) ** 2)
        a = min(1.0, max(0.0, a))
        return math.degrees(2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))

    @classmethod
    def calculate_alt_az(
        cls,
        ra_deg: float,
        dec_deg: float,
        lat_deg: float,
        lst_hours: float,
    ) -> Tuple[float, float]:
        """
        Computes horizontal coordinates: Altitude (-90 to +90 deg) and Azimuth (0 to 360 deg, North=0, East=90).
        """
        phi = math.radians(lat_deg)
        dec = math.radians(dec_deg)

        # Hour angle H in radians
        ha_deg = (lst_hours * 15.0 - ra_deg) % 360.0
        if ha_deg > 180.0:
            ha_deg -= 360.0
        h = math.radians(ha_deg)

        # Altitude
        sin_alt = math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(h)
        sin_alt = min(1.0, max(-1.0, sin_alt))
        alt_rad = math.asin(sin_alt)
        alt_deg = math.degrees(alt_rad)

        # Azimuth
        cos_alt = math.cos(alt_rad)
        if cos_alt > 1e-6:
            cos_az = (math.sin(dec) - math.sin(phi) * math.sin(alt_rad)) / (math.cos(phi) * cos_alt)
            cos_az = min(1.0, max(-1.0, cos_az))
            az_deg = math.degrees(math.acos(cos_az))
            if math.sin(h) > 0:
                az_deg = 360.0 - az_deg
        else:
            az_deg = 0.0

        return alt_deg, az_deg

    @classmethod
    def calculate_refraction_airmass(cls, altitude_deg: float) -> float:
        """
        Calculates atmospheric airmass X(z) taking into account spherical curvature and refraction
        using the Kasten & Young (1989) empirical formulation.
        Returns 99.0 if altitude <= 0 deg.
        """
        if altitude_deg <= 0.0:
            return 99.0
        z_deg = 90.0 - altitude_deg
        # Kasten & Young (1989)
        denominator = math.cos(math.radians(z_deg)) + 0.50572 * ((96.07995 - z_deg) ** -1.6364)
        if denominator <= 0.0:
            return 99.0
        return 1.0 / denominator

    @classmethod
    def evaluate_target_visibility(
        cls,
        facility: ObservatoryFacility,
        target_ra_deg: float,
        target_dec_deg: float,
        epoch_mjd: float,
    ) -> TargetVisibilityState:
        """
        Computes comprehensive physical observability of target from facility at given MJD.
        """
        rejections: List[str] = []
        loc = facility.location

        # 1. Ephemeris of Sun and Moon
        sun_ra, sun_dec = cls.calculate_solar_ephemeris(epoch_mjd)
        moon_ra, moon_dec, moon_phase = cls.calculate_lunar_ephemeris(epoch_mjd)

        solar_elongation = cls.angular_separation_deg(target_ra_deg, target_dec_deg, sun_ra, sun_dec)
        lunar_distance = cls.angular_separation_deg(target_ra_deg, target_dec_deg, moon_ra, moon_dec)

        # 2. Space observatory special handling
        if loc.is_space_based:
            # Space observatories don't experience atmospheric extinction (airmass = 1.0)
            airmass = 1.0
            altitude_deg = 90.0
            azimuth_deg = 0.0
            sun_alt_deg = -90.0
            is_night = True

            # JWST Sun-Earth L2 Field of Regard: 85 to 135 deg from Sun
            if facility.facility_id.upper() == "JWST":
                if solar_elongation < 85.0 or solar_elongation > 135.0:
                    rejections.append(f"JWST Field of Regard violation: solar elongation {solar_elongation:.1f} deg outside [85, 135] deg.")
            elif facility.facility_id.upper() in ["SWIFT", "ASTROSAT"]:
                # LEO Sun avoidance angle
                if solar_elongation < 45.0:
                    rejections.append(f"Space telescope Sun pointing constraint: elongation {solar_elongation:.1f} deg < 45.0 deg.")
                if lunar_distance < 20.0:
                    rejections.append(f"Space telescope Moon pointing constraint: distance {lunar_distance:.1f} deg < 20.0 deg.")

            is_observable = len(rejections) == 0
            return TargetVisibilityState(
                facility_id=facility.facility_id,
                target_ra_deg=target_ra_deg,
                target_dec_deg=target_dec_deg,
                epoch_mjd=epoch_mjd,
                is_observable=is_observable,
                altitude_deg=altitude_deg,
                azimuth_deg=azimuth_deg,
                airmass=airmass,
                solar_altitude_deg=sun_alt_deg,
                solar_elongation_deg=solar_elongation,
                lunar_distance_deg=lunar_distance,
                lunar_phase=moon_phase,
                is_astronomical_night=is_night,
                rejection_reasons=rejections,
            )

        # 3. Ground-based observatory calculations
        lst_hours = cls.calculate_lst_hours(epoch_mjd, loc.longitude_deg)
        alt_deg, az_deg = cls.calculate_alt_az(target_ra_deg, target_dec_deg, loc.latitude_deg, lst_hours)
        airmass = cls.calculate_refraction_airmass(alt_deg)

        sun_alt, _ = cls.calculate_alt_az(sun_ra, sun_dec, loc.latitude_deg, lst_hours)
        is_astro_night = sun_alt <= -18.0

        # Horizon limit check
        if alt_deg < facility.horizon_min_alt_deg:
            rejections.append(f"Target below horizon: altitude {alt_deg:.1f} deg < {facility.horizon_min_alt_deg:.1f} deg limit.")

        # Airmass limit check
        if airmass > facility.max_airmass:
            rejections.append(f"Airmass exceeded: X={airmass:.2f} > max {facility.max_airmass:.2f}.")

        # Waveband daytime constraints:
        # Optical and NIR require astronomical or nautical night; Radio is day/night operable
        has_optical_nir = any(
            inst.regime in [WavelengthRegime.OPTICAL, WavelengthRegime.NIR, WavelengthRegime.UV]
            for inst in facility.instruments.values()
        )
        is_pure_radio = all(inst.regime == WavelengthRegime.RADIO for inst in facility.instruments.values())

        if has_optical_nir and not is_pure_radio:
            if sun_alt > -12.0:
                rejections.append(f"Daylight / twilight: Sun altitude {sun_alt:.1f} deg > -12.0 deg.")

        # Lunar proximity check (avoid pointing within 10 deg of bright Moon)
        if lunar_distance < 10.0 and moon_phase > 0.3:
            rejections.append(f"Lunar blinding risk: target {lunar_distance:.1f} deg from Moon (phase {moon_phase:.2f}).")

        is_observable = len(rejections) == 0

        return TargetVisibilityState(
            facility_id=facility.facility_id,
            target_ra_deg=target_ra_deg,
            target_dec_deg=target_dec_deg,
            epoch_mjd=epoch_mjd,
            is_observable=is_observable,
            altitude_deg=alt_deg,
            azimuth_deg=az_deg,
            airmass=airmass,
            solar_altitude_deg=sun_alt,
            solar_elongation_deg=solar_elongation,
            lunar_distance_deg=lunar_distance,
            lunar_phase=moon_phase,
            is_astronomical_night=is_astro_night,
            rejection_reasons=rejections,
        )
