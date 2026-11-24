"""
ASTRAEUS Scientific Data Fabric - Multi-Wavelength Archive Connectors (WS7.4).
Implements client connectors for:
1. Gaia DR3 Astrometric TAP Archive (RUWE, parallax, proper motion star vetting)
2. GraceDB Gravitational-Wave Archive (LVK multi-messenger notices and skymaps)
3. NASA HEASARC / AstroSat Multi-Wavelength Archive (X-ray/UV upper limits and counterparts)

All connectors enforce strict Gate 3 Quarantine Firewall checks on coordinates.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import time
import math
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from engines.quarantine_firewall import QuarantineFirewall, QuarantineViolationError


class GaiaSourceRecord(BaseModel):
    """Astrometric and photometric measurement record from Gaia DR3."""
    source_id: int
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    ra_error_mas: float = Field(default=0.1)
    dec_error_mas: float = Field(default=0.1)
    parallax_mas: Optional[float] = None
    parallax_error_mas: Optional[float] = None
    pmra_mas_yr: Optional[float] = None
    pmdec_mas_yr: Optional[float] = None
    ruwe: Optional[float] = Field(default=None, description="Renormalized Unit Weight Error (<1.4 indicates single well-behaved star)")
    phot_g_mean_mag: Optional[float] = None
    phot_bp_mean_mag: Optional[float] = None
    phot_rp_mean_mag: Optional[float] = None

    @property
    def total_proper_motion_mas_yr(self) -> Optional[float]:
        if self.pmra_mas_yr is not None and self.pmdec_mas_yr is not None:
            return math.hypot(self.pmra_mas_yr, self.pmdec_mas_yr)
        return None

    @property
    def parallax_snr(self) -> float:
        if self.parallax_mas is not None and self.parallax_error_mas and self.parallax_error_mas > 0:
            return self.parallax_mas / self.parallax_error_mas
        return 0.0

    def is_confirmed_galactic_star(self) -> bool:
        """
        Determines whether source is an unambiguous Galactic foreground star
        ruling out an extragalactic transient progenitor.
        Criteria:
        - Parallax SNR >= 3.0, or
        - Total proper motion > 5.0 mas/yr, or
        - RUWE < 1.40 with detected non-zero parallax and bright G mag (<19.0).
        """
        if self.parallax_snr >= 3.0 and self.parallax_mas and self.parallax_mas > 0.1:
            return True
        pm = self.total_proper_motion_mas_yr
        if pm is not None and pm >= 5.0:
            return True
        if self.ruwe is not None and self.ruwe < 1.40 and self.phot_g_mean_mag and self.phot_g_mean_mag < 19.0:
            if pm is not None and pm >= 2.0:
                return True
        return False


class GaiaTapClient:
    """
    Client for querying Gaia DR3 astrometry via TAP or local cached catalogs,
    with Gate 3 Quarantine Firewall checks and star-vetting logic.
    """

    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self._cached_sources: Dict[int, GaiaSourceRecord] = {}
        self._spatial_index: List[GaiaSourceRecord] = []

    def add_source(self, src: GaiaSourceRecord) -> None:
        self._cached_sources[src.source_id] = src
        self._spatial_index.append(src)

    def query_cone(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_arcsec: float = 3.0,
    ) -> List[GaiaSourceRecord]:
        """Queries Gaia sources around celestial coordinates with Gate 3 Quarantine check."""
        QuarantineFirewall.assert_not_quarantined(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            identifier=f"gaia_cone_{ra_deg:.4f}_{dec_deg:.4f}",
            context="gaia_tap_query",
        )

        radius_deg = radius_arcsec / 3600.0
        matches = []
        for src in self._spatial_index:
            dra = (src.ra_deg - ra_deg) * math.cos(math.radians(dec_deg))
            ddec = src.dec_deg - dec_deg
            sep = math.hypot(dra, ddec)
            if sep <= radius_deg:
                matches.append(src)
        return matches

    def check_galactic_star_contamination(
        self,
        ra_deg: float,
        dec_deg: float,
        search_radius_arcsec: float = 1.5,
    ) -> Tuple[bool, Optional[GaiaSourceRecord]]:
        """
        Returns (is_star, matched_gaia_source).
        If coincident with a confirmed Galactic star, extragalactic transient hypothesis is falsified.
        """
        matches = self.query_cone(ra_deg, dec_deg, radius_arcsec=search_radius_arcsec)
        for m in matches:
            if m.is_confirmed_galactic_star():
                return True, m
        return False, (matches[0] if matches else None)


class GraceDbSuperevent(BaseModel):
    """Gravitational-Wave superevent summary from GraceDB."""
    superevent_id: str
    event_time_mjd: float
    far_hz: float
    prob_bns: float = Field(default=0.0, ge=0.0, le=1.0)
    prob_nsbh: float = Field(default=0.0, ge=0.0, le=1.0)
    prob_bbh: float = Field(default=0.0, ge=0.0, le=1.0)
    prob_terrestrial: float = Field(default=0.0, ge=0.0, le=1.0)
    prob_has_ns: float = Field(default=0.0, ge=0.0, le=1.0)
    prob_has_remnant: float = Field(default=0.0, ge=0.0, le=1.0)
    area_90_deg2: float = Field(default=100.0, ge=0.0)
    distance_mean_mpc: Optional[float] = None
    distance_std_mpc: Optional[float] = None
    central_ra_deg: Optional[float] = None
    central_dec_deg: Optional[float] = None
    skymap_url: Optional[str] = None

    @property
    def is_electromagnetic_promising(self) -> bool:
        """True if event has significant neutron-star remnant probability and low terrestrial false-alarm."""
        remnant_chance = max(self.prob_bns, self.prob_nsbh, self.prob_has_remnant)
        return remnant_chance > 0.40 and self.prob_terrestrial < 0.10


class GraceDbClient:
    """Client for querying GraceDB LVK alerts and skymaps."""

    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self._superevents: Dict[str, GraceDbSuperevent] = {}

    def add_superevent(self, event: GraceDbSuperevent) -> None:
        self._superevents[event.superevent_id] = event

    def get_superevent(self, superevent_id: str) -> Optional[GraceDbSuperevent]:
        ev = self._superevents.get(superevent_id)
        if ev and ev.central_ra_deg is not None and ev.central_dec_deg is not None:
            QuarantineFirewall.assert_not_quarantined(
                ra_deg=ev.central_ra_deg,
                dec_deg=ev.central_dec_deg,
                identifier=f"gracedb_{superevent_id}",
                context="gracedb_query",
            )
        return ev

    def query_events_in_window(
        self,
        mjd_min: float,
        mjd_max: float,
    ) -> List[GraceDbSuperevent]:
        """Finds GW superevents occurring within a specified time window."""
        matches = []
        for ev in self._superevents.values():
            if mjd_min <= ev.event_time_mjd <= mjd_max:
                if ev.central_ra_deg is not None and ev.central_dec_deg is not None:
                    try:
                        QuarantineFirewall.assert_not_quarantined(
                            ra_deg=ev.central_ra_deg,
                            dec_deg=ev.central_dec_deg,
                            identifier=f"gracedb_{ev.superevent_id}",
                            context="gracedb_window_query",
                        )
                    except QuarantineViolationError:
                        continue  # Skip quarantined regions
                matches.append(ev)
        return matches


class MultiWavelengthRecord(BaseModel):
    """High-energy (X-ray/UV/Gamma) observation record from AstroSat/HEASARC."""
    observation_id: str
    telescope: str = Field(..., description="e.g. 'AstroSat', 'Swift', 'Chandra', 'Fermi'")
    instrument: str = Field(..., description="e.g. 'CZTI', 'UVIT', 'XRT', 'SXT'")
    ra_deg: float
    dec_deg: float
    epoch_mjd: float
    exposure_sec: float
    is_detection: bool = False
    flux_erg_cm2_s: Optional[float] = None
    flux_err_erg_cm2_s: Optional[float] = None
    flux_upper_limit_3sigma: Optional[float] = None
    energy_band_kev: Optional[str] = "0.3-10 keV"


class AstroSatHeasarcClient:
    """
    Client for high-energy follow-up catalogs and pointed observations
    from AstroSat (CZTI, SXT, UVIT) and NASA HEASARC (Swift-XRT, Chandra).
    """

    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self._observations: List[MultiWavelengthRecord] = []

    def add_observation(self, obs: MultiWavelengthRecord) -> None:
        self._observations.append(obs)

    def query_counterparts(
        self,
        ra_deg: float,
        dec_deg: float,
        epoch_mjd: float,
        time_window_days: float = 7.0,
        radius_arcsec: float = 30.0,
    ) -> List[MultiWavelengthRecord]:
        """
        Finds coincident multi-wavelength observations within spatial and temporal tolerance.
        Checks Gate 3 Quarantine on coordinates.
        """
        QuarantineFirewall.assert_not_quarantined(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            identifier=f"heasarc_{ra_deg:.4f}_{dec_deg:.4f}",
            context="astrosat_heasarc_query",
        )

        radius_deg = radius_arcsec / 3600.0
        matches = []
        for obs in self._observations:
            dt = abs(obs.epoch_mjd - epoch_mjd)
            if dt > time_window_days:
                continue

            dra = (obs.ra_deg - ra_deg) * math.cos(math.radians(dec_deg))
            ddec = obs.dec_deg - dec_deg
            sep = math.hypot(dra, ddec)
            if sep <= radius_deg:
                matches.append(obs)

        return matches
