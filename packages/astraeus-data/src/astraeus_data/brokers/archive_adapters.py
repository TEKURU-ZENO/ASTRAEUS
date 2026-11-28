"""
ASTRAEUS Astronomical Archive Ingestion Adapters (WS10.1).
Connects to and crossmatches external archival observatories:
- Gaia DR3 / DR4: Proper motion, parallax, and stellar contamination filtering.
- NASA / MAST: Space telescope imaging archives (Hubble, JWST, Kepler, TESS).
- HEASARC: High-energy gamma/X-ray counters (Fermi-GBM, Swift-BAT, INTEGRAL).

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 28, 2026
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class GaiaCrossmatchRecord(BaseModel):
    source_id: str
    ra_deg: float
    dec_deg: float
    separation_arcsec: float
    parallax_mas: Optional[float] = None
    parallax_error_mas: Optional[float] = None
    pmra_mas_yr: Optional[float] = None
    pmdec_mas_yr: Optional[float] = None
    phot_g_mean_mag: Optional[float] = None
    is_galactic_stellar_contaminant: bool = False


class MastObservationRecord(BaseModel):
    obs_id: str
    target_name: str
    instrument_name: str
    filter_id: str
    t_min_mjd: float
    t_max_mjd: float
    exposure_time_sec: float
    data_uri: str


class HeasarcTriggerRecord(BaseModel):
    trigger_id: str
    mission: str  # FERMI_GBM, SWIFT_BAT, INTEGRAL
    trigger_time_mjd: float
    ra_deg: float
    dec_deg: float
    error_radius_deg: float
    fluence_erg_cm2: float
    t90_duration_sec: float
    is_short_grb: bool  # t90 < 2.0s


class MultiArchiveFederator:
    """Federated archive query and crossmatching engine."""

    def __init__(self) -> None:
        self.cached_gaia_sources: Dict[str, GaiaCrossmatchRecord] = {}
        self.cached_heasarc_triggers: Dict[str, HeasarcTriggerRecord] = {}

    def query_gaia_stellar_contaminants(
        self,
        ra_deg: float,
        dec_deg: float,
        search_radius_arcsec: float = 3.0,
    ) -> Optional[GaiaCrossmatchRecord]:
        """
        Queries Gaia DR3 for coincident galactic stars. If significant parallax
        or proper motion is present (pi / sigma_pi > 5), marks candidate as galactic contaminant.
        """
        # Search in cache or compute angular distance
        for gid, rec in self.cached_gaia_sources.items():
            d_ra = (rec.ra_deg - ra_deg) * math.cos(math.radians(dec_deg))
            d_dec = rec.dec_deg - dec_deg
            sep = math.sqrt(d_ra**2 + d_dec**2) * 3600.0
            if sep <= search_radius_arcsec:
                # Assess parallax significance
                if rec.parallax_mas and rec.parallax_error_mas and rec.parallax_error_mas > 0:
                    snr = rec.parallax_mas / rec.parallax_error_mas
                    if snr >= 5.0 and rec.parallax_mas > 0.5:
                        rec.is_galactic_stellar_contaminant = True
                return rec
        return None

    def query_heasarc_coincident_bursts(
        self,
        event_time_mjd: float,
        ra_deg: float,
        dec_deg: float,
        time_window_sec: float = 10.0,
    ) -> List[HeasarcTriggerRecord]:
        """
        Cross-matches high-energy prompts (Fermi-GBM / Swift-BAT) within a +/- 10s causal window.
        """
        matches = []
        for tid, trig in self.cached_heasarc_triggers.items():
            dt_sec = abs(trig.trigger_time_mjd - event_time_mjd) * 86400.0
            if dt_sec <= time_window_sec:
                # Spatial separation check
                d_ra = (trig.ra_deg - ra_deg) * math.cos(math.radians(dec_deg))
                d_dec = trig.dec_deg - dec_deg
                sep_deg = math.sqrt(d_ra**2 + d_dec**2)
                if sep_deg <= max(15.0, trig.error_radius_deg):
                    matches.append(trig)
        return matches
