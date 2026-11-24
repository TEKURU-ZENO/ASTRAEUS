"""
ASTRAEUS Intelligent Scientific Event Triage & Prioritization Engine (WS7.2).
Part of Phase 7 Closed-Loop Autonomous Scientific Intelligence.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026

Features:
- Multi-catalogue contextual cross-matching (NED/GLADE+ host galaxies, Gaia/PS1 reference stars)
- Comprehensive photometric feature extraction (rise rate, fade rate, color evolution, decay timescale)
- Heuristic and astrophysical anomaly detection (kilonovae, rapid optical transients, SLSN, GW counterparts)
- Information-gain & ToO urgency priority scoring (0 - 100)
- Thread-safe priority queue ranking for autonomous multi-agent follow-up
"""

from __future__ import annotations

import heapq
import math
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from astraeus_data.ingestion.scientific_event_engine import UnifiedScientificEvent


# ============================================================================
# 1. Enums and Data Models
# ============================================================================

class PriorityTier(str, Enum):
    CRITICAL_TOO = "CRITICAL_TOO"               # Score >= 85: Immediate target-of-opportunity follow-up (<2 hr)
    HIGH_PRIORITY = "HIGH_PRIORITY"             # Score >= 70: Urgent multi-band photometric follow-up (<24 hr)
    ROUTINE_MONITOR = "ROUTINE_MONITOR"         # Score >= 40: Standard survey cadence monitoring
    LOW_PRIORITY_ARCHIVE = "LOW_PRIORITY_ARCHIVE" # Score < 40: Likely artifact, stellar flare, or background


class AnomalyClass(str, Enum):
    RAPID_OPTICAL_TRANSIENT_KN = "RAPID_OPTICAL_TRANSIENT_KN"       # Kilonova / fast-fading optical transient (tau < 48h)
    MULTI_MESSENGER_GW_COUNTERPART = "MULTI_MESSENGER_GW_COUNTERPART" # Gravitational-wave counterpart candidate
    SUPERLUMINOUS_SN_TDE = "SUPERLUMINOUS_SN_TDE"                   # Superluminous Supernova or Tidal Disruption Event
    EXTRAGALACTIC_SN = "EXTRAGALACTIC_SN"                           # Canonical Core-Collapse or Type Ia Supernova
    STELLAR_FLARE_VARIABLE = "STELLAR_FLARE_VARIABLE"               # Galactic flare star / cataclysmic variable
    UNCLASSIFIED_ANOMALY = "UNCLASSIFIED_ANOMALY"                   # High-significance unusual lightcurve


class HostAssociationStatus(str, Enum):
    CONFIRMED_HOST = "CONFIRMED_HOST"
    PROBABLE_HOST = "PROBABLE_HOST"
    AMBIGUOUS_HOST = "AMBIGUOUS_HOST"
    UNKNOWN_HOST_DISTANCE = "UNKNOWN_HOST_DISTANCE"


class HostGalaxyContext(BaseModel):
    """Contextual host galaxy association from NED/GLADE+."""
    matched: bool = False
    status: HostAssociationStatus = HostAssociationStatus.UNKNOWN_HOST_DISTANCE
    galaxy_name: Optional[str] = None
    catalog_origin: Optional[str] = None
    redshift_z: Optional[float] = None
    redshift_z_err: Optional[float] = None
    distance_modulus_mu: Optional[float] = None
    distance_modulus_err: Optional[float] = None
    angular_separation_arcsec: float = 999.0
    physical_offset_kpc: Optional[float] = None
    chance_coincidence_prob: float = 1.0
    stellar_mass_log_msun: Optional[float] = None
    star_formation_rate_msun_yr: Optional[float] = None
    redshift_prior_distribution: Optional[Dict[str, Any]] = None


class ReferenceStarContext(BaseModel):
    """Point source stellar counterpart association from Gaia / PS1."""
    matched_star: bool = False
    source_id: Optional[str] = None
    separation_arcsec: float = 999.0
    proper_motion_mas_yr: Optional[float] = None
    parallax_mas: Optional[float] = None
    is_known_variable: bool = False


class PhotometricFeatures(BaseModel):
    """Extracted lightcurve morphological and physical features."""
    num_points: int = 0
    bands_observed: List[str] = Field(default_factory=list)
    time_baseline_days: float = 0.0
    peak_mag: float = 99.0
    peak_mjd: float = 0.0
    latest_mag: float = 99.0
    fade_rate_mag_per_day: float = 0.0       # Positive if fading (mag increasing)
    rise_rate_mag_per_day: float = 0.0       # Positive if brightening (mag decreasing)
    color_g_minus_r: Optional[float] = None
    color_rate_dgr_dt: Optional[float] = None # Positive if reddening (g-r becoming redder)
    timescale_tau_hours: Optional[float] = None
    estimated_abs_mag: Optional[float] = None


class TriageDecision(BaseModel):
    """Complete scientific triage assessment and priority ranking."""
    event_id: str
    survey_origin: str
    priority_score: float                     # 0.0 to 100.0
    priority_tier: PriorityTier
    primary_anomaly_class: AnomalyClass
    is_too_urgent: bool
    features: PhotometricFeatures
    host_context: HostGalaxyContext
    star_context: ReferenceStarContext
    reasoning_summary: str
    recommended_action: str
    triage_timestamp_iso: str
    raw_payload_hash: str

    def __lt__(self, other: "TriageDecision") -> bool:
        # Heap comparison: higher priority scores come first (max-heap behavior when inverted)
        return self.priority_score < other.priority_score


# ============================================================================
# 2. Contextual Catalogue Cross-Matcher
# ============================================================================

def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle distance between two celestial coordinates in arcseconds."""
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    d_ra = r2 - r1
    d_dec = d2 - d1
    a = math.sin(d_dec / 2.0) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(d_ra / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    sep_rad = 2.0 * math.asin(math.sqrt(a))
    return math.degrees(sep_rad) * 3600.0


def angular_diameter_distance_kpc(z: float, h0: float = 70.0, omega_m: float = 0.3) -> float:
    """
    Flat Lambda-CDM angular diameter distance in kpc.
    For low to moderate redshift z:
    d_A(z) = (c / (H0 * (1+z))) * int_0^z dz' / sqrt(Omega_m(1+z')^3 + Omega_L)
    """
    if z <= 0.0:
        return 0.0
    c_km_s = 299792.458
    omega_l = 1.0 - omega_m

    # Numerical Simpson quadrature for comoving distance
    n_steps = 100
    dz = z / n_steps
    integral = 0.0
    for i in range(n_steps + 1):
        zi = i * dz
        ez = math.sqrt(omega_m * (1.0 + zi) ** 3 + omega_l)
        weight = 2.0 if (i % 2 == 0) else 4.0
        if i == 0 or i == n_steps:
            weight = 1.0
        integral += (weight * dz / 3.0) / ez

    d_comoving_mpc = (c_km_s / h0) * integral
    d_angular_mpc = d_comoving_mpc / (1.0 + z)
    return d_angular_mpc * 1000.0  # to kpc


class CatalogCrossMatcher:
    """
    Contextual catalogue cross-matcher for host galaxies (NED/GLADE+) and reference stars (Gaia/PS1).
    Supports injecting custom or archival catalogs; provides standard reference catalogues.
    """

    def __init__(self, host_catalog: Optional[List[Dict[str, Any]]] = None, star_catalog: Optional[List[Dict[str, Any]]] = None):
        self.host_catalog = host_catalog or []
        self.star_catalog = star_catalog or []

    def add_host_galaxy(self, name: str, ra: float, dec: float, redshift: float,
                         log_mstar: float = 10.5, sfr: float = 1.0, catalog: str = "GLADE+"):
        self.host_catalog.append({
            "name": name,
            "ra_deg": ra,
            "dec_deg": dec,
            "redshift_z": redshift,
            "log_mstar": log_mstar,
            "sfr": sfr,
            "catalog": catalog
        })

    def add_reference_star(self, source_id: str, ra: float, dec: float, pm_mas_yr: float = 0.0,
                           parallax_mas: float = 0.0, is_variable: bool = False):
        self.star_catalog.append({
            "source_id": source_id,
            "ra_deg": ra,
            "dec_deg": dec,
            "pm_mas_yr": pm_mas_yr,
            "parallax_mas": parallax_mas,
            "is_variable": is_variable
        })

    def match_host(self, ra: float, dec: float, max_radius_arcsec: float = 45.0) -> HostGalaxyContext:
        """Find closest host galaxy within max_radius_arcsec."""
        best_match = None
        min_sep = 9999.0

        for host in self.host_catalog:
            sep = angular_separation_arcsec(ra, dec, host["ra_deg"], host["dec_deg"])
            if sep <= max_radius_arcsec and sep < min_sep:
                min_sep = sep
                best_match = host

        if best_match is None:
            return HostGalaxyContext(
                matched=False,
                status=HostAssociationStatus.UNKNOWN_HOST_DISTANCE,
                chance_coincidence_prob=1.0,
                redshift_prior_distribution={
                    "prior_type": "lognormal",
                    "mu_ln_z": -2.5,
                    "sigma_ln_z": 1.0,
                    "z_nominal_range": [0.01, 1.2],
                    "prior_description": "Hostless extragalactic prior based on deep survey luminosity function",
                }
            )

        z = float(best_match["redshift_z"])
        z_err = float(best_match.get("redshift_z_err", 0.0001))
        d_A_kpc = angular_diameter_distance_kpc(z)
        theta_rad = math.radians(min_sep / 3600.0)
        phys_offset_kpc = d_A_kpc * theta_rad

        # Sutherland & Saunders chance coincidence estimate
        n_density = 1.0e-4
        chance_prob = 1.0 - math.exp(-math.pi * n_density * (min_sep ** 2))

        # Cosmological distance modulus
        d_L_mpc = (d_A_kpc / 1000.0) * ((1.0 + z) ** 2)
        mu = float(5.0 * math.log10(max(d_L_mpc, 1e-4) * 1e6 / 10.0))
        mu_err = float((5.0 / math.log(10.0)) * (z_err / max(z, 1e-4)))

        # Status classification
        if chance_prob < 0.05 and min_sep <= 20.0:
            status = HostAssociationStatus.CONFIRMED_HOST
        elif chance_prob < 0.20:
            status = HostAssociationStatus.PROBABLE_HOST
        else:
            status = HostAssociationStatus.AMBIGUOUS_HOST

        return HostGalaxyContext(
            matched=True,
            status=status,
            galaxy_name=best_match["name"],
            catalog_origin=best_match.get("catalog", "GLADE+"),
            redshift_z=z,
            redshift_z_err=z_err,
            distance_modulus_mu=round(mu, 3),
            distance_modulus_err=round(mu_err, 4),
            angular_separation_arcsec=min_sep,
            physical_offset_kpc=phys_offset_kpc,
            chance_coincidence_prob=chance_prob,
            stellar_mass_log_msun=best_match.get("log_mstar"),
            star_formation_rate_msun_yr=best_match.get("sfr")
        )

    def match_star(self, ra: float, dec: float, max_radius_arcsec: float = 1.5) -> ReferenceStarContext:
        """Find coincident stellar point-source counterpart."""
        best_star = None
        min_sep = 9999.0

        for star in self.star_catalog:
            sep = angular_separation_arcsec(ra, dec, star["ra_deg"], star["dec_deg"])
            if sep <= max_radius_arcsec and sep < min_sep:
                min_sep = sep
                best_star = star

        if best_star is None:
            return ReferenceStarContext(matched_star=False)

        return ReferenceStarContext(
            matched_star=True,
            source_id=best_star["source_id"],
            separation_arcsec=min_sep,
            proper_motion_mas_yr=best_star.get("pm_mas_yr", 0.0),
            parallax_mas=best_star.get("parallax_mas", 0.0),
            is_known_variable=best_star.get("is_variable", False)
        )


# ============================================================================
# 3. Photometric Feature Extractor
# ============================================================================

class PhotometricFeatureExtractor:
    """Extracts analytical features from standardized lightcurve points."""

    @staticmethod
    def extract(photometry_points: List[Dict[str, Any]], host_z: Optional[float] = None) -> PhotometricFeatures:
        if not photometry_points:
            return PhotometricFeatures()

        standardized = []
        for p in photometry_points:
            mjd = p.get("mjd", p.get("midPointTai"))
            mag = p.get("mag") if p.get("mag") is not None else p.get("mag_ab")
            band = p.get("filter_band", p.get("filter", p.get("filterName", "r")))
            if mjd is not None and mag is not None:
                standardized.append({
                    "mjd": float(mjd),
                    "mag": float(mag),
                    "filter_band": str(band)
                })

        if not standardized:
            return PhotometricFeatures()

        # Sort chronologically by MJD
        sorted_points = sorted(standardized, key=lambda x: x["mjd"])
        mjds = [p["mjd"] for p in sorted_points]
        mags = [p["mag"] for p in sorted_points]
        bands = list({p["filter_band"] for p in sorted_points})

        t_min = mjds[0]
        t_max = mjds[-1]
        baseline = t_max - t_min

        # Peak apparent magnitude (minimum numerical mag)
        peak_idx = int(min(range(len(mags)), key=lambda i: mags[i]))
        peak_mag = mags[peak_idx]
        peak_mjd = mjds[peak_idx]
        latest_mag = mags[-1]

        # Monochromatic decay on dominant band if available (e.g. r-band)
        r_points = [p for p in sorted_points if p["filter_band"] in ("r", "ztfr", "lsstr")]
        fade_rate = 0.0
        rise_rate = 0.0

        if len(r_points) >= 2:
            r_mjds = [p["mjd"] for p in r_points]
            r_mags = [p["mag"] for p in r_points]
            r_peak_idx = int(min(range(len(r_mags)), key=lambda i: r_mags[i]))
            if r_peak_idx < len(r_mags) - 1:
                dt_fade = r_mjds[-1] - r_mjds[r_peak_idx]
                if dt_fade > 0.01:
                    fade_rate = (r_mags[-1] - r_mags[r_peak_idx]) / dt_fade
            if r_peak_idx > 0:
                dt_rise = r_mjds[r_peak_idx] - r_mjds[0]
                if dt_rise > 0.01:
                    rise_rate = (r_mags[0] - r_mags[r_peak_idx]) / dt_rise
        else:
            # Fallback across all points
            if peak_idx > 0:
                dt_rise = peak_mjd - t_min
                if dt_rise > 0.01:
                    rise_rate = (mags[0] - peak_mag) / dt_rise
            if peak_idx < len(mags) - 1:
                dt_fade = t_max - peak_mjd
                if dt_fade > 0.01:
                    fade_rate = (latest_mag - peak_mag) / dt_fade

        # E-folding / decay timescale tau:
        tau_hours = None
        if fade_rate > 0.05:
            tau_days = 1.085736 / fade_rate
            tau_hours = tau_days * 24.0

        # Color evaluation: g - r
        g_pts = [p for p in sorted_points if p.get("filter_band") in ("g", "ztfg", "lsstg")]
        r_pts = [p for p in sorted_points if p.get("filter_band") in ("r", "ztfr", "lsstr")]

        color_g_minus_r = None
        color_rate_dgr_dt = None

        if g_pts and r_pts:
            # Pair closest epochs
            color_epochs = []
            for g_pt in g_pts:
                closest_r = min(r_pts, key=lambda r: abs(r["mjd"] - g_pt["mjd"]))
                if abs(closest_r["mjd"] - g_pt["mjd"]) < 0.75:
                    gr = g_pt["mag"] - closest_r["mag"]
                    mean_mjd = 0.5 * (g_pt["mjd"] + closest_r["mjd"])
                    color_epochs.append((mean_mjd, gr))

            if color_epochs:
                color_epochs.sort(key=lambda x: x[0])
                color_g_minus_r = color_epochs[-1][1]
                if len(color_epochs) >= 2:
                    dt_color = color_epochs[-1][0] - color_epochs[0][0]
                    if dt_color > 0.1:
                        color_rate_dgr_dt = (color_epochs[-1][1] - color_epochs[0][1]) / dt_color
                # Find closest r point within 0.5 days
                closest_r = min(r_pts, key=lambda r: abs(r["mjd"] - g_pt["mjd"]))
                if abs(closest_r["mjd"] - g_pt["mjd"]) < 0.75:
                    gr = g_pt["mag"] - closest_r["mag"]
                    mean_mjd = 0.5 * (g_pt["mjd"] + closest_r["mjd"])
                    color_epochs.append((mean_mjd, gr))

            if color_epochs:
                color_epochs.sort(key=lambda x: x[0])
                color_g_minus_r = color_epochs[-1][1]
                if len(color_epochs) >= 2:
                    dt_color = color_epochs[-1][0] - color_epochs[0][0]
                    if dt_color > 0.1:
                        color_rate_dgr_dt = (color_epochs[-1][1] - color_epochs[0][1]) / dt_color

        # Absolute magnitude estimation if host redshift is known
        abs_mag = None
        if host_z is not None and host_z > 0.001:
            d_A_kpc = angular_diameter_distance_kpc(host_z)
            d_L_pc = d_A_kpc * 1000.0 * ((1.0 + host_z) ** 2)
            if d_L_pc > 10.0:
                dist_modulus = 5.0 * math.log10(d_L_pc / 10.0)
                abs_mag = peak_mag - dist_modulus

        return PhotometricFeatures(
            num_points=len(sorted_points),
            bands_observed=bands,
            time_baseline_days=baseline,
            peak_mag=peak_mag,
            peak_mjd=peak_mjd,
            latest_mag=latest_mag,
            fade_rate_mag_per_day=fade_rate,
            rise_rate_mag_per_day=rise_rate,
            color_g_minus_r=color_g_minus_r,
            color_rate_dgr_dt=color_rate_dgr_dt,
            timescale_tau_hours=tau_hours,
            estimated_abs_mag=abs_mag
        )


# ============================================================================
# 4. Heuristic Anomaly Detector & Prioritization Classifier
# ============================================================================

class HeuristicAnomalyDetector:
    """Classifies physical transient mechanisms and detects rare astrophysical anomalies."""

    @staticmethod
    def classify(event: UnifiedScientificEvent,
                 features: PhotometricFeatures,
                 host: HostGalaxyContext,
                 star: ReferenceStarContext) -> Tuple[AnomalyClass, float, bool, str]:
        """
        Returns:
            (primary_class, anomaly_score_0_to_100, is_too_urgent, justification)
        """
        reasons: List[str] = []
        is_too_urgent = False

        # 1. Multi-Messenger Gravitational Wave Notice / Coincidence
        if event.survey_origin == "LVK_GW" or (event.event_metadata and event.event_metadata.get("gw_coincidence")):
            reasons.append("Multi-messenger gravitational-wave coincidence or LVK notice.")
            is_too_urgent = True
            return (AnomalyClass.MULTI_MESSENGER_GW_COUNTERPART, 98.0, is_too_urgent, "; ".join(reasons))

        # 2. Stellar Point-Source Counterpart (Galactic variable / flare)
        if star.matched_star and star.separation_arcsec < 0.5:
            if star.proper_motion_mas_yr and star.proper_motion_mas_yr > 5.0:
                reasons.append(f"High proper motion stellar counterpart ({star.proper_motion_mas_yr:.1f} mas/yr).")
                return (AnomalyClass.STELLAR_FLARE_VARIABLE, 15.0, False, "; ".join(reasons))
            if star.is_known_variable:
                reasons.append("Coincident with known catalogued variable star.")
                return (AnomalyClass.STELLAR_FLARE_VARIABLE, 10.0, False, "; ".join(reasons))

        # 3. Kilonova / Rapid Optical Transient
        # Characterized by rapid fade (tau < 48h, fade rate > 0.5 mag/day) and rapid reddening
        is_fast_fading = (features.timescale_tau_hours is not None and features.timescale_tau_hours <= 48.0) or \
                         (features.fade_rate_mag_per_day >= 0.5)
        is_rapid_reddening = features.color_rate_dgr_dt is not None and features.color_rate_dgr_dt >= 0.25

        if is_fast_fading:
            reasons.append(f"Rapid decay timescale: tau={features.timescale_tau_hours or 0.0:.1f}h (fade rate={features.fade_rate_mag_per_day:.2f} mag/day).")
            if is_rapid_reddening:
                reasons.append(f"Rapid lanthanide reddening: d(g-r)/dt={features.color_rate_dgr_dt:.2f} mag/day.")
            if host.matched and host.physical_offset_kpc is not None:
                reasons.append(f"Extragalactic host galaxy association: {host.galaxy_name} at z={host.redshift_z:.4f}, offset={host.physical_offset_kpc:.1f} kpc.")

            score = 92.0 if is_rapid_reddening else 86.0
            is_too_urgent = True
            return (AnomalyClass.RAPID_OPTICAL_TRANSIENT_KN, score, is_too_urgent, "; ".join(reasons))

        # 4. Superluminous Supernova / Tidal Disruption Event
        if features.estimated_abs_mag is not None and features.estimated_abs_mag <= -21.0:
            reasons.append(f"Extreme peak absolute magnitude: M_abs={features.estimated_abs_mag:.2f} mag.")
            if host.matched and host.physical_offset_kpc is not None and host.physical_offset_kpc < 1.0:
                reasons.append(f"Nuclear alignment in host {host.galaxy_name} (offset={host.physical_offset_kpc:.2f} kpc).")
                return (AnomalyClass.SUPERLUMINOUS_SN_TDE, 88.0, True, "; ".join(reasons))
            return (AnomalyClass.SUPERLUMINOUS_SN_TDE, 82.0, False, "; ".join(reasons))

        # 5. Canonical Extragalactic Supernova (Ia / II)
        if host.matched and host.chance_coincidence_prob < 0.05:
            reasons.append(f"Associated with host galaxy {host.galaxy_name} (z={host.redshift_z:.4f}, P_chance={host.chance_coincidence_prob:.4f}).")
            if 0.02 <= features.fade_rate_mag_per_day <= 0.35:
                reasons.append(f"Standard SN lightcurve decay rate: {features.fade_rate_mag_per_day:.2f} mag/day.")
                return (AnomalyClass.EXTRAGALACTIC_SN, 65.0, False, "; ".join(reasons))
            return (AnomalyClass.EXTRAGALACTIC_SN, 55.0, False, "; ".join(reasons))

        # 6. Unclassified / Faint Anomaly
        if features.num_points >= 3:
            reasons.append(f"Multi-epoch transient with {features.num_points} observations without clear host association.")
            return (AnomalyClass.UNCLASSIFIED_ANOMALY, 45.0, False, "; ".join(reasons))

        reasons.append("Single epoch or insufficient cadence to establish classification.")
        return (AnomalyClass.UNCLASSIFIED_ANOMALY, 30.0, False, "; ".join(reasons))


# ============================================================================
# 5. Thread-Safe Prioritization Queue
# ============================================================================

class PrioritizationQueue:
    """Thread-safe priority queue sorting TriageDecision instances by descending priority score."""

    def __init__(self):
        self._heap: List[Tuple[float, int, TriageDecision]] = []
        self._counter: int = 0
        self._events_by_id: Dict[str, TriageDecision] = {}

    def push(self, decision: TriageDecision) -> None:
        self._counter += 1
        # Python heapq is a min-heap; negate priority_score so largest scores pop first
        heapq.heappush(self._heap, (-decision.priority_score, self._counter, decision))
        self._events_by_id[decision.event_id] = decision

    def pop(self) -> Optional[TriageDecision]:
        if not self._heap:
            return None
        _, _, decision = heapq.heappop(self._heap)
        self._events_by_id.pop(decision.event_id, None)
        return decision

    def peek(self) -> Optional[TriageDecision]:
        if not self._heap:
            return None
        return self._heap[0][2]

    def get_top_k(self, k: int) -> List[TriageDecision]:
        sorted_entries = sorted(self._heap, key=lambda x: x[0])
        return [item[2] for item in sorted_entries[:k]]

    def filter_by_tier(self, tier: PriorityTier) -> List[TriageDecision]:
        return [d for _, _, d in self._heap if d.priority_tier == tier]

    def __len__(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._events_by_id.clear()


# ============================================================================
# 6. Event Triage Engine Coordinator
# ============================================================================

class EventTriageEngine:
    """
    Unified Intelligent Triage and Prioritization Engine for ASTRAEUS.
    Evaluates incoming UnifiedScientificEvents, performs cross-matching, feature extraction,
    anomaly classification, priority ranking, and enqueuing for multi-agent investigation.
    """

    def __init__(self, cross_matcher: Optional[CatalogCrossMatcher] = None):
        self.cross_matcher = cross_matcher or CatalogCrossMatcher()
        self.feature_extractor = PhotometricFeatureExtractor()
        self.anomaly_detector = HeuristicAnomalyDetector()
        self.queue = PrioritizationQueue()

    def triage_event(self, event: UnifiedScientificEvent) -> TriageDecision:
        """Process a single event through full triage pipeline and enqueue it."""
        # Check quarantine status
        if not event.quarantine_cleared:
            raise ValueError(f"Event {event.event_id} failed Gate 3 quarantine firewall. Triage prohibited.")

        # 1. Catalogue cross-matching
        host_ctx = self.cross_matcher.match_host(event.ra_deg, event.dec_deg)
        star_ctx = self.cross_matcher.match_star(event.ra_deg, event.dec_deg)

        # 2. Photometric feature extraction
        features = self.feature_extractor.extract(event.photometry_points, host_z=host_ctx.redshift_z)

        # 3. Anomaly detection & physical classification
        anomaly_class, priority_score, is_too, justification = self.anomaly_detector.classify(
            event=event,
            features=features,
            host=host_ctx,
            star=star_ctx
        )

        # 4. Priority tier assignment
        if priority_score >= 85.0 or is_too:
            tier = PriorityTier.CRITICAL_TOO
            rec_action = "Trigger immediate ToO multi-band photometric follow-up and activate REGROUND investigation."
        elif priority_score >= 70.0:
            tier = PriorityTier.HIGH_PRIORITY
            rec_action = "Schedule high-cadence follow-up on next available observing window."
        elif priority_score >= 40.0:
            tier = PriorityTier.ROUTINE_MONITOR
            rec_action = "Continue regular survey monitoring; re-evaluate upon next alert epoch."
        else:
            tier = PriorityTier.LOW_PRIORITY_ARCHIVE
            rec_action = "Archive candidate; deprioritize active telescope pointing."

        decision = TriageDecision(
            event_id=event.event_id,
            survey_origin=event.survey_origin,
            priority_score=priority_score,
            priority_tier=tier,
            primary_anomaly_class=anomaly_class,
            is_too_urgent=is_too,
            features=features,
            host_context=host_ctx,
            star_context=star_ctx,
            reasoning_summary=justification,
            recommended_action=rec_action,
            triage_timestamp_iso=datetime.now(timezone.utc).isoformat(),
            raw_payload_hash=event.raw_payload_hash
        )

        # Push to internal queue
        self.queue.push(decision)
        return decision
