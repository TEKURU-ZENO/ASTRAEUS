"""
ASTRAEUS Gaia DR3 Probabilistic Cross-Matching Engine.
Implements Bayesian association accounting for proper-motion epoch propagation,
positional covariance error ellipses, local stellar background density, and chance coincidence.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field


class MatchClassification(str, Enum):
    SECURE_MATCH = "secure_match"
    AMBIGUOUS_MATCH = "ambiguous_match"
    NO_RELIABLE_MATCH = "no_reliable_match"


class GaiaCandidate(BaseModel):
    """
    Gaia DR3 source record at reference epoch J2016.0.
    """
    source_id: int = Field(..., description="Gaia DR3 64-bit source identifier.")
    ra_deg: float = Field(..., ge=0.0, lt=360.0)
    dec_deg: float = Field(..., ge=-90.0, le=90.0)
    ra_error_mas: float = Field(..., ge=0.0)
    dec_error_mas: float = Field(..., ge=0.0)
    pmra_mas_yr: Optional[float] = Field(default=0.0, description="pmra * cos(dec) in mas/yr.")
    pmdec_mas_yr: Optional[float] = Field(default=0.0, description="pmdec in mas/yr.")
    parallax_mas: Optional[float] = Field(default=None)
    phot_g_mean_mag: Optional[float] = Field(default=None)
    ref_epoch_jyr: float = 2016.0


class CrossMatchResult(BaseModel):
    """
    Probabilistic cross-match outcome containing posterior probability,
    separation, chance coincidence, and classification.
    """
    classification: MatchClassification
    best_source_id: Optional[int] = None
    posterior_match_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    separation_arcsec: Optional[float] = None
    chance_coincidence_prob: float = Field(default=0.0, ge=0.0, le=1.0)
    local_source_density_sq_arcsec: float
    propagated_gaia_coords: Optional[Tuple[float, float]] = None
    alternative_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty_assumptions: Dict[str, Any] = Field(default_factory=dict)


class ProbabilisticGaiaMatcher:
    """
    Bayesian counterpart matcher between astronomical transient alerts and Gaia DR3.
    """

    def __init__(self, prior_match_prob: float = 0.50):
        self.p0 = prior_match_prob

    @staticmethod
    def propagate_epoch(
        ra_deg: float,
        dec_deg: float,
        pmra_mas_yr: float,
        pmdec_mas_yr: float,
        t_ref_jyr: float,
        t_target_jyr: float
    ) -> Tuple[float, float]:
        """
        Propagates celestial coordinates from reference epoch to target observation epoch.
        Delta_t in years.
        """
        delta_t = t_target_jyr - t_ref_jyr
        dec_rad = np.radians(dec_deg)
        cos_dec = max(np.cos(dec_rad), 1e-6)

        # Convert mas to degrees: 1 mas = 1 / (3600 * 1000) deg
        mas_to_deg = 1.0 / (3600.0 * 1000.0)
        delta_ra_deg = (pmra_mas_yr * delta_t / cos_dec) * mas_to_deg
        delta_dec_deg = (pmdec_mas_yr * delta_t) * mas_to_deg

        new_ra = (ra_deg + delta_ra_deg) % 360.0
        new_dec = np.clip(dec_deg + delta_dec_deg, -90.0, 90.0)
        return float(new_ra), float(new_dec)

    def match(
        self,
        alert_ra: float,
        alert_dec: float,
        alert_ra_err_arcsec: float,
        alert_dec_err_arcsec: float,
        alert_epoch_jyr: float,
        gaia_candidates: List[GaiaCandidate],
        local_density_sources_per_sq_arcsec: float,
        search_radius_arcsec: float = 3.0
    ) -> CrossMatchResult:
        """
        Computes Bayesian counterpart probabilities over candidate Gaia sources.
        """
        rho = max(local_density_sources_per_sq_arcsec, 1e-8)
        if not gaia_candidates:
            return CrossMatchResult(
                classification=MatchClassification.NO_RELIABLE_MATCH,
                local_source_density_sq_arcsec=rho,
                uncertainty_assumptions={"reason": "No Gaia candidates within search cone"},
            )

        candidate_evaluations: List[Dict[str, Any]] = []

        for cand in gaia_candidates:
            # Step 1: Propagate Gaia coordinates to alert epoch
            pmra = cand.pmra_mas_yr or 0.0
            pmdec = cand.pmdec_mas_yr or 0.0
            prop_ra, prop_dec = self.propagate_epoch(
                cand.ra_deg, cand.dec_deg, pmra, pmdec, cand.ref_epoch_jyr, alert_epoch_jyr
            )

            # Step 2: Angular separation in arcseconds
            dec_rad = np.radians((alert_dec + prop_dec) / 2.0)
            dra = (alert_ra - prop_ra) * np.cos(dec_rad) * 3600.0  # arcsec
            ddec = (alert_dec - prop_dec) * 3600.0                 # arcsec
            sep = float(np.sqrt(dra**2 + ddec**2))

            if sep > search_radius_arcsec:
                continue

            # Step 3: Convolved positional covariance in arcsec^2
            gaia_ra_err = (cand.ra_error_mas / 1000.0)
            gaia_dec_err = (cand.dec_error_mas / 1000.0)
            sigma_ra2 = alert_ra_err_arcsec**2 + gaia_ra_err**2
            sigma_dec2 = alert_dec_err_arcsec**2 + gaia_dec_err**2
            det_sigma = sigma_ra2 * sigma_dec2

            # Step 4: Mahalanobis separation distance
            chi2 = (dra**2 / sigma_ra2) + (ddec**2 / sigma_dec2)
            likelihood_position = np.exp(-0.5 * chi2)

            # Chance coincidence in area pi * sep^2
            p_chance = float(1.0 - np.exp(-np.pi * (sep**2) * rho))

            # Bayesian likelihood ratio:
            # L_ratio = p(dra, ddec | match) / p(dra, ddec | background)
            # where background density = rho, match model = 2D Gaussian
            p_match_given_pos = likelihood_position / (2.0 * np.pi * np.sqrt(det_sigma))
            p_background = rho

            # Posterior probability via Bayes rule with prior p0
            numerator = p_match_given_pos * self.p0
            denominator = numerator + p_background * (1.0 - self.p0)
            p_posterior = float(numerator / max(denominator, 1e-12))
            p_posterior = np.clip(p_posterior, 0.0, 1.0)

            candidate_evaluations.append({
                "source_id": cand.source_id,
                "posterior_prob": p_posterior,
                "separation_arcsec": sep,
                "chance_coincidence": p_chance,
                "prop_ra": prop_ra,
                "prop_dec": prop_dec,
                "phot_g_mag": cand.phot_g_mean_mag,
            })

        if not candidate_evaluations:
            return CrossMatchResult(
                classification=MatchClassification.NO_RELIABLE_MATCH,
                local_source_density_sq_arcsec=rho,
                uncertainty_assumptions={"reason": f"No candidate within search radius {search_radius_arcsec} arcsec"},
            )

        # Sort descending by posterior probability
        candidate_evaluations.sort(key=lambda x: x["posterior_prob"], reverse=True)
        best = candidate_evaluations[0]

        # Tri-state classification
        if best["posterior_prob"] >= 0.90:
            # Check for ambiguity: is second candidate also plausible?
            if len(candidate_evaluations) > 1 and candidate_evaluations[1]["posterior_prob"] >= 0.15:
                classification = MatchClassification.AMBIGUOUS_MATCH
            else:
                classification = MatchClassification.SECURE_MATCH
        elif best["posterior_prob"] >= 0.50:
            classification = MatchClassification.AMBIGUOUS_MATCH
        else:
            classification = MatchClassification.NO_RELIABLE_MATCH

        alternatives = candidate_evaluations[1:] if len(candidate_evaluations) > 1 else []

        return CrossMatchResult(
            classification=classification,
            best_source_id=best["source_id"] if classification != MatchClassification.NO_RELIABLE_MATCH else None,
            posterior_match_prob=best["posterior_prob"],
            separation_arcsec=best["separation_arcsec"],
            chance_coincidence_prob=best["chance_coincidence"],
            local_source_density_sq_arcsec=rho,
            propagated_gaia_coords=(best["prop_ra"], best["prop_dec"]),
            alternative_candidates=alternatives,
            uncertainty_assumptions={
                "search_radius_arcsec": search_radius_arcsec,
                "prior_p0": self.p0,
                "epoch_propagated": True,
            }
        )
