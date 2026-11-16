"""
ASTRAEUS Cross-Survey Competing-Hypothesis Bayesian Matcher Module (v2.0).

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-XMATCH-002
Implements Sutherland & Saunders (1992) and Budavári & Szalay (2008) competing-hypothesis
Bayesian counterpart association with normalized background likelihood and no-counterpart hypothesis (H0).
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


class CrossSurveyMatchStatus(str, Enum):
    SECURE_MATCH = "SECURE_MATCH"
    PROBABLE_MATCH = "PROBABLE_MATCH"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    NO_RELIABLE_MATCH = "NO_RELIABLE_MATCH"


class SurveyCandidate(BaseModel):
    catalog_name: str       # 'ATLAS', 'DECam', 'Pan-STARRS1', 'ZTF'
    source_id: str
    ra_deg: float
    dec_deg: float
    ra_err_arcsec: float = 0.05
    dec_err_arcsec: float = 0.05
    mag: float
    filter_band: str


class CrossSurveyAssociation(BaseModel):
    query_entity_id: str
    matched_candidate: Optional[SurveyCandidate] = None
    match_status: CrossSurveyMatchStatus
    separation_arcsec: float = 999.0
    posterior_match_probability: float = 0.0
    posterior_no_counterpart_probability: float = 1.0
    likelihood_ratio: float = 0.0
    chance_coincidence_probability: float = 1.0
    local_source_density_per_sq_arcsec: float = 0.0
    competing_candidates_count: int = 0
    is_training_overlap_quarantined: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class CrossSurveyMatcher:
    """
    Competing-Hypothesis Bayesian counterpart matcher and identity audit engine.
    Enforces normalized background likelihood and candidate competition under H0, H1, ..., HK.
    """

    def __init__(self, completeness_prior_q: float = 0.90, max_search_radius_arcsec: float = 5.0):
        self.q_prior = completeness_prior_q
        self.max_search_radius_arcsec = max_search_radius_arcsec

    @staticmethod
    def angular_separation_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Great circle distance in arcseconds using Vincenty-like haversine."""
        d_ra = np.radians(ra2 - ra1)
        d_dec = np.radians(dec2 - dec1)
        lat1 = np.radians(dec1)
        lat2 = np.radians(dec2)

        a = np.sin(d_dec / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_ra / 2.0) ** 2
        c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
        return float(np.degrees(c) * 3600.0)

    def match_source(
        self,
        query_id: str,
        query_ra: float,
        query_dec: float,
        query_pos_err_arcsec: float = 0.05,
        candidates: Optional[List[SurveyCandidate]] = None,
        local_density_sq_arcsec: Optional[float] = None,
        query_dec_err_arcsec: Optional[float] = None,
    ) -> CrossSurveyAssociation:
        """
        Calculates competing-hypothesis Bayesian association probabilities under:
          H_j: candidate j is the physical counterpart
          H_0: no candidate in search region is the physical counterpart

        Implements full 2D Gaussian astrometric likelihood with covariance:
          Sigma_j = Sigma_query + Sigma_cand,j (utilizing both RA and Dec uncertainties)
        """
        if candidates is None:
            candidates = []

        # Prior odds factor for the no-counterpart hypothesis: (1 - Q) / Q
        prior_null_factor = (1.0 - self.q_prior) / self.q_prior

        # Local background source density (sources / arcsec^2)
        rho = local_density_sq_arcsec or 0.005

        if not candidates:
            return CrossSurveyAssociation(
                query_entity_id=query_id,
                match_status=CrossSurveyMatchStatus.NO_RELIABLE_MATCH,
                separation_arcsec=999.0,
                posterior_match_probability=0.0,
                posterior_no_counterpart_probability=1.0,
                likelihood_ratio=0.0,
                chance_coincidence_probability=1.0,
                local_source_density_per_sq_arcsec=rho,
                competing_candidates_count=0,
            )

        # 2D Astrometric error modeling
        sigma_alpha_query = query_pos_err_arcsec
        sigma_dec_query = query_dec_err_arcsec if query_dec_err_arcsec is not None else query_pos_err_arcsec

        # Filter candidates within search radius and evaluate 2D likelihoods
        evaluated_candidates: List[Tuple[SurveyCandidate, float, float]] = []  # (cand, sep, lr)
        for cand in candidates:
            mean_dec_rad = np.radians(0.5 * (query_dec + cand.dec_deg))
            dx_arcsec = (cand.ra_deg - query_ra) * np.cos(mean_dec_rad) * 3600.0
            dy_arcsec = (cand.dec_deg - query_dec) * 3600.0
            sep = np.sqrt(dx_arcsec ** 2 + dy_arcsec ** 2)

            if sep > self.max_search_radius_arcsec:
                continue

            # 2x2 Astrometric Covariance Matrix: Sigma = Sigma_query + Sigma_cand
            var_x = (sigma_alpha_query ** 2) + (cand.ra_err_arcsec ** 2) + 1e-6
            var_y = (sigma_dec_query ** 2) + (cand.dec_err_arcsec ** 2) + 1e-6
            cov_xy = 0.0  # Off-diagonal covariance (uncorrelated astrometric reductions)
            det_sigma = (var_x * var_y) - (cov_xy ** 2)

            # Mahalanobis distance squared: Delta_x^T Sigma^-1 Delta_x
            mahalanobis_sq = (
                (dx_arcsec ** 2) * var_y +
                (dy_arcsec ** 2) * var_x -
                (2.0 * cov_xy * dx_arcsec * dy_arcsec)
            ) / max(det_sigma, 1e-8)

            # 2D Positional Gaussian likelihood:
            # p(Delta_x | H_j) = (1 / 2*pi*sqrt(det_Sigma)) * exp(-0.5 * mahalanobis_sq)
            f_r = (1.0 / (2.0 * np.pi * np.sqrt(max(det_sigma, 1e-8)))) * np.exp(-0.5 * mahalanobis_sq)

            # Likelihood ratio: ell_j = p(Delta_x | H_j) / rho
            lr = f_r / max(rho, 1e-6)
            evaluated_candidates.append((cand, sep, lr))

        if not evaluated_candidates:
            return CrossSurveyAssociation(
                query_entity_id=query_id,
                match_status=CrossSurveyMatchStatus.NO_RELIABLE_MATCH,
                separation_arcsec=999.0,
                posterior_match_probability=0.0,
                posterior_no_counterpart_probability=1.0,
                likelihood_ratio=0.0,
                chance_coincidence_probability=1.0,
                local_source_density_per_sq_arcsec=rho,
                competing_candidates_count=0,
            )

        # Competing-hypothesis denominator: sum(ell_k) + (1 - Q) / Q
        sum_lr = sum(item[2] for item in evaluated_candidates)
        denom = sum_lr + prior_null_factor

        # Calculate posterior probabilities
        posteriors: List[Tuple[SurveyCandidate, float, float, float, float]] = []
        for cand, sep, lr in evaluated_candidates:
            p_hj = float(lr / denom)
            p_chance = float(1.0 - np.exp(-np.pi * (sep ** 2) * rho))
            posteriors.append((cand, sep, lr, p_hj, p_chance))

        # Posterior probability of no counterpart in the region (H0)
        p_h0 = float(prior_null_factor / denom)

        # Sort candidates descending by posterior probability P(H_j | D)
        posteriors.sort(key=lambda x: x[3], reverse=True)
        best_cand, best_sep, best_lr, best_post, best_chance = posteriors[0]

        # Check candidate competition
        competing = len(posteriors) > 1 and posteriors[1][3] >= 0.15

        if best_post >= 0.90 and not competing and best_sep <= 2.5:
            status = CrossSurveyMatchStatus.SECURE_MATCH
        elif best_post >= 0.70 and not competing and best_sep <= 3.5:
            status = CrossSurveyMatchStatus.PROBABLE_MATCH
        elif competing or best_post >= 0.15:
            status = CrossSurveyMatchStatus.AMBIGUOUS_MATCH
        else:
            status = CrossSurveyMatchStatus.NO_RELIABLE_MATCH

        return CrossSurveyAssociation(
            query_entity_id=query_id,
            matched_candidate=best_cand if status in (CrossSurveyMatchStatus.SECURE_MATCH, CrossSurveyMatchStatus.PROBABLE_MATCH) else None,
            match_status=status,
            separation_arcsec=best_sep,
            posterior_match_probability=best_post,
            posterior_no_counterpart_probability=p_h0,
            likelihood_ratio=best_lr,
            chance_coincidence_probability=best_chance,
            local_source_density_per_sq_arcsec=rho,
            competing_candidates_count=len(posteriors),
            details={
                "evaluated_candidates_count": len(posteriors),
                "null_posterior_p_h0": p_h0,
                "all_candidate_posteriors": [
                    {"source_id": c.source_id, "sep": s, "p_hj": p, "lr": l}
                    for c, s, l, p, _ in posteriors
                ],
            }
        )
