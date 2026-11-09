"""
ASTRAEUS Cross-Survey Bayesian Counterpart Matcher & Identity Exclusion Module.
Implements Sutherland & Saunders (1992) likelihood ratio matching across ZTF, Pan-STARRS1,
and DECam, handles ambiguous associations, and enforces cross-survey training/test object exclusion.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field

from astraeus_core.coordinates import CelestialCoordinates


class CrossSurveyMatchStatus(str, Enum):
    SECURE_MATCH = "SECURE_MATCH"
    AMBIGUOUS_MATCH = "AMBIGUOUS_MATCH"
    NO_RELIABLE_MATCH = "NO_RELIABLE_MATCH"


class SurveyCandidate(BaseModel):
    catalog_name: str       # 'Pan-STARRS1', 'DECam', 'ZTF'
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
    chance_coincidence_probability: float = 1.0
    local_source_density_per_sq_arcsec: float = 0.0
    is_training_overlap_quarantined: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)


class CrossSurveyMatcher:
    """
    Bayesian cross-matcher and identity audit engine.
    """

    def __init__(self, completeness_prior_q: float = 0.90, exclusion_radius_arcsec: float = 3.0):
        self.q_prior = completeness_prior_q
        self.exclusion_radius_arcsec = exclusion_radius_arcsec

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
        query_pos_err_arcsec: float,
        candidates: List[SurveyCandidate],
        local_density_sq_arcsec: Optional[float] = None
    ) -> CrossSurveyAssociation:
        """
        Calculates Bayesian likelihood ratio match probabilities for survey candidates.
        """
        if not candidates:
            return CrossSurveyAssociation(
                query_entity_id=query_id,
                match_status=CrossSurveyMatchStatus.NO_RELIABLE_MATCH,
                separation_arcsec=999.0,
                posterior_match_probability=0.0,
                chance_coincidence_probability=1.0,
            )

        # Estimate local source density if not provided:
        # typical extragalactic field density ~ 0.005 sources / arcsec^2 (18,000 / sq deg)
        rho = local_density_sq_arcsec or 0.005

        evaluated: List[Tuple[SurveyCandidate, float, float, float]] = [] # (cand, sep, lr, post)
        for cand in candidates:
            sep = self.angular_separation_arcsec(query_ra, query_dec, cand.ra_deg, cand.dec_deg)
            if sep > 10.0:
                continue

            sigma_tot = np.sqrt(query_pos_err_arcsec ** 2 + cand.ra_err_arcsec ** 2 + 1e-4)
            # Positional Gaussian likelihood: f(r) = (1 / 2*pi*sigma^2) * exp(-r^2 / 2*sigma^2)
            f_r = (1.0 / (2.0 * np.pi * (sigma_tot ** 2))) * np.exp(-0.5 * (sep / sigma_tot) ** 2)

            # Likelihood ratio: L = f(r) / rho
            lr = f_r / max(rho, 1e-6)

            # Sutherland & Saunders posterior match probability:
            # P = L / (L + (1 - Q) / Q)
            denom = lr + ((1.0 - self.q_prior) / self.q_prior)
            post_prob = float(lr / denom) if denom > 0 else 0.0

            # Chance coincidence: 1 - exp(-pi * r^2 * rho)
            p_chance = float(1.0 - np.exp(-np.pi * (sep ** 2) * rho))

            evaluated.append((cand, sep, post_prob, p_chance))

        if not evaluated:
            return CrossSurveyAssociation(
                query_entity_id=query_id,
                match_status=CrossSurveyMatchStatus.NO_RELIABLE_MATCH,
                separation_arcsec=999.0,
                posterior_match_probability=0.0,
                chance_coincidence_probability=1.0,
            )

        # Sort by posterior probability descending
        evaluated.sort(key=lambda x: x[2], reverse=True)
        best_cand, best_sep, best_post, best_chance = evaluated[0]

        # Check for ambiguity: multiple candidates with significant posterior
        ambiguous = len(evaluated) > 1 and evaluated[1][2] >= 0.20

        if best_post >= 0.85 and not ambiguous and best_sep <= 2.5:
            status = CrossSurveyMatchStatus.SECURE_MATCH
        elif best_post >= 0.20 or ambiguous:
            status = CrossSurveyMatchStatus.AMBIGUOUS_MATCH
        else:
            status = CrossSurveyMatchStatus.NO_RELIABLE_MATCH

        return CrossSurveyAssociation(
            query_entity_id=query_id,
            matched_candidate=best_cand if status != CrossSurveyMatchStatus.NO_RELIABLE_MATCH else None,
            match_status=status,
            separation_arcsec=best_sep,
            posterior_match_probability=best_post,
            chance_coincidence_probability=best_chance,
            local_source_density_per_sq_arcsec=rho,
            details={"competing_matches_count": len(evaluated) - 1},
        )

    def audit_training_test_overlap(
        self,
        training_entities: List[Dict[str, float]], # List of {'entity_id', 'ra_deg', 'dec_deg'}
        target_eval_entities: List[Dict[str, float]]
    ) -> List[str]:
        """
        Cross-matches evaluation entities against training entities to identify any
        spatial overlaps within exclusion_radius_arcsec. Returns list of quarantined target IDs.
        """
        quarantined_ids = []
        for target in target_eval_entities:
            t_ra = target["ra_deg"]
            t_dec = target["dec_deg"]
            t_id = target.get("entity_id", "unknown")

            for train in training_entities:
                sep = self.angular_separation_arcsec(t_ra, t_dec, train["ra_deg"], train["dec_deg"])
                if sep <= self.exclusion_radius_arcsec:
                    quarantined_ids.append(t_id)
                    break

        return quarantined_ids
