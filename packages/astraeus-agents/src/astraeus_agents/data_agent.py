"""
ASTRAEUS Data & Cross-Match Agent.
Responsible for alert retrieval, photometric measurement validation,
Bayesian counterpart association, upper-limit preservation, and IVOA provenance recording.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field

from astraeus_agents.contracts import AgentRole, AgentMessage
from astraeus_core.epistemic import EpistemicTier, ClaimType
from astraeus_core.provenance import create_provenance_record, ProvEntity
from astraeus_data.crossmatch.cross_survey_matcher import (
    CrossSurveyMatcher,
    CrossSurveyAssociation,
    SurveyCandidate,
    CrossSurveyMatchStatus,
)


class DataAgentOutput(BaseModel):
    entity_id: str
    ra_deg: float
    dec_deg: float
    num_detections: int
    num_upper_limits: int
    time_baseline_days: float
    quality_flags: List[int]
    cross_match_result: CrossSurveyAssociation
    provenance_record: ProvEntity
    has_subtraction_defect: bool = False
    is_spatially_clean: bool = True


class DataCrossMatchAgent:
    """
    Executes photometric verification and multi-survey Bayesian counterpart matching.
    """

    def __init__(self, matcher: Optional[CrossSurveyMatcher] = None):
        self.matcher = matcher or CrossSurveyMatcher(completeness_prior_q=0.90, exclusion_radius_arcsec=3.0)

    def process_candidate_data(
        self,
        entity_id: str,
        ra_deg: float,
        dec_deg: float,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray,
        bands: List[str],
        flags: Optional[List[int]] = None,
        survey_candidates: Optional[List[SurveyCandidate]] = None
    ) -> DataAgentOutput:
        """
        Validates measurements, evaluates cross-matches, and records provenance.
        """
        n_epochs = len(times_mjd)
        duration = float(times_mjd[-1] - times_mjd[0]) if n_epochs > 1 else 0.0
        q_flags = flags or [0] * n_epochs

        # Detections vs upper limits (e.g. error > 0.5 or flag != 0)
        num_det = sum(1 for m, e in zip(magnitudes, errors) if e < 0.40)
        num_uplim = n_epochs - num_det

        # Bayesian cross-match against provided archival or target survey candidates
        cross_res = self.matcher.match_source(
            query_id=entity_id,
            query_ra=ra_deg,
            query_dec=dec_deg,
            query_pos_err_arcsec=0.15,
            candidates=survey_candidates or [],
        )

        # Check for subtraction dipole or spatial blending
        has_defect = any(f == 1 for f in q_flags)
        is_clean = (cross_res.local_source_density_per_sq_arcsec < 0.05) and not has_defect

        # Record IVOA provenance
        prov = create_provenance_record(
            entity_id=entity_id,
            entity_type="transient_candidate",
            activity_id=f"activity:data_agent:{entity_id}",
            agent_id="agent:astraeus_data_agent",
        )

        return DataAgentOutput(
            entity_id=entity_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            num_detections=num_det,
            num_upper_limits=num_uplim,
            time_baseline_days=duration,
            quality_flags=q_flags,
            cross_match_result=cross_res,
            provenance_record=prov,
            has_subtraction_defect=has_defect,
            is_spatially_clean=is_clean,
        )
