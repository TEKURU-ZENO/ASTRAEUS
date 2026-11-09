"""
ASTRAEUS Investigation Orchestrator.
Coordinates the 4-agent collaborative runtime across Data, Literature, Physics, and Skeptic roles.
Feeds all intermediate outputs into deterministic scientific evidence gates,
enforcing that follow-up recommendations are auditable and grounded in empirical evidence.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np

from astraeus_agents.contracts import (
    AgentRole,
    AgentTask,
    AgentMessage,
    TerminalInvestigationStatus,
)
from astraeus_agents.data_agent import DataCrossMatchAgent, DataAgentOutput
from astraeus_agents.literature_agent import LiteratureContextAgent
from astraeus_agents.skeptic_agent import SkepticValidationAgent
from astraeus_agents.evidence_gates import DeterministicEvidenceGateEngine, GateEvaluationResult
from astraeus_agents.dossier import InvestigationDossier
from astraeus_models.features import AstronomicalFeatureExtractor
from astraeus_models.physics.engine import PhysicalModelEngine, PhysicsFitResult
from astraeus_data.crossmatch.cross_survey_matcher import SurveyCandidate


class InvestigationOrchestrator:
    """
    Orchestrates the 4-agent candidate investigation lifecycle.
    """

    def __init__(
        self,
        data_agent: Optional[DataCrossMatchAgent] = None,
        literature_agent: Optional[LiteratureContextAgent] = None,
        skeptic_agent: Optional[SkepticValidationAgent] = None,
        physics_engine: Optional[PhysicalModelEngine] = None,
    ):
        self.data_agent = data_agent or DataCrossMatchAgent()
        self.literature_agent = literature_agent or LiteratureContextAgent()
        self.skeptic_agent = skeptic_agent or SkepticValidationAgent()
        self.physics_engine = physics_engine or PhysicalModelEngine()
        self.feature_extractor = AstronomicalFeatureExtractor()

    def investigate_candidate(
        self,
        entity_id: str,
        ra_deg: float,
        dec_deg: float,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray,
        bands: List[str],
        quality_flags: Optional[List[int]] = None,
        candidate_hypothesis: str = "unknown",
        survey_candidates: Optional[List[SurveyCandidate]] = None
    ) -> InvestigationDossier:
        """
        Executes the complete 4-agent investigation pipeline for an incoming candidate.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        q_flags = quality_flags or [0] * len(times_mjd)

        # -------------------------------------------------------------
        # Step 1: Data & Cross-Match Agent
        # -------------------------------------------------------------
        data_output = self.data_agent.process_candidate_data(
            entity_id=entity_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            times_mjd=times_mjd,
            magnitudes=magnitudes,
            errors=errors,
            bands=bands,
            flags=q_flags,
            survey_candidates=survey_candidates,
        )

        # -------------------------------------------------------------
        # Step 2: Feature Extraction & Physical Forward Fitting
        # -------------------------------------------------------------
        n_epochs = len(times_mjd)
        duration = float(times_mjd[-1] - times_mjd[0]) if n_epochs > 1 else 0.0

        if n_epochs >= 3:
            # Pseudo-flux conversion for physical fitting: F ~ 10^(-(m - 25) / 2.5)
            fluxes = np.clip(10.0 ** (-(magnitudes - 25.0) / 2.5), 1e-4, 1e5)
            flux_errs = np.clip(0.921 * fluxes * errors, 1e-4, None)
            physics_res = self.physics_engine.fit_all_applicable(
                times_mjd=times_mjd,
                fluxes=fluxes,
                flux_errors=flux_errs,
                candidate_hypothesis=candidate_hypothesis,
            )
        else:
            physics_res = None

        # -------------------------------------------------------------
        # Step 3: Literature & Context Agent
        # -------------------------------------------------------------
        lit_citations = self.literature_agent.query_context(
            entity_id=entity_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )

        # -------------------------------------------------------------
        # Step 4: Skeptic & Adversarial Validation Agent
        # -------------------------------------------------------------
        skeptic_challenges = self.skeptic_agent.challenge_candidate(
            entity_id=entity_id,
            times_mjd=times_mjd,
            magnitudes=magnitudes,
            errors=errors,
            quality_flags=q_flags,
            local_source_density_sq_arcsec=data_output.cross_match_result.local_source_density_per_sq_arcsec,
            physics_fit=physics_res,
            literature_citations=lit_citations,
        )

        # -------------------------------------------------------------
        # Step 5: Deterministic Evidence Gates
        # -------------------------------------------------------------
        gate_res = DeterministicEvidenceGateEngine.evaluate_candidate(
            entity_id=entity_id,
            n_epochs=n_epochs,
            duration_days=duration,
            quality_flags=q_flags,
            physics_fit=physics_res,
            literature_citations=lit_citations,
            skeptic_challenges=skeptic_challenges,
            has_ivoa_provenance=True,
        )

        # -------------------------------------------------------------
        # Step 6: Dossier Compilation
        # -------------------------------------------------------------
        phys_summary: Dict[str, Any] = {}
        if physics_res is not None:
            phys_summary = {
                "preferred_model": physics_res.preferred_model,
                "best_bic": physics_res.best_bic,
                "model_fit_quality": physics_res.model_fit_quality,
                "physical_parameters": physics_res.physical_parameters,
                "parameter_space_deviation": physics_res.parameter_space_deviation,
            }

        return InvestigationDossier(
            dossier_id=f"dossier:{entity_id}:{int(datetime.now(timezone.utc).timestamp())}",
            target_entity_id=entity_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            timestamp_utc=now_utc,
            terminal_status=gate_res.terminal_status,
            recommended_action=gate_res.recommended_followup_action,
            data_summary={
                "num_detections": data_output.num_detections,
                "num_upper_limits": data_output.num_upper_limits,
                "time_baseline_days": data_output.time_baseline_days,
            },
            cross_match=data_output.cross_match_result,
            physical_modeling=phys_summary,
            literature_context=lit_citations,
            skeptic_challenges=skeptic_challenges,
            gate_evaluation=gate_res,
            provenance=data_output.provenance_record,
        )
