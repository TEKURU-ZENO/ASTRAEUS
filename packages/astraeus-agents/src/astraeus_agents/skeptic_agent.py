"""
ASTRAEUS Skeptic & Adversarial Validation Agent.
Actively executes structured scientific challenges against candidate anomalies.
Verifies PSF shape, subtraction dipoles, crowded-field blending, physical parameter bounds,
and alternative astrophysical explanations.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from astraeus_agents.contracts import (
    SkepticChallenge,
    ChallengeCategory,
    ChallengeStatus,
    LiteratureCitation,
    CitationEvidenceType,
)
from astraeus_models.physics.engine import PhysicsFitResult


class SkepticValidationAgent:
    """
    Structured adversarial challenger that attempts to disqualify candidate discoveries.
    """

    def challenge_candidate(
        self,
        entity_id: str,
        times_mjd: np.ndarray,
        magnitudes: np.ndarray,
        errors: np.ndarray,
        quality_flags: List[int],
        local_source_density_sq_arcsec: float,
        physics_fit: Optional[PhysicsFitResult],
        literature_citations: List[LiteratureCitation]
    ) -> List[SkepticChallenge]:
        """
        Executes systematic adversarial tests and returns structured challenges.
        """
        challenges: List[SkepticChallenge] = []
        n_epochs = len(times_mjd)

        # -------------------------------------------------------------
        # Test 1: PSF Point-Source Shape & Subtraction Dipole Audit
        # -------------------------------------------------------------
        has_subtraction_defect = any(f == 1 for f in quality_flags)
        if has_subtraction_defect:
            challenges.append(
                SkepticChallenge(
                    challenge_id=f"chk:psf_defect:{entity_id}",
                    category=ChallengeCategory.SUBTRACTION_DIPOLE_ARTIFACT,
                    assertion="Measurement flags indicate image subtraction dipole residual or bad pixel mask.",
                    evidence_data={"quality_flags": quality_flags},
                    status=ChallengeStatus.FATAL_DISQUALIFICATION,
                    resolution_notes="Disqualified: Instrumental subtraction error, not an astrophysical transient.",
                )
            )

        # Test single-epoch unphysical jump (> 4 mag jump in 0 days with no follow-up)
        if n_epochs >= 3:
            max_mag_diff = float(np.max(magnitudes) - np.min(magnitudes))
            min_dt = float(np.min(np.diff(times_mjd))) if len(times_mjd) > 1 else 1.0
            if max_mag_diff > 5.0 and min_dt < 0.1 and n_epochs <= 4:
                challenges.append(
                    SkepticChallenge(
                        challenge_id=f"chk:single_epoch_spike:{entity_id}",
                        category=ChallengeCategory.PSF_SHAPE_FAILURE,
                        assertion="Light curve exhibits instantaneous >5 mag jump on sub-hour timescale without sustained rise.",
                        evidence_data={"max_mag_diff": max_mag_diff, "min_dt": min_dt},
                        status=ChallengeStatus.FATAL_DISQUALIFICATION,
                        resolution_notes="Disqualified: Profile consistent with single-epoch cosmic ray hit.",
                    )
                )

        # -------------------------------------------------------------
        # Test 2: Crowded Field & Blending Audit
        # -------------------------------------------------------------
        if local_source_density_sq_arcsec > 0.04:  # > 144,000 stars / sq deg (Galactic plane)
            challenges.append(
                SkepticChallenge(
                    challenge_id=f"chk:crowded_field:{entity_id}",
                    category=ChallengeCategory.STELLAR_DENSITY_BLENDING,
                    assertion=f"Local stellar density is extreme ({local_source_density_sq_arcsec:.3f} / arcsec^2); high blending risk.",
                    evidence_data={"density": local_source_density_sq_arcsec},
                    status=ChallengeStatus.RAISED,
                    resolution_notes="Warning: Photometry may be contaminated by neighboring stellar PSFs.",
                )
            )

        # -------------------------------------------------------------
        # Test 3: Physical Parameter Plausibility Audit
        # -------------------------------------------------------------
        if physics_fit is not None and physics_fit.preferred_model is not None:
            params = physics_fit.physical_parameters

            # Nickel mass check
            if "m_ni" in params:
                m_ni = params["m_ni"]
                if m_ni > 2.5:
                    challenges.append(
                        SkepticChallenge(
                            challenge_id=f"chk:unphysical_mni:{entity_id}",
                            category=ChallengeCategory.UNPHYSICAL_PARAMETERS,
                            assertion=f"Inferred Nickel-56 mass ({m_ni:.2f} M_sun) exceeds Chandrasekhar mass limit for standard radioactive SNe.",
                            evidence_data={"m_ni": m_ni},
                            status=ChallengeStatus.FATAL_DISQUALIFICATION,
                            resolution_notes="Disqualified: Inferred parameters violate physical thermonuclear limits.",
                        )
                    )
                else:
                    # Validated parameter plausibility
                    challenges.append(
                        SkepticChallenge(
                            challenge_id=f"chk:valid_mni:{entity_id}",
                            category=ChallengeCategory.UNPHYSICAL_PARAMETERS,
                            assertion=f"Nickel-56 mass ({m_ni:.2f} M_sun) is within acceptable physical bounds.",
                            evidence_data={"m_ni": m_ni},
                            status=ChallengeStatus.RESOLVED,
                            resolution_notes="Physical bounds verified.",
                        )
                    )

            # Rise timescale check
            if "tau_rise" in params:
                t_rise = params["tau_rise"]
                if t_rise < 0.2:
                    challenges.append(
                        SkepticChallenge(
                            challenge_id=f"chk:unphysical_rise:{entity_id}",
                            category=ChallengeCategory.UNPHYSICAL_PARAMETERS,
                            assertion=f"Rise timescale ({t_rise:.2f} days) is unphysically fast for standard stellar envelope expansion.",
                            evidence_data={"tau_rise": t_rise},
                            status=ChallengeStatus.RAISED,
                            resolution_notes="Requires central engine or relativistic shock cooling hypothesis.",
                        )
                    )

        # -------------------------------------------------------------
        # Test 4: Alternative Known Explanations Audit
        # -------------------------------------------------------------
        for cit in literature_citations:
            if cit.evidence_type in {CitationEvidenceType.DIRECT_MEASUREMENT, CitationEvidenceType.CATALOG_ASSOCIATION}:
                txt = cit.relevant_excerpt.lower()
                if "rr lyrae" in txt or "variable" in txt:
                    challenges.append(
                        SkepticChallenge(
                            challenge_id=f"chk:known_variable:{entity_id}",
                            category=ChallengeCategory.ALTERNATIVE_EXPLANATION,
                            assertion=f"Archival literature identifies object as known variable ({cit.title}).",
                            evidence_data={"citation_id": cit.citation_id},
                            status=ChallengeStatus.RESOLVED,
                            resolution_notes="Classified as known periodic astronomical object; not an unexplained transient.",
                        )
                    )

        return challenges
