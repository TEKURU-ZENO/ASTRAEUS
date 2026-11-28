"""
ASTRAEUS 2.0 Comparative Benchmark Scorecard Engine (WS-B & WS-C).
Evaluates systems across all 10 mandated scientific dimensions with paired statistical tests,
Benjamini-Hochberg FDR multiplicity corrections, effect sizes, and non-compensatory superiority criteria.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from scipy import stats


class EvaluationDimension(str, Enum):
    SCIENTIFIC_PERFORMANCE = "Scientific Performance"
    UNCERTAINTY = "Uncertainty & Calibration"
    GENERALIZATION = "Generalization & Domain Shift"
    REASONING = "Scientific Reasoning & Grounding"
    DISCOVERY = "Discovery & Falsification"
    AUTONOMY = "Autonomy & Task Completion"
    EFFICIENCY = "Computational & Resource Efficiency"
    SECURITY = "Cybersecurity & Supply Chain Integrity"
    GOVERNANCE = "AI Governance & Oversight"
    REPRODUCIBILITY = "Independent Reproducibility"


class DimensionScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    dimension: EvaluationDimension
    raw_metric_name: str
    astraeus_value: float
    competitor_value: float
    competitor_name: str
    higher_is_better: bool = True
    
    # Statistical validation metrics
    delta: float
    relative_improvement_pct: float
    raw_p_value: float
    bh_fdr_q_value: float = 1.0
    is_statistically_significant: bool = False
    effect_size: float = 0.0
    effect_size_type: str = "Cohen's d"
    is_practically_significant: bool = False
    is_non_inferior: bool = True
    superiority_verdict: str = "INCONCLUSIVE"  # "SUPERIOR", "PARITY", "INFERIOR", "INCONCLUSIVE"
    notes: str = ""


class ComparativeScorecard(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_suite_id: str
    target_competitor: str
    dimension_scores: Dict[str, DimensionScore]
    overall_verdict: str
    is_non_compensatory_valid: bool
    governance_and_security_passed: bool
    falsification_passed: bool
    recommendations: List[str] = Field(default_factory=list)


class ScorecardEngine:
    """
    Computes rigorous comparative benchmark scorecards with multiple testing adjustments.
    """

    def __init__(self, alpha: float = 0.05, min_effect_size_cohen_d: float = 0.5):
        self.alpha = alpha
        self.min_effect_size_cohen_d = min_effect_size_cohen_d

    def evaluate_paired_dimension(
        self,
        dimension: EvaluationDimension,
        metric_name: str,
        astraeus_samples: List[float],
        competitor_samples: List[float],
        competitor_name: str,
        higher_is_better: bool = True
    ) -> DimensionScore:
        a_arr = np.array(astraeus_samples, dtype=float)
        c_arr = np.array(competitor_samples, dtype=float)

        if len(a_arr) != len(c_arr) or len(a_arr) < 2:
            raise ValueError("Sample sizes must be identical and >= 2 for paired evaluation.")

        mean_a = float(np.mean(a_arr))
        mean_c = float(np.mean(c_arr))
        diff = a_arr - c_arr if higher_is_better else c_arr - a_arr
        delta = mean_a - mean_c

        # Relative improvement
        denom = abs(mean_c) if abs(mean_c) > 1e-9 else 1.0
        rel_imp = ((mean_a - mean_c) / denom) * 100.0 if higher_is_better else ((mean_c - mean_a) / denom) * 100.0

        # Paired Wilcoxon or t-test
        diff_nonzero = diff[diff != 0]
        if len(diff_nonzero) < 2:
            p_val = 1.0
            effect_size = 0.0
        else:
            try:
                stat_res = stats.wilcoxon(a_arr, c_arr, alternative="greater" if higher_is_better else "less")
                p_val = float(stat_res.pvalue)
            except Exception:
                t_res = stats.ttest_rel(a_arr, c_arr)
                p_val = float(t_res.pvalue / 2.0) if ((delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better)) else 1.0

            pooled_sd = float(np.std(diff, ddof=1))
            effect_size = float(np.mean(diff) / pooled_sd) if pooled_sd > 1e-9 else 0.0

        return DimensionScore(
            dimension=dimension,
            raw_metric_name=metric_name,
            astraeus_value=mean_a,
            competitor_value=mean_c,
            competitor_name=competitor_name,
            higher_is_better=higher_is_better,
            delta=delta,
            relative_improvement_pct=rel_imp,
            raw_p_value=p_val,
            bh_fdr_q_value=p_val,  # will be updated in multiplicity adjustment
            is_statistically_significant=(p_val < self.alpha),
            effect_size=effect_size,
            effect_size_type="Cohen's d (paired)",
            is_practically_significant=(effect_size >= self.min_effect_size_cohen_d),
            is_non_inferior=(delta >= -0.05 * denom),
            superiority_verdict="SUPERIOR" if (p_val < self.alpha and effect_size >= self.min_effect_size_cohen_d) else ("INFERIOR" if delta < 0 and p_val < self.alpha else "PARITY")
        )

    def apply_benjamini_hochberg(self, scores: List[DimensionScore]) -> List[DimensionScore]:
        """
        Controls the False Discovery Rate across all evaluated dimensions.
        """
        m = len(scores)
        if m == 0:
            return []

        sorted_indices = sorted(range(m), key=lambda i: scores[i].raw_p_value)
        p_vals = [scores[i].raw_p_value for i in sorted_indices]

        # Benjamini-Hochberg step-up adjustment
        q_vals = [1.0] * m
        min_q = 1.0
        for rank in range(m, 0, -1):
            idx = rank - 1
            raw_p = p_vals[idx]
            q = min(1.0, (raw_p * m) / rank)
            min_q = min(min_q, q)
            q_vals[idx] = min_q

        adjusted_scores = []
        for orig_idx in range(m):
            score = scores[orig_idx]
            rank_pos = sorted_indices.index(orig_idx)
            bh_q = q_vals[rank_pos]
            is_sig = (bh_q <= self.alpha)
            is_superior = is_sig and (score.effect_size >= self.min_effect_size_cohen_d)

            verdict = "SUPERIOR" if is_superior else ("INFERIOR" if score.delta < 0 and is_sig else "PARITY")

            adjusted_scores.append(DimensionScore(
                dimension=score.dimension,
                raw_metric_name=score.raw_metric_name,
                astraeus_value=score.astraeus_value,
                competitor_value=score.competitor_value,
                competitor_name=score.competitor_name,
                higher_is_better=score.higher_is_better,
                delta=score.delta,
                relative_improvement_pct=score.relative_improvement_pct,
                raw_p_value=score.raw_p_value,
                bh_fdr_q_value=bh_q,
                is_statistically_significant=is_sig,
                effect_size=score.effect_size,
                effect_size_type=score.effect_size_type,
                is_practically_significant=score.is_practically_significant,
                is_non_inferior=score.is_non_inferior,
                superiority_verdict=verdict,
                notes=score.notes
            ))

        return adjusted_scores

    def build_scorecard(
        self,
        benchmark_id: str,
        competitor_name: str,
        dimension_scores: List[DimensionScore],
        governance_passed: bool,
        security_passed: bool
    ) -> ComparativeScorecard:
        adjusted = self.apply_benjamini_hochberg(dimension_scores)
        score_dict = {s.dimension.value: s for s in adjusted}

        # Check non-compensatory condition:
        # ASTRAEUS must NOT fail any safety/governance dimension, and must not be inferior in core reasoning.
        gov_sec_passed = governance_passed and security_passed
        inferior_count = sum(1 for s in adjusted if s.superiority_verdict == "INFERIOR")
        superior_count = sum(1 for s in adjusted if s.superiority_verdict == "SUPERIOR")

        is_valid = gov_sec_passed and (inferior_count == 0)

        if not gov_sec_passed:
            overall = "FAILED: GOVERNANCE OR SECURITY VIOLATION"
        elif superior_count >= 1 and is_valid:
            overall = f"CERTIFIED: DEMONSTRATED SUPERIORITY IN {superior_count} DIMENSIONS (FDR q <= {self.alpha})"
        elif is_valid:
            overall = "COMPETITIVE PARITY: NON-INFERIOR ACROSS ALL EVALUATED DIMENSIONS"
        else:
            overall = f"INCONCLUSIVE: {inferior_count} INFERIOR DIMENSIONS DETECTED"

        return ComparativeScorecard(
            benchmark_suite_id=benchmark_id,
            target_competitor=competitor_name,
            dimension_scores=score_dict,
            overall_verdict=overall,
            is_non_compensatory_valid=is_valid,
            governance_and_security_passed=gov_sec_passed,
            falsification_passed=is_valid,
            recommendations=[
                f"Statistically validated under Benjamini-Hochberg FDR multiplicity (m={len(dimension_scores)}).",
                "Non-compensatory rule enforced: zero governance/security trade-offs permitted."
            ]
        )
