"""
ASTRAEUS Stage 2 Statistical Multiplicity & Testing Framework (Day 111).
Implements the Reconciled 32-Test Multiplicity Family under Benjamini-Hochberg FDR (alpha = 0.05),
Two-Sided Paired Wilcoxon Signed-Rank Tests for continuous tasks (T1, T2, T4),
Exact Binomial McNemar Tests for discrete early classification (T3),
and the Secondary Top-10 Permutation Diagnostic.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
from scipy import stats
from pydantic import BaseModel, ConfigDict, Field


class StatisticalContrast(BaseModel):
    """
    Standardized result for an individual primary hypothesis test.
    The primary family is strictly bounded to m = 32 tests (C01 through C32).
    """
    model_config = ConfigDict(frozen=True)

    contrast_id: str                      # e.g. "C01", "C12", "C32"
    task_id: str                          # "T1", "T2", "T3", "T4"
    task_name: str                        # "Masked Reconstruction", "Multi-Horizon Forecast", etc.
    baseline_id: str                      # "B1_Akima", "B2_GP", "B3_Transformer", "B4_ASTROMER"
    cohort_id: str                        # "Cohort_1_DatasetA_Test", "Cohort_2_Track3_Archival"
    sample_size: int                      # N = 45 or N = 17
    test_type: str                        # "Two-Sided Paired Wilcoxon", "Exact Binomial McNemar"
    test_statistic: float                 # W, chi2, or discordant sum
    raw_p_value: float                    # Unadjusted p-value
    bh_q_value: float = 1.0               # Adjusted q-value after Benjamini-Hochberg FDR
    is_significant_fdr: bool = False      # True if bh_q_value <= alpha (0.05)
    mean_difference: float = 0.0          # Metric delta (baseline - candidate or pairwise)
    effect_size: float = 0.0              # Rank-biserial correlation or odds ratio
    interpretation: str = ""


class Top10PermutationDiagnostic(BaseModel):
    """
    Secondary operational sensitivity diagnostic for Top-10 queue selection.
    Strictly separated from the primary 32-test multiplicity family.
    """
    model_config = ConfigDict(frozen=True)

    baseline_id: str
    cohort_id: str
    observed_p_at_10_baseline: float
    observed_p_at_10_candidate: float
    delta_p_at_10: float                  # candidate - baseline
    num_permutations: int = 10000
    empirical_p_value: float
    ci_lower_95: float
    ci_upper_95: float
    is_exchangeable: bool                 # True if p > 0.05


def compute_paired_wilcoxon(
    diffs: np.ndarray,
    zero_method: str = "wilcox"
) -> Tuple[float, float, float]:
    """
    Computes two-sided paired Wilcoxon signed-rank test on differences:
    diffs = metric_base - metric_cand (or candidate - baseline).
    Returns: (statistic, p_value, effect_size_r).
    """
    diff_arr = np.asarray(diffs, dtype=np.float64)
    # Remove NaN values if any
    diff_arr = diff_arr[~np.isnan(diff_arr)]
    n = len(diff_arr)

    if n < 5 or np.all(diff_arr == 0.0):
        # Degenerate case: no differences or insufficient sample size
        return 0.0, 1.0, 0.0

    try:
        res = stats.wilcoxon(diff_arr, alternative="two-sided", zero_method=zero_method)
        stat = float(res.statistic)
        p_val = float(res.pvalue)

        # Effect size: rank-biserial correlation r = 1 - (2 * W / total_rank_sum)
        # For Wilcoxon signed-rank: total ranks = n * (n + 1) / 2
        total_rank = n * (n + 1) / 2.0
        effect_size = float(1.0 - (2.0 * stat / max(total_rank, 1.0)))
        return stat, p_val, effect_size
    except Exception:
        # Fallback if ties or zero differences make exact test non-computable
        return 0.0, 1.0, 0.0


def compute_exact_mcnemar(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    y_pred_cand: np.ndarray
) -> Tuple[int, int, float, float]:
    """
    Computes Exact Binomial McNemar Test on paired 0/1 correctness indicators.
    b: candidate correct, baseline incorrect
    c: candidate incorrect, baseline correct
    Returns: (b, c, p_value, odds_ratio).
    """
    corr_base = (np.asarray(y_pred_base) == np.asarray(y_true))
    corr_cand = (np.asarray(y_pred_cand) == np.asarray(y_true))

    b = int(np.sum(corr_cand & ~corr_base))  # candidate won
    c = int(np.sum(~corr_cand & corr_base))  # baseline won

    n_discordant = b + c
    if n_discordant == 0:
        return 0, 0, 1.0, 1.0

    # Exact two-sided binomial test under H0: p = 0.5
    binom_res = stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
    p_val = float(binom_res.pvalue)

    odds_ratio = float(b / max(c, 1)) if c > 0 else float(b + 1)
    return b, c, p_val, odds_ratio


def apply_benjamini_hochberg(
    contrasts: List[StatisticalContrast],
    alpha: float = 0.05
) -> List[StatisticalContrast]:
    """
    Applies strict Benjamini-Hochberg False Discovery Rate (FDR) control across
    the primary 32-test multiplicity family.
    Computes monotonic q-values:
    q_(i) = min_{j >= i} [ min(1.0, (m / j) * p_(j)) ].
    """
    m = len(contrasts)
    if m == 0:
        return []

    # Sort contrasts ascending by raw p-value
    indexed_contrasts = sorted(enumerate(contrasts), key=lambda item: item[1].raw_p_value)

    # Compute unconstrained BH values: (m / rank) * p
    raw_q_values = []
    for rank, (orig_idx, c) in enumerate(indexed_contrasts, start=1):
        raw_q = (m / rank) * c.raw_p_value
        raw_q_values.append(min(1.0, raw_q))

    # Enforce monotonicity backwards: q_(i) = min(raw_q_(i), q_(i+1))
    adjusted_q = [1.0] * m
    running_min = 1.0
    for i in reversed(range(m)):
        running_min = min(running_min, raw_q_values[i])
        adjusted_q[i] = running_min

    # Reconstruct updated contrasts in original order
    updated_contrasts = [None] * m
    for sort_idx, (orig_idx, c) in enumerate(indexed_contrasts):
        q_val = float(adjusted_q[sort_idx])
        is_sig = q_val <= alpha
        interp = (
            f"Statistically significant under BH-FDR (q={q_val:.4f} <= {alpha})"
            if is_sig
            else f"Not significant under BH-FDR (q={q_val:.4f} > {alpha})"
        )

        updated = StatisticalContrast(
            contrast_id=c.contrast_id,
            task_id=c.task_id,
            task_name=c.task_name,
            baseline_id=c.baseline_id,
            cohort_id=c.cohort_id,
            sample_size=c.sample_size,
            test_type=c.test_type,
            test_statistic=c.test_statistic,
            raw_p_value=c.raw_p_value,
            bh_q_value=q_val,
            is_significant_fdr=is_sig,
            mean_difference=c.mean_difference,
            effect_size=c.effect_size,
            interpretation=interp,
        )
        updated_contrasts[orig_idx] = updated

    return updated_contrasts


def compute_top10_permutation_test(
    scores_baseline: np.ndarray,
    scores_candidate: np.ndarray,
    is_rare_target: np.ndarray,
    num_permutations: int = 10000,
    seed: int = 42
) -> Top10PermutationDiagnostic:
    """
    Executes secondary operational sensitivity diagnostic:
    Permutation test (B=10,000) for Top-10 follow-up queue exchangeability.
    """
    rng = np.random.RandomState(seed)
    n = len(is_rare_target)
    k = min(10, n)

    # Observed Top-10 precision
    top_base = np.argsort(scores_baseline)[::-1][:k]
    top_cand = np.argsort(scores_candidate)[::-1][:k]

    p10_base = float(np.sum(is_rare_target[top_base])) / k
    p10_cand = float(np.sum(is_rare_target[top_cand])) / k
    delta_obs = p10_cand - p10_base

    # Permutation loop
    null_deltas = np.zeros(num_permutations, dtype=np.float64)
    for b in range(num_permutations):
        # Swap model scores with p = 0.5 per object
        swap_mask = rng.rand(n) < 0.5
        perm_base = np.where(swap_mask, scores_candidate, scores_baseline)
        perm_cand = np.where(swap_mask, scores_baseline, scores_candidate)

        perm_top_base = np.argsort(perm_base)[::-1][:k]
        perm_top_cand = np.argsort(perm_cand)[::-1][:k]

        perm_p_base = float(np.sum(is_rare_target[perm_top_base])) / k
        perm_p_cand = float(np.sum(is_rare_target[perm_top_cand])) / k
        null_deltas[b] = perm_p_cand - perm_p_base

    # Empirical two-sided p-value
    emp_p = float((1 + np.sum(np.abs(null_deltas) >= np.abs(delta_obs))) / (num_permutations + 1))

    # Bootstrap 95% confidence interval of delta
    ci_low = float(np.percentile(null_deltas, 2.5))
    ci_high = float(np.percentile(null_deltas, 97.5))

    return Top10PermutationDiagnostic(
        baseline_id="baseline",
        cohort_id="cohort",
        observed_p_at_10_baseline=p10_base,
        observed_p_at_10_candidate=p10_cand,
        delta_p_at_10=delta_obs,
        num_permutations=num_permutations,
        empirical_p_value=emp_p,
        ci_lower_95=ci_low,
        ci_upper_95=ci_high,
        is_exchangeable=(emp_p > 0.05),
    )
