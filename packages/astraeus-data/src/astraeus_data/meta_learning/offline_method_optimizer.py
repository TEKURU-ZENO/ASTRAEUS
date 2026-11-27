"""
ASTRAEUS Self-Improving Scientific Methods & Offline Meta-Learning (WS9.4).
Optimizes scientific inference and planning methods strictly offline.
Enforces the mandatory Covenant: Streaming live alerts are barred from updating operational weights online.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 25, 2026
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class OnlineWeightUpdateProhibitedError(PermissionError):
    """Raised when an agent attempts to mutate operational weights directly from streaming live alerts."""
    pass


class MethodVersionStatus(str, Enum):
    """Operational promotion status of a scientific method configuration."""
    CANDIDATE = "CANDIDATE"
    PROMOTED_OPERATIONAL = "PROMOTED_OPERATIONAL"
    REJECTED_REGRESSION = "REJECTED_REGRESSION"
    DEPRECATED = "DEPRECATED"


class MethodBenchmarkResult(BaseModel):
    """Benchmark performance metrics for a method configuration on a specific task."""
    task_id: str
    accuracy_score: float
    information_gain_nats: float
    execution_time_sec: float


class ScientificMethodConfig(BaseModel):
    """Versioned configuration of an algorithmic or statistical method."""
    method_id: str
    method_name: str
    version: str
    hyperparameters: Dict[str, Any]
    status: MethodVersionStatus = MethodVersionStatus.CANDIDATE
    regression_detected: bool = False
    parent_version: Optional[str] = None
    historical_benchmark_scores: Dict[str, float] = Field(default_factory=dict)
    baseline_provenance_hash: Optional[str] = None
    evaluator_identity: Optional[str] = None
    evaluator_version: Optional[str] = None
    sha256_hash: str = Field(default="")
    created_at_iso: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_hash(self) -> str:
        content = f"{self.method_id}:{self.version}:{sorted(self.hyperparameters.items())}:{self.parent_version}:{self.evaluator_identity}:{self.evaluator_version}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def compute_baseline_provenance_hash(self) -> str:
        import json
        scores_payload = json.dumps(self.historical_benchmark_scores, sort_keys=True)
        content = f"{self.method_id}:{self.version}:{scores_payload}:{self.evaluator_identity}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        if not self.sha256_hash:
            self.sha256_hash = self.compute_hash()
        if not self.baseline_provenance_hash and self.historical_benchmark_scores:
            self.baseline_provenance_hash = self.compute_baseline_provenance_hash()


class OfflineMethodOptimizer:
    """
    Offline meta-learning optimizer for scientific methods.
    Audits candidate improvements across retrospective benchmark tasks,
    detects regressions, and manages versioned promotion to production.
    """

    def __init__(self, is_online_stream_active: bool = False) -> None:
        self.is_online_stream_active = is_online_stream_active
        self._operational_methods: Dict[str, ScientificMethodConfig] = {}
        self._historical_archive: Dict[str, List[MethodBenchmarkResult]] = {}

    def assert_offline_gate(self) -> None:
        """Enforces the Offline Learning Covenant: streaming live data cannot update weights."""
        if self.is_online_stream_active:
            raise OnlineWeightUpdateProhibitedError(
                "GOVERNANCE COVENANT BREACH: Online learning from unconfirmed streaming alerts is strictly barred. "
                "Method improvements must execute exclusively in offline batched verification cycles."
            )

    def register_operational_method(self, method: ScientificMethodConfig) -> None:
        """Registers the baseline operational method with cryptographic score provenance."""
        method.status = MethodVersionStatus.PROMOTED_OPERATIONAL
        if not method.baseline_provenance_hash and method.historical_benchmark_scores:
            method.baseline_provenance_hash = method.compute_baseline_provenance_hash()
        self._operational_methods[method.method_id] = method

    def get_operational_method(self, method_id: str) -> Optional[ScientificMethodConfig]:
        """Retrieves the current operational configuration."""
        return self._operational_methods.get(method_id)

    def evaluate_candidate_improvement(
        self,
        candidate: ScientificMethodConfig,
        benchmark_tasks: List[Dict[str, Any]],
        max_allowable_regression_pct: float = 0.50,
        verify_evaluator_identity: bool = True,
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Evaluates a candidate method against historical tasks.
        Ensures zero regression (> 0.5% degradation on any historical task blocks promotion).
        Verifies baseline score provenance and evaluator identity consistency.
        """
        self.assert_offline_gate()

        baseline = self._operational_methods.get(candidate.method_id)
        candidate_scores: Dict[str, float] = {}
        has_regression = False
        regression_notes = []

        # 1. Verify baseline provenance integrity
        if baseline and baseline.historical_benchmark_scores:
            expected_prov = baseline.compute_baseline_provenance_hash()
            if baseline.baseline_provenance_hash and baseline.baseline_provenance_hash != expected_prov:
                has_regression = True
                regression_notes.append("Baseline provenance breach: stored baseline scores tampered or corrupted.")

        for task in benchmark_tasks:
            tid = task["task_id"]
            base_score = baseline.historical_benchmark_scores.get(tid, 80.0) if baseline else 80.0

            # 2. Verify evaluator identity consistency
            task_eval_id = task.get("evaluator_identity")
            if verify_evaluator_identity and task_eval_id and baseline and baseline.evaluator_identity:
                if task_eval_id != baseline.evaluator_identity:
                    has_regression = True
                    regression_notes.append(
                        f"Task {tid}: Evaluator mismatch ({task_eval_id} vs baseline {baseline.evaluator_identity}). "
                        "Regression gate strictly prohibits comparing scores across differently evaluated baselines."
                    )
            tid = task["task_id"]
            base_score = baseline.historical_benchmark_scores.get(tid, 80.0) if baseline else 80.0

            # Execute real task evaluator if provided
            if "evaluator" in task and callable(task["evaluator"]):
                cand_score = float(task["evaluator"](candidate.hyperparameters))
            elif "historical_episodes" in task and "episode_evaluator" in task and callable(task["episode_evaluator"]):
                # Retrospective evaluation executing callable across historical episodes
                episodes = task["historical_episodes"]
                scores = [float(task["episode_evaluator"](candidate.hyperparameters, ep)) for ep in episodes]
                cand_score = float(sum(scores) / max(1, len(scores)))
            elif "historical_episodes" in task and "evaluator" in task and callable(task["evaluator"]):
                cand_score = float(task["evaluator"](candidate.hyperparameters))
            else:
                # Synthetic scoring fallback is strictly prohibited in scientific optimization
                has_regression = True
                regression_notes.append(
                    f"Task {tid}: missing executable evaluator callable. "
                    "Synthetic scoring arithmetic is strictly prohibited in scientific method optimization."
                )
                cand_score = 0.0

            candidate_scores[tid] = round(cand_score, 2)

            # Check regression
            if baseline and tid in baseline.historical_benchmark_scores and cand_score > 0.0:
                b_score = baseline.historical_benchmark_scores[tid]
                drop_pct = ((b_score - cand_score) / b_score) * 100.0
                if drop_pct > max_allowable_regression_pct:
                    has_regression = True
                    regression_notes.append(f"Task {tid}: score dropped by {drop_pct:.2f}% ({b_score:.2f} -> {cand_score:.2f})")

        candidate.historical_benchmark_scores = candidate_scores

        if has_regression:
            candidate.status = MethodVersionStatus.REJECTED_REGRESSION
            candidate.regression_detected = True
            msg = f"Promotion Blocked: Method regression detected on {len(regression_notes)} tasks. " + "; ".join(regression_notes)
            return False, msg, candidate_scores
        else:
            candidate.status = MethodVersionStatus.PROMOTED_OPERATIONAL
            candidate.regression_detected = False
            self._operational_methods[candidate.method_id] = candidate
            msg = f"Promotion Approved: Method {candidate.method_id} v{candidate.version} promoted to operational."
            return True, msg, candidate_scores
