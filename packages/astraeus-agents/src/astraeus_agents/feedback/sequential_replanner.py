"""
ASTRAEUS Sequential Scientific Feedback - Dynamic Sequential Replanning Engine (WS7.6-F).
Re-evaluates expected information gain and regenerates feasible scheduling horizons
upon ingesting new observational measurements, preserving complete revision history.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
import copy
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from astraeus_agents.observatory.facility_registry import (
    FacilityRegistry,
    ObservatoryFacility,
)
from astraeus_agents.observatory.schedulers import (
    TargetState,
    ObservingSchedule,
    ScheduledObservation,
    RollingHorizonOptimizer,
    HybridOptimizer,
)
from astraeus_agents.feedback.observation_models import CalibratedObservation


@dataclass
class ReplanningRevisionRecord:
    """Immutable audit record documenting a sequential schedule replanning cycle."""
    revision_number: int
    triggering_observation_id: str
    replan_epoch_mjd: float
    prior_schedule_id: str
    new_schedule_id: str
    prior_entropy_nats: float
    updated_entropy_nats: float
    entropy_reduction_nats: float
    executed_observations_count: int
    revised_observations_count: int
    optimal_next_observation: Optional[ScheduledObservation]
    full_new_schedule: ObservingSchedule
    revision_hash_sha256: str


class SequentialReplanningEngine:
    """
    Dynamic sequential experimental planner.
    Recalculates EIG and regenerates future observing sequences in response to incoming evidence.
    """

    def __init__(
        self,
        facility_registry: Optional[FacilityRegistry] = None,
        optimizer: Optional[RollingHorizonOptimizer] = None,
    ):
        self.facility_registry = facility_registry or FacilityRegistry()
        self.optimizer = optimizer or RollingHorizonOptimizer(horizon_depth=2)
        self.schedule_revisions: List[ReplanningRevisionRecord] = []
        self._revision_counter: int = 0

    def trigger_sequential_replan(
        self,
        current_schedule: ObservingSchedule,
        target: TargetState,
        facilities: List[ObservatoryFacility],
        triggering_observation: CalibratedObservation,
        updated_posteriors: Dict[str, float],
        current_mjd: float,
        remaining_duration_hours: float,
    ) -> ReplanningRevisionRecord:
        """
        Executes sequential dynamic replanning cycle:
        1. Updates target's hypothesis belief state.
        2. Recalculates remaining visibility and dynamic EIG.
        3. Optimizes future observing sequence without overwriting historical plans.
        """
        self._revision_counter += 1

        # 1. Update target hypothesis state
        updated_target = copy.deepcopy(target)
        updated_target.hypothesis_posteriors = updated_posteriors
        updated_target.visits_completed += 1

        # 2. Prior vs updated entropy
        prior_h = 0.0
        for p in target.hypothesis_posteriors.values():
            if p > 1e-12:
                prior_h -= p * math.log(p)

        new_h = 0.0
        for p in updated_posteriors.values():
            if p > 1e-12:
                new_h -= p * math.log(p)

        delta_h = prior_h - new_h

        # 3. Generate revised schedule over remaining duration
        revised_schedule = self.optimizer.plan_schedule(
            targets=[updated_target],
            facilities=facilities,
            start_mjd=current_mjd,
            duration_hours=remaining_duration_hours,
            slew_origin_ra_deg=triggering_observation.target_ra_deg,
            slew_origin_dec_deg=triggering_observation.target_dec_deg,
        )
        revised_schedule.schedule_id = f"sched_rev_{self._revision_counter}_{int(current_mjd)}"

        optimal_next = revised_schedule.observations[0] if revised_schedule.observations else None

        # 4. Cryptographic provenance seal
        rev_raw = f"{self._revision_counter}:{triggering_observation.observation_id}:{current_mjd:.6f}:{new_h:.4f}"
        rev_hash = hashlib.sha256(rev_raw.encode()).hexdigest()

        record = ReplanningRevisionRecord(
            revision_number=self._revision_counter,
            triggering_observation_id=triggering_observation.observation_id,
            replan_epoch_mjd=current_mjd,
            prior_schedule_id=current_schedule.schedule_id,
            new_schedule_id=revised_schedule.schedule_id,
            prior_entropy_nats=round(prior_h, 4),
            updated_entropy_nats=round(new_h, 4),
            entropy_reduction_nats=round(delta_h, 4),
            executed_observations_count=len(current_schedule.observations),
            revised_observations_count=len(revised_schedule.observations),
            optimal_next_observation=optimal_next,
            full_new_schedule=revised_schedule,
            revision_hash_sha256=rev_hash,
        )

        self.schedule_revisions.append(record)
        return record

    def get_revision_history(self) -> List[ReplanningRevisionRecord]:
        return list(self.schedule_revisions)
