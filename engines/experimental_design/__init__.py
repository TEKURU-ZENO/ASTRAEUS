"""
ASTRAEUS Experimental Intelligence Engine (Phase 6 - Track C).
Autonomous information-theoretic scheduling, Expected Information Gain (EIG),
and Constrained Proximal Policy Optimization (PPO) telescope control.

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 23, 2026
"""

from engines.experimental_design.eig_objective import (
    TransientTarget,
    ExpectedInformationGain,
)
from engines.experimental_design.telescope_env import (
    TelescopeSchedulingEnv,
    angular_distance_deg,
)
from engines.experimental_design.ppo_scheduler import (
    ActorCriticNetwork,
    BaseScheduler,
    GreedyAirmassScheduler,
    RoundRobinScheduler,
    ConstrainedPPOScheduler,
    evaluate_baseline,
)

__all__ = [
    "TransientTarget",
    "ExpectedInformationGain",
    "TelescopeSchedulingEnv",
    "angular_distance_deg",
    "ActorCriticNetwork",
    "BaseScheduler",
    "GreedyAirmassScheduler",
    "RoundRobinScheduler",
    "ConstrainedPPOScheduler",
    "evaluate_baseline",
]
