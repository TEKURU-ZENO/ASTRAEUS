"""
ASTRAEUS Data Meta-Learning Submodule (WS9.4).
Self-improving scientific methods, regression detection, and offline promotion.
"""

from astraeus_data.meta_learning.offline_method_optimizer import (
    OnlineWeightUpdateProhibitedError,
    MethodVersionStatus,
    ScientificMethodConfig,
    OfflineMethodOptimizer,
)

__all__ = [
    "OnlineWeightUpdateProhibitedError",
    "MethodVersionStatus",
    "ScientificMethodConfig",
    "OfflineMethodOptimizer",
]
