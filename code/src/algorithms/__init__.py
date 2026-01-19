"""RL algorithm modules (Phase 3+).

This package contains RL4CO-compatible algorithm classes that use the Max@K
estimators from :mod:`src.estimators`.
"""

from src.algorithms.maxk_pomo import MaxKPOMO

__all__ = [
    "MaxKPOMO",
]

