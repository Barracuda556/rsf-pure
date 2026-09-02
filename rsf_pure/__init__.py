"""
Pure-Python Random Survival Forest with pluggable split criteria.

All public names use the ``Pure*`` / ``pure_*`` prefix so this package
can sit next to scikit-survival without name clashes::

    from sksurv.ensemble import RandomSurvivalForest
    from rsf_pure import PureRandomSurvivalForest

Quick start
-----------
>>> from rsf_pure import PureRandomSurvivalForest, pure_concordance_index
>>> rsf = PureRandomSurvivalForest(n_estimators=50, random_state=0)
>>> rsf.fit(X, time, event)
>>> risk = rsf.predict(X_test)                      # (n,) like sksurv
>>> S = rsf.predict_survival_function(X_test)       # StepFunction array
>>> S_arr = rsf.predict_survival_function(X_test, return_array=True)
>>> H = rsf.predict_cumulative_hazard_function(X_test, return_array=True)

Custom criterion
----------------
>>> from rsf_pure import PureSplitCriterion, PureRandomSurvivalForest
>>> class MyCrit(PureSplitCriterion):
...     name = "my"
...     def score(self, time, event, left_mask):
...         return score  # higher = better
>>> rsf = PureRandomSurvivalForest(criterion=MyCrit(), n_estimators=30)
"""

from .criterion import (
    PureSplitCriterion,
    PureLogRankCriterion,
    get_criterion,
    CRITERIA,
    pure_logrank_statistic,
)
from .tree import PureSurvivalTree, PureTreeNode
from .rsf import PureRandomSurvivalForest, pure_concordance_index

__all__ = [
    "PureSplitCriterion",
    "PureLogRankCriterion",
    "get_criterion",
    "CRITERIA",
    "pure_logrank_statistic",
    "PureSurvivalTree",
    "PureTreeNode",
    "PureRandomSurvivalForest",
    "pure_concordance_index",
]
