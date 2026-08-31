"""
Pure-Python Random Survival Forest with pluggable split criteria.

Quick start
-----------
>>> from rsf_pure import RandomSurvivalForest, LogRankCriterion
>>> rsf = RandomSurvivalForest(n_estimators=50, random_state=0)
>>> rsf.fit(X, time, event)
>>> S = rsf.predict_survival_function(X_test)
>>> risk = rsf.predict(X_test)

To add a custom criterion
-------------------------
>>> from rsf_pure.criterion import SplitCriterion
>>> class MyCrit(SplitCriterion):
...     name = "my"
...     def score(self, time, event, left_mask):
...         ...
...         return score   # higher = better
>>> rsf = RandomSurvivalForest(criterion=MyCrit(), n_estimators=30)
"""

from .criterion import (
    SplitCriterion,
    LogRankCriterion,
    get_criterion,
    CRITERIA,
)
from .tree import SurvivalTree, Node
from .rsf import RandomSurvivalForest, concordance_index

__all__ = [
    "SplitCriterion",
    "LogRankCriterion",
    "get_criterion",
    "CRITERIA",
    "SurvivalTree",
    "Node",
    "RandomSurvivalForest",
    "concordance_index",
]
