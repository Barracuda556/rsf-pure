# rsf-pure

Pure-Python **Random Survival Forest** (Ishwaran-style) with pluggable split criteria.

All public names use the `Pure*` / `pure_*` prefix so this package can be used **alongside scikit-survival** without name clashes.

## Install

```bash
pip install numpy
# optional: pip install numba   # speeds up log-rank
```

## Structure

- `rsf_pure/criterion.py` — `PureSplitCriterion`, `PureLogRankCriterion`
- `rsf_pure/tree.py` — `PureSurvivalTree`, `PureTreeNode`
- `rsf_pure/rsf.py` — `PureRandomSurvivalForest`, `pure_concordance_index`
- `rsf_pure/example_custom_criterion.py` — custom criterion example

## Quick start

```python
from rsf_pure import PureRandomSurvivalForest, pure_concordance_index

rsf = PureRandomSurvivalForest(n_estimators=100, criterion="logrank", random_state=0)
rsf.fit(X, time, event)
risk = rsf.predict(X_test)
S = rsf.predict_survival_function(X_test)
c = pure_concordance_index(time_test, event_test, risk)
```

## Side-by-side with sksurv

```python
from sksurv.ensemble import RandomSurvivalForest
from rsf_pure import PureRandomSurvivalForest
```

## Custom criterion

```python
from rsf_pure import PureSplitCriterion, PureRandomSurvivalForest

class MyCrit(PureSplitCriterion):
    name = "my"
    def score(self, time, event, left_mask):
        return float_score  # higher = better

rsf = PureRandomSurvivalForest(criterion=MyCrit(), n_estimators=50)
```

Requires only NumPy (Numba optional).
