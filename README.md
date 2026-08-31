# rsf-pure

Pure-Python **Random Survival Forest** (Ishwaran-style) with pluggable split criteria.

## Install

```bash
pip install numpy
# then use the rsf_pure package from this repo
```

## Structure

- `rsf_pure/criterion.py` — `SplitCriterion` interface + classic log-rank
- `rsf_pure/tree.py` — `SurvivalTree`
- `rsf_pure/rsf.py` — `RandomSurvivalForest`
- `rsf_pure/example_custom_criterion.py` — how to plug in a custom criterion

## Quick start

```python
from rsf_pure import RandomSurvivalForest, concordance_index

rsf = RandomSurvivalForest(n_estimators=100, criterion="logrank", random_state=0)
rsf.fit(X, time, event)
risk = rsf.predict(X_test)
S = rsf.predict_survival_function(X_test)
c = concordance_index(time_test, event_test, risk)
```

## Custom criterion

```python
from rsf_pure import SplitCriterion, RandomSurvivalForest

class MyCrit(SplitCriterion):
    name = "my"
    def score(self, time, event, left_mask):
        # higher = better split
        return float_score

rsf = RandomSurvivalForest(criterion=MyCrit(), n_estimators=50)
```

Requires only NumPy.
