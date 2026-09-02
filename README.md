# rsf-pure

Pure-Python **Random Survival Forest** (Ishwaran-style) with pluggable split criteria.

All public names use the `Pure*` / `pure_*` prefix so this package can be used **alongside scikit-survival** without name clashes.

Prediction API matches sksurv (`predict`, `predict_survival_function`, `predict_cumulative_hazard_function` with `return_array` and `StepFunction`).

## Install

```bash
pip install numpy
# optional: pip install numba scikit-survival
```

## Quick start

```python
from rsf_pure import PureRandomSurvivalForest, pure_concordance_index

rsf = PureRandomSurvivalForest(n_estimators=100, criterion="logrank", random_state=0)
rsf.fit(X, time, event)

risk = rsf.predict(X_test)  # (n,) risk scores
S = rsf.predict_survival_function(X_test, return_array=True)
H = rsf.predict_cumulative_hazard_function(X_test, return_array=True)
fns = rsf.predict_survival_function(X_test)  # array of StepFunction
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
