"""
Pure Random Survival Forest (Ishwaran-style) — NumPy only.

Class names use the Pure* prefix so they never clash with
sksurv.ensemble.RandomSurvivalForest or sklearn estimators.

Prediction API mirrors scikit-survival:
  - predict(X) -> (n_samples,) risk scores
  - predict_survival_function(X, return_array=False)
  - predict_cumulative_hazard_function(X, return_array=False)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np

from .criterion import PureSplitCriterion
from .tree import PureSurvivalTree

# Prefer sksurv StepFunction when available (needed for sksurv metrics)
try:
    from sksurv.functions import StepFunction as _StepFunction

    _HAS_SKSURV_STEP = True
except ImportError:  # pragma: no cover
    _HAS_SKSURV_STEP = False

    class _StepFunction:  # type: ignore
        """Minimal drop-in if sksurv is not installed."""

        def __init__(self, x, y, a=1.0, b=0.0, domain=(0, None)):
            self.x = np.asarray(x, dtype=np.float64)
            self.y = np.asarray(y, dtype=np.float64)
            self.a = float(a)
            self.b = float(b)

        def __call__(self, t):
            t = np.asarray(t, dtype=np.float64)
            idx = np.searchsorted(self.x, t, side="right") - 1
            out = np.empty_like(t, dtype=np.float64)
            out[idx < 0] = self.a * self.y[0] + self.b if self.y.size else self.b
            valid = idx >= 0
            out[valid] = self.a * self.y[np.minimum(idx[valid], len(self.y) - 1)] + self.b
            return out


def _array_to_step_functions(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Convert (n_samples, n_times) to array of StepFunction (sksurv-compatible)."""
    n = values.shape[0]
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = _StepFunction(times, values[i])
    return out


class PureRandomSurvivalForest:
    """
    Random Survival Forest with pluggable split criteria.

    Prediction methods match ``sksurv.ensemble.RandomSurvivalForest`` so you
    can use ``sksurv.metrics`` directly.

    Safe to import alongside::

        from sksurv.ensemble import RandomSurvivalForest
        from rsf_pure import PureRandomSurvivalForest
    """

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: Union[str, PureSplitCriterion] = "logrank",
        max_depth: Optional[int] = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Union[str, int, float, None] = "sqrt",
        min_events_leaf: int = 1,
        bootstrap: bool = True,
        max_samples: Optional[Union[int, float]] = None,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        max_candidate_splits: Optional[int] = 32,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_events_leaf = min_events_leaf
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_candidate_splits = max_candidate_splits

        self.estimators_: List[PureSurvivalTree] = []
        self.feature_importances_: Optional[np.ndarray] = None
        self.unique_times_: Optional[np.ndarray] = None
        self.event_times_: Optional[np.ndarray] = None  # alias of unique_times_
        self.oob_score_: Optional[float] = None
        self.n_features_in_: Optional[int] = None

    def _tree_params(self) -> dict:
        return dict(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            min_events_leaf=self.min_events_leaf,
            max_candidate_splits=self.max_candidate_splits,
        )

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
    ) -> "PureRandomSurvivalForest":
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event).ravel().astype(bool)

        n, p = X.shape
        self.n_features_in_ = p
        self.unique_times_ = np.unique(time[event])
        self.event_times_ = self.unique_times_

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)

        if self.max_samples is None:
            sample_size = n
        elif isinstance(self.max_samples, float):
            sample_size = max(1, int(self.max_samples * n))
        else:
            sample_size = int(self.max_samples)

        def _fit_one(seed: int):
            local_rng = np.random.default_rng(seed)
            if self.bootstrap:
                idx = local_rng.choice(n, size=sample_size, replace=True)
            else:
                idx = np.arange(n)
                if sample_size < n:
                    idx = local_rng.choice(n, size=sample_size, replace=False)

            tree = PureSurvivalTree(random_state=int(seed), **self._tree_params())
            tree.fit(X[idx], time[idx], event[idx])
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[idx] = False
            return tree, oob_mask

        n_jobs = self.n_jobs
        if n_jobs is None or n_jobs == 1:
            results = [_fit_one(int(s)) for s in seeds]
        else:
            max_workers = None if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(_fit_one, [int(s) for s in seeds]))

        self.estimators_ = [t for t, _ in results]
        oob_masks = [m for _, m in results]

        imp = np.zeros(p, dtype=np.float64)
        for t in self.estimators_:
            if t.feature_importances_ is not None:
                imp += t.feature_importances_
        s = imp.sum()
        self.feature_importances_ = imp / s if s > 0 else imp

        if self.oob_score:
            self.oob_score_ = self._oob_concordance(X, time, event, oob_masks)

        return self

    def _oob_concordance(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        oob_masks: List[np.ndarray],
    ) -> float:
        n = X.shape[0]
        risk = np.zeros(n, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)

        for tree, mask in zip(self.estimators_, oob_masks):
            if not mask.any():
                continue
            r = tree.predict(X[mask])
            risk[mask] += r
            counts[mask] += 1.0

        valid = counts > 0
        if valid.sum() < 2:
            return np.nan
        risk[valid] /= counts[valid]
        return pure_concordance_index(time[valid], event[valid], risk[valid])

    # ---------- sksurv-compatible predictions ----------

    def _ensemble_chf_array(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        H = np.zeros((X.shape[0], times.size), dtype=np.float64)
        for tree in self.estimators_:
            H += tree.predict_cumulative_hazard(X, times)
        H /= max(len(self.estimators_), 1)
        return H

    def _ensemble_sf_array(
        self, X: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        S = np.zeros((X.shape[0], times.size), dtype=np.float64)
        for tree in self.estimators_:
            S += tree.predict_survival_function(X, times)
        S /= max(len(self.estimators_), 1)
        return S

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Risk score for each sample (higher = higher risk).

        Returns
        -------
        ndarray, shape (n_samples,)
            Ensemble cumulative hazard at the last event time
            (same role as sksurv RSF ``predict``).
        """
        X = np.asarray(X, dtype=np.float64)
        times = self.unique_times_
        if times is None or times.size == 0:
            return np.zeros(X.shape[0], dtype=np.float64)
        H = self._ensemble_chf_array(X, times)
        return H[:, -1]

    def predict_cumulative_hazard_function(
        self,
        X: np.ndarray,
        return_array: bool = False,
    ):
        """
        Predict cumulative hazard function (Nelson–Aalen ensemble).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        return_array : bool, default False
            If True, return ndarray (n_samples, n_event_times).
            If False, return ndarray of StepFunction (sksurv-compatible).

        Returns
        -------
        cum_hazard : ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        times = self.unique_times_
        if times is None:
            raise RuntimeError("Model is not fitted")
        arr = self._ensemble_chf_array(X, times)
        if return_array:
            return arr
        return _array_to_step_functions(times, arr)

    def predict_survival_function(
        self,
        X: np.ndarray,
        return_array: bool = False,
    ):
        """
        Predict survival function (Kaplan–Meier ensemble average).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        return_array : bool, default False
            If True, return ndarray (n_samples, n_event_times).
            If False, return ndarray of StepFunction (sksurv-compatible).

        Returns
        -------
        survival : ndarray
        """
        X = np.asarray(X, dtype=np.float64)
        times = self.unique_times_
        if times is None:
            raise RuntimeError("Model is not fitted")
        arr = self._ensemble_sf_array(X, times)
        if return_array:
            return arr
        return _array_to_step_functions(times, arr)

    # Backwards-compatible aliases (previous pure-API names)
    def predict_cumulative_hazard(
        self, X: np.ndarray, times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Alias: always returns 2-d array (optionally on custom ``times``)."""
        X = np.asarray(X, dtype=np.float64)
        if times is None:
            times = self.unique_times_
        times = np.asarray(times, dtype=np.float64)
        return self._ensemble_chf_array(X, times)


def pure_concordance_index(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    """
    Harrell's C-index (higher risk = worse survival).

    Named ``pure_concordance_index`` so it does not shadow
    ``sksurv.metrics.concordance_index_censored``.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event).astype(bool)
    risk = np.asarray(risk, dtype=np.float64)

    order = np.argsort(time, kind="mergesort")
    time = time[order]
    event = event[order]
    risk = risk[order]

    n = time.size
    concordant = 0.0
    permissible = 0.0

    for i in range(n):
        if not event[i]:
            continue
        ti = time[i]
        ri = risk[i]
        for j in range(i + 1, n):
            if time[j] == ti:
                continue
            permissible += 1.0
            rj = risk[j]
            if ri > rj:
                concordant += 1.0
            elif ri == rj:
                concordant += 0.5

    if permissible == 0:
        return np.nan
    return concordant / permissible
