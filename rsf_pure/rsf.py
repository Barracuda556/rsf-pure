"""
Pure Random Survival Forest (Ishwaran-style) — NumPy only.

Class names use the Pure* prefix so they never clash with
sksurv.ensemble.RandomSurvivalForest or sklearn estimators.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np

from .criterion import PureSplitCriterion
from .tree import PureSurvivalTree


class PureRandomSurvivalForest:
    """
    Random Survival Forest with pluggable split criteria.

    Safe to import alongside::

        from sksurv.ensemble import RandomSurvivalForest
        from rsf_pure import PureRandomSurvivalForest

    Parameters
    ----------
    n_estimators : int
    criterion : str or PureSplitCriterion
        "logrank" or a custom PureSplitCriterion instance.
    max_depth, min_samples_split, min_samples_leaf, max_features
    min_events_leaf : int
    bootstrap : bool
    max_samples : float, int or None
    oob_score : bool
    n_jobs : int
        Parallel tree builds (-1 = all CPUs). Threads are used so NumPy work
        can release the GIL without pickling issues.
    random_state : int or None
    max_candidate_splits : int or None
        Default 32 — limits thresholds tried per feature (speed).
        Set None to scan all unique values.
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

    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if times is None:
            times = self.unique_times_
        times = np.asarray(times, dtype=np.float64)

        H = np.zeros((X.shape[0], times.size), dtype=np.float64)
        for tree in self.estimators_:
            H += tree.predict_cumulative_hazard(X, times)
        H /= len(self.estimators_)
        return H

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if times is None:
            times = self.unique_times_
        times = np.asarray(times, dtype=np.float64)

        S = np.zeros((X.shape[0], times.size), dtype=np.float64)
        for tree in self.estimators_:
            S += tree.predict_survival_function(X, times)
        S /= len(self.estimators_)
        return S

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble risk = mean CHF at last grid time (higher = higher risk)."""
        H = self.predict_cumulative_hazard(X)
        return H[:, -1]


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

    # Sort by time ascending — then only need to look at later samples
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
        # j with time[j] > time[i]
        # (ties in time with events are skipped, same as before)
        ti = time[i]
        ri = risk[i]
        for j in range(i + 1, n):
            if time[j] == ti:
                continue
            # time[j] > ti
            permissible += 1.0
            rj = risk[j]
            if ri > rj:
                concordant += 1.0
            elif ri == rj:
                concordant += 0.5

    if permissible == 0:
        return np.nan
    return concordant / permissible
