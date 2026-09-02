"""
Split criteria for pure survival trees.

Any criterion must expose:
    score(time, event, left_mask) -> float
where higher score = better split.

Names are prefixed with Pure* to avoid clashes with sksurv / sklearn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

# Optional Numba acceleration (used if installed)
try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return deco


class PureSplitCriterion(ABC):
    """Base class for node-splitting criteria (pure-RSF)."""

    name: str = "base"

    @abstractmethod
    def score(
        self,
        time: np.ndarray,
        event: np.ndarray,
        left_mask: np.ndarray,
    ) -> float:
        """
        Parameters
        ----------
        time : (n,) float
        event : (n,) bool / 0-1
        left_mask : (n,) bool — True = goes to left child

        Returns
        -------
        float : higher is better. Return -inf / 0 if split is invalid.
        """
        ...

    def best_split_on_feature(
        self,
        x: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        min_leaf: int = 3,
        max_candidates: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, Optional[float], Optional[np.ndarray]]:
        """
        Find best threshold on a single continuous/ordinal feature.

        Returns
        -------
        best_score, best_threshold, left_mask  (or -inf, None, None)
        """
        n = x.shape[0]
        if n < 2 * min_leaf:
            return -np.inf, None, None

        order = np.argsort(x, kind="mergesort")
        x_s = x[order]
        t_s = time[order]
        e_s = event[order]

        unique_idx = np.where(x_s[1:] != x_s[:-1])[0]
        valid = (unique_idx + 1 >= min_leaf) & (n - (unique_idx + 1) >= min_leaf)
        cand = unique_idx[valid]
        if cand.size == 0:
            return -np.inf, None, None

        if max_candidates is not None and cand.size > max_candidates:
            if rng is None:
                rng = np.random.default_rng()
            cand = np.sort(rng.choice(cand, size=max_candidates, replace=False))

        # Sort by time once; evaluate each candidate without re-sorting
        time_order = np.argsort(t_s, kind="mergesort")
        t_time = t_s[time_order]
        e_time = e_s[time_order].astype(np.float64)

        best_score = -np.inf
        best_thr = None
        best_i = None

        for i in cand:
            # left in feature-sorted order: indices 0..i
            left_feat = np.zeros(n, dtype=np.bool_)
            left_feat[: i + 1] = True
            left_time = left_feat[time_order]
            sc = _logrank_statistic_sorted(t_time, e_time, left_time)
            if sc > best_score:
                best_score = sc
                best_thr = 0.5 * (x_s[i] + x_s[i + 1])
                best_i = i

        if best_i is None:
            return -np.inf, None, None

        left_mask = np.zeros(n, dtype=bool)
        left_mask[order[: best_i + 1]] = True
        return best_score, best_thr, left_mask


class PureLogRankCriterion(PureSplitCriterion):
    """
    Classic two-sample log-rank statistic (|L|).

    L = sum (d_L - E[d_L]) / sqrt(sum Var)
    Larger |L| = stronger survival difference between child nodes.
    """

    name = "logrank"

    def score(
        self,
        time: np.ndarray,
        event: np.ndarray,
        left_mask: np.ndarray,
    ) -> float:
        return pure_logrank_statistic(time, event, left_mask)

    def best_split_on_feature(
        self,
        x: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        min_leaf: int = 3,
        max_candidates: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[float, Optional[float], Optional[np.ndarray]]:
        # Use optimized path from base (single time-sort + sorted log-rank)
        return PureSplitCriterion.best_split_on_feature(
            self, x, time, event, min_leaf, max_candidates, rng
        )


@njit(cache=True)
def _logrank_statistic_sorted(
    t: np.ndarray,
    e: np.ndarray,
    left_s: np.ndarray,
) -> float:
    """
    |log-rank| when `t` is already sorted ascending and `e`, `left_s`
    are aligned with `t`. Pure loops for Numba compatibility.
    """
    n = t.shape[0]
    n_left = 0
    for i in range(n):
        if left_s[i]:
            n_left += 1
    if n_left == 0 or n_left == n:
        return 0.0

    num = 0.0
    den = 0.0
    Y = float(n)
    Y_L = float(n_left)

    i = 0
    while i < n:
        # group identical times
        j = i + 1
        while j < n and t[j] == t[i]:
            j += 1
        d = 0.0
        d_L = 0.0
        drop = 0.0
        drop_L = 0.0
        for k in range(i, j):
            drop += 1.0
            if left_s[k]:
                drop_L += 1.0
            if e[k] > 0.0:
                d += 1.0
                if left_s[k]:
                    d_L += 1.0
        if d > 0.0 and Y > 1.0:
            E = Y_L * (d / Y)
            num += d_L - E
            factor = (Y - d) / (Y - 1.0)
            den += (Y_L / Y) * (1.0 - Y_L / Y) * d * factor
        Y -= drop
        Y_L -= drop_L
        i = j

    if den <= 1e-12:
        return 0.0
    return abs(num) / np.sqrt(den)


def pure_logrank_statistic(
    time: np.ndarray,
    event: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    """Public log-rank helper (sorts by time, then calls sorted core)."""
    n = time.shape[0]
    left = np.asarray(left_mask, dtype=np.bool_)
    n_left = int(left.sum())
    if n_left == 0 or n_left == n:
        return 0.0

    order = np.argsort(time, kind="mergesort")
    t = np.asarray(time, dtype=np.float64)[order]
    e = np.asarray(event, dtype=np.float64)[order]
    left_s = left[order]
    return float(_logrank_statistic_sorted(t, e, left_s))


# Back-compat alias used in examples
_logrank_statistic = pure_logrank_statistic


CRITERIA = {
    "logrank": PureLogRankCriterion,
}


def get_criterion(name_or_obj) -> PureSplitCriterion:
    if isinstance(name_or_obj, PureSplitCriterion):
        return name_or_obj
    if isinstance(name_or_obj, str):
        key = name_or_obj.lower()
        if key not in CRITERIA:
            raise ValueError(
                f"Unknown criterion '{name_or_obj}'. Available: {list(CRITERIA)}"
            )
        return CRITERIA[key]()
    raise TypeError("criterion must be str or PureSplitCriterion instance")
