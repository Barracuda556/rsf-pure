"""
Split criteria for survival trees.

Any criterion must expose:
    score(time, event, left_mask) -> float
where higher score = better split.

Also provides fast vectorized log-rank for a sorted feature.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class SplitCriterion(ABC):
    """Base class for node-splitting criteria."""

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

        # candidate split indices: after position i (0-based), left has i+1 samples
        # skip identical values
        unique_idx = np.where(x_s[1:] != x_s[:-1])[0]
        # left size = idx+1 must be >= min_leaf and n-(idx+1) >= min_leaf
        valid = (unique_idx + 1 >= min_leaf) & (n - (unique_idx + 1) >= min_leaf)
        cand = unique_idx[valid]
        if cand.size == 0:
            return -np.inf, None, None

        if max_candidates is not None and cand.size > max_candidates:
            if rng is None:
                rng = np.random.default_rng()
            cand = rng.choice(cand, size=max_candidates, replace=False)
            cand.sort()

        best_score = -np.inf
        best_thr = None
        best_i = None

        for i in cand:
            left_mask_sorted = np.zeros(n, dtype=bool)
            left_mask_sorted[: i + 1] = True
            # map back? score only needs the groups in original order of t_s,e_s
            sc = self.score(t_s, e_s, left_mask_sorted)
            if sc > best_score:
                best_score = sc
                best_thr = 0.5 * (x_s[i] + x_s[i + 1])
                best_i = i

        if best_i is None:
            return -np.inf, None, None

        # reconstruct left_mask in original sample order
        left_mask = np.zeros(n, dtype=bool)
        left_mask[order[: best_i + 1]] = True
        return best_score, best_thr, left_mask


class LogRankCriterion(SplitCriterion):
    """
    Classic two-sample log-rank statistic (absolute value).

    L = sum (d_L - E[d_L]) / sqrt(sum Var)
    We return |L| (or L^2); larger = stronger difference.
    """

    name = "logrank"

    def score(
        self,
        time: np.ndarray,
        event: np.ndarray,
        left_mask: np.ndarray,
    ) -> float:
        return _logrank_statistic(time, event, left_mask)


def _logrank_statistic(
    time: np.ndarray,
    event: np.ndarray,
    left_mask: np.ndarray,
) -> float:
    """
    Compute |log-rank| for groups defined by left_mask.
    Efficient single-pass over unique event times.
    """
    n = time.shape[0]
    left = left_mask.astype(bool)
    n_left = int(left.sum())
    n_right = n - n_left
    if n_left == 0 or n_right == 0:
        return 0.0

    # sort by time
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    e = event[order].astype(np.float64)
    left_s = left[order]

    # unique event times (only times where at least one event occurred)
    event_times = np.unique(t[e > 0])
    if event_times.size == 0:
        return 0.0

    # at-risk and deaths via reverse cumsum style
    # We walk from earliest to latest
    num = 0.0
    den = 0.0

    # pointers / precompute order indices
    # For speed: compute Y and d at each distinct time using search
    # Simple O(n + M) approach:
    idx = 0
    n_at_risk = n
    n_at_risk_L = n_left

    # group by unique times (all times, not only events)
    unique_t, inv = np.unique(t, return_inverse=True)
    # counts per unique time
    n_unique = unique_t.size
    deaths = np.zeros(n_unique, dtype=np.float64)
    deaths_L = np.zeros(n_unique, dtype=np.float64)
    drop = np.zeros(n_unique, dtype=np.float64)  # samples leaving at this time (event or censor)
    drop_L = np.zeros(n_unique, dtype=np.float64)

    for i in range(n):
        j = inv[i]
        drop[j] += 1.0
        if left_s[i]:
            drop_L[j] += 1.0
        if e[i] > 0:
            deaths[j] += 1.0
            if left_s[i]:
                deaths_L[j] += 1.0

    Y = float(n)
    Y_L = float(n_left)

    for j in range(n_unique):
        d = deaths[j]
        d_L = deaths_L[j]
        if d > 0 and Y > 1:
            # expected deaths in left
            E = Y_L * (d / Y)
            num += d_L - E
            # hypergeometric variance
            factor = (Y - d) / (Y - 1.0) if Y > 1 else 0.0
            den += (Y_L / Y) * (1.0 - Y_L / Y) * d * factor
        # remove those who had time = unique_t[j]
        Y -= drop[j]
        Y_L -= drop_L[j]

    if den <= 1e-12:
        return 0.0
    return abs(num) / np.sqrt(den)


# Registry for easy extension
CRITERIA = {
    "logrank": LogRankCriterion,
}


def get_criterion(name_or_obj) -> SplitCriterion:
    if isinstance(name_or_obj, SplitCriterion):
        return name_or_obj
    if isinstance(name_or_obj, str):
        key = name_or_obj.lower()
        if key not in CRITERIA:
            raise ValueError(f"Unknown criterion '{name_or_obj}'. Available: {list(CRITERIA)}")
        return CRITERIA[key]()
    raise TypeError("criterion must be str or SplitCriterion instance")
