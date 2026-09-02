"""
Single survival tree used inside PureRandomSurvivalForest.

Names are prefixed with Pure* to avoid clashes with sksurv / sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from .criterion import PureSplitCriterion, get_criterion


@dataclass
class PureTreeNode:
    """Tree node. Leaf stores KM / Nelson–Aalen estimates."""

    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["PureTreeNode"] = None
    right: Optional["PureTreeNode"] = None

    is_leaf: bool = False
    n_samples: int = 0
    n_events: int = 0
    times: Optional[np.ndarray] = None
    survival: Optional[np.ndarray] = None
    cumhaz: Optional[np.ndarray] = None


def _kaplan_meier_nelson_aalen(
    time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns unique_times, S(t), H(t) for the sample in a leaf."""
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    e = event[order].astype(np.float64)

    unique_t, inv = np.unique(t, return_inverse=True)
    n_u = unique_t.size
    deaths = np.zeros(n_u, dtype=np.float64)
    drop = np.zeros(n_u, dtype=np.float64)
    for i in range(t.size):
        j = inv[i]
        drop[j] += 1.0
        if e[i] > 0:
            deaths[j] += 1.0

    n = float(t.size)
    S = np.ones(n_u, dtype=np.float64)
    H = np.zeros(n_u, dtype=np.float64)
    at_risk = n
    surv = 1.0
    haz = 0.0
    for j in range(n_u):
        d = deaths[j]
        if at_risk > 0 and d > 0:
            haz += d / at_risk
            surv *= 1.0 - d / at_risk
        S[j] = surv
        H[j] = haz
        at_risk -= drop[j]

    return unique_t, S, H


class PureSurvivalTree:
    """
    Binary survival tree with pluggable split criterion.

    Parameters
    ----------
    criterion : str or PureSplitCriterion
        Default "logrank".
    max_depth : int or None
    min_samples_split : int
    min_samples_leaf : int
    max_features : int, float, "sqrt", "log2" or None
    min_events_leaf : int
    random_state : int or None
    max_candidate_splits : int or None
        Cap on candidate thresholds per feature (default 32).
        None = try all unique values.
    """

    def __init__(
        self,
        criterion: Union[str, PureSplitCriterion] = "logrank",
        max_depth: Optional[int] = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: Union[int, float, str, None] = "sqrt",
        min_events_leaf: int = 1,
        random_state: Optional[int] = None,
        max_candidate_splits: Optional[int] = 32,
    ):
        self.criterion = get_criterion(criterion)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_events_leaf = min_events_leaf
        self.random_state = random_state
        self.max_candidate_splits = max_candidate_splits

        self.root_: Optional[PureTreeNode] = None
        self.n_features_in_: Optional[int] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self._leaves_: List[PureTreeNode] = []

    def fit(
        self, X: np.ndarray, time: np.ndarray, event: np.ndarray
    ) -> "PureSurvivalTree":
        X = np.asarray(X, dtype=np.float64)
        time = np.asarray(time, dtype=np.float64).ravel()
        event = np.asarray(event).ravel().astype(bool)

        if X.ndim != 2:
            raise ValueError("X must be 2-d")
        n, p = X.shape
        if time.shape[0] != n or event.shape[0] != n:
            raise ValueError("time/event length mismatch")

        self.n_features_in_ = p
        self.feature_importances_ = np.zeros(p, dtype=np.float64)
        self._leaves_ = []
        rng = np.random.default_rng(self.random_state)

        self.root_ = self._build(X, time, event, depth=0, rng=rng)
        s = self.feature_importances_.sum()
        if s > 0:
            self.feature_importances_ /= s
        return self

    def _n_features_to_try(self, p: int) -> int:
        mf = self.max_features
        if mf is None:
            return p
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(p)))
            if mf == "log2":
                return max(1, int(np.log2(p)))
            raise ValueError(f"Unknown max_features: {mf}")
        if isinstance(mf, float):
            return max(1, int(mf * p))
        return max(1, min(int(mf), p))

    def _build(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        depth: int,
        rng: np.random.Generator,
    ) -> PureTreeNode:
        n, p = X.shape
        node = PureTreeNode(n_samples=n, n_events=int(event.sum()))

        stop = (
            n < self.min_samples_split
            or node.n_events < self.min_events_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        )
        if stop:
            return self._make_leaf(node, time, event)

        n_try = self._n_features_to_try(p)
        feat_idx = rng.choice(p, size=n_try, replace=False)

        best_score = -np.inf
        best_feat = None
        best_thr = None
        best_left = None

        for j in feat_idx:
            score, thr, left_mask = self.criterion.best_split_on_feature(
                X[:, j],
                time,
                event,
                min_leaf=self.min_samples_leaf,
                max_candidates=self.max_candidate_splits,
                rng=rng,
            )
            if score > best_score and left_mask is not None:
                best_score = score
                best_feat = j
                best_thr = thr
                best_left = left_mask

        if best_feat is None or best_left is None:
            return self._make_leaf(node, time, event)

        self.feature_importances_[best_feat] += best_score * n

        node.feature = best_feat
        node.threshold = best_thr
        left_idx = best_left
        right_idx = ~best_left

        node.left = self._build(
            X[left_idx], time[left_idx], event[left_idx], depth + 1, rng
        )
        node.right = self._build(
            X[right_idx], time[right_idx], event[right_idx], depth + 1, rng
        )
        return node

    def _make_leaf(
        self, node: PureTreeNode, time: np.ndarray, event: np.ndarray
    ) -> PureTreeNode:
        node.is_leaf = True
        times, S, H = _kaplan_meier_nelson_aalen(time, event)
        node.times = times
        node.survival = S
        node.cumhaz = H
        self._leaves_.append(node)
        return node

    def _apply_batch(self, X: np.ndarray) -> List[PureTreeNode]:
        """Route every row to its leaf (still Python loop; leaves are few)."""
        n = X.shape[0]
        leaves: List[PureTreeNode] = [self.root_] * n  # type: ignore
        for i in range(n):
            node = self.root_
            x = X[i]
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            leaves[i] = node
        return leaves

    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        n = X.shape[0]
        m = times.size
        out = np.zeros((n, m), dtype=np.float64)
        leaves = self._apply_batch(X)
        for i, leaf in enumerate(leaves):
            idx = np.searchsorted(leaf.times, times, side="right") - 1
            valid = idx >= 0
            out[i, valid] = leaf.cumhaz[idx[valid]]
        return out

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        n = X.shape[0]
        m = times.size
        out = np.ones((n, m), dtype=np.float64)
        leaves = self._apply_batch(X)
        for i, leaf in enumerate(leaves):
            idx = np.searchsorted(leaf.times, times, side="right") - 1
            valid = idx >= 0
            out[i, valid] = leaf.survival[idx[valid]]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Risk score = leaf cumulative hazard at last time (higher = higher risk)."""
        X = np.asarray(X, dtype=np.float64)
        leaves = self._apply_batch(X)
        risk = np.empty(X.shape[0], dtype=np.float64)
        for i, leaf in enumerate(leaves):
            risk[i] = leaf.cumhaz[-1] if leaf.cumhaz.size else 0.0
        return risk
