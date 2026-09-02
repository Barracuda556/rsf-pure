"""
Example: plug in a custom split criterion and compare with log-rank.
"""

from __future__ import annotations

import numpy as np

from rsf_pure import (
    PureRandomSurvivalForest,
    PureLogRankCriterion,
    PureSplitCriterion,
    pure_concordance_index,
    pure_logrank_statistic,
)


class PurePetoLogRankCriterion(PureSplitCriterion):
    """
    Placeholder custom criterion (re-uses classic log-rank).
    Replace ``score`` with your statistic.
    """

    name = "peto_logrank"

    def score(self, time, event, left_mask):
        return pure_logrank_statistic(time, event, left_mask)


def make_synthetic(n=300, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    log_hazard = 0.8 * X[:, 0] + 0.5 * X[:, 1]
    time = rng.exponential(1.0 / np.exp(log_hazard.clip(-5, 5)))
    cens_time = rng.exponential(scale=np.median(time) * 1.2, size=n)
    event = time <= cens_time
    time = np.minimum(time, cens_time)
    return X, time, event


def main():
    X, time, event = make_synthetic()
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    train, test = idx[:200], idx[200:]

    print("=== Pure log-rank RSF ===")
    rsf_lr = PureRandomSurvivalForest(
        n_estimators=40,
        max_depth=6,
        min_samples_leaf=5,
        criterion="logrank",
        max_candidate_splits=32,
        n_jobs=-1,
        random_state=0,
    )
    rsf_lr.fit(X[train], time[train], event[train])
    risk_lr = rsf_lr.predict(X[test])
    c_lr = pure_concordance_index(time[test], event[test], risk_lr)
    print(f"C-index (logrank): {c_lr:.4f}")
    print("Feature importances:", np.round(rsf_lr.feature_importances_, 3))

    print("\n=== Custom criterion RSF ===")
    rsf_custom = PureRandomSurvivalForest(
        n_estimators=40,
        max_depth=6,
        min_samples_leaf=5,
        criterion=PurePetoLogRankCriterion(),
        max_candidate_splits=32,
        n_jobs=-1,
        random_state=0,
    )
    rsf_custom.fit(X[train], time[train], event[train])
    risk_c = rsf_custom.predict(X[test])
    c_c = pure_concordance_index(time[test], event[test], risk_c)
    print(f"C-index (custom):  {c_c:.4f}")

    times_grid = np.linspace(0, np.percentile(time, 90), 20)
    S = rsf_lr.predict_survival_function(X[test[:1]], times_grid)
    print("\nS(t) first test sample (first 5 pts):", np.round(S[0, :5], 3))


if __name__ == "__main__":
    main()
