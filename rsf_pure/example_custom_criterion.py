"""
Example: how to plug in a new split criterion and compare with log-rank.
"""

from __future__ import annotations

import numpy as np

from rsf_pure import (
    RandomSurvivalForest,
    LogRankCriterion,
    SplitCriterion,
    concordance_index,
)


class PetoLogRankCriterion(SplitCriterion):
    """
    Illustration of a custom criterion.
    Here we simply re-use classic log-rank (placeholder).
    Replace the body with your own statistic.
    """

    name = "peto_logrank"

    def score(self, time, event, left_mask):
        # --- replace this with your criterion ---
        from rsf_pure.criterion import _logrank_statistic
        return _logrank_statistic(time, event, left_mask)


def make_synthetic(n=300, p=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # risk depends on first two features
    log_hazard = 0.8 * X[:, 0] + 0.5 * X[:, 1]
    # exponential times with censoring
    time = rng.exponential(1.0 / np.exp(log_hazard.clip(-5, 5)))
    cens_time = rng.exponential(scale=np.median(time) * 1.2, size=n)
    event = time <= cens_time
    time = np.minimum(time, cens_time)
    return X, time, event


def main():
    X, time, event = make_synthetic()
    # train / test split
    n = X.shape[0]
    idx = np.random.default_rng(0).permutation(n)
    train, test = idx[:200], idx[200:]

    print("=== Log-rank RSF ===")
    rsf_lr = RandomSurvivalForest(
        n_estimators=40,
        max_depth=6,
        min_samples_leaf=5,
        criterion="logrank",
        n_jobs=-1,
        random_state=0,
    )
    rsf_lr.fit(X[train], time[train], event[train])
    risk_lr = rsf_lr.predict(X[test])
    c_lr = concordance_index(time[test], event[test], risk_lr)
    print(f"C-index (logrank): {c_lr:.4f}")
    print("Feature importances:", np.round(rsf_lr.feature_importances_, 3))

    print("\n=== Custom criterion RSF ===")
    rsf_custom = RandomSurvivalForest(
        n_estimators=40,
        max_depth=6,
        min_samples_leaf=5,
        criterion=PetoLogRankCriterion(),
        n_jobs=-1,
        random_state=0,
    )
    rsf_custom.fit(X[train], time[train], event[train])
    risk_c = rsf_custom.predict(X[test])
    c_c = concordance_index(time[test], event[test], risk_c)
    print(f"C-index (custom):  {c_c:.4f}")

    # survival curves for first test sample
    times_grid = np.linspace(0, np.percentile(time, 90), 20)
    S = rsf_lr.predict_survival_function(X[test[:1]], times_grid)
    print("\nS(t) for first test sample (first 5 points):", np.round(S[0, :5], 3))


if __name__ == "__main__":
    main()
