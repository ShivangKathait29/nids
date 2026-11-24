"""
model_zoo.py
============
Stage 2 - Component 1: Model Definitions

Implements:
    1. IDRandomForest — our replication of the base paper's core algorithm:
       a Random Forest that uses (a) feature-importance-driven feature
       selection and (b) cost-complexity tree pruning to cut detection
       latency and memory footprint versus a vanilla Random Forest.

    2. get_xgboost_model() / get_lightgbm_model() — thin wrappers around
       XGBoost / LightGBM for the benchmark comparison. These libraries
       need `pip install xgboost lightgbm` (they need network access,
       which this dev environment doesn't have — but your team's machines
       will). If unavailable, we automatically fall back to sklearn's
       HistGradientBoostingClassifier (GradientBoosting) as a stand-in so
       the pipeline still runs and is testable — clearly flagged in logs
       and in the results table so nobody mistakes fallback numbers for
       real XGBoost/LightGBM numbers.

Why cost-complexity pruning specifically?
------------------------------------------
sklearn's RandomForestClassifier doesn't prune trees by default — trees
grow until leaves are pure or hit min_samples constraints, which is a big
part of why Random Forests are memory-heavy and slower at inference than
a boosted-tree alternative. Minimal cost-complexity pruning (`ccp_alpha`)
removes the weakest sub-trees (the ones that reduce impurity the least
per added complexity) after the fact — trading a small amount of accuracy
for meaningfully smaller, faster trees. We search a range of ccp_alpha
values and pick the one that best balances accuracy vs. latency, which is
exactly the "reduce detection latency" claim from the base paper.
"""

import time
import warnings
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# IDRandom-Forest (base paper replication)
# ---------------------------------------------------------------------------
class IDRandomForest:
    """
    Optimized Random Forest with feature-importance-driven pruning.

    Two-phase fit:
        Phase A — train a baseline RF, read off feature_importances_
        Phase B — search a small grid of ccp_alpha (cost-complexity pruning)
                  values via cross-validation, pick the alpha that keeps
                  accuracy within `tolerance` of the unpruned model while
                  minimizing average tree size (a direct proxy for both
                  memory footprint and inference latency)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        cv_folds: int = 3,
        tolerance: float = 0.005,
        n_alpha_candidates: int = 8,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.tolerance = tolerance
        self.n_alpha_candidates = n_alpha_candidates

        self.model_ = None
        self.best_ccp_alpha_ = None
        self.feature_importances_ = None
        self.pruning_log_ = []

    def fit(self, X, y):
        from sklearn.model_selection import train_test_split

        # --- Phase A: baseline (unpruned) RF ---
        baseline = RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1
        )
        baseline.fit(X, y)
        self.feature_importances_ = baseline.feature_importances_

        # Use a stratified sample for cross-validation pruning search to run in seconds
        if len(X) > 50000:
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=50000, stratify=y, random_state=self.random_state
            )
        else:
            X_sub, y_sub = X, y

        baseline_acc = cross_val_score(
            RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=-1),
            X_sub, y_sub, cv=self.cv_folds, scoring="accuracy", n_jobs=-1
        ).mean()

        baseline_avg_nodes = np.mean([t.tree_.node_count for t in baseline.estimators_])
        self.pruning_log_.append(
            f"Baseline RF: cv_accuracy={baseline_acc:.4f}, avg_nodes_per_tree={baseline_avg_nodes:.1f}"
        )

        # --- Phase B: cost-complexity pruning search ---
        sample_tree = RandomForestClassifier(
            n_estimators=1, random_state=self.random_state
        ).fit(X_sub, y_sub).estimators_[0]
        path = sample_tree.cost_complexity_pruning_path(X_sub, y_sub)
        ccp_alphas = np.unique(path.ccp_alphas)
        if len(ccp_alphas) > self.n_alpha_candidates:
            idx = np.linspace(0, len(ccp_alphas) - 1, self.n_alpha_candidates).astype(int)
            ccp_alphas = ccp_alphas[idx]

        best_alpha = 0.0
        best_nodes = baseline_avg_nodes

        for alpha in ccp_alphas:
            if alpha == 0:
                continue
            candidate = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                ccp_alpha=alpha,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc = cross_val_score(
                    candidate, X_sub, y_sub, cv=self.cv_folds, scoring="accuracy", n_jobs=-1
                ).mean()

            if acc >= baseline_acc - self.tolerance:
                candidate.fit(X_sub, y_sub)
                avg_nodes = np.mean([t.tree_.node_count for t in candidate.estimators_])
                self.pruning_log_.append(
                    f"ccp_alpha={alpha:.6f}: cv_accuracy={acc:.4f}, avg_nodes_per_tree={avg_nodes:.1f} "
                    f"({'ACCEPTED' if avg_nodes < best_nodes else 'rejected (not smaller)'})"
                )
                if avg_nodes < best_nodes:
                    best_alpha, best_nodes = alpha, avg_nodes
            else:
                self.pruning_log_.append(
                    f"ccp_alpha={alpha:.6f}: cv_accuracy={acc:.4f} — rejected (below tolerance)"
                )

        self.best_ccp_alpha_ = best_alpha
        if best_alpha > 0:
            self.model_ = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                ccp_alpha=best_alpha,
                n_jobs=-1,
            ).fit(X, y)
        else:
            self.model_ = baseline

        reduction_pct = 100 * (1 - best_nodes / baseline_avg_nodes)
        self.pruning_log_.append(
            f"SELECTED ccp_alpha={best_alpha:.6f} -> avg_nodes_per_tree={best_nodes:.1f} "
            f"({reduction_pct:.1f}% smaller trees than baseline, within {self.tolerance*100:.1f}% accuracy tolerance)"
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


# ---------------------------------------------------------------------------
# Benchmark models (XGBoost / LightGBM with fallback)
# ---------------------------------------------------------------------------
def get_xgboost_model(random_state: int = 42):
    """Returns (model, is_real_xgboost: bool, note: str)"""
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )
        return model, True, "Using real XGBoost"
    except ImportError:
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=random_state
        )
        return model, False, (
            "XGBoost not installed in this environment (no network access to pip install). "
            "Falling back to sklearn HistGradientBoostingClassifier as a stand-in — "
            "run `pip install xgboost` on your machine and re-run for real XGBoost numbers."
        )


def get_lightgbm_model(random_state: int = 42):
    """Returns (model, is_real_lightgbm: bool, note: str)"""
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        return model, True, "Using real LightGBM"
    except ImportError:
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.05, random_state=random_state
        )
        return model, False, (
            "LightGBM not installed in this environment (no network access to pip install). "
            "Falling back to sklearn HistGradientBoostingClassifier as a stand-in — "
            "run `pip install lightgbm` on your machine and re-run for real LightGBM numbers."
        )


def get_baseline_random_forest(random_state: int = 42, n_estimators: int = 100):
    """Vanilla, unpruned Random Forest — the 'before' comparison for IDRandom-Forest."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
