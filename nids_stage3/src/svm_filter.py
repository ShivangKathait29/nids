"""
svm_filter.py
=============
Stage 3 - Component 1: Fast First-Stage Filter

Implements the "Two-Stage Hybrid Filtering" architecture from the base
paper brief: a cheap, fast filter that confidently clears obviously-normal
traffic without invoking the expensive Stage 2 classifier, and only passes
"suspicious" traffic on for deep analysis.

Naming correction
-------------------
An earlier version of this module defaulted to `SGDClassifier(loss=
"log_loss")` and called it an "SVM filter" — that's inaccurate.
`SGDClassifier` with log_loss trains a plain logistic-regression-style
linear classifier via stochastic gradient descent; it is NOT a Support
Vector Machine (which specifically optimizes hinge loss / a max-margin
objective). This module now offers BOTH, named honestly:

    - LinearFilter (default, kernel="sgd_log_loss"): the original fast
      choice. Cheap, gives native probability estimates, appropriate for
      a first-stage filter. Correctly called a "Linear Filter" or
      "SGD-based Linear Filter" in your report — NOT an SVM.
    - A genuine Linear SVM option (kernel="linear_svm"): uses
      `sklearn.svm.LinearSVC` (hinge loss, true max-margin SVM objective),
      wrapped in `CalibratedClassifierCV` to get probability estimates
      (LinearSVC has no native predict_proba). If your report says "SVM
      filter," use THIS option so the claim is accurate.
    - kernel="rbf": true kernel SVM via `SVC`, included only as a slower
      comparison point / ablation — NOT recommended as the deployed
      filter, since kernel-SVM inference cost scales with the number of
      support vectors.

Whichever kernel you choose, the filter outputs a probability of "attack"
for every sample. Whether a sample counts as "confidently normal" (and can
skip Stage 2) is controlled by a threshold tuned in threshold_tuning.py —
NOT hardcoded here.
"""

import time
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV


class LinearFilter:
    """
    Fast first-stage traffic filter.

    Parameters
    ----------
    kernel : "sgd_log_loss" (default — fast linear classifier via SGD,
                 NOT an SVM, native predict_proba)
             "linear_svm" (a genuine Linear SVM — LinearSVC + calibration
                 for probability estimates — use this if your report
                 specifically claims "SVM filter")
             "rbf" (true kernel SVM via SVC — slow, comparison/ablation
                 only, not recommended as the deployed filter)
    """

    def __init__(self, kernel: str = "sgd_log_loss", random_state: int = 42):
        self.kernel = kernel
        self.random_state = random_state
        self.model_ = None
        self.train_time_sec_ = None

    def fit(self, X, y):
        t0 = time.time()
        if self.kernel == "linear":
            # backward-compatible alias for the old default name
            self.kernel = "sgd_log_loss"
        if self.kernel == "sgd_log_loss":
            # Fast linear classifier via stochastic gradient descent, native
            # predict_proba. This is a logistic-regression-family model,
            # NOT a Support Vector Machine — call it a "Linear Filter" or
            # "SGD-based Linear Filter" in your report, not an SVM.
            self.model_ = SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                tol=1e-3,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model_.fit(X, y)
        elif self.kernel == "linear_svm":
            # A GENUINE linear SVM: LinearSVC optimizes hinge loss with a
            # max-margin objective, which is what "SVM" actually means.
            # LinearSVC has no native predict_proba, so we calibrate it
            # via Platt scaling (CalibratedClassifierCV) to get usable
            # probability estimates for the threshold search downstream.
            base = LinearSVC(random_state=self.random_state, max_iter=5000, dual="auto")
            self.model_ = CalibratedClassifierCV(base, cv=3)
            self.model_.fit(X, y)
        elif self.kernel == "rbf":
            # true kernel SVM, calibrated for probability estimates.
            # Included for comparison only — NOT recommended as the actual
            # deployed filter, since kernel SVM inference cost scales with
            # the number of support vectors and is far slower than linear.
            base = SVC(kernel="rbf", probability=False, random_state=self.random_state)
            self.model_ = CalibratedClassifierCV(base, cv=3)
            self.model_.fit(X, y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        self.train_time_sec_ = time.time() - t0
        return self

    def predict_proba_attack(self, X) -> np.ndarray:
        """Returns P(attack) for each sample — the score threshold_tuning.py uses."""
        proba = self.model_.predict_proba(X)
        # class order from sklearn is sorted(unique(y)) -> [0, 1] for our binary labels
        classes = list(self.model_.classes_)
        attack_idx = classes.index(1)
        return proba[:, attack_idx]

    def predict(self, X):
        return self.model_.predict(X)


# Backward-compatible alias — earlier versions of this project called this
# class "SVMFilter" and defaulted to kernel="linear" (now "sgd_log_loss").
# Kept so any existing imports/scripts don't break; new code should use
# LinearFilter directly and pick kernel="linear_svm" if a true SVM claim is needed.
SVMFilter = LinearFilter
