"""
two_stage_pipeline.py
======================
Stage 3 - Component 3: The Hybrid Cascade

Wires the LinearFilter (Stage 1) and a chosen Stage 2 classifier (from
nids_stage2 — IDRandom-Forest, XGBoost, LightGBM, or baseline RF) into a
single cascading predictor:

    for each sample:
        p_attack = linear_filter.predict_proba_attack(sample)
        if p_attack < threshold:
            final_prediction = 0 (Normal)          # FAST PATH — Stage 2 skipped
        else:
            final_prediction = stage2_model.predict_at_tuned_threshold(sample)  # SLOW PATH

This class exposes a `.predict()` matching the sklearn interface so it can
be dropped into the same `metrics.py` evaluation functions from Stage 2
with zero changes.

Stage 2 decision threshold — important
-----------------------------------------
A model trained on SMOTE-balanced data and evaluated with the default 0.5
decision threshold is very often mis-calibrated on the real test
distribution (see Stage 2's methodology-review fix). This class accepts
an OPTIONAL `stage2_threshold` — if provided (tuned via Stage 2's
`find_best_threshold()` against the same validation set used for Stage
1's threshold search), it's applied to every slow-path prediction instead
of the model's raw `.predict()`. Skipping this is what caused an earlier
version of this project's Stage 3 benchmark to silently evaluate a
collapsed, near-always-predicts-Normal detector — always pass a tuned
threshold in `train_stage3.py`.
"""

import time
import numpy as np
import pandas as pd


class TwoStagePipeline:
    def __init__(self, svm_filter, stage2_model, threshold: float,
                 stage2_model_name: str = "stage2_model", stage2_threshold: float = None):
        self.svm_filter = svm_filter
        self.stage2_model = stage2_model
        self.threshold = threshold
        self.stage2_model_name = stage2_model_name
        self.stage2_threshold = stage2_threshold  # None -> falls back to model's default .predict()

    def _stage2_predict(self, X_slow):
        """Applies the tuned Stage 2 decision threshold if one was provided."""
        if self.stage2_threshold is not None and hasattr(self.stage2_model, "predict_proba"):
            proba = self.stage2_model.predict_proba(X_slow)
            proba_attack = proba[:, 1] if proba.ndim > 1 else proba
            return (proba_attack >= self.stage2_threshold).astype(int)
        return self.stage2_model.predict(X_slow)

    def predict(self, X):
        """Standard sklearn-style predict — returns final 0/1 labels."""
        preds, _ = self.predict_with_diagnostics(X)
        return preds

    def predict_with_diagnostics(self, X):
        """
        Returns
        -------
        final_predictions : np.ndarray of 0/1
        diagnostics : dict with routing info (how many samples took each path)
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X
        else:
            X_arr = X

        proba_attack = self.svm_filter.predict_proba_attack(X_arr)
        short_circuit_mask = proba_attack < self.threshold  # confidently Normal -> fast path

        final_predictions = np.zeros(len(proba_attack), dtype=int)  # default: Normal

        n_fast = short_circuit_mask.sum()
        n_slow = (~short_circuit_mask).sum()

        if n_slow > 0:
            if isinstance(X, pd.DataFrame):
                X_slow = X_arr.loc[~short_circuit_mask]
            else:
                X_slow = X_arr[~short_circuit_mask]
            stage2_preds = self._stage2_predict(X_slow)
            final_predictions[~short_circuit_mask] = stage2_preds

        diagnostics = {
            "n_total": len(proba_attack),
            "n_fast_path": int(n_fast),
            "n_slow_path": int(n_slow),
            "fraction_fast_path": n_fast / len(proba_attack),
        }
        return final_predictions, diagnostics

    def predict_proba(self, X):
        """
        Approximate probability output for compatibility with metrics that
        expect predict_proba (e.g. ROC-AUC). Fast-path samples get a hard
        0.0 attack-probability (SVM was confident); slow-path samples use
        Stage 2's own probability if available, else a hard prediction.
        """
        if isinstance(X, pd.DataFrame):
            X_arr = X
        else:
            X_arr = X

        proba_attack_svm = self.svm_filter.predict_proba_attack(X_arr)
        short_circuit_mask = proba_attack_svm < self.threshold

        final_proba_attack = np.zeros(len(proba_attack_svm))

        if (~short_circuit_mask).sum() > 0:
            if isinstance(X, pd.DataFrame):
                X_slow = X_arr.loc[~short_circuit_mask]
            else:
                X_slow = X_arr[~short_circuit_mask]

            if hasattr(self.stage2_model, "predict_proba"):
                stage2_proba = self.stage2_model.predict_proba(X_slow)
                classes = list(getattr(self.stage2_model, "classes_", [0, 1]))
                attack_idx = classes.index(1) if 1 in classes else 1
                final_proba_attack[~short_circuit_mask] = stage2_proba[:, attack_idx]
            else:
                final_proba_attack[~short_circuit_mask] = self.stage2_model.predict(X_slow)

        return np.column_stack([1 - final_proba_attack, final_proba_attack])

    def measure_latency(self, X, n_repeats: int = 5) -> dict:
        """
        Measures END-TO-END average per-sample latency of the cascade —
        this is the headline number for Stage 3: the two-stage pipeline's
        latency should be lower than running Stage 2 alone on 100% of
        traffic, because most samples take the cheap fast path.
        """
        n_samples = len(X)
        per_sample_times_ms = []
        diagnostics = None

        for _ in range(n_repeats):
            start = time.perf_counter()
            _, diagnostics = self.predict_with_diagnostics(X)
            elapsed = time.perf_counter() - start
            per_sample_times_ms.append((elapsed / n_samples) * 1000)

        return {
            "avg_latency_ms_per_sample": float(np.mean(per_sample_times_ms)),
            "std_latency_ms_per_sample": float(np.std(per_sample_times_ms)),
            "n_test_samples": n_samples,
            "fraction_fast_path": diagnostics["fraction_fast_path"],
            "n_fast_path": diagnostics["n_fast_path"],
            "n_slow_path": diagnostics["n_slow_path"],
        }
