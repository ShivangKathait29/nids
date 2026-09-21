"""
metrics.py
==========
Stage 2 - Component 2: Evaluation Metrics

Computes the things that actually matter for an imbalanced intrusion
detection problem — NOT just accuracy, which is misleading when ~80-85%
of real traffic is benign (a model that predicts "normal" for everything
scores 80%+ accuracy while catching zero attacks):
    - Classification quality: precision, recall, F1, ROC-AUC, PR-AUC
      (PR-AUC in particular — ROC-AUC can look deceptively reasonable on
      imbalanced data; PR-AUC is the harder, more honest metric here),
      with accuracy reported LAST, as a secondary/sanity-check number only
    - Inference latency: average prediction time per sample (ms) — critical
      for a system claiming to be "real-time"
    - Memory footprint: serialized model size on disk (proxy for RAM usage
      when loaded, and directly relevant to the "Hardware-Aware /
      Edge-IoT" discussion even though that's not in your current scope)

A note on accuracy for this problem
------------------------------------
On a test set that's ~80% benign / ~20% attack, a trivial "always predict
Normal" classifier already scores ~80% accuracy while catching ZERO
attacks — recall = 0. If your model's accuracy is close to the majority-
class fraction and recall is low, that is a red flag that the model has
collapsed to (near-)always-predict-normal, NOT a sign of good performance.
Always check recall and PR-AUC before trusting an accuracy number on this
kind of data.
"""

import os
import pickle
import tempfile
import time
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, precision_recall_curve,
)


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict:
    """
    Standard classification metrics for binary (or multiclass) labels.
    Ordered with recall/precision/F1 first — accuracy is included but
    should be treated as secondary for imbalanced IDS data (see module
    docstring). Also flags majority-class collapse explicitly.
    """
    average = "binary" if len(np.unique(y_true)) == 2 else "macro"
    metrics = {
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),  # secondary metric — see module docstring
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            proba_positive = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
            metrics["roc_auc"] = roc_auc_score(y_true, proba_positive)
            metrics["pr_auc"] = average_precision_score(y_true, proba_positive)
        except Exception:
            pass

    # Majority-class collapse check: warn if the model's accuracy is close
    # to (or below) what a trivial "always predict majority class" baseline
    # would achieve, AND recall is low — the classic false signal of an
    # imbalanced-data model that learned nothing useful.
    y_true_arr = np.asarray(y_true)
    majority_baseline_acc = max(np.mean(y_true_arr == 0), np.mean(y_true_arr == 1))
    metrics["majority_class_baseline_accuracy"] = majority_baseline_acc
    metrics["accuracy_above_majority_baseline"] = metrics["accuracy"] - majority_baseline_acc
    if metrics["accuracy"] <= majority_baseline_acc + 0.01 and metrics["recall"] < 0.3:
        metrics["_warning"] = (
            "Accuracy is at/near the trivial majority-class baseline AND recall is low "
            "(<30%) — this model has likely collapsed to near-always-predicting the "
            "majority class rather than learning real separation. Do not report this "
            "accuracy figure as evidence of good detection performance."
        )
    return metrics


def find_best_threshold(y_true, y_proba, metric: str = "f1", min_precision: float = None) -> dict:
    """
    Sweeps decision thresholds against precision_recall_curve and picks the
    one that optimizes the requested metric, INSTEAD of relying on the
    classifier's default 0.5 threshold (which is rarely optimal for
    imbalanced data and SMOTE-trained models evaluated on a real-world
    distribution).

    IMPORTANT: call this on a VALIDATION set (real distribution, untouched
    by training/resampling) — never on the test set. The selected
    threshold should then be applied ONCE to the held-out test set for
    final reporting.

    Parameters
    ----------
    metric : "f1" (default) — maximize F1
             "recall_at_precision" — maximize recall subject to
             precision >= min_precision (set min_precision when using this)

    Returns
    -------
    dict with the selected threshold and the validation-set metrics at
    that threshold, plus the full sweep for plotting.
    """
    proba_positive = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba_positive)
    # precision_recall_curve returns one more precision/recall point than
    # thresholds (the last point is precision=1,recall=0 with no threshold);
    # trim to align.
    precisions, recalls = precisions[:-1], recalls[:-1]

    f1_scores = np.where(
        (precisions + recalls) > 0, 2 * precisions * recalls / (precisions + recalls + 1e-12), 0
    )

    if metric == "f1":
        best_idx = int(np.argmax(f1_scores))
    elif metric == "recall_at_precision":
        if min_precision is None:
            raise ValueError("min_precision must be set when metric='recall_at_precision'")
        eligible = np.where(precisions >= min_precision)[0]
        if len(eligible) == 0:
            # nothing meets the precision bar — fall back to best-F1 and warn
            best_idx = int(np.argmax(f1_scores))
            fallback_note = (
                f"No threshold achieved precision >= {min_precision}; fell back to "
                f"best-F1 threshold instead."
            )
        else:
            best_idx = int(eligible[np.argmax(recalls[eligible])])
            fallback_note = None
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return {
        "threshold": float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5,
        "val_precision_at_threshold": float(precisions[best_idx]),
        "val_recall_at_threshold": float(recalls[best_idx]),
        "val_f1_at_threshold": float(f1_scores[best_idx]),
        "sweep": {
            "thresholds": thresholds.tolist(),
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
            "f1_scores": f1_scores.tolist(),
        },
    }


def predict_with_threshold(model, X, threshold: float) -> np.ndarray:
    """Applies a CUSTOM decision threshold instead of the model's default 0.5."""
    proba = model.predict_proba(X)
    proba_positive = proba[:, 1] if proba.ndim > 1 else proba
    return (proba_positive >= threshold).astype(int)


def measure_inference_latency(model, X_test, n_repeats: int = 5) -> dict:
    """
    Measures average per-sample inference latency in milliseconds.

    Runs `n_repeats` full passes over X_test and averages, to smooth out
    noise from a single run (JIT warmup, OS scheduling jitter, etc).
    """
    n_samples = len(X_test)
    per_sample_times_ms = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        model.predict(X_test)
        elapsed = time.perf_counter() - start
        per_sample_times_ms.append((elapsed / n_samples) * 1000)

    return {
        "avg_latency_ms_per_sample": float(np.mean(per_sample_times_ms)),
        "std_latency_ms_per_sample": float(np.std(per_sample_times_ms)),
        "total_batch_predict_ms": float(np.mean(per_sample_times_ms) * n_samples),
        "n_test_samples": n_samples,
    }


def measure_model_memory(model) -> dict:
    """
    Serializes the model to disk and measures its size — a practical,
    reproducible proxy for the memory footprint claim in the base paper
    (loading a smaller serialized model uses proportionally less RAM).
    """
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name

    size_bytes = os.path.getsize(tmp_path)
    os.remove(tmp_path)

    return {
        "model_size_kb": round(size_bytes / 1024, 2),
        "model_size_mb": round(size_bytes / (1024 * 1024), 4),
    }


def full_model_report(model, X_test, y_test, model_name: str, latency_repeats: int = 5,
                       decision_threshold: float = None) -> dict:
    """
    Runs everything above and returns one flat dict — one row of your
    comparison table.

    decision_threshold : if provided (recommended — tune this on a
                          validation set first via find_best_threshold),
                          predictions use this threshold instead of the
                          model's default 0.5 cutoff.
    """
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    if decision_threshold is not None and y_proba is not None:
        y_pred = predict_with_threshold(model, X_test, decision_threshold)
    else:
        y_pred = model.predict(X_test)

    report = {"model_name": model_name, "decision_threshold": decision_threshold if decision_threshold is not None else 0.5}
    report.update(evaluate_classification(y_test, y_pred, y_proba))
    report.update(measure_inference_latency(model, X_test, n_repeats=latency_repeats))
    report.update(measure_model_memory(model))
    return report


def confusion_matrix_and_text_report(model, X_test, y_test) -> tuple:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    text_report = classification_report(y_test, y_pred, zero_division=0)
    return cm, text_report
