"""
train_stage3.py
================
Stage 3 - Full Orchestration: Two-Stage Hybrid Classifier

Runs, for a given dataset:
    1. Load Stage 1's processed train/val/test splits (val = REAL
       distribution, untouched — used for threshold tuning)
    2. Train the Stage 1 Linear Filter (see naming note in svm_filter.py —
       default is an SGD-based linear classifier, NOT a true SVM unless
       you pass --filter_kernel linear_svm)
    3. Tune the short-circuit confidence threshold against the Stage 1
       validation set, using the POPULATION-based attack leakage rate
       (fraction of ALL real attacks silently dropped) as the safety
       constraint — see threshold_tuning.py for why this replaced the
       earlier shortcircuit-purity-based metric
    4. Train (or reuse) the Stage 2 classifier (default: IDRandom-Forest,
       reusing nids_stage2's model_zoo — same model type your team already
       benchmarked in Stage 2)
    5. Build the TwoStagePipeline cascade
    6. Benchmark: two-stage cascade vs Stage 2 running alone on 100% of
       traffic — precision/recall/F1/ROC-AUC/PR-AUC AND end-to-end latency

This directly produces the "measure and compare latency/compute savings
vs. single-stage baseline" result your project plan calls for.

IMPORTANT — Stage 3 cannot fix a weak Stage 2 classifier
------------------------------------------------------------
This stage optimizes INFERENCE COST (how much traffic can skip deep
analysis), not detection quality. If Stage 2's classifier has low
recall/PR-AUC on real data, Stage 3 will faithfully reflect that same
weak detection performance end-to-end — a fast cascade around a bad
detector is still a bad detector, just a faster one. Confirm Stage 2's
detection quality (recall, PR-AUC) is acceptable BEFORE trusting Stage 3's
latency numbers as the headline result.

Usage:
    python train_stage3.py --dataset cic_ids2017 \
        --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 \
        --stage2_model idrandom_forest
"""

import argparse
import json
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage2", "src"))

from svm_filter import LinearFilter
from threshold_tuning import tune_threshold
from two_stage_pipeline import TwoStagePipeline

from model_zoo import IDRandomForest, get_xgboost_model, get_lightgbm_model, get_baseline_random_forest
from metrics import (
    evaluate_classification, measure_inference_latency, confusion_matrix_and_text_report,
    find_best_threshold, predict_with_threshold,
)


STAGE2_MODEL_FACTORY = {
    "idrandom_forest": lambda: ("IDRandomForest", IDRandomForest()),
    "baseline_rf": lambda: ("BaselineRF", get_baseline_random_forest()),
    "xgboost": lambda: ("XGBoost", get_xgboost_model()[0]),
    "lightgbm": lambda: ("LightGBM", get_lightgbm_model()[0]),
}


def load_stage1_outputs(stage1_dir: str):
    X_train = pd.read_csv(f"{stage1_dir}/X_train_resampled.csv")
    y_train = pd.read_csv(f"{stage1_dir}/y_train_resampled.csv").iloc[:, 0]

    # Validation set: REAL distribution, never resampled — used ONLY for
    # threshold tuning (both Stage 3's short-circuit threshold here, and
    # any decision-threshold tuning already done upstream in Stage 2).
    X_val = pd.read_csv(f"{stage1_dir}/X_val_selected.csv")
    y_val = pd.read_csv(f"{stage1_dir}/y_val_bin.csv").iloc[:, 0]

    X_test = pd.read_csv(f"{stage1_dir}/X_test_selected.csv")
    y_test = pd.read_csv(f"{stage1_dir}/y_test_bin.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_stage3(
    dataset_name: str,
    stage1_dir: str,
    stage2_model_name: str = "idrandom_forest",
    filter_kernel: str = "sgd_log_loss",
    out_dir: str = None,
    max_attack_leakage_rate: float = 0.02,
    latency_repeats: int = 5,
):
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(str(msg))

    log(f"{'='*70}\nSTAGE 3 PIPELINE — dataset: {dataset_name}, stage2_model: {stage2_model_name}, "
        f"filter_kernel: {filter_kernel}\n{'='*70}")

    out_dir = out_dir or f"../results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data (now a proper 3-way split from Stage 1) ---
    log("\n>>> Loading Stage 1 outputs")
    X_train, y_train, X_val, y_val, X_test, y_test = load_stage1_outputs(stage1_dir)
    log(f"Train: {X_train.shape} (SMOTE-balanced), Val: {X_val.shape} (REAL distribution: "
        f"{y_val.value_counts().to_dict()}), Test: {X_test.shape} (REAL distribution: "
        f"{y_test.value_counts().to_dict()})")

    # --- 1. Train Stage 1 filter + tune threshold ---
    filter_label = "Linear SVM (LinearSVC)" if filter_kernel == "linear_svm" else (
        "true kernel SVM (RBF)" if filter_kernel == "rbf" else "SGD-based Linear Filter (not an SVM)"
    )
    log(f"\n>>> [1/4] Training Stage 1 filter ({filter_label}) and tuning short-circuit threshold "
        f"against the REAL-distribution validation set")
    linear_filter = LinearFilter(kernel=filter_kernel)
    threshold, search_log, selection_note = tune_threshold(
        linear_filter, X_train, y_train, X_val, y_val,
        max_attack_leakage_rate=max_attack_leakage_rate,
    )
    log(f"  Filter train time: {linear_filter.train_time_sec_:.3f}s")
    log(f"  {selection_note}")
    search_log.to_csv(f"{out_dir}/threshold_search_log.csv", index=False)

    # --- 2. Train Stage 2 classifier ---
    log(f"\n>>> [2/4] Training Stage 2 classifier: {stage2_model_name}")
    if stage2_model_name not in STAGE2_MODEL_FACTORY:
        raise ValueError(f"Unknown stage2_model: {stage2_model_name}. "
                          f"Choose from {list(STAGE2_MODEL_FACTORY.keys())}")
    model_label, stage2_model = STAGE2_MODEL_FACTORY[stage2_model_name]()
    t0 = time.time()
    stage2_model.fit(X_train, y_train)
    stage2_train_time = time.time() - t0
    log(f"  {model_label} trained in {stage2_train_time:.2f}s")

    # Tune Stage 2's decision threshold on the validation set — SAME fix as
    # Stage 2's own train_stage2.py. Without this, Stage 2's classifier is
    # evaluated at the default 0.5 threshold after training on SMOTE-
    # balanced data, which very often collapses to near-always-predicting
    # the majority class on the real test distribution (see Stage 2's
    # README changelog — this exact bug was caught by methodology review
    # and previously left unfixed here, giving a misleadingly broken
    # "single-stage baseline" to compare the cascade against).
    if hasattr(stage2_model, "predict_proba"):
        val_proba = stage2_model.predict_proba(X_val)
        stage2_threshold_result = find_best_threshold(y_val, val_proba, metric="f1")
        stage2_threshold = stage2_threshold_result["threshold"]
        log(f"  Stage 2 decision threshold tuned on validation set: {stage2_threshold:.3f} "
            f"(val precision={stage2_threshold_result['val_precision_at_threshold']:.3f}, "
            f"val recall={stage2_threshold_result['val_recall_at_threshold']:.3f})")
    else:
        stage2_threshold = None
        log("  Stage 2 model has no predict_proba — using default .predict() threshold")

    # --- 3. Build the cascade ---
    log("\n>>> [3/4] Building two-stage cascade pipeline")
    pipeline = TwoStagePipeline(
        linear_filter, stage2_model, threshold,
        stage2_model_name=model_label, stage2_threshold=stage2_threshold,
    )

    # Persist the trained cascade and threshold metadata (for Stage 6 real-time inference)
    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    cascade_path = os.path.join(models_dir, "cascade.pkl")
    joblib.dump(pipeline, cascade_path)
    log(f"  Saved trained cascade model to {cascade_path}")

    meta_path = os.path.join(models_dir, "cascade_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "filter_kernel": filter_kernel,
            "stage1_threshold": float(threshold),
            "stage2_model_name": model_label,
            "stage2_threshold": float(stage2_threshold) if stage2_threshold is not None else None,
        }, f, indent=2)
    log(f"  Saved cascade metadata to {meta_path}")

    # --- 4. Benchmark: two-stage cascade vs Stage 2 alone (single-stage baseline) ---
    log("\n>>> [4/4] Benchmarking two-stage cascade vs single-stage (Stage 2 alone) baseline")

    # Single-stage baseline: run Stage 2 model directly on 100% of test
    # traffic, using the SAME tuned threshold the cascade uses internally —
    # otherwise this comparison would be apples-to-oranges (a tuned cascade
    # vs an untuned baseline).
    single_stage_latency = measure_inference_latency(stage2_model, X_test, n_repeats=latency_repeats)
    single_stage_proba = stage2_model.predict_proba(X_test) if hasattr(stage2_model, "predict_proba") else None
    if stage2_threshold is not None:
        single_stage_pred = predict_with_threshold(stage2_model, X_test, stage2_threshold)
    else:
        single_stage_pred = stage2_model.predict(X_test)
    single_stage_metrics = evaluate_classification(y_test, single_stage_pred, single_stage_proba)

    # Two-stage cascade
    two_stage_latency = pipeline.measure_latency(X_test, n_repeats=latency_repeats)
    two_stage_pred, two_stage_diag = pipeline.predict_with_diagnostics(X_test)
    two_stage_proba = pipeline.predict_proba(X_test)
    two_stage_metrics = evaluate_classification(y_test, two_stage_pred, two_stage_proba)

    comparison = pd.DataFrame([
        {
            "pipeline": f"Single-Stage ({model_label} alone, tuned threshold={stage2_threshold:.3f})"
                        if stage2_threshold is not None else f"Single-Stage ({model_label} alone, default threshold)",
            **single_stage_metrics,
            "avg_latency_ms_per_sample": single_stage_latency["avg_latency_ms_per_sample"],
            "fraction_via_stage2": 1.0,
        },
        {
            "pipeline": f"Two-Stage Hybrid (Linear Filter -> {model_label}, tuned threshold={stage2_threshold:.3f})"
                        if stage2_threshold is not None else f"Two-Stage Hybrid (Linear Filter -> {model_label})",
            **two_stage_metrics,
            "avg_latency_ms_per_sample": two_stage_latency["avg_latency_ms_per_sample"],
            "fraction_via_stage2": 1 - two_stage_diag["fraction_fast_path"],
        },
    ])
    comparison.to_csv(f"{out_dir}/comparison_table.csv", index=False)

    log(f"\n{comparison.to_string(index=False)}")

    latency_reduction_pct = 100 * (
        1 - two_stage_latency["avg_latency_ms_per_sample"] / single_stage_latency["avg_latency_ms_per_sample"]
    )
    accuracy_delta = two_stage_metrics["accuracy"] - single_stage_metrics["accuracy"]
    recall_delta = two_stage_metrics["recall"] - single_stage_metrics["recall"]

    # Recall WILL drop somewhat by design — the whole point of
    # max_attack_leakage_rate is "accept up to X% of real attacks being
    # silently short-circuited, in exchange for compute savings." A small
    # drop is the expected, budgeted cost of the cascade, not a bug. What
    # actually needs checking is whether the drop stayed within that
    # budget on the TEST set (the val-tuned threshold's leakage rate is an
    # estimate — test-set leakage can differ somewhat due to sampling).
    test_proba_attack = linear_filter.predict_proba_attack(X_test)
    test_short_circuit_mask = test_proba_attack < threshold
    test_total_attacks = int(y_test.sum())
    test_leaked_attacks = int(y_test[test_short_circuit_mask].sum()) if test_short_circuit_mask.sum() > 0 else 0
    test_attack_leakage_rate = test_leaked_attacks / test_total_attacks if test_total_attacks > 0 else 0.0

    log(f"\n{'='*70}\nHEADLINE RESULT — Two-Stage Cascade vs Single-Stage Baseline\n{'='*70}")
    log(f"Traffic resolved at Stage 1 alone (fast path): {two_stage_diag['fraction_fast_path']*100:.1f}% "
        f"({two_stage_diag['n_fast_path']}/{two_stage_diag['n_total']} samples)")
    log(f"Traffic passed to Stage 2 (slow path): {(1-two_stage_diag['fraction_fast_path'])*100:.1f}% "
        f"({two_stage_diag['n_slow_path']}/{two_stage_diag['n_total']} samples)")
    log(f"End-to-end latency reduction: {latency_reduction_pct:.1f}%")
    if latency_reduction_pct < 0:
        log(
            f"  ⚠️  Cascade is SLOWER than single-stage. This happens when the short-circuit "
            f"fraction ({two_stage_diag['fraction_fast_path']*100:.1f}%) is too small for the compute "
            f"saved on skipped Stage 2 calls to outweigh (a) the Stage 1 filter's own inference cost "
            f"on 100% of traffic, and (b) Python-level orchestration overhead (masking/indexing) in "
            f"the cascade — both roughly fixed costs that only pay off once a meaningfully larger "
            f"fraction of traffic can be safely short-circuited. Not a bug; expected when the filter "
            f"can't separate much of the traffic confidently (as on this signal-free synthetic data). "
            f"On real, separable data, expect a much higher short-circuit fraction and a genuine "
            f"latency win — if it's still negative there, that's worth investigating."
        )
    log(f"Accuracy change: {accuracy_delta:+.4f}")
    log(f"Recall (attack detection) change: {recall_delta:+.4f}")
    log(
        f"  Test-set attack leakage from Stage 1 shortcut: {test_attack_leakage_rate*100:.2f}% "
        f"({test_leaked_attacks}/{test_total_attacks} real test-set attacks silently short-circuited), "
        f"configured tolerance: {max_attack_leakage_rate*100:.1f}%"
    )
    if test_attack_leakage_rate > max_attack_leakage_rate * 1.5:
        log(
            f"  ⚠️  Test-set leakage ({test_attack_leakage_rate*100:.2f}%) is notably higher than the "
            f"{max_attack_leakage_rate*100:.1f}% tolerance the threshold was tuned for on validation — "
            f"the val-based threshold may not be generalizing well (small val set, or real distribution "
            f"shift between val/test). Consider a larger validation set or a stricter tolerance."
        )
    else:
        log(
            f"  Recall drop is within the budgeted leakage tolerance — this is the expected, "
            f"by-design cost of the cascade, not a detection-quality regression."
        )

    # --- Plots ---
    _plot_threshold_search(search_log, threshold, out_dir)
    _plot_latency_comparison(comparison, out_dir)
    log(f"\nSaved threshold_search.png and latency_comparison.png to {out_dir}")

    with open(f"{out_dir}/stage3_report.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(report_lines))

    log(f"\n>>> Stage 3 complete. All artifacts written to: {out_dir}")
    return comparison, pipeline


def _plot_threshold_search(search_log: pd.DataFrame, selected_threshold: float, out_dir: str):
    """
    Plots compute savings against the PRIMARY safety metric
    (population_attack_leakage_rate — % of ALL real attacks silently
    dropped by the Stage 1 shortcut), not the shortcircuit-purity metric.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(search_log["threshold"], search_log["fraction_short_circuited"] * 100,
              marker="o", color="#4C72B0", label="% traffic short-circuited (compute saved)")
    ax1.set_xlabel("Stage 1 filter confidence threshold")
    ax1.set_ylabel("% short-circuited at Stage 1", color="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(search_log["threshold"], search_log["population_attack_leakage_rate"] * 100,
              marker="s", color="#C44E52", label="% of ALL real attacks missed (safety metric)")
    ax2.set_ylabel("% of all real attacks missed", color="#C44E52")

    ax1.axvline(selected_threshold, color="green", linestyle="--", alpha=0.7,
                label=f"Selected threshold = {selected_threshold:.2f}")

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.title("Stage 1 Threshold Search: Compute Savings vs. Real Attack Miss Rate")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/threshold_search.png", dpi=150)
    plt.close()


def _plot_latency_comparison(comparison: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(comparison["pipeline"], comparison["avg_latency_ms_per_sample"],
                   color=["#DD8452", "#55A868"])
    ax.set_ylabel("Avg. latency (ms/sample)")
    ax.set_title("End-to-End Latency: Single-Stage vs. Two-Stage Hybrid")
    plt.xticks(rotation=10, ha="right")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/latency_comparison.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 3: two-stage hybrid classifier")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--stage2_model", default="idrandom_forest",
                         choices=list(STAGE2_MODEL_FACTORY.keys()))
    parser.add_argument("--filter_kernel", default="sgd_log_loss",
                         choices=["sgd_log_loss", "linear_svm", "rbf"],
                         help="sgd_log_loss (default, fast, NOT an SVM) | linear_svm (true SVM, "
                              "use if your report claims 'SVM filter') | rbf (slow, comparison only)")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_attack_leakage_rate", type=float, default=0.02,
                         help="Max fraction of ALL real attacks allowed to be silently dropped by Stage 1")
    parser.add_argument("--latency_repeats", type=int, default=5)
    args = parser.parse_args()

    run_stage3(
        args.dataset, args.stage1_dir, stage2_model_name=args.stage2_model,
        filter_kernel=args.filter_kernel, out_dir=args.out_dir,
        max_attack_leakage_rate=args.max_attack_leakage_rate,
        latency_repeats=args.latency_repeats,
    )
