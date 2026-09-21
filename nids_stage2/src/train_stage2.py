"""
train_stage2.py
================
Stage 2 - Full Orchestration: Core Detection Model Training & Benchmarking

Runs, for a given dataset processed by Stage 1:
    1. Baseline Random Forest (vanilla, unpruned)      <- "before" comparison
    2. IDRandom-Forest (our base-paper replication)     <- feature-importance
                                                             + ccp_alpha pruning
    3. XGBoost (or HistGradientBoosting fallback)
    4. LightGBM (or HistGradientBoosting fallback)

For each model: trains on the Stage 1 SMOTE-resampled training set, TUNES
its decision threshold on the untouched, real-distribution VALIDATION set
(maximizing F1 — not accuracy), then evaluates ONCE on the untouched test
set using that tuned threshold. Reports precision/recall/F1/ROC-AUC/PR-AUC
as the primary metrics (accuracy is secondary — see metrics.py's docstring
for why accuracy is misleading on ~80/20 imbalanced IDS data). Also
measures inference latency (ms/sample) and serialized model size, then
logs everything via ExperimentTracker (MLflow if available, else local JSON).

Why tune the threshold at all?
--------------------------------
A model trained on SMOTE-balanced (50/50) data, evaluated with the default
0.5 decision threshold on a real ~80/20 test distribution, is very often
mis-calibrated — it can end up almost always predicting the majority
class, producing a deceptively high accuracy with near-zero recall (few
or no attacks actually caught). Tuning the threshold on a held-out
validation set (kept at the REAL distribution, never resampled) corrects
for this and is standard practice for imbalanced classification.

Produces:
    results/<dataset>/comparison_table.csv     <- the headline results table
    results/<dataset>/pruning_log.txt          <- IDRandom-Forest's pruning search log
    results/<dataset>/confusion_matrices.png   <- 4-panel confusion matrix grid
    results/<dataset>/comparison_chart.png     <- accuracy vs latency chart
    results/<dataset>/stage2_report.txt        <- full run log

Usage:
    python train_stage2.py --dataset cic_ids2017 \
        --stage1_dir ../../nids_stage1/data/processed/cic_ids2017
"""

import argparse
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from model_zoo import IDRandomForest, get_xgboost_model, get_lightgbm_model, get_baseline_random_forest
from metrics import full_model_report, confusion_matrix_and_text_report, find_best_threshold
from experiment_tracker import ExperimentTracker


def load_stage1_outputs(stage1_dir: str):
    """
    Loads the artifacts produced by Stage 1's pipeline.py — now a proper
    THREE-WAY split: train (SMOTE-resampled) / val (real distribution,
    for threshold tuning) / test (real distribution, final eval only).
    """
    X_train = pd.read_csv(f"{stage1_dir}/X_train_resampled.csv")
    y_train = pd.read_csv(f"{stage1_dir}/y_train_resampled.csv").iloc[:, 0]

    # Validation set: real distribution, NEVER resampled — used only to
    # tune each model's decision threshold, never to train.
    X_val = pd.read_csv(f"{stage1_dir}/X_val_selected.csv")
    y_val = pd.read_csv(f"{stage1_dir}/y_val_bin.csv").iloc[:, 0]

    # Test set: real distribution, untouched, used exactly once for final eval.
    X_test = pd.read_csv(f"{stage1_dir}/X_test_selected.csv")
    y_test = pd.read_csv(f"{stage1_dir}/y_test_bin.csv").iloc[:, 0]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_tune_evaluate(
    model, model_key: str, model_label: str, params: dict,
    X_train, y_train, X_val, y_val, X_test, y_test,
    tracker, out_dir: str, dataset_name: str, latency_repeats: int,
    log,
):
    """
    Shared training flow for every model in this stage:
        1. fit on (resampled) train
        2. tune decision threshold on validation (real distribution, maximize F1)
        3. evaluate ONCE on test using the tuned threshold
        4. log everything to the experiment tracker
    """
    with tracker.start_run(run_name=f"{model_key}_{dataset_name}"):
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        if hasattr(model, "predict_proba"):
            val_proba = model.predict_proba(X_val)
            threshold_result = find_best_threshold(y_val, val_proba, metric="f1")
            tuned_threshold = threshold_result["threshold"]
            log(f"  Threshold tuned on validation set: {tuned_threshold:.3f} "
                f"(val precision={threshold_result['val_precision_at_threshold']:.3f}, "
                f"val recall={threshold_result['val_recall_at_threshold']:.3f}, "
                f"val F1={threshold_result['val_f1_at_threshold']:.3f})")
        else:
            tuned_threshold = None
            log("  Model has no predict_proba — using default .predict() threshold")

        report = full_model_report(
            model, X_test, y_test, model_label, latency_repeats,
            decision_threshold=tuned_threshold,
        )
        report["train_time_sec"] = round(train_time, 3)
        report.update(params)

        tracker.log_params({**params, "dataset": dataset_name, "tuned_threshold": tuned_threshold})
        tracker.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})
        tracker.log_model(model, model_key)

        warning = report.get("_warning")
        log(f"  TEST SET (threshold={tuned_threshold if tuned_threshold else 0.5:.3f}): "
            f"precision={report['precision']:.4f}, recall={report['recall']:.4f}, "
            f"f1={report['f1_score']:.4f}, roc_auc={report.get('roc_auc', float('nan')):.4f}, "
            f"pr_auc={report.get('pr_auc', float('nan')):.4f}, accuracy={report['accuracy']:.4f} (secondary), "
            f"latency={report['avg_latency_ms_per_sample']:.4f}ms/sample, size={report['model_size_kb']:.1f}KB")
        if warning:
            log(f"  ⚠️  {warning}")

    return report, model


def run_stage2(dataset_name: str, stage1_dir: str, out_dir: str = None, latency_repeats: int = 5):
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(str(msg))

    log(f"{'='*70}\nSTAGE 2 PIPELINE — dataset: {dataset_name}\n{'='*70}")

    out_dir = out_dir or f"../results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Load Stage 1 outputs (now a proper 3-way split) ---
    log("\n>>> Loading Stage 1 outputs")
    X_train, y_train, X_val, y_val, X_test, y_test = load_stage1_outputs(stage1_dir)
    log(f"Train: {X_train.shape}, SMOTE-balanced classes: {y_train.value_counts().to_dict()}")
    log(f"Val:   {X_val.shape}, REAL class distribution: {y_val.value_counts().to_dict()} "
        f"(used ONLY for threshold tuning)")
    log(f"Test:  {X_test.shape}, REAL class distribution: {y_test.value_counts().to_dict()} "
        f"(used ONLY for final evaluation)")

    tracker = ExperimentTracker(experiment_name=f"nids_{dataset_name}", out_dir=out_dir)

    results = []
    trained_models = {}

    # --- 1. Baseline Random Forest ---
    log("\n>>> Training [1/4] Baseline Random Forest (unpruned)")
    baseline_rf = get_baseline_random_forest()
    report, baseline_rf = train_tune_evaluate(
        baseline_rf, "baseline_rf", "Baseline Random Forest",
        {"model": "RandomForest", "n_estimators": 100},
        X_train, y_train, X_val, y_val, X_test, y_test,
        tracker, out_dir, dataset_name, latency_repeats, log,
    )
    results.append(report)
    trained_models["baseline_rf"] = baseline_rf

    # --- 2. IDRandom-Forest (base paper replication) ---
    log("\n>>> Training [2/4] IDRandom-Forest (feature-importance + ccp_alpha pruning)")
    idrf = IDRandomForest()
    report, idrf = train_tune_evaluate(
        idrf, "idrandom_forest", "IDRandom-Forest", {"model": "IDRandomForest"},
        X_train, y_train, X_val, y_val, X_test, y_test,
        tracker, out_dir, dataset_name, latency_repeats, log,
    )
    report["selected_ccp_alpha"] = idrf.best_ccp_alpha_
    results.append(report)
    trained_models["idrandom_forest"] = idrf
    with open(f"{out_dir}/pruning_log.txt", "w") as f:
        f.write("\n".join(idrf.pruning_log_))
    for line in idrf.pruning_log_:
        log("    " + line)

    # --- 3. XGBoost ---
    log("\n>>> Training [3/4] XGBoost")
    xgb_model, is_real_xgb, note = get_xgboost_model()
    log(f"  NOTE: {note}")
    model_label = "XGBoost" if is_real_xgb else "XGBoost [FALLBACK: HistGradientBoosting]"
    report, xgb_model = train_tune_evaluate(
        xgb_model, "xgboost", model_label, {"model": model_label, "is_real_library": is_real_xgb},
        X_train, y_train, X_val, y_val, X_test, y_test,
        tracker, out_dir, dataset_name, latency_repeats, log,
    )
    results.append(report)
    trained_models["xgboost"] = xgb_model

    # --- 4. LightGBM ---
    log("\n>>> Training [4/4] LightGBM")
    lgb_model, is_real_lgb, note = get_lightgbm_model()
    log(f"  NOTE: {note}")
    model_label = "LightGBM" if is_real_lgb else "LightGBM [FALLBACK: HistGradientBoosting]"
    report, lgb_model = train_tune_evaluate(
        lgb_model, "lightgbm", model_label, {"model": model_label, "is_real_library": is_real_lgb},
        X_train, y_train, X_val, y_val, X_test, y_test,
        tracker, out_dir, dataset_name, latency_repeats, log,
    )
    results.append(report)
    trained_models["lightgbm"] = lgb_model

    # --- Comparison table --- precision/recall/F1/ROC-AUC/PR-AUC lead; accuracy is secondary
    log("\n>>> Building comparison table (recall/precision/F1/AUC first — accuracy is secondary)")
    comparison_df = pd.DataFrame(results)
    col_order = [
        "model_name", "decision_threshold", "recall", "precision", "f1_score",
        "roc_auc", "pr_auc", "accuracy", "majority_class_baseline_accuracy",
        "avg_latency_ms_per_sample", "model_size_kb", "train_time_sec",
    ]
    col_order = [c for c in col_order if c in comparison_df.columns]
    comparison_df = comparison_df[col_order + [c for c in comparison_df.columns if c not in col_order]]
    comparison_df.to_csv(f"{out_dir}/comparison_table.csv", index=False)
    log(comparison_df[col_order].to_string(index=False))

    for _, row in comparison_df.iterrows():
        if isinstance(row.get("_warning"), str):
            log(f"  ⚠️  {row['model_name']}: {row['_warning']}")

    # --- Visualizations ---
    log("\n>>> Generating comparison visualizations")
    # results and trained_models were appended in the same order (baseline_rf,
    # idrandom_forest, xgboost, lightgbm), so zip them directly rather than
    # matching on display-name strings.
    thresholds_by_key = {key: r["decision_threshold"] for key, r in zip(trained_models.keys(), results)}
    _plot_confusion_matrices(trained_models, X_test, y_test, thresholds_by_key, out_dir)
    _plot_recall_vs_latency(comparison_df, out_dir)
    log(f"  Saved confusion_matrices.png and comparison_chart.png to {out_dir}")

    # --- IDRandom-Forest vs Baseline RF headline comparison ---
    baseline_row = comparison_df[comparison_df["model_name"] == "Baseline Random Forest"].iloc[0]
    idrf_row = comparison_df[comparison_df["model_name"] == "IDRandom-Forest"].iloc[0]
    latency_reduction = 100 * (1 - idrf_row["avg_latency_ms_per_sample"] / baseline_row["avg_latency_ms_per_sample"])
    size_reduction = 100 * (1 - idrf_row["model_size_kb"] / baseline_row["model_size_kb"])
    recall_delta = idrf_row["recall"] - baseline_row["recall"]
    accuracy_delta = idrf_row["accuracy"] - baseline_row["accuracy"]

    log(f"\n{'='*70}\nHEADLINE RESULT — IDRandom-Forest vs Baseline Random Forest\n{'='*70}")
    log(f"Latency reduction: {latency_reduction:.1f}%")
    log(f"Model size reduction: {size_reduction:.1f}%")
    log(f"Recall (attack detection) change: {recall_delta:+.4f}  <- the metric that actually matters here")
    log(f"Accuracy change: {accuracy_delta:+.4f} (secondary metric, see metrics.py docstring)")
    log(
        "\nCORRECT CLAIM FOR YOUR REPORT: \"IDRandom-Forest reduces model complexity and "
        "inference latency while maintaining comparable detection performance (recall/F1) to "
        "the baseline Random Forest.\" Do NOT claim IDRandom-Forest improves detection quality "
        "unless recall_delta above is clearly positive on REAL data — pruning is a "
        "compression/latency technique, not a detection-quality technique."
    )
    if idrf_row.get("recall", 0) < 0.3 or baseline_row.get("recall", 0) < 0.3:
        log(
            "\n⚠️  Both models show low recall — before reporting ANY latency/pruning results as "
            "the headline, first confirm the classifier is actually separating attacks from "
            "normal traffic (check recall, PR-AUC, and the majority-class-collapse warning above). "
            "A fast, small, low-recall model is not a good IDS."
        )

    with open(f"{out_dir}/stage2_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    log(f"\n>>> Stage 2 complete. All artifacts written to: {out_dir}")
    return comparison_df, trained_models


def _plot_confusion_matrices(trained_models, X_test, y_test, thresholds_by_key, out_dir):
    """Confusion matrices computed using each model's TUNED decision threshold, not the default 0.5."""
    from metrics import predict_with_threshold
    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, len(trained_models), figsize=(5 * len(trained_models), 4))
    if len(trained_models) == 1:
        axes = [axes]
    for ax, (key, model) in zip(axes, trained_models.items()):
        threshold = thresholds_by_key.get(key)
        if hasattr(model, "predict_proba") and threshold is not None:
            y_pred = predict_with_threshold(model, X_test, threshold)
        else:
            y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{key}\n(threshold={threshold:.2f})" if threshold is not None else key)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrices.png", dpi=150)
    plt.close()


def _plot_recall_vs_latency(comparison_df, out_dir):
    """Recall (the metric that matters for IDS) vs. latency — accuracy is deliberately NOT the headline chart."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comparison_df))
    width = 0.35

    ax1.bar(x - width / 2, comparison_df["recall"], width, label="Recall (attack detection)", color="#55A868")
    ax1.set_ylabel("Recall")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df["model_name"], rotation=20, ha="right")

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, comparison_df["avg_latency_ms_per_sample"], width,
            label="Latency (ms/sample)", color="#DD8452")
    ax2.set_ylabel("Latency (ms/sample)")

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Model Comparison: Recall (Attack Detection) vs. Inference Latency")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_chart.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2: core model training & benchmarking")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. cic_ids2017, unsw_nb15)")
    parser.add_argument("--stage1_dir", required=True, help="Path to Stage 1 processed output directory")
    parser.add_argument("--out_dir", default=None, help="Output directory for Stage 2 results")
    parser.add_argument("--latency_repeats", type=int, default=5)
    args = parser.parse_args()

    run_stage2(args.dataset, args.stage1_dir, out_dir=args.out_dir, latency_repeats=args.latency_repeats)
