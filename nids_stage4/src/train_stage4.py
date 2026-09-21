"""
train_stage4.py
================
Stage 4 - Full Orchestration: Cross-Dataset Transfer Learning

Runs the full transfer learning evaluation:
    1. Load & align both datasets into the common feature space
    2. Split TARGET dataset into a SMALL fine-tuning sample (e.g. 10%) and
       a TEST set (the remaining 90%, untouched by any training)
    3. Train three models and evaluate all three on the SAME target test set:
         a) Source-only (zero-shot)  — trained on source only, applied
            directly to target with NO adaptation. This is the "before"
            baseline that shows the raw domain gap.
         b) Target-only-small        — trained from scratch using ONLY the
            small target fine-tuning sample. Shows what's achievable
            without any transferred knowledge.
         c) Transfer (fine-tuned)    — pretrained on source, THEN
            fine-tuned on the same small target sample used in (b). This
            is the actual innovation: if transfer learning works, (c)
            should beat both (a) and (b).
    4. A data-efficiency sweep: repeat the (b) vs (c) comparison across
       several fine-tuning sample sizes (5%, 10%, 20%, 40%) to show HOW
       MUCH LESS target data transfer learning needs to match a given
       accuracy — the classic, most convincing transfer-learning result.

Usage:
    python train_stage4.py \
        --source_dataset cic_ids2017 --source_path ../../nids_stage1/data/sample_cic_ids2017.csv \
        --target_dataset unsw_nb15   --target_path ../../nids_stage1/data/sample_unsw_nb15.csv \
        --finetune_fraction 0.10
"""

import argparse
import copy
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage1", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage2", "src"))

from data_loader import load_dataset
from common_feature_space import extract_common_features, COMMON_FEATURES
from transfer_model import TransferRandomForest, TransferXGBoost
from metrics import evaluate_classification


def load_aligned(dataset_name: str, raw_path: str):
    """Loads raw data and projects it into the common feature space."""
    df = load_dataset(dataset_name, raw_path)
    common = extract_common_features(df, dataset_name)
    X = common[COMMON_FEATURES]
    y = common["label"].astype(int)
    return X, y


def run_stage4(
    source_dataset: str, source_path: str,
    target_dataset: str, target_path: str,
    finetune_fraction: float = 0.10,
    model_type: str = "random_forest",
    out_dir: str = None,
    random_state: int = 42,
):
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(str(msg))

    log(f"{'='*70}\nSTAGE 4 PIPELINE — source: {source_dataset}, target: {target_dataset}\n{'='*70}")

    out_dir = out_dir or f"../results/{source_dataset}_to_{target_dataset}"
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Load & align to common feature space ---
    log(f"\n>>> [1/5] Loading and aligning both datasets to common feature space: {COMMON_FEATURES}")
    X_source, y_source = load_aligned(source_dataset, source_path)
    X_target, y_target = load_aligned(target_dataset, target_path)
    log(f"Source ({source_dataset}): {X_source.shape}, class balance: {y_source.value_counts().to_dict()}")
    log(f"Target ({target_dataset}): {X_target.shape}, class balance: {y_target.value_counts().to_dict()}")

    # --- 2. Scale each domain independently (reduces raw scale/unit mismatch
    #        between datasets, on top of the semantic alignment already done) ---
    source_scaler = StandardScaler()
    X_source_scaled = pd.DataFrame(
        source_scaler.fit_transform(X_source), columns=COMMON_FEATURES, index=X_source.index
    )
    target_scaler = StandardScaler()
    X_target_scaled = pd.DataFrame(
        target_scaler.fit_transform(X_target), columns=COMMON_FEATURES, index=X_target.index
    )

    # --- 3. Split target into small fine-tune sample + untouched test set ---
    X_target_finetune, X_target_test, y_target_finetune, y_target_test = train_test_split(
        X_target_scaled, y_target, train_size=finetune_fraction,
        random_state=random_state, stratify=y_target,
    )
    log(f"\n>>> [2/5] Target split: fine-tune sample = {len(X_target_finetune)} rows "
        f"({finetune_fraction*100:.0f}%), test set = {len(X_target_test)} rows (untouched)")

    # --- 4. Train the three comparison models ---
    log(f"\n>>> [3/5] Training source model ({model_type}) on {source_dataset}")
    ModelClass = TransferRandomForest if model_type == "random_forest" else TransferXGBoost
    transfer_model = ModelClass(random_state=random_state)
    t0 = time.time()
    transfer_model.fit_source(X_source_scaled, y_source)
    log(f"  Source model trained in {time.time()-t0:.2f}s")

    # (a) Zero-shot: snapshot the source-trained model BEFORE fine-tuning mutates it
    zero_shot_model = copy.deepcopy(transfer_model)

    # (b) Target-only-small: fresh model trained from scratch on the small target sample only
    log(f"\n>>> [4/5] Training target-only-small baseline (no source knowledge, "
        f"{finetune_fraction*100:.0f}% target data only)")
    target_only_model = ModelClass(random_state=random_state)
    target_only_model.fit_source(X_target_finetune, y_target_finetune)  # "source" here = its only data

    # (c) Transfer (fine-tuned): continue training the source model on the small target sample
    log(f"\n>>> [5/5] Fine-tuning source model on the same {finetune_fraction*100:.0f}% target sample")
    transfer_model.fine_tune(X_target_finetune, y_target_finetune)

    # --- 5. Evaluate all three on the SAME untouched target test set ---
    results = []
    for name, model in [
        ("Source-only (zero-shot)", zero_shot_model),
        ("Target-only (small sample, no transfer)", target_only_model),
        ("Transfer Learning (pretrained + fine-tuned)", transfer_model),
    ]:
        y_pred = model.predict(X_target_test)
        y_proba = model.predict_proba(X_target_test)
        m = evaluate_classification(y_target_test, y_pred, y_proba)
        m["approach"] = name
        results.append(m)

    comparison_df = pd.DataFrame(results)[
        ["approach", "accuracy", "precision", "recall", "f1_score"] +
        (["roc_auc"] if "roc_auc" in results[0] else [])
    ]
    comparison_df.to_csv(f"{out_dir}/comparison_table.csv", index=False)
    log(f"\n{'='*70}\nRESULTS on {target_dataset} test set (untouched, {len(X_target_test)} samples)\n{'='*70}")
    log(comparison_df.to_string(index=False))

    zero_shot_acc = comparison_df.iloc[0]["accuracy"]
    target_only_acc = comparison_df.iloc[1]["accuracy"]
    transfer_acc = comparison_df.iloc[2]["accuracy"]

    log(f"\n{'='*70}\nHEADLINE RESULT — Transfer Learning Value\n{'='*70}")
    log(f"Domain gap (source zero-shot on target): {zero_shot_acc:.4f} accuracy")
    log(f"Target-only, small sample only:          {target_only_acc:.4f} accuracy")
    log(f"Transfer (pretrained + fine-tuned):       {transfer_acc:.4f} accuracy")
    log(f"Transfer vs zero-shot improvement: {transfer_acc - zero_shot_acc:+.4f}")
    log(f"Transfer vs target-only improvement: {transfer_acc - target_only_acc:+.4f} "
        f"({'transfer learning helps' if transfer_acc > target_only_acc else 'no clear benefit over target-only at this sample size — see data-efficiency sweep'})")

    # --- 6. Data-efficiency sweep ---
    log(f"\n>>> Running data-efficiency sweep (varying fine-tune sample size)")
    sweep_df = _data_efficiency_sweep(
        X_source_scaled, y_source, X_target_scaled, y_target, ModelClass, random_state, log
    )
    sweep_df.to_csv(f"{out_dir}/data_efficiency_sweep.csv", index=False)

    # --- Plots ---
    _plot_comparison(comparison_df, out_dir)
    _plot_data_efficiency(sweep_df, out_dir)
    log(f"\nSaved comparison_chart.png and data_efficiency_curve.png to {out_dir}")

    with open(f"{out_dir}/stage4_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    log(f"\n>>> Stage 4 complete. All artifacts written to: {out_dir}")
    return comparison_df, sweep_df


def _data_efficiency_sweep(X_source, y_source, X_target, y_target, ModelClass, random_state, log):
    fractions = [0.02, 0.05, 0.10, 0.20, 0.40]
    rows = []

    # Fixed test set across the whole sweep (largest possible, held out once)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X_target, y_target, test_size=0.5, random_state=random_state, stratify=y_target
    )

    source_model = ModelClass(random_state=random_state)
    source_model.fit_source(X_source, y_source)

    for frac in fractions:
        n_samples = max(10, int(frac * len(X_target)))
        n_samples = min(n_samples, len(X_pool))
        X_ft, y_ft = X_pool.iloc[:n_samples], y_pool.iloc[:n_samples]

        # Target-only at this sample size
        t_only = ModelClass(random_state=random_state)
        t_only.fit_source(X_ft, y_ft)
        acc_target_only = evaluate_classification(y_test, t_only.predict(X_test))["accuracy"]

        # Transfer at this sample size
        transfer = copy.deepcopy(source_model)
        transfer.fine_tune(X_ft, y_ft)
        acc_transfer = evaluate_classification(y_test, transfer.predict(X_test))["accuracy"]

        rows.append({
            "fraction": frac, "n_samples": n_samples,
            "target_only_accuracy": acc_target_only, "transfer_accuracy": acc_transfer,
        })
        log(f"  fraction={frac:.2f} (n={n_samples}): target_only_acc={acc_target_only:.4f}, "
            f"transfer_acc={acc_transfer:.4f}")

    return pd.DataFrame(rows)


def _plot_comparison(comparison_df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#C44E52", "#DD8452", "#55A868"]
    bars = ax.bar(comparison_df["approach"], comparison_df["accuracy"], color=colors)
    ax.set_ylabel("Accuracy on target test set")
    ax.set_ylim(0, 1.05)
    ax.set_title("Transfer Learning: Zero-Shot vs Target-Only vs Fine-Tuned")
    plt.xticks(rotation=12, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width()/2, h), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparison_chart.png", dpi=150)
    plt.close()


def _plot_data_efficiency(sweep_df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["fraction"] * 100, sweep_df["target_only_accuracy"] * 100,
            marker="o", label="Target-only (no transfer)", color="#DD8452")
    ax.plot(sweep_df["fraction"] * 100, sweep_df["transfer_accuracy"] * 100,
            marker="s", label="Transfer (pretrained + fine-tuned)", color="#55A868")
    ax.set_xlabel("% of target data used for training")
    ax.set_ylabel("Accuracy on held-out target test set (%)")
    ax.set_title("Data Efficiency: How Much Target Data Does Transfer Learning Save?")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/data_efficiency_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 4: cross-dataset transfer learning")
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--target_dataset", required=True)
    parser.add_argument("--target_path", required=True)
    parser.add_argument("--finetune_fraction", type=float, default=0.10)
    parser.add_argument("--model_type", default="random_forest", choices=["random_forest", "xgboost"])
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    run_stage4(
        args.source_dataset, args.source_path,
        args.target_dataset, args.target_path,
        finetune_fraction=args.finetune_fraction,
        model_type=args.model_type,
        out_dir=args.out_dir,
    )
