"""
run_stage5_demo.py
===================
Stage 5 - Full Orchestration: End-to-End XAI Demo

Trains a Stage 2 classifier on Stage 1's processed data, picks a handful
of flagged (predicted-attack) samples plus a few normal samples from the
test set, and generates a natural-language explanation for each — this is
the concrete "Chat with your Network" building block from your project plan.

Produces:
    results/<dataset>/explained_alerts.json   <- full structured output
    results/<dataset>/sample_report.txt        <- human-readable report

Usage:
    python run_stage5_demo.py --dataset cic_ids2017 \
        --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 \
        --llm_preference auto
"""

import argparse
import json
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage2", "src"))

from model_zoo import IDRandomForest
from xai_pipeline import XAIExplainer


def load_stage1_outputs(stage1_dir: str):
    X_train = pd.read_csv(f"{stage1_dir}/X_train_resampled.csv")
    y_train = pd.read_csv(f"{stage1_dir}/y_train_resampled.csv").iloc[:, 0]
    X_test = pd.read_csv(f"{stage1_dir}/X_test_selected.csv")
    y_test = pd.read_csv(f"{stage1_dir}/y_test_bin.csv").iloc[:, 0]
    return X_train, y_train, X_test, y_test


def run_stage5_demo(
    dataset_name: str, stage1_dir: str, out_dir: str = None,
    n_attack_samples: int = 3, n_normal_samples: int = 2, llm_preference: str = "auto",
):
    out_dir = out_dir or f"../results/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*70}\nSTAGE 5 DEMO — dataset: {dataset_name}\n{'='*70}")

    print("\n>>> Loading Stage 1 outputs and training a detector")
    X_train, y_train, X_test, y_test = load_stage1_outputs(stage1_dir)
    feature_names = list(X_train.columns)

    model = IDRandomForest()
    model.fit(X_train, y_train)
    print(f"  Model trained. Test accuracy: {(model.predict(X_test) == y_test).mean():.4f}")

    print("\n>>> Initializing XAI explainer (context builder + knowledge base + LLM provider)")
    explainer = XAIExplainer(model, feature_names, X_train, y_train, llm_preference=llm_preference)

    print("\n>>> Selecting sample alerts to explain")
    test_predictions = model.predict(X_test)
    attack_idx = X_test.index[test_predictions == 1][:n_attack_samples]
    normal_idx = X_test.index[test_predictions == 0][:n_normal_samples]
    selected_idx = list(attack_idx) + list(normal_idx)
    print(f"  Selected {len(attack_idx)} predicted-attack and {len(normal_idx)} predicted-normal samples")

    print("\n>>> Generating explanations")
    explained_results = []
    report_lines = []
    for i, idx in enumerate(selected_idx):
        sample = X_test.loc[idx]
        result = explainer.explain(sample)
        result["sample_index"] = int(idx)
        explained_results.append(result)

        report_lines.append(f"\n{'-'*70}\nALERT #{i+1} (test sample index {idx})")
        report_lines.append(f"Verdict: {result['prediction']}  |  Confidence: {result['confidence']*100:.1f}%")
        report_lines.append(f"Top contributing features:")
        for f in result["contributing_features"]:
            report_lines.append(
                f"    - {f['feature']}: {f['value']} (typical benign: {f['typical_benign_value']}, "
                f"{f['deviation_std']} std devs {f['direction']})"
            )
        report_lines.append(f"\nExplanation ({result['llm_provider_used']}):")
        report_lines.append(f"  {result['explanation_text']}")
        print(f"  [{i+1}/{len(selected_idx)}] {result['prediction']} (conf={result['confidence']*100:.1f}%) — explained")

    with open(f"{out_dir}/explained_alerts.json", "w") as f:
        json.dump(explained_results, f, indent=2)

    with open(f"{out_dir}/sample_report.txt", "w") as f:
        f.write(f"XAI SAMPLE REPORT — {dataset_name}\n" + "\n".join(report_lines))

    print(f"\n>>> Stage 5 demo complete. Artifacts written to: {out_dir}")
    print(f"    (LLM provider used: {explained_results[0]['llm_provider_used'] if explained_results else 'n/a'})")
    return explained_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 5: XAI explanation demo")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--n_attack_samples", type=int, default=3)
    parser.add_argument("--n_normal_samples", type=int, default=2)
    parser.add_argument("--llm_preference", default="auto", choices=["auto", "openai", "gemini", "local"])
    args = parser.parse_args()

    run_stage5_demo(
        args.dataset, args.stage1_dir, out_dir=args.out_dir,
        n_attack_samples=args.n_attack_samples, n_normal_samples=args.n_normal_samples,
        llm_preference=args.llm_preference,
    )
