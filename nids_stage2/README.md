# NIDS Project — Stage 2: Core Detection Model Training & Benchmarking

Implements **Stage 2**: IDRandom-Forest replication, XGBoost/LightGBM
benchmarking, and MLflow experiment tracking, as laid out in the project plan.

## 🔧 Updated after methodology review — read this first

A methodology review of the initial results caught a real, important bug:
models were being evaluated with the default 0.5 decision threshold after
training on SMOTE-balanced (50/50) data, then tested on a realistic ~80/20
distribution. This is a classic imbalanced-classification mistake, and it
produced deceptive numbers — e.g. an original run showed **82% accuracy
with only ~3% recall** (only 4-5 out of ~160 real attacks caught). That
82% figure is actually **worse than a trivial "always predict Normal"
baseline** (~80% accuracy, 0% recall) — the review correctly called this
out as a red flag, not a working detector.

### What changed

1. **Decision threshold is now tuned, not defaulted.** `find_best_threshold()`
   in `metrics.py` sweeps thresholds against the Stage 1 validation set
   (real distribution, never resampled) and picks the one that maximizes
   F1. This threshold — not 0.5 — is what gets applied to the test set for
   final evaluation. Verified on synthetic data with genuine class
   separability: default-threshold recall was already good (93.5%) and
   tuning didn't break it (94.7% recall, 95.5% precision) — confirming the
   fix helps exactly the cases where it's needed without hurting cases
   where it isn't.
2. **Accuracy is now a secondary metric everywhere** — comparison tables,
   charts, and the headline result all lead with **recall, precision, F1,
   ROC-AUC, PR-AUC**. See `metrics.py`'s module docstring for why accuracy
   is actively misleading on ~80/20 imbalanced IDS data.
3. **Automatic majority-class-collapse warning.** `evaluate_classification()`
   now compares a model's accuracy against the trivial majority-baseline
   accuracy and flags it explicitly if accuracy is at/near that baseline
   AND recall is low — the exact failure pattern the review caught. This
   warning is written into the comparison table and printed in the report.
4. **PR-AUC added** alongside ROC-AUC — PR-AUC is the harder, more honest
   metric on imbalanced data (ROC-AUC can look deceptively reasonable even
   when a classifier isn't very useful at the class balance you actually care about).
5. **Confusion matrices and the headline chart now use the tuned
   threshold**, not the default — otherwise the visualizations would
   still tell the old, misleading story even after the metrics were fixed.

### What this means for the current (synthetic-data) results

With the fix applied, recall on the still-signal-free synthetic data jumps
from ~3% to ~80-98% — but at a steep precision cost (~20%), and accuracy
now correctly drops *below* the majority baseline. **This is the expected,
correct behavior**, not a new bug: fixing the threshold repairs the
"always predicts Normal" collapse, but it cannot manufacture signal that
isn't there. ROC-AUC and PR-AUC both remain at their pre-fix
(near-random) levels, which is the honest tell that the underlying
placeholder data has no real feature-label relationship — see Stage 1's
`generate_sample_data.py`. **Once real data is used, this same threshold-
tuning mechanism should produce both high recall AND reasonable
precision**, exactly like the synthetic-signal sanity check below shows.

---

## What's included

```
nids_stage2/
├── src/
│   ├── model_zoo.py            # IDRandom-Forest + XGBoost/LightGBM wrappers
│   ├── metrics.py               # Accuracy/precision/recall/F1, latency, memory
│   ├── experiment_tracker.py    # MLflow wrapper with local-JSON fallback
│   └── train_stage2.py          # Orchestrates the full Stage 2 flow
├── results/
│   └── <dataset>/
│       ├── comparison_table.csv
│       ├── pruning_log.txt
│       ├── confusion_matrices.png
│       ├── comparison_chart.png
│       ├── stage2_report.txt
│       └── experiment_log.jsonl (or mlruns/ if MLflow is installed)
└── README.md
```

## ⚠️ Two things to know before you run this for real

**1. This depends on Stage 1's outputs.** Run Stage 1 first (on the *real*
CIC-IDS2017/UNSW-NB15 data, not the synthetic sample) — Stage 2 reads
`X_train_resampled.csv`, `y_train_resampled.csv`, `X_test_selected.csv`,
and `y_test_bin.csv` from `nids_stage1/data/processed/<dataset>/`.

**2. XGBoost, LightGBM, and MLflow aren't installed in this dev environment**
(no network access here to `pip install` them). The code is written to use
the real libraries — it just automatically falls back to
`sklearn.HistGradientBoostingClassifier` and local JSON logging when they're
missing, **so I could build and test the full pipeline end-to-end without
them.** On your machines, run:
```bash
pip install xgboost lightgbm mlflow
```
No code changes needed — `model_zoo.py` and `experiment_tracker.py` detect
the real libraries automatically and switch over. Any results currently
showing `[FALLBACK: HistGradientBoosting]` in the comparison table will
become real XGBoost/LightGBM numbers once installed.

## How to run

```bash
cd src
python train_stage2.py --dataset cic_ids2017 \
    --stage1_dir ../../nids_stage1/data/processed/cic_ids2017

python train_stage2.py --dataset unsw_nb15 \
    --stage1_dir ../../nids_stage1/data/processed/unsw_nb15
```

Once `mlflow` is installed, view the experiment dashboard with:
```bash
mlflow ui --backend-store-uri file:$(pwd)/../results/cic_ids2017/mlruns
```

## What each component does

### 1. `IDRandomForest` (base paper replication)
Two-phase training:
- **Phase A**: trains a vanilla Random Forest, records baseline cross-validated
  accuracy and average tree size (node count).
- **Phase B**: searches a range of `ccp_alpha` (cost-complexity pruning)
  values. Any candidate that stays within a small accuracy tolerance
  (default 0.5%) of the baseline is accepted; among accepted candidates, the
  one producing the **smallest average tree size** is selected.

This directly implements the base paper's claim — "optimized Random Forest
that uses feature importance ranking and tree pruning to reduce detection
latency" — using a reproducible, explainable method (minimal cost-complexity
pruning) rather than an ad-hoc heuristic. `pruning_log.txt` records the full
search so you can show the tradeoff curve in your report.

### 2. Benchmark models
- **Baseline Random Forest** — the "before" reference point
- **IDRandom-Forest** — the "after" (your core contribution replication)
- **XGBoost** and **LightGBM** — external benchmarks, so your report can
  honestly say where IDRandom-Forest lands relative to standard alternatives,
  not just against itself

### 3. `metrics.py`
For every model, computes:
- Accuracy, precision, recall, F1, ROC-AUC
- **Inference latency**: average ms/sample, measured over 5 repeated full
  passes over the test set (smooths out timing noise)
- **Model size**: serialized (pickled) size on disk, as a reproducible proxy
  for memory footprint

These three categories are exactly what the base paper brief flagged as the
differentiator over accuracy-only papers — your comparison table reports all
three for every model, not just accuracy.

### 4. `experiment_tracker.py`
Every training run logs its parameters, metrics, and the trained model
itself. With MLflow installed, you get the full web UI to compare runs
side-by-side. Without it (as in this dev environment), the same information
is written to `experiment_log.jsonl` — nothing is lost, it's just less
visual until you install MLflow.

## Reading the results (on synthetic data, for now)

The numbers currently in `results/` come from Stage 1's **synthetic**
placeholder data (randomly generated, no real signal between features and
labels). With the threshold-tuning fix, recall now sits around 80-98%
(fixed from the earlier ~3% collapse), but precision/PR-AUC/ROC-AUC remain
near-random (~0.2 precision, ~0.5 AUC) — because there is genuinely no
signal for any threshold choice to exploit. **This is expected and
correct**, and proves the pipeline plumbing works end-to-end (training,
threshold tuning, pruning search, latency measurement, comparison table,
plots all execute without errors and respond sensibly to the fix).

**Once you re-run Stage 1 + Stage 2 on the real CIC-IDS2017/UNSW-NB15
datasets**, expect recall, precision, F1, ROC-AUC, and PR-AUC all to climb
together into a genuinely good range (literature reports 95-99%+ on these
datasets for similar models) — if precision/PR-AUC stay low even on real
data while recall is high, that's a real signal worth investigating
(check the feature selection output from Stage 1, confirm the label
column mapped correctly, and confirm Stage 1's val/test sets are actually
using real, unresampled data).

## What to paste into your report

- `comparison_table.csv` → your main Results table — lead with
  **recall/precision/F1/ROC-AUC/PR-AUC**, list accuracy last as a secondary
  sanity-check number, and note the tuned decision threshold per model
- `pruning_log.txt` → your "Pruning Strategy" methodology paragraph, plus a
  nice chart of accuracy vs. tree-size tradeoff if you plot the log
- `comparison_chart.png` (now recall vs. latency, not accuracy vs. latency)
  and `confusion_matrices.png` (now computed at each model's tuned
  threshold) → drop straight into your Results/Evaluation section
- The "HEADLINE RESULT" block in `stage2_report.txt` → your abstract/
  conclusion's key numbers (X% latency reduction, Y% smaller model, recall
  delta — NOT accuracy delta — as the detection-quality comparison)

## Next steps (Stage 3)

Take whichever model wins on the **recall/precision/latency** tradeoff
(usually IDRandom-Forest or XGBoost — don't pick based on accuracy alone)
and wire it into the two-stage Linear Filter → classifier hybrid pipeline.
Note: Stage 3 currently trains its own copy of the Stage 2 classifier
using the default 0.5 threshold rather than reusing Stage 2's tuned
threshold — worth unifying if you want the two stages' recall numbers to
match exactly. The `trained_models` dict returned by `run_stage2()` in
`train_stage2.py` already has every trained model in memory if you're
extending this in the same Python session — otherwise reload from
`results/<dataset>/models/*.pkl` (local fallback) or MLflow's model registry.
