"""
threshold_tuning.py
====================
Stage 3 - Component 2: Confidence Threshold Selection

The Stage 1 SVM filter outputs P(attack) for every sample. We need a
threshold `t` such that:
    P(attack) < t  ->  classify as Normal directly, SKIP Stage 2 (fast path)
    P(attack) >= t ->  send to Stage 2 for deep classification (slow path)

This is the single most important design decision in the two-stage
architecture: set `t` too low, and almost nothing gets short-circuited
(no compute savings — defeats the purpose). Set `t` too high, and real
attacks with a merely "moderate" SVM confidence score get incorrectly
finalized as Normal and NEVER reach Stage 2 to be caught — a false
negative the second stage never gets a chance to fix.

We evaluate on Stage 1's dedicated VALIDATION split (kept at the real,
untouched class distribution — never SMOTE-resampled, never used for
training) rather than carving an ad-hoc split out of the resampled
training set. This matters: tuning against a resampled 50/50 split would
calibrate the threshold against an unrealistic traffic mix, giving a false
sense of how much can safely be short-circuited in production.

We search a grid of thresholds and select the one that maximizes compute
savings (fraction of traffic resolved at Stage 1) SUBJECT TO keeping
attack "leakage" below a strict safety tolerance — using the same
tolerance-based philosophy as Stage 2's ccp_alpha pruning search.

Two different leakage metrics — use the POPULATION one for safety decisions
------------------------------------------------------------------------------
There are two different questions you could ask about a short-circuit
threshold, and they give very different numbers:

    1. "Of the traffic we short-circuited, what fraction was secretly an
       attack?" -> shortcircuit_purity_leak_rate = leaked / n_short_circuited
       This tells you how CONTAMINATED the fast-path bucket is — useful as
       an audit/diagnostic number, but NOT the right basis for a safety
       constraint, because it's silent about how many attacks total this
       represents.

    2. "Of ALL the attacks actually present in this traffic, what fraction
       did Stage 1 silently drop without ever reaching Stage 2?" ->
       population_attack_leakage_rate = leaked / total_actual_attacks
       THIS is the safety-relevant number: it directly answers "what
       fraction of real attacks does this system permanently miss because
       of the Stage 1 shortcut?" — which is what an admin actually needs
       to know before accepting a threshold. A threshold could look very
       "pure" by metric #1 while still discarding a dangerous fraction of
       total attacks if the attack base rate is high.

We use population_attack_leakage_rate as the PRIMARY safety constraint,
and report shortcircuit_purity_leak_rate alongside it as a secondary
diagnostic only.
"""

import numpy as np
import pandas as pd


def evaluate_threshold(y_val, proba_attack_val, threshold: float) -> dict:
    """
    For a candidate threshold, compute:
        - fraction_short_circuited: % of validation traffic resolved at
          Stage 1 alone (this is your compute-savings number)
        - population_attack_leakage_rate (PRIMARY SAFETY METRIC): of ALL
          actual attacks in the validation set, what fraction got
          incorrectly short-circuited as "Normal" and will NEVER reach
          Stage 2. This is what a leakage safety threshold should be set
          against.
        - shortcircuit_purity_leak_rate (secondary diagnostic): of the
          traffic short-circuited, what fraction was secretly an attack —
          useful for auditing, not for the safety decision itself.
    """
    y_val = np.asarray(y_val)
    total_actual_attacks = int(y_val.sum())
    short_circuit_mask = proba_attack_val < threshold  # confidently "Normal"

    n_short_circuited = int(short_circuit_mask.sum())
    fraction_short_circuited = n_short_circuited / len(y_val)

    leaked_attacks = int(y_val[short_circuit_mask].sum()) if n_short_circuited > 0 else 0

    population_attack_leakage_rate = (
        leaked_attacks / total_actual_attacks if total_actual_attacks > 0 else 0.0
    )
    shortcircuit_purity_leak_rate = (
        leaked_attacks / n_short_circuited if n_short_circuited > 0 else 0.0
    )

    return {
        "threshold": threshold,
        "fraction_short_circuited": fraction_short_circuited,
        "population_attack_leakage_rate": population_attack_leakage_rate,
        "shortcircuit_purity_leak_rate": shortcircuit_purity_leak_rate,
        "leaked_attack_count": leaked_attacks,
        "total_actual_attacks": total_actual_attacks,
        "n_short_circuited": n_short_circuited,
    }


def tune_threshold(
    svm_filter,
    X_train,
    y_train,
    X_val,
    y_val,
    max_attack_leakage_rate: float = 0.02,
    n_candidates: int = 19,
):
    """
    Fits the filter on the (resampled) training set, evaluates a grid of
    candidate thresholds against the REAL-distribution validation set, and
    selects the one that maximizes compute savings (fraction_short_circuited)
    while keeping population_attack_leakage_rate — the fraction of ALL
    actual attacks that get silently dropped — at or below
    `max_attack_leakage_rate` (default: at most 2% of real attacks may be
    missed by the Stage 1 shortcut).

    Parameters
    ----------
    X_train, y_train : training data (may be SMOTE-resampled — used only
                        to fit the filter itself)
    X_val, y_val : Stage 1's dedicated validation split — REAL class
                   distribution, never resampled. This is what the
                   threshold search is evaluated against.

    Returns
    -------
    best_threshold : float
    search_log : pd.DataFrame — full grid results, for your report's
                 threshold-selection chart / methodology section
    selection_note : str
    """
    svm_filter.fit(X_train, y_train)
    proba_attack_val = svm_filter.predict_proba_attack(X_val)

    candidates = np.linspace(0.05, 0.95, n_candidates)
    rows = [evaluate_threshold(y_val, proba_attack_val, t) for t in candidates]
    search_log = pd.DataFrame(rows)

    safe = search_log[search_log["population_attack_leakage_rate"] <= max_attack_leakage_rate]
    if len(safe) > 0:
        best_row = safe.loc[safe["fraction_short_circuited"].idxmax()]
        selection_note = (
            f"Selected threshold={best_row['threshold']:.3f}: "
            f"{best_row['fraction_short_circuited']*100:.1f}% of traffic short-circuited, "
            f"{best_row['population_attack_leakage_rate']*100:.2f}% of ALL real attacks "
            f"missed ({best_row['leaked_attack_count']}/{best_row['total_actual_attacks']}) "
            f"(within {max_attack_leakage_rate*100:.1f}% safety tolerance; "
            f"short-circuit-bucket purity leak rate: {best_row['shortcircuit_purity_leak_rate']*100:.2f}%, diagnostic only)"
        )
    else:
        # nothing met the safety bar — fall back to the safest (lowest-leakage)
        # candidate rather than silently picking something unsafe
        best_row = search_log.loc[search_log["population_attack_leakage_rate"].idxmin()]
        selection_note = (
            f"WARNING: no threshold kept population attack leakage at or below "
            f"{max_attack_leakage_rate*100:.1f}% of all real attacks. Falling back to the "
            f"lowest-leakage candidate: threshold={best_row['threshold']:.3f}, "
            f"{best_row['population_attack_leakage_rate']*100:.2f}% of real attacks missed "
            f"({best_row['leaked_attack_count']}/{best_row['total_actual_attacks']}), "
            f"only {best_row['fraction_short_circuited']*100:.1f}% short-circuited — "
            f"the filter may need better features, more training data, or this dataset may not "
            f"support a safe two-stage shortcut at all."
        )

    print(selection_note)

    return float(best_row["threshold"]), search_log, selection_note
