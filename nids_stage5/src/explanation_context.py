"""
explanation_context.py
=======================
Stage 5 - Component 1: Structured Explanation Context

Before any LLM can explain "why was this flagged," we need to turn a raw
feature vector + model prediction into structured, human-interpretable
context: which features look abnormal, how abnormal, and in which
direction. This is the actual "explainability" logic — the LLM's job in
Stage 5 is just to turn this structured data into fluent prose, not to
figure out what's suspicious itself.

Why not SHAP?
-------------
SHAP is the standard tool for this in real deployments, but it isn't
installed in this dev environment (no network access to pip install).
Instead we use a lightweight, fully-explainable substitute:

    For each feature, compute how many standard deviations the flagged
    sample's value is from the BENIGN class's mean (a z-score), then
    weight that by the model's global feature_importances_. Features with
    both (a) a large deviation from normal AND (b) high importance to the
    model rank highest as "explanatory" — a defensible, simple heuristic
    that doesn't require any extra dependency.

This is NOT a claim of Shapley-value-level rigor — it's a transparent,
reproducible approximation, and should be described as such in your
report. `pip install shap` and swapping in `TreeExplainer` is a natural
"future work" upgrade once your team has network access.
"""

import numpy as np
import pandas as pd


class ExplanationContextBuilder:
    def __init__(self, feature_names: list, X_train_benign: pd.DataFrame, feature_importances: np.ndarray = None):
        """
        Parameters
        ----------
        feature_names : list of feature column names, in model input order
        X_train_benign : training rows where label == 0 (benign only) —
                          used to compute the "what does normal look like"
                          baseline (mean, std per feature)
        feature_importances : optional array of per-feature importances
                               (e.g. model.feature_importances_) to weight
                               the ranking. If None, all features weighted equally.
        """
        self.feature_names = feature_names
        self.benign_mean = X_train_benign[feature_names].mean()
        self.benign_std = X_train_benign[feature_names].std().replace(0, 1e-6)  # avoid div-by-zero
        self.feature_importances = (
            pd.Series(feature_importances, index=feature_names)
            if feature_importances is not None
            else pd.Series(1.0, index=feature_names)
        )

    def build_context(
        self,
        sample: pd.Series,
        prediction: int,
        confidence: float,
        top_k: int = 5,
        routing_info: dict = None,
    ) -> dict:
        """
        Builds the structured context dict that gets fed into the LLM prompt.

        Parameters
        ----------
        sample : the flagged row's feature values (pd.Series indexed by feature_names)
        prediction : 0 (Normal) or 1 (Attack) — the model's final verdict
        confidence : model's probability estimate for the predicted class
        top_k : how many top contributing features to include
        routing_info : optional dict from Stage 3's TwoStagePipeline
                       diagnostics (e.g. {"path": "fast" or "slow"}) — adds
                       useful context about HOW the verdict was reached
        """
        z_scores = (sample[self.feature_names] - self.benign_mean) / self.benign_std
        # "explanatory power" = |deviation from normal| * global importance
        explanatory_power = z_scores.abs() * self.feature_importances
        top_features = explanatory_power.sort_values(ascending=False).head(top_k)

        contributing_features = []
        for feat_name in top_features.index:
            contributing_features.append({
                "feature": feat_name,
                "value": round(float(sample[feat_name]), 3),
                "typical_benign_value": round(float(self.benign_mean[feat_name]), 3),
                "deviation_std": round(float(z_scores[feat_name]), 2),
                "direction": "above normal" if z_scores[feat_name] > 0 else "below normal",
            })

        context = {
            "prediction": "Attack" if prediction == 1 else "Normal",
            "confidence": round(float(confidence), 4),
            "contributing_features": contributing_features,
        }
        if routing_info:
            context["routing_info"] = routing_info
        return context
