"""
xai_pipeline.py
================
Stage 5 - Component 5: XAI Orchestration

Wires together everything above into a single interface:

    explainer = XAIExplainer(model, feature_names, X_train, y_train)
    result = explainer.explain(sample_row, prediction, confidence)
    print(result["explanation_text"])

Design note — decoupling from the detection pipeline
--------------------------------------------------------
This module NEVER blocks or slows down detection itself. It's meant to be
called AFTER a prediction has already been made (by Stage 2's classifier
or Stage 3's cascade), as a separate enrichment step — e.g. only for
flagged/attack traffic, or on-demand when an admin clicks into an alert.
A slow or temporarily-unavailable LLM API should never be able to hold up
real-time packet classification.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from explanation_context import ExplanationContextBuilder
from security_knowledge_base import SecurityKnowledgeBase
from prompt_templates import SYSTEM_PROMPT, HUMAN_PROMPT_TEMPLATE, build_prompt_variables
from llm_provider import get_llm_provider


class XAIExplainer:
    def __init__(
        self,
        model,
        feature_names: list,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        llm_preference: str = "auto",
    ):
        """
        Parameters
        ----------
        model : any fitted classifier with .predict()/.predict_proba() —
                works with Stage 2's IDRandomForest, Stage 3's
                TwoStagePipeline, or any sklearn-style model
        feature_names : feature columns in model input order
        X_train, y_train : training data, used to compute the "what does
                            normal traffic look like" baseline
        llm_preference : "auto", "openai", "gemini", or "local"
        """
        self.model = model
        self.feature_names = feature_names

        X_benign = X_train[y_train == 0]
        feature_importances = getattr(model, "feature_importances_", None)
        if feature_importances is None:
            # Stage 2/3 wrapper classes expose the underlying sklearn model
            # via .model_ — try that before giving up on weighting
            inner = getattr(model, "model_", None)
            feature_importances = getattr(inner, "feature_importances_", None)

        self.context_builder = ExplanationContextBuilder(feature_names, X_benign, feature_importances)
        self.knowledge_base = SecurityKnowledgeBase()
        self.llm_provider = get_llm_provider(preference=llm_preference)

    def explain(self, sample: pd.Series, routing_info: dict = None) -> dict:
        """
        Runs the model on `sample`, then generates a natural-language
        explanation for the verdict.

        Returns
        -------
        dict with: prediction, confidence, contributing_features,
                   retrieved_knowledge, explanation_text, llm_provider_used
        """
        X_row = sample[self.feature_names].to_frame().T
        prediction = int(self.model.predict(X_row)[0])

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_row)[0]
            confidence = proba[prediction] if len(proba) > prediction else max(proba)
        else:
            confidence = 1.0

        context = self.context_builder.build_context(
            sample, prediction, confidence, routing_info=routing_info
        )

        # Retrieve grounding knowledge: build a short query from the top
        # contributing feature names, biased toward an "attack-like" query
        # when the verdict is Attack
        query_terms = " ".join(f["feature"] for f in context["contributing_features"])
        if prediction == 1:
            query = f"attack pattern high rate anomalous {query_terms}"
        else:
            query = f"normal benign typical traffic {query_terms}"
        retrieved = self.knowledge_base.retrieve(query, top_k=2)

        prompt_vars = build_prompt_variables(context, retrieved)
        human_prompt = HUMAN_PROMPT_TEMPLATE.format(**prompt_vars)
        try:
            explanation_text = self.llm_provider.generate(SYSTEM_PROMPT, human_prompt)
        except Exception as llm_err:
            import sys
            from llm_provider import LocalTemplateProvider
            print(
                f"[xai_pipeline] LLM error ({type(llm_err).__name__}: {llm_err}). "
                "Falling back to LocalTemplateProvider.",
                file=sys.stderr
            )
            explanation_text = LocalTemplateProvider().generate(SYSTEM_PROMPT, human_prompt)

        return {
            "prediction": context["prediction"],
            "confidence": context["confidence"],
            "contributing_features": context["contributing_features"],
            "retrieved_knowledge": retrieved,
            "explanation_text": explanation_text.strip(),
            "llm_provider_used": self.llm_provider.name,
        }

    def explain_batch(self, X_samples: pd.DataFrame, routing_infos: list = None) -> list:
        """Convenience wrapper for explaining multiple flagged samples at once."""
        results = []
        for i, (idx, row) in enumerate(X_samples.iterrows()):
            routing = routing_infos[i] if routing_infos else None
            results.append(self.explain(row, routing_info=routing))
        return results
