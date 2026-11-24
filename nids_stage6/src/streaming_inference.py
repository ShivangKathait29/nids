"""
streaming_inference.py
========================
Stage 6 - Component 3: Real-Time Inference

Takes a flow-feature dict (from flow_feature_extractor.py) and runs it
through the EXACT SAME preprocessing (scaler) and model (Stage 3's
TwoStagePipeline, or a plain Stage 2 model) used during training — this is
what makes "real-time" mean something more than just "fast": the live
inference path must match the trained pipeline exactly, or predictions
are meaningless regardless of latency.

Loads three artifacts produced by Stage 1:
    - scaler.pkl            : fitted StandardScaler (fit on train only)
    - full_feature_names.json : the exact column order the scaler expects
    - selected_features.json  : which of those columns the model actually uses
"""

import json
import os
import pickle
import sys
import time
import joblib
import numpy as np
import pandas as pd

# Ensure Stage 2 and Stage 3 source paths are in sys.path so pickled models can unpickle
_stage3_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage3", "src"))
_stage2_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage2", "src"))
if _stage3_src not in sys.path:
    sys.path.append(_stage3_src)
if _stage2_src not in sys.path:
    sys.path.append(_stage2_src)


class StreamingInferenceEngine:
    def __init__(self, stage1_dir: str, model=None, model_path: str = None, explainer=None):
        """
        Parameters
        ----------
        stage1_dir : path to Stage 1's processed output directory
                     (needs scaler.pkl, full_feature_names.json, selected_features.json)
        model : optional fitted classifier with .predict()/.predict_proba() —
                Stage 3's TwoStagePipeline or a plain Stage 2 model
        model_path : optional path to a persisted model (.pkl) saved via joblib.dump
        explainer : optional Stage 5 XAIExplainer — if provided, flagged
                    (attack) traffic gets an automatic natural-language
                    explanation generated (decoupled — see note below)
        """
        with open(f"{stage1_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{stage1_dir}/full_feature_names.json") as f:
            self.full_feature_names = json.load(f)
        with open(f"{stage1_dir}/selected_features.json") as f:
            self.selected_features = json.load(f)

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = joblib.load(model_path)
        else:
            raise ValueError("Must provide either a 'model' instance or a 'model_path' string.")

        self.explainer = explainer

        missing = set(self.selected_features) - set(self.full_feature_names)
        if missing:
            raise ValueError(
                f"selected_features contains columns not in full_feature_names: {missing}. "
                f"Stage 1 artifacts may be from an inconsistent run."
            )

    def _flow_dict_to_scaled_row(self, flow_features) -> pd.DataFrame:
        """
        Phase 1.4: Builds an aligned DataFrame with exact column order matching
        full_feature_names.json, cleans inf/NaN to 0.0, applies the scaler,
        then subsets to the selected features the model was trained on.
        """
        from flow_feature_extractor import align_features
        raw_row = align_features(flow_features, self.full_feature_names)
        scaled = pd.DataFrame(
            self.scaler.transform(raw_row), columns=self.full_feature_names
        )
        return scaled[self.selected_features]

    def predict_flow(self, flow_features: dict, flow_metadata: dict = None) -> dict:
        """
        Runs one flow through the full inference path.

        Parameters
        ----------
        flow_features : dict from FlowAggregator._compute_features() (or
                         a real CICFlowMeter CSV row parsed into a dict)
        flow_metadata : optional extra context (src/dst IP, port) purely
                         for logging/display — never fed to the model

        Returns
        -------
        dict: prediction, confidence, inference_latency_ms, and (if an
              explainer was provided and the verdict is Attack) an
              explanation.
        """
        t0 = time.perf_counter()
        X_row = self._flow_dict_to_scaled_row(flow_features)

        prediction = int(self.model.predict(X_row)[0])
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_row)[0]
            confidence = float(proba[prediction]) if len(proba) > prediction else float(max(proba))
        else:
            confidence = 1.0
        latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "prediction": "Attack" if prediction == 1 else "Normal",
            "confidence": confidence,
            "inference_latency_ms": latency_ms,
            "flow_metadata": flow_metadata or {},
        }

        # XAI is called ONLY for flagged traffic, and ONLY after the
        # verdict is already final — this keeps a slow/unavailable LLM
        # from ever blocking real-time classification (per Stage 5's
        # decoupling design).
        if self.explainer is not None and prediction == 1:
            sample_row = X_row.iloc[0]
            routing_info = None
            if hasattr(self.model, "predict_with_diagnostics"):
                _, diag = self.model.predict_with_diagnostics(X_row)
                routing_info = {"path": "fast" if diag["fraction_fast_path"] > 0.5 else "slow"}
            explanation = self.explainer.explain(sample_row, routing_info=routing_info)
            result["explanation"] = explanation["explanation_text"]
            result["contributing_features"] = explanation["contributing_features"]

        return result
