"""
test_feature_alignment.py
=========================
Phase 1.4 Verification Test: Feature Alignment & Cleaning

Spec requirements:
1. Print FlowAggregator output columns and compare to full_feature_names.json.
2. Missing columns -> fill with 0. Extra columns -> drop.
3. Replace inf/NaN with 0 before scaling.
4. Done when: assert list(flow_df.columns) == feature_names passes.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from live_capture import SimulatedCapture
from flow_feature_extractor import FlowAggregator, align_features
from streaming_inference import StreamingInferenceEngine


def run_feature_alignment_test(stage1_dir: str = None, cascade_path: str = None):
    print("=" * 70)
    print("STAGE 6 — PHASE 1.4: FEATURE ALIGNMENT VERIFICATION")
    print("=" * 70)

    # 1. Locate Stage 1 full_feature_names.json
    if stage1_dir is None:
        stage1_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage1", "data", "processed", "cic_ids2017")
        )

    json_path = os.path.join(stage1_dir, "full_feature_names.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find full_feature_names.json at {json_path}")

    with open(json_path) as f:
        feature_names = json.load(f)

    print(f"\n[Step 1] Loaded Stage 1 training schema ({len(feature_names)} features expected)")

    # 2. Capture simulated packets and aggregate into a flow
    print("\n[Step 2] Capturing simulated packets with FlowAggregator...")
    capture = SimulatedCapture(packets_per_second=100)
    aggregator = FlowAggregator(flow_timeout=0.5)

    packets = []
    for pkt in capture.capture(duration_seconds=1.0):
        aggregator.add_packet(pkt)
        packets.append(pkt)

    flows = aggregator.flush_all()
    assert len(flows) > 0, "FlowAggregator produced no flows!"
    raw_flow = flows[0]
    aggregator_columns = list(raw_flow.keys())

    # 3. Print FlowAggregator output columns and compare to full_feature_names.json
    print(f"\n[Step 3] Comparing FlowAggregator columns vs Stage 1 feature schema:")
    common_cols = [c for c in aggregator_columns if c in feature_names]
    missing_cols = [c for c in feature_names if c not in aggregator_columns]
    extra_cols = [c for c in aggregator_columns if c not in feature_names]

    print(f"  * FlowAggregator output column count: {len(aggregator_columns)}")
    print(f"  * Stage 1 full_feature_names count : {len(feature_names)}")
    print(f"  * Common features present in both   : {len(common_cols)}")
    print(f"  * Missing features (filled with 0)  : {len(missing_cols)}")
    print(f"  * Extra columns (dropped)           : {len(extra_cols)} -> {extra_cols}")

    # 4. Inject edge-case inf and NaN to test dirty input cleaning
    test_flow_dirty = dict(raw_flow)
    test_flow_dirty["flow_bytes_s"] = np.inf
    test_flow_dirty["flow_packets_s"] = -np.inf
    test_flow_dirty["flow_iat_std"] = np.nan

    # 5. Execute align_features
    flow_df = align_features(test_flow_dirty, feature_names)

    # 6. Formal Assertions
    print("\n[Step 4] Running formal Phase 1.4 verification checks:")

    # Check A: Exact column match and ordering
    assert list(flow_df.columns) == feature_names, (
        f"Column mismatch! Expected {len(feature_names)} columns, got {len(flow_df.columns)}"
    )
    print("  [PASS] list(flow_df.columns) == feature_names (exact names and ordering)")

    # Check B: Missing columns filled with 0.0
    for col in missing_cols:
        assert (flow_df[col] == 0.0).all(), f"Missing column {col} was not filled with 0.0"
    print(f"  [PASS] All {len(missing_cols)} missing columns filled with 0.0")

    # Check C: Extra columns dropped
    for col in extra_cols:
        assert col not in flow_df.columns, f"Extra column {col} was not dropped!"
    print(f"  [PASS] All {len(extra_cols)} extra columns successfully dropped")

    # Check D: Inf and NaN replaced with 0.0
    assert not np.isinf(flow_df.values).any(), "Found remaining inf values!"
    assert not flow_df.isna().any().any(), "Found remaining NaN values!"
    assert flow_df["flow_bytes_s"].iloc[0] == 0.0, "Injected +inf was not cleaned to 0.0"
    assert flow_df["flow_packets_s"].iloc[0] == 0.0, "Injected -inf was not cleaned to 0.0"
    assert flow_df["flow_iat_std"].iloc[0] == 0.0, "Injected NaN was not cleaned to 0.0"
    print("  [PASS] Replaced inf/-inf and NaN with 0.0 before scaling")

    # 7. End-to-end integration check with StreamingInferenceEngine
    print("\n[Step 5] Testing end-to-end with StreamingInferenceEngine...")
    if cascade_path is None:
        candidate = os.path.join(os.path.dirname(__file__), "..", "models", "cascade.pkl")
        if os.path.exists(candidate):
            cascade_path = candidate
        else:
            cascade_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "nids_stage3", "results", "cic_ids2017", "models", "cascade.pkl"
            )

    engine = StreamingInferenceEngine(stage1_dir, model_path=cascade_path)
    result = engine.predict_flow(test_flow_dirty)
    print(f"  Inference result: verdict={result['prediction']}, confidence={result['confidence']:.2f}, "
          f"latency={result['inference_latency_ms']:.3f}ms")
    print("  [PASS] StreamingInferenceEngine processed dirty flow without scaling errors")

    print("\n" + "=" * 70)
    print("PHASE 1.4 COMPLETE: All feature alignment criteria verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 1.4 Feature Alignment Test")
    parser.add_argument("--stage1_dir", default=None)
    parser.add_argument("--cascade_path", default=None)
    args = parser.parse_args()

    run_feature_alignment_test(stage1_dir=args.stage1_dir, cascade_path=args.cascade_path)
