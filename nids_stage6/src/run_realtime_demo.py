"""
run_realtime_demo.py
======================
Stage 6 - Full Orchestration: Real-Time Pipeline Demo

Two modes, each testing a different, honest slice of "real-time":

    --mode simulate   Simulated raw packets -> flow aggregation -> scaling
                       -> inference -> live console alerts. Exercises the
                       FULL raw-capture-to-alert path exactly as it would
                       run in production, but on fabricated (unlabeled)
                       traffic — measures real end-to-end throughput and
                       structural correctness, NOT detection accuracy
                       (there's no ground truth for synthetic traffic).

    --mode replay      Streams Stage 1's REAL labeled test set through a
                       Kafka-style producer/consumer split (in-memory
                       queue backend by default; pass --kafka_backend
                       kafka once you have a broker) at a configurable
                       arrival rate. Exercises the trained model against
                       real ground truth — measures both latency AND
                       actual accuracy/recall, and demonstrates the
                       producer/consumer streaming architecture concretely.

Usage:
    python run_realtime_demo.py --mode simulate --stage1_dir <path> \
        --duration_seconds 10 --packets_per_second 100

    python run_realtime_demo.py --mode replay --stage1_dir <path> \
        --n_samples 50 --arrival_rate_per_sec 20 --kafka_backend in_memory
"""

import argparse
import os
import sys
import threading
import time
import json
import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage2", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage3", "src"))

from live_capture import SimulatedCapture, TsharkCapture
from flow_feature_extractor import FlowAggregator
from streaming_inference import StreamingInferenceEngine
from kafka_pipeline import make_producer_consumer_pair

import joblib
from model_zoo import IDRandomForest


def get_or_build_cascade(stage1_dir: str, cascade_path: str = None):
    """Loads an existing persisted cascade model, or falls back to training on the fly."""
    candidate_paths = []
    if cascade_path:
        candidate_paths.append(cascade_path)
    
    dataset = os.path.basename(os.path.normpath(stage1_dir))
    candidate_paths.extend([
        os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage3", "results", dataset, "models", "cascade.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "cascade.pkl"),
        os.path.join("models", "cascade.pkl"),
    ])

    for path in candidate_paths:
        if os.path.exists(path):
            print(f">>> Loading pre-trained cascade from: {os.path.abspath(path)}")
            return joblib.load(path)

    print("⚠️ Pre-trained cascade not found; falling back to on-the-fly training (Phase 1.3)...")
    X_train, y_train, X_val, y_val = _load_stage1_train(stage1_dir)
    return build_stage3_cascade(stage1_dir, X_train, y_train, X_val, y_val)


def build_stage3_cascade(stage1_dir: str, X_train, y_train, X_val, y_val):
    """Trains a fresh Stage 2 model + Stage 3 cascade for the demo."""
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "nids_stage3", "src"))
    from svm_filter import LinearFilter
    from threshold_tuning import tune_threshold
    from two_stage_pipeline import TwoStagePipeline
    from metrics import find_best_threshold

    linear_filter = LinearFilter(kernel="sgd_log_loss")
    threshold, _, _ = tune_threshold(linear_filter, X_train, y_train, X_val, y_val)

    stage2_model = IDRandomForest()
    stage2_model.fit(X_train, y_train)
    val_proba = stage2_model.predict_proba(X_val)
    stage2_threshold = find_best_threshold(y_val, val_proba, metric="f1")["threshold"]

    return TwoStagePipeline(
        linear_filter, stage2_model, threshold,
        stage2_model_name="IDRandomForest", stage2_threshold=stage2_threshold,
    )


# ---------------------------------------------------------------------------
# Mode 1: simulate — full raw pipeline, no ground truth
# ---------------------------------------------------------------------------
def run_simulate_mode(stage1_dir: str, duration_seconds: float, packets_per_second: float,
                       flow_timeout: float = 2.0, cascade_path: str = None):
    print(f"{'='*70}\nSTAGE 6 — SIMULATE MODE (full raw pipeline, no ground truth)\n{'='*70}")
    print(f"Duration: {duration_seconds}s, target rate: {packets_per_second} packets/sec\n")

    cascade = get_or_build_cascade(stage1_dir, cascade_path)
    engine = StreamingInferenceEngine(stage1_dir, model=cascade)

    capture = SimulatedCapture(packets_per_second=packets_per_second)
    aggregator = FlowAggregator(flow_timeout=flow_timeout)

    print(f"\n>>> Starting simulated capture...\n")
    n_packets, n_flows, n_alerts = 0, 0, 0
    start = time.time()
    last_flush = start

    for pkt in capture.capture(duration_seconds=duration_seconds):
        aggregator.add_packet(pkt)
        n_packets += 1

        now = time.time()
        if now - last_flush > 1.0:  # check for completed flows once per second
            completed = aggregator.flush_expired_flows(now)
            for flow_features in completed:
                n_flows += 1
                result = engine.predict_flow(
                    flow_features,
                    flow_metadata={"dst_port": flow_features["destination_port"]},
                )
                if result["prediction"] == "Attack":
                    n_alerts += 1
                    print(f"  [ALERT #{n_alerts}] flow #{n_flows}: {result['prediction']} "
                          f"(confidence {result['confidence']*100:.1f}%, "
                          f"inference {result['inference_latency_ms']:.3f}ms)")
            last_flush = now

    # flush anything still in-flight at the end
    for flow_features in aggregator.flush_all():
        n_flows += 1
        result = engine.predict_flow(flow_features)
        if result["prediction"] == "Attack":
            n_alerts += 1
            print(f"  [ALERT #{n_alerts}] flow #{n_flows} (final flush): {result['prediction']}")

    elapsed = time.time() - start
    print(f"\n>>> Simulate mode complete: {n_packets} packets -> {n_flows} flows -> {n_alerts} alerts "
          f"in {elapsed:.1f}s ({n_packets/elapsed:.1f} packets/sec actual throughput)")


# ---------------------------------------------------------------------------
# Mode 2: replay — real labeled data through a producer/consumer split
# ---------------------------------------------------------------------------
def run_replay_mode(stage1_dir: str, n_samples: int, arrival_rate_per_sec: float,
                     kafka_backend: str = "in_memory", cascade_path: str = None):
    print(f"{'='*70}\nSTAGE 6 — REPLAY MODE (real labeled data, producer/consumer split)\n{'='*70}")
    print(f"Backend: {kafka_backend}, samples: {n_samples}, arrival rate: {arrival_rate_per_sec}/sec\n")

    cascade = get_or_build_cascade(stage1_dir, cascade_path)
    X_test, y_test = _load_stage1_test(stage1_dir)

    # Replay mode feeds ALREADY-SCALED, ALREADY-SELECTED rows directly to the
    # model (bypassing StreamingInferenceEngine's raw->scaled step, since
    # Stage 1's test set is scaled data, not raw flow dicts) — this
    # specifically measures model+cascade latency and accuracy against real
    # ground truth, complementing simulate mode's raw-pipeline coverage.
    sample_idx = X_test.index[:n_samples]
    producer, consumer = make_producer_consumer_pair(backend=kafka_backend)

    results = {"latencies_ms": [], "predictions": [], "actuals": []}

    def producer_thread_fn():
        for idx in sample_idx:
            record = {"index": int(idx), "features": X_test.loc[idx].to_dict(), "actual_label": int(y_test.loc[idx])}
            producer.send(record)
            time.sleep(1.0 / arrival_rate_per_sec)
        producer.flush()

    def consumer_thread_fn():
        n_consumed = 0
        for record in consumer:
            row = pd.DataFrame([record["features"]])
            t0 = time.perf_counter()
            pred = int(cascade.predict(row)[0])
            latency_ms = (time.perf_counter() - t0) * 1000

            results["latencies_ms"].append(latency_ms)
            results["predictions"].append(pred)
            results["actuals"].append(record["actual_label"])

            verdict = "Attack" if pred == 1 else "Normal"
            actual = "Attack" if record["actual_label"] == 1 else "Normal"
            match = "MATCH" if pred == record["actual_label"] else "MISMATCH"
            print(f"  flow #{record['index']}: predicted={verdict}, actual={actual} [{match}], "
                  f"latency={latency_ms:.3f}ms")

            n_consumed += 1
            if n_consumed >= n_samples:
                break

    print(">>> Starting producer/consumer threads...\n")
    p_thread = threading.Thread(target=producer_thread_fn)
    c_thread = threading.Thread(target=consumer_thread_fn)
    start = time.time()
    p_thread.start()
    c_thread.start()
    p_thread.join()
    c_thread.join(timeout=30)
    elapsed = time.time() - start

    n = len(results["predictions"])
    if n > 0:
        correct = sum(1 for p, a in zip(results["predictions"], results["actuals"]) if p == a)
        avg_latency = sum(results["latencies_ms"]) / n
        print(f"\n>>> Replay mode complete: {n} flows processed in {elapsed:.1f}s "
              f"({n/elapsed:.1f} flows/sec end-to-end throughput)")
        print(f"    Accuracy on this batch: {correct}/{n} ({100*correct/n:.1f}%)")
        print(f"    Avg inference latency: {avg_latency:.3f}ms/flow")
    else:
        print("\n>>> No flows were consumed — check arrival_rate_per_sec / consumer timeout settings")


def _load_stage1_test(stage1_dir: str):
    X_test = pd.read_csv(f"{stage1_dir}/X_test_selected.csv")
    y_test = pd.read_csv(f"{stage1_dir}/y_test_bin.csv").iloc[:, 0]
    return X_test, y_test


def _load_stage1_train(stage1_dir: str):
    X_train = pd.read_csv(f"{stage1_dir}/X_train_resampled.csv")
    y_train = pd.read_csv(f"{stage1_dir}/y_train_resampled.csv").iloc[:, 0]
    X_val = pd.read_csv(f"{stage1_dir}/X_val_selected.csv")
    y_val = pd.read_csv(f"{stage1_dir}/y_val_bin.csv").iloc[:, 0]
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: real-time pipeline demo")
    parser.add_argument("--mode", required=True, choices=["simulate", "replay"])
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--cascade_path", default=None, help="Optional path to persisted cascade.pkl")
    # simulate mode args
    parser.add_argument("--duration_seconds", type=float, default=10)
    parser.add_argument("--packets_per_second", type=float, default=80)
    # replay mode args
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--arrival_rate_per_sec", type=float, default=10)
    parser.add_argument("--kafka_backend", default="in_memory", choices=["in_memory", "kafka"])
    args = parser.parse_args()

    if args.mode == "simulate":
        run_simulate_mode(args.stage1_dir, args.duration_seconds, args.packets_per_second, cascade_path=args.cascade_path)
    else:
        run_replay_mode(args.stage1_dir, args.n_samples, args.arrival_rate_per_sec, args.kafka_backend, cascade_path=args.cascade_path)
