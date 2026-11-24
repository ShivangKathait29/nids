"""
api_server.py
=============
Stage 6 Unified Backend: Real-time Live API, WebSockets, XAI & Security Chatbot

Unifies the complete NIDS pipeline into a single API server:
  - GET /: Serves the Unified NIDS Master Dashboard
  - WS /ws/alerts: Real-time streaming WebSocket (Stage 6)
  - POST /start & POST /stop: Streaming engine controls (Stage 6)
  - GET /stats: Real-time throughput & latency telemetry (Stage 6)
  - GET /alerts: Live recent alerts buffer (Stage 6)
  - POST /api/chat: "Chat with Your Network" security assistant (Stage 5)
  - GET /api/xai/alerts: Detailed Explainable AI alerts with feature deviations (Stage 5)
  - GET /api/benchmarks: Model zoo and cascade benchmarks across Stages 1–4
"""

import os
import sys
import time
import json
import asyncio
import threading
import subprocess
import shutil
import traceback
from contextlib import asynccontextmanager
from collections import deque
from typing import Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

# Paths setup
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE6_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(STAGE6_DIR, ".."))

# Auto-load .env for Gemini/OpenAI API keys if present
for env_file in [os.path.join(ROOT_DIR, ".env"), os.path.join(STAGE6_DIR, ".env")]:
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as _ef:
                for _line in _ef:
                    _line = _line.strip()
                    if _line and not _line.startswith("#") and "=" in _line:
                        _k, _v = _line.split("=", 1)
                        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        except Exception:
            pass

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
for stage in ["nids_stage1", "nids_stage2", "nids_stage3", "nids_stage4", "nids_stage5"]:
    stage_path = os.path.join(ROOT_DIR, stage, "src")
    if os.path.exists(stage_path) and stage_path not in sys.path:
        sys.path.append(stage_path)

from streaming_inference import StreamingInferenceEngine
from flow_feature_extractor import FlowAggregator, align_features
from live_capture import SimulatedCapture, TsharkCapture

# Stage 5 Imports for Chat & XAI
try:
    from security_knowledge_base import SecurityKnowledgeBase
    from llm_provider import get_llm_provider
    stage5_available = True
except Exception as e:
    print(f"[API] Warning: Stage 5 components import error: {e}")
    stage5_available = False


def infer_attack_type(flow_features: dict, verdict: str) -> str:
    """
    Identifies specific attack types (PortScan, DoS/DDoS, Brute Force, Web Attack, etc.)
    for live Wi-Fi and simulated flows based on protocol, port, packet rates, and flow metrics.
    """
    if verdict != "Attack":
        return "Normal"

    dst_port = int(flow_features.get("destination_port", 0) or 0)
    pkt_rate = float(flow_features.get("flow_packets_s", 0.0) or 0.0)
    byte_rate = float(flow_features.get("flow_bytes_s", 0.0) or 0.0)
    fwd_pkts = int(flow_features.get("total_fwd_packets", 0) or 0)
    bwd_pkts = int(flow_features.get("total_backward_packets", 0) or 0)
    duration = float(flow_features.get("flow_duration", 0.0) or 0.0)
    protocol = int(flow_features.get("protocol", 6) or 6)

    # 1. DoS / DDoS Flooding (extremely high packet/byte rate or one-sided flood)
    if pkt_rate > 200 or (fwd_pkts > 40 and bwd_pkts == 0) or byte_rate > 350000:
        if dst_port in [80, 443, 8080]:
            return "DoS (HTTP Flood)"
        elif dst_port == 53:
            return "DDoS (DNS Amp)"
        elif protocol == 17:
            return "DDoS (UDP Flood)"
        return "DoS (SYN Flood)"

    # 2. PortScan / Reconnaissance (short probe, 1-3 fwd packets, 0 bwd packets, or rapid connection attempt)
    if (fwd_pkts <= 3 and bwd_pkts == 0) or (duration < 8000 and fwd_pkts <= 5):
        return "PortScan / Recon"

    # 3. Brute Force Attempts on authentication ports
    if dst_port == 22:
        return "Brute Force (SSH)"
    if dst_port == 21:
        return "Brute Force (FTP)"
    if dst_port == 3389:
        return "Brute Force (RDP)"
    if dst_port in [1433, 3306, 5432]:
        return "Database Brute Force"

    # 4. Web Application Exploits
    if dst_port in [80, 443, 8080, 8443]:
        return "Web Attack (SQLi/XSS)"

    # 5. Suspicious high-port command & control / botnet beacon
    if dst_port > 1024 and bwd_pkts > 0:
        return "Botnet / C2"

    return "Anomalous Traffic"


class PipelineManager:
    def __init__(self):
        self.is_running = False
        self.mode = "stopped"
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Statistics
        self.total_flows = 0
        self.attack_flows = 0
        self.normal_flows = 0
        self.latencies = deque(maxlen=200)
        self.recent_alerts = deque(maxlen=200)
        self.start_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.lock = threading.Lock()

        # Engine & Models
        self.stage1_dir = os.path.join(ROOT_DIR, "nids_stage1", "data", "processed", "cic_ids2017")
        self.cascade_path = os.path.join(STAGE6_DIR, "models", "cascade.pkl")
        self.engine: Optional[StreamingInferenceEngine] = None
        self.cascade = None
        self.cached_X_test = None
        self.cached_y_test = None
        self.cached_y_multi = None
        self.attack_pool = None
        self.multi_labels = None

        # Stage 5 Assistant
        self.knowledge_base = None
        self.llm_provider = None

        # WebSockets
        self.active_websockets: Set[WebSocket] = set()
        self.event_queue: Optional[asyncio.Queue] = None
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None

        self._init_models()
        self._init_chat()

    def _init_models(self):
        try:
            if os.path.exists(self.cascade_path):
                print(f"[API] Loading pre-trained cascade from: {self.cascade_path}")
                self.cascade = joblib.load(self.cascade_path)
            elif os.path.exists(os.path.join(ROOT_DIR, "nids_stage3", "results", "cic_ids2017", "models", "cascade.pkl")):
                alt_path = os.path.join(ROOT_DIR, "nids_stage3", "results", "cic_ids2017", "models", "cascade.pkl")
                print(f"[API] Loading cascade from: {alt_path}")
                self.cascade = joblib.load(alt_path)

            if os.path.exists(self.stage1_dir):
                self.engine = StreamingInferenceEngine(self.stage1_dir, model=self.cascade)
                print("[API] StreamingInferenceEngine initialized successfully.")

                # Preload dataset samples for instant, zero-latency streaming startup
                test_path = os.path.join(self.stage1_dir, "X_test_selected.csv")
                label_path = os.path.join(self.stage1_dir, "y_test_bin.csv")
                multi_path = os.path.join(self.stage1_dir, "y_test_multi.csv")
                if os.path.exists(test_path) and os.path.exists(label_path):
                    self.cached_X_test = pd.read_csv(test_path, nrows=25000)
                    self.cached_y_test = pd.read_csv(label_path, nrows=25000).iloc[:, 0]
                    if os.path.exists(multi_path):
                        self.cached_y_multi = pd.read_csv(multi_path, nrows=25000).iloc[:, 0]
                    atk_indices = self.cached_y_test[self.cached_y_test == 1].index
                    self.attack_pool = self.cached_X_test.iloc[atk_indices].reset_index(drop=True)
                    if self.cached_y_multi is not None:
                        self.multi_labels = self.cached_y_multi.iloc[atk_indices].reset_index(drop=True)
                    print(f"[API] Pre-loaded {len(self.cached_X_test)} test flow samples for instant replay/simulation.")
        except Exception as e:
            print(f"[API] Warning during model initialization: {e}")

    def _init_chat(self):
        if stage5_available:
            try:
                self.knowledge_base = SecurityKnowledgeBase()
                pref = os.environ.get("LLM_PREFERENCE", "gemini" if os.environ.get("GOOGLE_API_KEY") else "auto")
                self.llm_provider = get_llm_provider(pref)
                print(f"[API] Stage 5 Chat Assistant ready (Provider: {self.llm_provider.name})")
            except Exception as e:
                print(f"[API] Chat Assistant init warning: {e}")

    def register_loop(self, loop: asyncio.AbstractEventLoop):
        self.main_loop = loop
        self.event_queue = asyncio.Queue()

    def broadcast_event_threadsafe(self, event: dict):
        with self.lock:
            self.total_flows += 1
            if event.get("verdict") == "Attack":
                self.attack_flows += 1
            else:
                self.normal_flows += 1

            latency = event.get("latency_ms", 0.0)
            self.latencies.append(latency)
            self.recent_alerts.append(event)

        if self.main_loop and self.event_queue:
            self.main_loop.call_soon_threadsafe(self.event_queue.put_nowait, event)

    def get_stats(self) -> dict:
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0.0
            flows_sec = round(self.total_flows / elapsed, 1) if elapsed > 0 else 0.0
            avg_lat = round(sum(self.latencies) / len(self.latencies), 2) if self.latencies else 0.0

            return {
                "status": "running" if self.is_running else "stopped",
                "mode": self.mode,
                "total_flows": self.total_flows,
                "attack_flows": self.attack_flows,
                "normal_flows": self.normal_flows,
                "attack_ratio": round(self.attack_flows / self.total_flows, 3) if self.total_flows > 0 else 0.0,
                "flows_per_sec": flows_sec,
                "avg_latency_ms": avg_lat,
                "elapsed_sec": round(elapsed, 1),
                "last_error": self.last_error,
                "connected_clients": len(self.active_websockets),
            }

    def start_pipeline(self, mode: str = "replay", interface: str = "Wi-Fi", rate: float = 20.0):
        if self.is_running:
            return False, "Pipeline is already running."

        # Quick check for live capture preconditions before launching
        if mode == "live":
            if shutil.which("tshark") is None:
                err_msg = "TShark not found on PATH. Install Wireshark (ensure TShark and Npcap are checked) and add Wireshark to PATH."
                self.last_error = err_msg
                print(f"[API Error] {err_msg}")
                return False, err_msg
            try:
                check_out = subprocess.check_output(["tshark", "-D"], text=True, stderr=subprocess.STDOUT)
                if "The NPF driver isn't running" in check_out:
                    err_msg = "Npcap (NPF) driver is not running. Please open PowerShell as Administrator and run: net start npcap"
                    self.last_error = err_msg
                    print(f"[API Error] {err_msg}")
                    return False, err_msg
            except Exception as e:
                pass

        self.is_running = True
        self.mode = mode
        self.stop_event.clear()
        self.last_error = None
        self.start_time = time.time()

        with self.lock:
            self.total_flows = 0
            self.attack_flows = 0
            self.normal_flows = 0
            self.latencies.clear()
            self.recent_alerts.clear()

        if mode == "replay":
            self.worker_thread = threading.Thread(target=self._run_replay_worker, args=(rate,), daemon=True)
        elif mode == "simulate":
            self.worker_thread = threading.Thread(target=self._run_simulate_worker, args=(rate,), daemon=True)
        elif mode == "live":
            self.worker_thread = threading.Thread(target=self._run_live_worker, args=(interface,), daemon=True)
        else:
            self.is_running = False
            return False, f"Unknown mode: {mode}"

        self.worker_thread.start()
        print(f"[API] Pipeline worker started successfully in '{mode}' mode.")
        return True, f"Started pipeline in {mode} mode"

    def stop_pipeline(self):
        if not self.is_running:
            return False, "Pipeline is not running."

        self.is_running = False
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        self.mode = "stopped"
        print(f"[API] Pipeline stopped. Total flows evaluated: {self.total_flows} (Attacks: {self.attack_flows}, Normal: {self.normal_flows})")
        return True, "Pipeline stopped"

    def _run_replay_worker(self, arrival_rate: float):
        try:
            print(f"[API] Replay mode active at target rate of {arrival_rate} flows/sec...")
            if self.cached_X_test is not None and self.cached_y_test is not None:
                X_test = self.cached_X_test
                y_test = self.cached_y_test
                y_multi = self.cached_y_multi
            else:
                test_path = os.path.join(self.stage1_dir, "X_test_selected.csv")
                label_path = os.path.join(self.stage1_dir, "y_test_bin.csv")
                multi_path = os.path.join(self.stage1_dir, "y_test_multi.csv")
                if not os.path.exists(test_path) or not os.path.exists(label_path):
                    err_msg = f"Replay test files missing in {self.stage1_dir}"
                    print(f"[API Error] {err_msg}")
                    self.last_error = err_msg
                    self.is_running = False
                    return
                X_test = pd.read_csv(test_path, nrows=25000)
                y_test = pd.read_csv(label_path, nrows=25000).iloc[:, 0]
                y_multi = pd.read_csv(multi_path, nrows=25000).iloc[:, 0] if os.path.exists(multi_path) else None

            n_samples = len(X_test)
            delay = 1.0 / max(arrival_rate, 1.0)
            idx = 0
            flow_id = 0

            while not self.stop_event.is_set():
                curr_idx = idx % n_samples
                row = X_test.iloc[[curr_idx]]
                actual_label = int(y_test.iloc[curr_idx])

                t0 = time.perf_counter()
                if hasattr(self.cascade, "predict_with_diagnostics"):
                    pred, diag = self.cascade.predict_with_diagnostics(row)
                    prediction = int(pred[0])
                    path = "fast" if diag.get("fraction_fast_path", 0) > 0.5 else "slow"
                else:
                    prediction = int(self.cascade.predict(row)[0])
                    path = "standard"

                if hasattr(self.cascade, "predict_proba"):
                    proba = self.cascade.predict_proba(row)[0]
                    confidence = float(proba[prediction]) if len(proba) > prediction else float(max(proba))
                else:
                    confidence = 0.95
                latency_ms = (time.perf_counter() - t0) * 1000

                verdict = "Attack" if prediction == 1 else "Normal"
                flow_id += 1

                attack_type = "Normal"
                if verdict == "Attack":
                    if y_multi is not None and curr_idx < len(y_multi):
                        raw_type = str(y_multi.iloc[curr_idx])
                        attack_type = raw_type if raw_type.upper() != "BENIGN" else "Malicious Traffic"
                    else:
                        attack_type = infer_attack_type(row.to_dict('records')[0], verdict)

                src_ip = f"192.168.1.{10 + (flow_id % 240)}:{1024 + (flow_id * 17) % 64000}"
                dst_port = 80 if flow_id % 3 == 0 else (443 if flow_id % 3 == 1 else 22)
                dst_ip = f"10.0.0.5:{dst_port}"

                explanation = None
                if verdict == "Attack":
                    explanation = (
                        f"Traffic flagged as {verdict} ({attack_type}) via {path}-path classifier. "
                        f"Confidence: {confidence*100:.1f}%. Pattern exhibits anomalous packet dynamics."
                    )

                event = {
                    "id": flow_id,
                    "ts": round(time.time(), 3),
                    "src": src_ip,
                    "dst": dst_ip,
                    "verdict": verdict,
                    "attack_type": attack_type,
                    "actual": "Attack" if actual_label == 1 else "Normal",
                    "confidence": round(confidence, 3),
                    "latency_ms": round(latency_ms, 2),
                    "path": path,
                    "explanation": explanation,
                }

                self.broadcast_event_threadsafe(event)
                if flow_id % 25 == 1:
                    print(f"[REPLAY] Flow #{flow_id}: {verdict} ({attack_type}) | Latency: {latency_ms:.2f}ms")
                idx += 1
                time.sleep(delay)
        except Exception as e:
            print(f"[API Error] Replay worker failed: {e}\n{traceback.format_exc()}")
            self.last_error = str(e)
            self.is_running = False

    def _run_simulate_worker(self, packets_per_second: float):
        try:
            print(f"[API] Simulation mode active at {packets_per_second} pkts/sec...")
            attack_pool = self.attack_pool
            multi_labels = self.multi_labels
            if attack_pool is None:
                test_path = os.path.join(self.stage1_dir, "X_test_selected.csv")
                label_path = os.path.join(self.stage1_dir, "y_test_bin.csv")
                multi_path = os.path.join(self.stage1_dir, "y_test_multi.csv")
                if os.path.exists(test_path) and os.path.exists(label_path):
                    try:
                        y_bin = pd.read_csv(label_path, nrows=10000).iloc[:, 0]
                        atk_indices = y_bin[y_bin == 1].index
                        X_all = pd.read_csv(test_path, nrows=10000)
                        attack_pool = X_all.iloc[atk_indices].reset_index(drop=True)
                        if os.path.exists(multi_path):
                            multi_labels = pd.read_csv(multi_path, nrows=10000).iloc[:, 0].iloc[atk_indices].reset_index(drop=True)
                    except Exception as e:
                        print(f"[API] Warning loading simulation attack pool: {e}")

            capture = SimulatedCapture(packets_per_second=packets_per_second, attack_burst_probability=0.3)
            aggregator = FlowAggregator(flow_timeout=0.3)
            last_flush = time.time()
            flow_id = 0
            atk_cursor = 0

            for pkt in capture.capture(duration_seconds=86400):
                if self.stop_event.is_set():
                    break

                aggregator.add_packet(pkt)
                now = time.time()
                if now - last_flush > 0.2:
                    completed = []
                    for k, f in list(aggregator._flows.items()):
                        if (now - f.last_time) > 0.3 or (len(f.fwd_packet_lengths) + len(f.bwd_packet_lengths)) >= 12:
                            completed.append(aggregator._compute_features(aggregator._flows.pop(k)))
                    for flow_features in completed:
                        if self.stop_event.is_set():
                            break
                        flow_id += 1

                        # Inject realistic attack flow every 4th flow if attack pool is available
                        is_injected_attack = (attack_pool is not None and len(attack_pool) > 0 and flow_id % 4 == 0)

                        if is_injected_attack:
                            row = attack_pool.iloc[[atk_cursor % len(attack_pool)]]
                            raw_label = str(multi_labels.iloc[atk_cursor % len(multi_labels)]) if multi_labels is not None else "Attack"
                            atk_cursor += 1

                            t0 = time.perf_counter()
                            if hasattr(self.cascade, "predict_with_diagnostics"):
                                pred, diag = self.cascade.predict_with_diagnostics(row)
                                prediction = int(pred[0])
                                path = "fast" if diag.get("fraction_fast_path", 0) > 0.5 else "slow"
                            else:
                                prediction = int(self.cascade.predict(row)[0])
                                path = "standard"
                            latency_ms = (time.perf_counter() - t0) * 1000

                            verdict = "Attack" if prediction == 1 else "Normal"
                            attack_type = raw_label if (verdict == "Attack" and raw_label.upper() != "BENIGN") else ("PortScan / Recon" if verdict == "Attack" else "Normal")
                            conf = 0.96

                            event = {
                                "id": flow_id,
                                "ts": round(now, 3),
                                "src": f"198.51.100.{10 + (flow_id % 200)}:{1024 + (flow_id * 37) % 64000}",
                                "dst": f"10.0.0.9:80",
                                "verdict": verdict,
                                "attack_type": attack_type,
                                "confidence": conf,
                                "latency_ms": round(latency_ms, 2),
                                "path": path,
                                "explanation": f"Simulation intercepted {attack_type} pattern via {path}-path classifier." if verdict == "Attack" else None,
                            }
                        else:
                            meta = {"dst_port": flow_features.get("destination_port", 80)}
                            result = self.engine.predict_flow(flow_features, flow_metadata=meta)
                            verdict = result["prediction"]
                            attack_type = infer_attack_type(flow_features, verdict)

                            event = {
                                "id": flow_id,
                                "ts": round(now, 3),
                                "src": f"192.168.1.{20 + (flow_id % 200)}:{1024 + (flow_id * 31) % 64000}",
                                "dst": f"10.0.0.9:{flow_features.get('destination_port', 80)}",
                                "verdict": verdict,
                                "attack_type": attack_type,
                                "confidence": round(result.get("confidence", 0.95), 3),
                                "latency_ms": round(result.get("inference_latency_ms", 1.0), 2),
                                "path": "cascade",
                                "explanation": result.get("explanation"),
                            }
                        self.broadcast_event_threadsafe(event)
                        if flow_id % 20 == 1:
                            print(f"[SIMULATE] Flow #{flow_id}: {verdict} ({attack_type}) | Latency: {event['latency_ms']:.2f}ms")
                    last_flush = now
        except Exception as e:
            print(f"[API Error] Simulation worker failed: {e}\n{traceback.format_exc()}")
            self.last_error = str(e)
            self.is_running = False

    def _run_live_worker(self, interface: str):
        try:
            print(f"[API] Starting live network capture on '{interface}'...")
            if shutil.which("tshark") is None:
                raise RuntimeError("tshark not found on PATH. Add C:\\Program Files\\Wireshark to PATH and restart the server.")
            # pyshark needs an asyncio event loop in this (non-main) thread
            asyncio.set_event_loop(asyncio.new_event_loop())
            capture = TsharkCapture(interface=interface)
            aggregator = FlowAggregator(flow_timeout=2.0)
            last_flush = time.time()
            flow_id = 0

            for pkt in capture.capture(duration_seconds=86400):
                if self.stop_event.is_set():
                    break
                aggregator.add_packet(pkt)
                now = time.time()
                if now - last_flush > 1.0:
                    completed = aggregator.flush_expired_flows(now)
                    for flow_features in completed:
                        flow_id += 1
                        result = self.engine.predict_flow(flow_features)
                        verdict = result["prediction"]
                        attack_type = infer_attack_type(flow_features, verdict)
                        event = {
                            "id": flow_id,
                            "ts": round(now, 3),
                            "src": f"{flow_features.get('_src_ip', '0.0.0.0')}:{flow_features.get('_src_port', 0)}",
                            "dst": f"{flow_features.get('_dst_ip', '0.0.0.0')}:{flow_features.get('destination_port', 0)}",
                            "verdict": verdict,
                            "attack_type": attack_type,
                            "confidence": round(result.get("confidence", 0.9), 3),
                            "latency_ms": round(result.get("inference_latency_ms", 1.0), 2),
                            "path": "live",
                            "explanation": result.get("explanation"),
                        }
                        self.broadcast_event_threadsafe(event)
                        if flow_id % 10 == 1:
                            print(f"[LIVE] Flow #{flow_id}: {event['src']} -> {event['dst']} | {verdict} ({attack_type}) | Latency: {event['latency_ms']:.2f}ms")
                    last_flush = now

            if not self.stop_event.is_set():
                if flow_id == 0:
                    err_msg = f"Live capture on '{interface}' captured 0 flows. If on Windows, please run PowerShell as Administrator and run 'net start npcap', then verify interface in 'tshark -D'."
                    print(f"[API Error] {err_msg}")
                    self.last_error = err_msg
                self.is_running = False
        except Exception as e:
            print(f"[API Error] Live capture failed: {e}\n{traceback.format_exc()}")
            self.last_error = str(e)
            self.is_running = False


manager = PipelineManager()


async def websocket_broadcaster():
    while True:
        try:
            if manager.event_queue is not None:
                event = await manager.event_queue.get()
                message_text = json.dumps(event)
                dead_sockets = set()
                for ws in list(manager.active_websockets):
                    try:
                        await ws.send_text(message_text)
                    except Exception:
                        dead_sockets.add(ws)
                for dead in dead_sockets:
                    manager.active_websockets.discard(dead)
            else:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    manager.register_loop(loop)
    broadcaster_task = asyncio.create_task(websocket_broadcaster())
    yield
    broadcaster_task.cancel()


app = FastAPI(title="NIDS Master Unified Control Center", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST Endpoints: Stage 6 Streaming ---

@app.post("/start")
def start_pipeline(
    mode: str = Query("replay", pattern="^(replay|simulate|live)$"),
    interface: str = Query("Wi-Fi"),
    rate: float = Query(20.0, ge=1.0, le=500.0),
):
    success, message = manager.start_pipeline(mode=mode, interface=interface, rate=rate)
    status_code = 200 if success else 400
    return JSONResponse(status_code=status_code, content={"success": success, "message": message, "stats": manager.get_stats()})


@app.post("/stop")
def stop_pipeline():
    success, message = manager.stop_pipeline()
    return {"success": success, "message": message, "stats": manager.get_stats()}


@app.get("/stats")
def get_stats():
    return manager.get_stats()


@app.get("/alerts")
def get_recent_alerts(limit: int = 50):
    with manager.lock:
        alerts = list(manager.recent_alerts)[-limit:]
        alerts.reverse()
        return {"alerts": alerts, "count": len(alerts)}


@app.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        manager.active_websockets.discard(websocket)


from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str = ""

class GeminiKeyRequest(BaseModel):
    api_key: str
    model: str = "gemini-2.0-flash"

class ExplainFlowRequest(BaseModel):
    flow_id: int = 1
    src: str = "192.168.1.10"
    dst: str = "10.0.0.5:80"
    verdict: str = "Attack"
    attack_type: Optional[str] = "Attack"
    confidence: float = 0.95
    path: str = "slow"
    latency_ms: float = 1.2


# --- REST Endpoints: Stage 5 Chat & XAI ---

@app.get("/api/llm_status")
def get_llm_status():
    """Returns current active LLM provider details."""
    provider_name = manager.llm_provider.name if manager.llm_provider else "Not Initialized"
    is_gemini = "Gemini" in provider_name and "canned" not in provider_name.lower()
    has_key = bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return {
        "provider": provider_name,
        "is_gemini": is_gemini,
        "has_key": has_key,
        "model": os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    }

@app.post("/api/set_gemini_key")
def set_gemini_key(req: GeminiKeyRequest):
    """Sets the Gemini API key in runtime and saves to .env for persistence."""
    key = req.api_key.strip()
    if not key:
        return {"success": False, "message": "API key cannot be empty"}

    os.environ["GOOGLE_API_KEY"] = key
    os.environ["GEMINI_MODEL"] = req.model.strip() or "gemini-2.0-flash"
    os.environ["LLM_PREFERENCE"] = "gemini"

    try:
        env_path = os.path.join(ROOT_DIR, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"GOOGLE_API_KEY={key}\nGEMINI_MODEL={os.environ['GEMINI_MODEL']}\nLLM_PREFERENCE=gemini\n")
    except Exception:
        pass

    try:
        from llm_provider import get_llm_provider
        manager.llm_provider = get_llm_provider("gemini")
        return {
            "success": True,
            "message": f"Successfully connected to Gemini! ({manager.llm_provider.name})",
            "provider": manager.llm_provider.name
        }
    except Exception as e:
        return {"success": False, "message": f"Failed to initialize Gemini: {str(e)}"}

@app.post("/api/explain_flow")
def explain_flow_incident(req: ExplainFlowRequest):
    """Deep incident analysis of a single flow using Gemini / LLM."""
    system_prompt = (
        "You are an expert SOC Incident Responder and Network Security Specialist. "
        "Analyze this specific network flow flagged by our NIDS real-time engine. "
        "Provide a crisp 3-part triage summary in under 4 sentences:\n"
        "1. Threat Identification (e.g. DoS, PortScan, Brute-Force, or Benign Spike)\n"
        "2. Anomaly Indicators (confidence level, fast vs slow path routing)\n"
        "3. Recommended Containment (exact firewall/iptables/ACL rule to mitigate)"
    )
    human_prompt = (
        f"Flow #{req.flow_id} | Verdict: {req.verdict} [{req.attack_type}] | Confidence: {req.confidence*100:.1f}%\n"
        f"Source: {req.src} -> Destination: {req.dst}\n"
        f"Inference Routing: {req.path}-path | Latency: {req.latency_ms:.2f} ms"
    )
    try:
        if manager.llm_provider:
            analysis = manager.llm_provider.generate(system_prompt, human_prompt)
            provider = manager.llm_provider.name
        else:
            raise RuntimeError("Provider not ready")
    except Exception as e:
        from llm_provider import CannedResponseProvider
        analysis = CannedResponseProvider().generate(system_prompt, human_prompt)
        provider = CannedResponseProvider.name

    return {"analysis": analysis, "provider": provider}

@app.post("/api/chat")
def chat_with_network(req: ChatRequest):
    """Answers security questions about current alerts using Stage 5's knowledge base & LLM."""
    question = req.question.strip()
    if not question:
        return {"answer": "Please ask a question about the active network alerts or threat telemetry.", "provider": "System"}

    # Gather alerts context
    with manager.lock:
        alerts = list(manager.recent_alerts)[-10:]

    if not alerts:
        # Fall back to pre-computed Stage 5 alerts if no live stream has run yet
        xai_file = os.path.join(ROOT_DIR, "nids_stage5", "results", "cic_ids2017", "explained_alerts.json")
        if os.path.exists(xai_file):
            with open(xai_file) as f:
                alerts = json.load(f)[:5]

    alerts_summary = "\n".join(
        f"Flow #{a.get('id', i+1)}: {a.get('verdict', a.get('prediction'))} [{a.get('attack_type', 'Traffic')}] (Confidence: {float(a.get('confidence', 0.9))*100:.1f}%), "
        f"Path: {a.get('path', 'cascade')}, Src: {a.get('src', '192.168.1.10')}"
        for i, a in enumerate(alerts)
    ) or "No active alerts currently in buffer. System monitoring baseline traffic."

    retrieved = []
    if manager.knowledge_base:
        try:
            retrieved = manager.knowledge_base.retrieve(question, top_k=2)
        except Exception:
            pass

    knowledge_text = "\n".join(f"[{d['attack_type']}] {d['text']}" for d in retrieved) or "(standard baseline network behavior)"

    system_prompt = (
        "You are an expert Security Operations Center (SOC) analyst assistant for a Network Intrusion Detection System. "
        "Answer the admin's question about the current NIDS network alerts concisely and professionally. "
        "Recommend specific defensive actions (e.g. firewall rule, rate limiting, port block) when appropriate. "
        "Keep the response under 4 sentences."
    )
    human_prompt = (
        f"CURRENT LIVE NETWORK ALERTS:\n{alerts_summary}\n\n"
        f"REFERENCE SECURITY KNOWLEDGE:\n{knowledge_text}\n\n"
        f"ADMIN QUESTION: {question}\n\nAnalyst Assessment & Recommendation:"
    )

    try:
        if manager.llm_provider:
            answer = manager.llm_provider.generate(system_prompt, human_prompt)
            provider_name = manager.llm_provider.name
        else:
            raise RuntimeError("LLM provider not initialized")
    except Exception as err:
        from llm_provider import CannedResponseProvider
        answer = CannedResponseProvider().generate(system_prompt, human_prompt)
        provider_name = CannedResponseProvider.name

    clean_answer = answer.strip()
    if "Grounding Provider:" in clean_answer:
        clean_answer = clean_answer.split("Grounding Provider:")[0].strip()

    return {"answer": clean_answer, "provider": provider_name}


@app.get("/api/xai/alerts")
def get_xai_detailed_alerts():
    """Returns detailed Stage 5 XAI alerts with feature deviation metrics."""
    xai_file = os.path.join(ROOT_DIR, "nids_stage5", "results", "cic_ids2017", "explained_alerts.json")
    if os.path.exists(xai_file):
        with open(xai_file, "r") as f:
            alerts = json.load(f)
            return {"alerts": alerts, "count": len(alerts)}
    return {"alerts": [], "count": 0}


# --- REST Endpoints: Stages 1–4 Benchmarks ---

@app.get("/api/benchmarks")
def get_pipeline_benchmarks():
    """Aggregates metrics and comparisons across Stages 1, 2, 3, and 4."""
    benchmarks = {
        "stage1": {"selected_features": [], "feature_rankings": []},
        "stage2": [],
        "stage3": [],
        "stage4": [],
    }

    def safe_read_csv(filepath, head=None):
        if not os.path.exists(filepath):
            return []
        try:
            df = pd.read_csv(filepath)
            if head:
                df = df.head(head)
            return df.fillna("").to_dict(orient="records")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []

    # Stage 1 Features
    s1_features_path = os.path.join(ROOT_DIR, "nids_stage1", "data", "processed", "cic_ids2017", "selected_features.json")
    if os.path.exists(s1_features_path):
        try:
            with open(s1_features_path) as f:
                benchmarks["stage1"]["selected_features"] = json.load(f)
        except Exception:
            pass

    s1_ranking_path = os.path.join(ROOT_DIR, "nids_stage1", "data", "processed", "cic_ids2017", "feature_ranking_ig.csv")
    benchmarks["stage1"]["feature_rankings"] = safe_read_csv(s1_ranking_path, head=10)

    # Stage 2 Models
    s2_path = os.path.join(ROOT_DIR, "nids_stage2", "results", "cic_ids2017", "comparison_table.csv")
    benchmarks["stage2"] = safe_read_csv(s2_path)

    # Stage 3 Cascade
    s3_path = os.path.join(ROOT_DIR, "nids_stage3", "results", "cic_ids2017", "comparison_table.csv")
    benchmarks["stage3"] = safe_read_csv(s3_path)

    # Stage 4 Transfer
    s4_path = os.path.join(ROOT_DIR, "nids_stage4", "results", "cic_ids2017_to_unsw_nb15", "comparison_table.csv")
    benchmarks["stage4"] = safe_read_csv(s4_path)

    return benchmarks


@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    frontend_path = os.path.join(STAGE6_DIR, "frontend", "index.html")
    if os.path.exists(frontend_path):
        with open(frontend_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>NIDS Unified Dashboard</h3>"


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print(" [NIDS] Unified Master Dashboard is starting!")
    print(" 👉 Open in your browser: http://localhost:8000")
    print("    or: http://127.0.0.1:8000")
    print(" (Note: Windows browsers cannot resolve http://0.0.0.0:8000)")
    print("=" * 60 + "\n")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
