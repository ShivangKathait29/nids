# Stage 6 — Implementation Guide (Fix Everything + Live Capture Demo)

Goal: a working real-time NIDS: **capture → flows → scale → cascade → alert → live dashboard**.

> Existing code lives in `src/`. See `README.md` for what is real vs. simulated.
> Do the phases **in order**. Each phase has a "Done when" check.

---

## Phase 0 — Prerequisites

1. Finish **Stage 1** on real data (produces `data/processed/cic_ids2017/` with
   `scaler.pkl`, `full_feature_names.json`, and train/test splits).
2. Finish **Stage 2/3** training on the same real data.
3. Python 3.10+ and a virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install numpy pandas scikit-learn joblib fastapi "uvicorn[standard]" websockets pyshark kafka-python
   ```

**Done when:** `scaler.pkl` and `full_feature_names.json` exist in the Stage 1 processed folder.

---

## Phase 1 — Fix the foundations

### 1.1 Clean the stray folder
Delete the empty folder named literally `{src,results}` inside `nids_stage6/`
(leftover from a bad shell command). Make sure `results/` exists:
```powershell
Remove-Item -Recurse -Force "{src,results}"
New-Item -ItemType Directory -Force results
```

### 1.2 Retrain Stage 1–3 on REAL data
The README notes the demos flag almost everything as "Attack" because the
threshold was tuned on synthetic data. Re-run Stage 1, 2, 3 on CIC-IDS2017
so precision/recall are meaningful. Do **not** judge Stage 6 until this is done.

### 1.3 Persist the trained cascade (stop retraining every run)
`build_stage3_cascade()` currently retrains on each demo run.
1. After training in Stage 3, save: `joblib.dump(cascade, "models/cascade.pkl")`
   (also save its threshold).
2. In `streaming_inference.py`, load it with `joblib.load` instead of calling
   `build_stage3_cascade()`.
3. Keep the scaler loading as-is (must use the **same** `scaler.pkl`).

**Done when:** the demo starts in seconds and gives identical predictions twice.

### 1.4 Verify feature alignment
Live flows must have the exact columns (names and order) in
`full_feature_names.json`.
1. Print `FlowAggregator` output columns and compare to the JSON list.
2. Missing columns → fill with `0`. Extra columns → drop.
3. Replace `inf`/`NaN` with `0` before scaling (same as Stage 1 cleaning).

**Done when:** a test asserting `list(flow_df.columns) == feature_names` passes.

### 1.5 Verify with the offline modes first
```powershell
cd src
python run_realtime_demo.py --mode replay --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 --n_samples 200 --arrival_rate_per_sec 20
python run_realtime_demo.py --mode simulate --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 --duration_seconds 10 --packets_per_second 100
```
**Done when:** replay accuracy is close to your Stage 3 test accuracy (not ~15%).

---

## Phase 2 — Build the live API (backend for the frontend)

Create `src/api_server.py` (FastAPI):

1. **Pipeline thread:** capture → `FlowAggregator` → `StreamingInferenceEngine.predict_flow()`.
2. **Event queue:** each result becomes JSON:
   ```json
   {"ts": 1728800000.1, "src": "10.0.0.5:51234", "dst": "10.0.0.9:80",
    "verdict": "Attack", "confidence": 0.93, "latency_ms": 4.2,
    "explanation": null}
   ```
3. **Endpoints:**
   - `POST /start?mode=replay|simulate|live&interface=Wi-Fi`
   - `POST /stop`
   - `GET /stats` → totals, attacks, flows/sec, avg latency
   - `WS /ws/alerts` → pushes each event to connected browsers
4. Use `asyncio.Queue` + `loop.call_soon_threadsafe` to move events from the
   pipeline thread into the WebSocket broadcaster.
5. Call the Stage 5 explainer **only** for `Attack` verdicts, asynchronously
   (never block classification).

Run:
```powershell
uvicorn api_server:app --host 0.0.0.0 --port 8000
```
**Done when:** `wscat`/browser console on `ws://localhost:8000/ws/alerts`
receives events after `POST /start?mode=replay`.

---

## Phase 3 — Build the frontend

Either extend the Stage 5 dashboard or add `frontend/index.html` (plain HTML/JS).

Components:
- **Counters:** total flows, attacks, flows/sec, average latency.
- **Live table:** newest first, red = Attack, green = Normal, click row → explanation.
- **Timeline chart:** attacks vs normal per second (Chart.js).
- **Controls:** Start / Stop, mode dropdown (Replay / Simulate / Live), interface name.

Core JS:
```js
const ws = new WebSocket("ws://localhost:8000/ws/alerts");
ws.onmessage = (m) => addRow(JSON.parse(m.data));
```
**Done when:** pressing Start in replay mode fills the table in real time.

---

## Phase 4 — Live capture on a real network

### 4.1 Install capture tooling (Windows)
1. Install **Wireshark** (tick *Npcap* and *TShark*). Reboot if asked.
2. Add `C:\Program Files\Wireshark` to `PATH`.
3. List interfaces and pick yours:
   ```powershell
   tshark -D
   ```
4. Sanity check capture works (run PowerShell **as Administrator**):
   ```powershell
   tshark -i "Wi-Fi" -c 20
   ```
(Linux: `sudo apt install tshark`, `sudo usermod -aG wireshark $USER`, re-login.)

### 4.2 Choose flow extraction
- **Easy (recommended):** `FlowAggregator` in-process (no Java).
- **Most accurate:** capture short pcap windows (e.g., 10 s) with tshark, then
  `run_cicflowmeter_cli(pcap_path, output_dir, jar_path)` (needs Java +
  CICFlowMeter jar). This matches training features best.

### 4.3 Switch the source to live
In `api_server.py` (live mode) use:
```python
source = TsharkCapture(interface="Wi-Fi")   # name from `tshark -D`
```
Everything downstream is unchanged.

### 4.4 Run the live demo
1. Start backend (as Administrator): `uvicorn api_server:app --port 8000`
2. Open `frontend/index.html`.
3. Select **Live**, enter the interface, click **Start**.
4. Generate traffic:
   - **Normal:** browse the web, `ping 8.8.8.8`, stream a video.
   - **Attack-like (own lab only!):** from a Kali/VM on an isolated
     host-only network against a machine you own:
     ```bash
     nmap -sS -p 1-1000 <target-ip>              # port scan
     hping3 -S --flood -p 80 <target-ip>          # SYN flood (DoS)
     slowhttptest -c 1000 -H -u http://<target-ip> # slow DoS
     ```
5. Watch rows turn red and the attack counter rise.

> ⚠️ Only attack systems you own or have permission to test.

### 4.5 Fallback plan
If capture fails mid-demo, switch the dropdown to **Replay** (real labeled
data). Always rehearse both.

---

## Phase 5 — Optional: Kafka backend

1. Start a broker: `docker run -p 9092:9092 apache/kafka`
2. Run with `--kafka_backend kafka` (or
   `make_producer_consumer_pair(backend="kafka", bootstrap_servers="localhost:9092")`).
3. Capture process = producer; inference/API process = consumer.

---

## Phase 6 — Testing checklist

| Test | Expected |
|---|---|
| Replay accuracy vs Stage 3 test accuracy | Within a few % |
| Feature columns == `full_feature_names.json` | Exact match |
| Model loads from `cascade.pkl` | No retraining |
| `tshark -D` lists interface | Yes |
| Live normal browsing | Mostly "Normal" |
| `nmap` scan from VM | Attack alerts appear |
| Per-flow latency | Roughly < 100 ms |
| Stop/Start repeatedly | No crashes/leaks |

## Common problems

| Symptom | Fix |
|---|---|
| Everything flagged Attack | Retrain on real data; re-tune threshold (Phase 1.2) |
| `tshark not found` | Add Wireshark to `PATH`, reopen terminal |
| No packets captured | Run as Administrator; check interface name |
| Verdicts noisy on live traffic | Use real CICFlowMeter (4.2); check feature alignment |
| Flows never complete | Lower flow inactivity timeout in `FlowAggregator` |
| Wrong predictions after changes | Ensure the same `scaler.pkl` is used |
| Frontend shows nothing | Check WebSocket URL/CORS and that `/start` was called |

## Final demo script (5 minutes)
1. Show architecture diagram.
2. Replay mode → show accuracy + latency.
3. Switch to Live → browse normally (green).
4. Run `nmap` from VM (red alerts + explanation).
5. Show Stage 5 explanation for one attack.

---

## ✅ Master Checklist (follow top to bottom)

### Phase 0 — Prerequisites
- [x] 1. Stage 1 finished on real CIC-IDS2017 data
- [x] 2. `scaler.pkl` and `full_feature_names.json` exist in `nids_stage1/data/processed/cic_ids2017`
- [x] 3. Stage 2 and Stage 3 trained on the same real data
- [x] 4. Virtual environment / Python 3.12 environment active
- [x] 5. Packages installed (`fastapi`, `uvicorn`, `websockets`, `pyshark`, `joblib`, `scikit-learn`, `pandas`, `numpy`)

### Phase 1 — Fix the foundations
- [x] 6. Deleted stray `{src,results}` folder; `results/` exists
- [x] 7. Cascade model + threshold saved to `models/cascade.pkl`
- [x] 8. `streaming_inference.py` loads the saved model (no retraining)
- [x] 9. Live flow columns match `full_feature_names.json` (names and order)
- [x] 10. `inf`/`NaN` replaced with 0 before scaling
- [x] 11. Replay mode runs; accuracy close to Stage 3 test accuracy (verified: 98.5% batch accuracy, 12.7ms avg latency)
- [x] 12. Simulate mode runs; flows aggregate (verified: 100% multi-packet flows, 21.0 pkts/flow avg)

### Phase 2 — Backend API
- [x] 13. `api_server.py` created (FastAPI)
- [x] 14. Pipeline runs in a background thread
- [x] 15. `/start`, `/stop`, `/stats` endpoints work
- [x] 16. `/ws/alerts` WebSocket streams JSON events
- [x] 17. XAI explanation called only for "Attack" and never blocks classification

### Phase 3 — Frontend
- [x] 18. Page connects to the WebSocket
- [x] 19. Counters update (total, attacks, flows/sec, latency)
- [x] 20. Live table shows red/green rows; clicking a row shows the explanation
- [x] 21. Timeline chart updates every second
- [x] 22. Start/Stop and mode dropdown work

### Phase 4 — Live capture
- [x] 23. Wireshark/tshark installed and on `PATH`
- [x] 24. `tshark -D` lists the interface; name noted
- [x] 25. `tshark -i "<iface>" -c 20` captures packets (run as Administrator)
- [x] 26. Source switched to `TsharkCapture(interface=...)`
- [x] 27. Normal browsing/ping shows mostly "Normal"
- [x] 28. Attack-like traffic simulated (`simulate_portscan.py` created for local test, or `nmap`/`hping3` from lab VM)
- [x] 29. Replay mode confirmed as fallback

### Phase 5 — Optional Kafka (Optional: Skip if Docker not installed)
- [ ] 30. Broker running (`docker run -p 9092:9092 apache/kafka`) [Optional]
- [ ] 31. Pipeline works with `--kafka_backend kafka` [Optional]

### Phase 6 — Final verification
- [x] 32. Per-flow latency is roughly under 100 ms (measured ~12-23 ms)
- [x] 33. Repeated Start/Stop causes no crash or leak
- [x] 34. Results/screenshots saved to `results/` (15 screenshots saved)
- [x] 35. Full demo script rehearsed (replay → live → attack → explanation)
- [x] 36. Backup plan ready (replay mode) in case live capture fails
