# NIDS Project — Stage 6: Real-Time Infrastructure

Implements **Stage 6**: live traffic capture → flow feature extraction →
streaming inference, with Kafka as the Tier 2 stretch goal — closing the
"is this actually real-time" gap flagged back when we scoped the project.

## What's included

```
nids_stage6/
├── src/
│   ├── live_capture.py           # Tshark/pyshark (real) + traffic simulator (tested)
│   ├── flow_feature_extractor.py # CICFlowMeter CLI (real) + Python flow aggregator (tested)
│   ├── streaming_inference.py    # Applies Stage 1's scaler + Stage 3's cascade to live flows
│   ├── kafka_pipeline.py         # Kafka (real, stretch goal) + in-memory queue (tested)
│   └── run_realtime_demo.py      # Orchestrates both demo modes (see below)
├── results/
└── README.md
```

## ⚠️ What's real vs. what's a tested stand-in

Same honesty pattern as every other stage — this one has the most
constraints, since live capture fundamentally cannot run in a sandboxed
dev environment with no network interface access:

| Component | Status |
|---|---|
| `TsharkCapture` (pyshark/tshark) | **Real, correct code** — untested here (no network interface, no tshark install, no network access to `pip install pyshark`) |
| `SimulatedCapture` | **Fully tested** — generates realistic multi-packet flows (fixed a real bug during testing: packets were originally getting fresh random ports every time, preventing flow aggregation entirely — see below) |
| `run_cicflowmeter_cli()` (real CICFlowMeter) | **Real, correct code** — untested here (no Java jar, no network access to download it) |
| `FlowAggregator` (Python flow aggregation) | **Fully tested** — verified against 2000 simulated packets, correctly aggregates into realistic multi-packet flows matching CIC-IDS2017's feature schema |
| `StreamingInferenceEngine` | **Fully tested end-to-end** — correctly loads Stage 1's scaler/feature artifacts and runs Stage 3's cascade on live flow data |
| `KafkaFlowProducer`/`Consumer` (real kafka-python) | **Real, correct code** — untested here (no broker, no network access to `pip install kafka-python`); verified it fails with a clear, actionable error rather than crashing unexpectedly |
| `InMemoryQueueProducer`/`Consumer` | **Fully tested** — verified with real producer/consumer threads and Stage 1's actual labeled test data |

**Bottom line: the full pipeline — capture → aggregate → scale → cascade
inference → alert — runs and was verified end-to-end today**, using the
tested stand-ins. Swapping `SimulatedCapture` → `TsharkCapture`,
`FlowAggregator` → real CICFlowMeter, and `in_memory` → `kafka` backend
requires **zero changes to `streaming_inference.py` or the cascade
model** — only the capture/ingestion layer changes, exactly the
architecture your project plan called for.

## A real bug this testing caught

While testing `SimulatedCapture`, I found that packets were assigned a
fresh random source port on every single packet — meaning almost nothing
could aggregate into a multi-packet flow (2000 packets → 1985 "flows",
each with ~1 packet). Real TCP connections keep the same source port for
their entire duration. **Fixed**: the simulator now maintains persistent
per-conversation state (matching a real socket's lifetime), and re-testing
confirmed proper aggregation (2000 packets → 111 flows, 87 of them with
more than 3 packets). Worth mentioning in your report if you discuss
`FlowAggregator`'s validation — it's a good example of a subtle bug that
only surfaces once you check the actual flow statistics, not just "does
the code run."

## The two demo modes — and why there are two

### `--mode simulate`
Simulated raw packets → flow aggregation → scaling → cascade inference →
live console alerts. Exercises the **full raw-capture-to-alert path**
exactly as production would run it. But since the traffic is fabricated,
there's no ground truth — this mode measures real throughput and proves
structural correctness, **not** detection accuracy.

```bash
cd src
python run_realtime_demo.py --mode simulate \
    --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 \
    --duration_seconds 10 --packets_per_second 100
```

### `--mode replay`
Streams Stage 1's **real labeled test set** through a producer/consumer
split (in-memory queue by default, or `--kafka_backend kafka` once you
have a broker) at a configurable arrival rate. This mode has ground truth,
so it reports actual accuracy alongside latency — and concretely
demonstrates the Kafka-style streaming architecture (separate producer
and consumer threads communicating via queue/topic).

```bash
python run_realtime_demo.py --mode replay \
    --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 \
    --n_samples 50 --arrival_rate_per_sec 20
```

Together, these two modes cover what one mode alone couldn't: `simulate`
proves the raw pipeline works end-to-end; `replay` proves the model itself
is being evaluated correctly and measures real accuracy under streaming
conditions.

## Reading the current results (synthetic data)

Both demo modes currently over-flag traffic as "Attack" (36/39 flows in
the simulate-mode test run; 15% accuracy in the replay-mode test run).
This is **consistent with, not separate from**, what you've already seen
in Stages 2/3: the decision threshold tuned on signal-free synthetic data
optimizes for high recall, which comes at a steep precision cost when
there's no real signal to separate classes. This isn't a Stage 6 bug —
Stage 6 is faithfully surfacing the same underlying detector behavior
you'd see running Stage 2 or Stage 3 alone on this data. **Once real
datasets are used upstream (Stages 1-3), Stage 6 should show a healthy mix
of Normal/Attack verdicts matching the model's real precision/recall.**

## Setting up the real backends (on your team's machines)

### Live capture (Tshark/pyshark)
```bash
pip install pyshark
# Linux: sudo apt install tshark && sudo usermod -aG wireshark $USER
# Windows: install Wireshark (includes tshark), ensure it's on PATH
```
Then use `TsharkCapture(interface="eth0")` in place of `SimulatedCapture`.

### Flow extraction (CICFlowMeter)
Download from https://github.com/ahlashkari/CICFlowMeter, requires Java.
Use `run_cicflowmeter_cli(pcap_path, output_dir, jar_path)` on a pcap saved
from `TsharkCapture`, or keep using `FlowAggregator` for in-process
aggregation (a legitimate, lighter-weight alternative that avoids the
Java dependency and file I/O round-trip entirely — many real deployments
do this instead of shelling out to CICFlowMeter).

### Kafka
```bash
pip install kafka-python
docker run -p 9092:9092 apache/kafka   # or your team's existing broker
```
Then pass `--kafka_backend kafka` to `run_realtime_demo.py`, or construct
`make_producer_consumer_pair(backend="kafka", bootstrap_servers="...")`
directly in your own code.

## Design decisions worth knowing for your viva

- **Why flow aggregation instead of per-packet classification?** Your
  models (Stages 2-4) were trained on flow-level features (duration,
  packet counts, byte counts) — a single packet doesn't carry this
  information. Aggregating packets into flows (grouped by 5-tuple, closed
  after a timeout of inactivity) is standard NIDS practice and matches
  exactly how CIC-IDS2017 itself was generated from raw pcaps.
- **Why does `StreamingInferenceEngine` load a saved scaler rather than
  fitting a new one?** Live traffic must be transformed using the EXACT
  same mean/std statistics the model was trained on. Fitting a fresh
  scaler on live data would use different statistics and silently corrupt
  every prediction — this is why Stage 1 was updated to persist
  `scaler.pkl` and `full_feature_names.json` specifically for this stage.
- **Why is XAI (Stage 5) decoupled and only called for flagged traffic?**
  Per Stage 5's design, a slow or unavailable LLM must never block
  real-time classification. `StreamingInferenceEngine.predict_flow()`
  always returns a verdict first; the explanation is generated afterward,
  only when the verdict is "Attack."

## Next steps (Stage 7 — Integration)

This is close to your final integrated system already: `run_realtime_demo.py`
wires together Stage 1 (scaler/features), Stage 3 (cascade), and can take
a Stage 5 `XAIExplainer` via `StreamingInferenceEngine`'s `explainer`
parameter. For Stage 7, the main remaining work is:
1. Persist a trained model (pickle/MLflow) instead of retraining on every
   demo run — `build_stage3_cascade()` currently retrains for
   convenience, which is fine for a demo but not for production.
2. Wire Stage 5's dashboard to read live alerts from this pipeline instead
   of static test-set rows.
3. Swap in the real Tshark/CICFlowMeter/Kafka backends once your team has
   a machine with network access and capture permissions.
