# NIDS Project — Stage 5: XAI / Explainability Layer

Implements **Stage 5**: a LangChain-based pipeline that turns flagged
predictions into human-readable explanations, grounded in retrieved
security knowledge, plus a "Chat with your Network" dashboard — exactly
the differentiator called out in your project plan.

## What's included

```
nids_stage5/
├── src/
│   ├── explanation_context.py     # Per-sample feature-deviation analysis (SHAP-lite)
│   ├── security_knowledge_base.py # Local retrieval over attack-type knowledge (LlamaIndex-lite)
│   ├── prompt_templates.py        # LangChain prompt engineering
│   ├── llm_provider.py            # Real LangChain+OpenAI/Gemini, with local fallback
│   ├── xai_pipeline.py            # Orchestrates context -> retrieval -> LLM -> explanation
│   ├── run_stage5_demo.py         # CLI demo: explains a batch of flagged alerts
│   └── dashboard/
│       ├── app.py                 # Flask "Chat with your Network" dashboard
│       └── templates/dashboard.html
├── results/
│   └── <dataset>/
│       ├── explained_alerts.json
│       └── sample_report.txt
└── README.md
```

## ⚠️ What's real vs. what's a tested stand-in

Same constraint as every prior stage: **no network access in this dev
environment**, so `langchain`, `langchain-openai`, `langchain-google-genai`,
and `shap` aren't installed, and there are no API keys to call Gemini/
OpenAI even if they were. Here's exactly what that means for each piece:

| Component | Status |
|---|---|
| Feature-deviation explainability | **Fully real** — hand-implemented (z-score × feature importance), no dependency needed |
| Knowledge retrieval | **Real retrieval mechanism** (TF-IDF + cosine similarity via sklearn), but the corpus is a small curated set I wrote — swap for real LlamaIndex + a live CVE feed when you have network access (interface won't change) |
| LangChain + OpenAI/Gemini calls | **Real, correct code** in `llm_provider.py`, verified importable/structured correctly, but UNTESTED against a live API here — no key, no network |
| Local template fallback | **Fully real and tested** — proves the entire plumbing (context → prompt → explanation → report → dashboard) works end-to-end right now |
| Flask dashboard | **Fully real and tested** via Flask's test client — index page, alerts API, and chat API all verified working |

**Bottom line: everything runs end-to-end today using the local fallback.
The moment your team runs `pip install langchain langchain-openai
langchain-google-genai` and sets `OPENAI_API_KEY` or `GOOGLE_API_KEY`, real
LLM explanations kick in automatically — zero code changes needed
anywhere.** This is a deliberate design choice (see `llm_provider.py`'s
`get_llm_provider()` factory) specifically so the project isn't blocked on
API access to develop and demo.

## How to run

### CLI demo (generates a report file)
```bash
cd src
python run_stage5_demo.py --dataset cic_ids2017 \
    --stage1_dir ../../nids_stage1/data/processed/cic_ids2017 \
    --llm_preference auto
```

### Dashboard (interactive, "Chat with your Network")
```bash
cd src/dashboard
python app.py --dataset cic_ids2017 \
    --stage1_dir ../../../nids_stage1/data/processed/cic_ids2017 \
    --llm_preference auto
```
Then open **http://localhost:5000**. Click any alert card to see its full
explanation and per-feature deviation breakdown; use the chat box at the
bottom to ask free-form questions about the current alerts.

### Enabling real LLM output
```bash
pip install langchain langchain-openai langchain-google-genai
export OPENAI_API_KEY="sk-..."      # or:
export GOOGLE_API_KEY="..."
```
`--llm_preference auto` tries OpenAI first, then Gemini, then falls back
to the local template — or pin one explicitly with `--llm_preference openai`
/ `gemini` / `local`.

## How the explanation pipeline works, end to end

```
flagged sample
      │
      ▼
explanation_context.py   →  which features deviate from "normal", how much,
                             weighted by the model's own feature importances
      │
      ▼
security_knowledge_base.py → retrieves the 1-2 most relevant attack-type
                              descriptions for this pattern (TF-IDF search)
      │
      ▼
prompt_templates.py      →  formats both into a structured prompt
      │
      ▼
llm_provider.py           →  sends to OpenAI/Gemini via LangChain (or the
                              local fallback), returns fluent prose
      │
      ▼
human-readable explanation + feature breakdown + retrieved context
```

### Why not SHAP for feature explanations?
SHAP is the standard tool for this, but isn't installed here (no network
access). Instead, `explanation_context.py` computes each feature's z-score
deviation from the benign-class mean, weighted by the model's global
`feature_importances_`. This is a transparent, reproducible approximation
— not Shapley-value-rigorous, but fully explainable in your viva, and
`pip install shap` + swapping in `TreeExplainer` is a natural upgrade once
your team has network access (documented in the module).

### Why not real LlamaIndex?
Same reason — no network access to install it here. `security_knowledge_base.py`
implements the same underlying retrieval concept (vectorize a query, find
the nearest documents) using sklearn's `TfidfVectorizer`, over a small
curated knowledge base I wrote describing each attack category present in
your datasets (DDoS, PortScan, Bot, Exploits, etc. — NOT copied from any
external CVE text, written fresh to avoid any copyright concerns). The
`retrieve()` interface is designed so swapping in real LlamaIndex + a live
CVE feed later wouldn't require changing any calling code.

## Sample output (from the tested local-fallback run)

```
ALERT #1 (test sample index 11)
Verdict: Attack  |  Confidence: 63.7%
Top contributing features:
    - total_fwd_packets: -1.252 (typical benign: 0.011, -1.27 std devs below normal)
    - destination_port: -0.654 (typical benign: -0.006, -0.65 std devs below normal)
    - fwd_packet_length_mean: -0.919 (typical benign: -0.006, -0.91 std devs below normal)
    - flow_duration: 0.93 (typical benign: 0.004, 0.92 std devs above normal)
    - protocol: -1.101 (typical benign: 0.004, -1.1 std devs below normal)

Explanation: This traffic was flagged as a potential ATTACK with 63.7% model
confidence. The strongest signal was total_fwd_packets, along with 4 other
feature(s) deviating from typical benign traffic patterns. This pattern is
consistent with known DoS-style traffic characteristics. Recommended action:
review the source of this flow and consider temporary monitoring or
rate-limiting if the pattern repeats.
```

This is the local template's output (no real LLM) — already coherent and
useful. A real LLM (OpenAI/Gemini) will produce noticeably more natural,
varied prose, but the **structure and grounding** (which features, how far
from normal, which known attack pattern it resembles) is identical either
way, because that logic lives in `explanation_context.py` and
`security_knowledge_base.py`, not in the LLM call itself.

## Dashboard design notes

The dashboard is a two-pane console (alert list + detail panel) with a
docked chat box, styled for extended monitoring use (dark background,
monospace for data fields, a teal/coral accent pair distinguishing
normal/attack). Each contributing feature gets a small horizontal bar
sized by its deviation magnitude (in standard deviations) — a lightweight,
readable stand-in for a proper SHAP force plot.

## Decoupling from the detection pipeline

Per the project plan, this layer must never block real-time detection.
Every explanation call in `xai_pipeline.py` happens strictly AFTER a
prediction is already made — it's an enrichment step, not part of the hot
path. In a real deployment, you'd only call `explainer.explain()` for
flagged/attack traffic (or on-demand when an admin clicks an alert), never
for every packet.

## What to paste into your report

- The pipeline diagram above → your Stage 5 methodology section
- `sample_report.txt` → example output for your Results/Demo section
- A dashboard screenshot (run it locally and screenshot it — I couldn't
  capture one directly in this sandboxed dev environment since it has no
  network access even to `localhost`, but the Flask test client confirms
  every route works correctly)
- The SHAP/LlamaIndex substitution notes above → honest "implementation
  constraints" paragraph, useful if a panel member asks about your XAI
  methodology specifically

## Next steps (Stage 6/7)

- Wire `XAIExplainer` into Stage 3's `TwoStagePipeline` — pass
  `routing_info={"path": "fast"/"slow"}` from the cascade's diagnostics
  into `explain()` so explanations can reference HOW a verdict was reached
  (already supported — see `explain(sample, routing_info=...)`).
- Once Stage 6 (real-time capture) is live, call `explainer.explain()` on
  each newly-flagged live sample instead of static test-set rows, and
  push results into the dashboard's alert list dynamically.
