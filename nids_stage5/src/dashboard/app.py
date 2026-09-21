"""
app.py
======
Stage 5 - Component 6: "Chat with Your Network" Dashboard

A small Flask app that:
    1. Shows the flagged alerts from run_stage5_demo.py's output as cards
       with their generated explanations (the XAI dashboard piece)
    2. Provides a chat box where an admin can ask free-form questions
       about the current alerts ("why was alert 2 flagged?", "which
       alerts are DDoS-like?", "what should I do about alert 1?") — this
       is the "Chat with your Network" feature from the project plan,
       built on the same XAIExplainer + knowledge base + LLM provider
       already used for individual alert explanations.

Run with:
    cd nids_stage5/src/dashboard
    python app.py --dataset cic_ids2017 \
        --stage1_dir ../../../nids_stage1/data/processed/cic_ids2017
Then open http://localhost:5000

Design note: this is deliberately a small, single-file Flask app (not a
full framework project) — appropriate scope for a demo-able final year
project component. It reuses run_stage5_demo.py's alert-generation logic
directly rather than duplicating it.
"""

import argparse
import os
import sys

from flask import Flask, render_template, request, jsonify

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "nids_stage2", "src"))

from run_stage5_demo import load_stage1_outputs
from model_zoo import IDRandomForest
from xai_pipeline import XAIExplainer

app = Flask(__name__)

# Populated at startup by init_app()
STATE = {"explainer": None, "alerts": [], "dataset_name": None}


def init_app(dataset_name: str, stage1_dir: str, llm_preference: str = "auto",
             n_attack_samples: int = 4, n_normal_samples: int = 3):
    print(f"[dashboard] Loading data and training detector for '{dataset_name}'...")
    X_train, y_train, X_test, y_test = load_stage1_outputs(stage1_dir)
    feature_names = list(X_train.columns)

    model = IDRandomForest()
    model.fit(X_train, y_train)

    explainer = XAIExplainer(model, feature_names, X_train, y_train, llm_preference=llm_preference)

    preds = model.predict(X_test)
    attack_idx = X_test.index[preds == 1][:n_attack_samples]
    normal_idx = X_test.index[preds == 0][:n_normal_samples]

    alerts = []
    for i, idx in enumerate(list(attack_idx) + list(normal_idx)):
        sample = X_test.loc[idx]
        result = explainer.explain(sample)
        result["alert_id"] = i + 1
        result["sample_index"] = int(idx)
        alerts.append(result)

    STATE["explainer"] = explainer
    STATE["alerts"] = alerts
    STATE["dataset_name"] = dataset_name
    print(f"[dashboard] Ready. {len(alerts)} alerts loaded "
          f"(LLM provider: {explainer.llm_provider.name})")


@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        dataset_name=STATE["dataset_name"],
        alerts=STATE["alerts"],
        llm_provider_name=STATE["explainer"].llm_provider.name if STATE["explainer"] else "n/a",
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Free-form Q&A about the current alerts. Retrieves relevant knowledge
    base context for the question, builds a prompt containing a summary
    of all current alerts, and asks the LLM provider to answer.
    """
    question = request.json.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please ask a question about the current alerts."})

    explainer = STATE["explainer"]
    alerts = STATE["alerts"]

    alerts_summary = "\n".join(
        f"Alert #{a['alert_id']}: {a['prediction']} (confidence {a['confidence']*100:.1f}%), "
        f"top feature: {a['contributing_features'][0]['feature']}"
        for a in alerts
    )
    retrieved = explainer.knowledge_base.retrieve(question, top_k=2)
    knowledge_text = "\n".join(f"[{d['attack_type']}] {d['text']}" for d in retrieved) or "(none directly relevant)"

    system_prompt = (
        "You are a network security analyst assistant. Answer the admin's question about the "
        "current NIDS alerts below using ONLY the information provided. If the question can't be "
        "answered from this information, say so plainly. Keep the answer under 4 sentences."
    )
    human_prompt = (
        f"CURRENT ALERTS:\n{alerts_summary}\n\n"
        f"REFERENCE KNOWLEDGE:\n{knowledge_text}\n\n"
        f"ADMIN QUESTION: {question}\n\nAnswer:"
    )

    try:
        answer = explainer.llm_provider.generate(system_prompt, human_prompt)
        provider_used = explainer.llm_provider.name
    except Exception:
        from llm_provider import CannedResponseProvider
        answer = CannedResponseProvider().generate(system_prompt, human_prompt)
        provider_used = CannedResponseProvider.name

    return jsonify({"answer": answer.strip(), "llm_provider_used": provider_used})


@app.route("/api/alerts")
def api_alerts():
    return jsonify(STATE["alerts"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stage1_dir", required=True)
    parser.add_argument("--llm_preference", default="auto", choices=["auto", "openai", "gemini", "local"])
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    init_app(args.dataset, args.stage1_dir, llm_preference=args.llm_preference)
    app.run(debug=False, port=args.port)
