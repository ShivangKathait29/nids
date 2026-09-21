"""
prompt_templates.py
====================
Stage 5 - Component 3: Prompt Engineering

Defines the LangChain prompt template used to turn structured detection
context (from explanation_context.py) plus retrieved grounding text (from
security_knowledge_base.py) into a request for a fluent, human-readable
security explanation.

Kept in its own module so prompt wording can be iterated on independently
of the plumbing in xai_pipeline.py — and so your report can show this file
directly as your "prompt engineering" artifact.
"""

SYSTEM_PROMPT = """You are a network security analyst assistant embedded in a Network \
Intrusion Detection System (NIDS). Your job is to translate a machine learning model's \
technical detection output into a clear, concise explanation for a network administrator \
who may not have a deep ML background.

Rules:
- Be concrete: reference the specific features and values provided, not generic statements.
- Be calibrated: if confidence is low or evidence is mixed, say so plainly — don't overstate certainty.
- Be actionable: end with one brief, concrete recommended next step appropriate to the verdict.
- Keep the explanation to 3-5 sentences. No headers, no bullet lists — plain prose an admin can read in 10 seconds.
- If reference knowledge about a matching attack type is provided, use it to explain WHY the \
flagged pattern matches that category, but do not simply repeat the reference text verbatim."""


HUMAN_PROMPT_TEMPLATE = """DETECTION RESULT
Verdict: {prediction}
Model confidence: {confidence}
{routing_line}

TOP CONTRIBUTING FEATURES (ranked by deviation from typical benign traffic, weighted by model importance)
{feature_lines}

REFERENCE KNOWLEDGE (retrieved, may or may not be directly relevant — use your judgment)
{knowledge_context}

Write the explanation now."""


def format_feature_lines(contributing_features: list) -> str:
    lines = []
    for f in contributing_features:
        lines.append(
            f"- {f['feature']}: observed value {f['value']} vs. typical benign value "
            f"{f['typical_benign_value']} ({f['deviation_std']} std devs {f['direction']})"
        )
    return "\n".join(lines)


def format_knowledge_context(retrieved_docs: list) -> str:
    if not retrieved_docs:
        return "(none retrieved / not applicable)"
    lines = []
    for doc in retrieved_docs:
        lines.append(f"[{doc['attack_type']}] {doc['text']}")
    return "\n".join(lines)


def build_prompt_variables(context: dict, retrieved_docs: list) -> dict:
    """
    Converts the structured context dict (from ExplanationContextBuilder)
    and retrieved knowledge base docs into the variables the LangChain
    prompt template expects.
    """
    routing_line = ""
    if "routing_info" in context:
        path = context["routing_info"].get("path", "unknown")
        if path == "fast":
            routing_line = "Routing: cleared by the fast Stage 1 filter alone (high-confidence normal, deep classifier was not invoked)."
        elif path == "slow":
            routing_line = "Routing: flagged as suspicious by the Stage 1 filter and confirmed by the Stage 2 deep classifier."

    return {
        "prediction": context["prediction"],
        "confidence": f"{context['confidence']*100:.1f}%",
        "routing_line": routing_line,
        "feature_lines": format_feature_lines(context["contributing_features"]),
        "knowledge_context": format_knowledge_context(retrieved_docs),
    }
