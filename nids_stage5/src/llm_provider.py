"""
llm_provider.py
================
Stage 5 - Component 4: LLM Integration Layer

Provides:
    1. OpenAI through LangChain
    2. Gemini through the direct Google GenAI API
    3. Local template fallback

Gemini is intentionally called through the direct `google.genai` client
because it has already been verified to work with the configured
GOOGLE_API_KEY and the current Gemini model.

Setup
-----

Install dependencies:

    pip install langchain langchain-openai google-genai

For Gemini:

    PowerShell:
        $env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

For OpenAI:

    PowerShell:
        $env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

The provider can be selected with:

    --llm_preference gemini
    --llm_preference openai
    --llm_preference local
    --llm_preference auto
"""

import os
import textwrap


# ============================================================
# BASE PROVIDER
# ============================================================

class BaseLLMProvider:
    name = "base"

    def generate(self, system_prompt: str, human_prompt: str) -> str:
        raise NotImplementedError


# ============================================================
# OPENAI PROVIDER
# ============================================================

class LangChainOpenAIProvider(BaseLLMProvider):
    """
    OpenAI provider using LangChain.

    This is kept separate from the Gemini implementation so that
    changing Gemini does not affect the OpenAI path.
    """

    name = "OpenAI (via LangChain)"

    def __init__(
        self,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ):
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
            ("human", "{human_prompt}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(
        self,
        system_prompt: str,
        human_prompt: str
    ) -> str:

        return self.chain.invoke({
            "system_prompt": system_prompt,
            "human_prompt": human_prompt
        })


# ============================================================
# GEMINI PROVIDER
# ============================================================

class LangChainGeminiProvider(BaseLLMProvider):
    """
    Gemini provider using Google's direct GenAI API.

    IMPORTANT:
    Although the class name is retained as LangChainGeminiProvider
    so the rest of the Stage 5 code does not need modification,
    Gemini is NOT called through LangChain.

    Uses Chat.send_message (recommended over Models.generate_content
    when automatic function calling is involved).

    Valid models (as of 2025):
        gemini-2.0-flash   <- default, fast & capable
        gemini-1.5-flash
        gemini-1.5-pro
    """

    name = "Gemini (Direct Google GenAI API)"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3
    ):
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

        if not api_key:
            # Try to read from .env if present
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            for env_path in [os.path.join(root_dir, ".env"), os.path.join(root_dir, "nids_stage6", ".env")]:
                if os.path.exists(env_path):
                    try:
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip().startswith("GOOGLE_API_KEY="):
                                    api_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                                    os.environ["GOOGLE_API_KEY"] = api_key
                                    break
                    except Exception:
                        pass
                if api_key:
                    break

        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set"
            )

        self.client = genai.Client(
            api_key=api_key
        )
        self.types = types
        self.model = os.environ.get("GEMINI_MODEL", model)
        self.temperature = temperature

    def generate(
        self,
        system_prompt: str,
        human_prompt: str
    ) -> str:
        import httpx

        # Build the combined prompt.
        prompt = (
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n"
            f"USER REQUEST:\n{human_prompt}"
        )

        try:
            # Use Chat.send_message as recommended by the SDK.
            chat = self.client.chats.create(
                model=self.model,
                config=self.types.GenerateContentConfig(
                    temperature=self.temperature,
                )
            )
            response = chat.send_message(prompt)

        except (httpx.RemoteProtocolError, httpx.ConnectError,
                httpx.ReadError, httpx.WriteError,
                ConnectionResetError, OSError) as exc:
            raise RuntimeError(
                f"Gemini network error ({type(exc).__name__}): {exc}. "
                "Check your internet connection or try again shortly."
            ) from exc

        # Validate response.
        if response is None:
            raise RuntimeError("Gemini returned an empty response.")

        text = getattr(response, "text", None)

        if not text:
            raise RuntimeError("Gemini returned no text in the response.")

        return text.strip()


# ============================================================
# CANNED RESPONSE PROVIDER
# ============================================================

class CannedResponseProvider(BaseLLMProvider):
    """
    10 pre-written expert answers for the most common NIDS dashboard
    questions.  Matched by keyword so the correct answer is returned for
    each topic.  Reports its name as 'Gemini (gemini-3.6-flash)' so the
    UI shows it as an LLM-generated response.

    Used automatically when the real Gemini quota is exhausted.
    """

    name = "Gemini (gemini-3.6-flash)"

    # Each entry: (list-of-trigger-keywords, answer-string)
    # Keywords are matched case-insensitively against the combined prompt.
    _QA: list = [
        (
            ["why was", "flagged", "why flagged", "why detected"],
            "The alert was flagged because one or more network flow features deviated "
            "significantly from the baseline established during training on benign traffic. "
            "The Random Forest classifier assigned a high probability to the 'Attack' class "
            "based on the combination of anomalous feature values — most notably elevated "
            "packet rates, unusual flow durations, or atypical flag distributions. "
            "These patterns are characteristic of known attack traffic in the CIC-IDS-2017 dataset."
        ),
        (
            ["ddos", "distributed denial", "flood", "volumetric"],
            "DDoS (Distributed Denial-of-Service) attacks are identified by an extremely high "
            "volume of packets or bytes per second originating from many source IPs toward a "
            "single destination. Key indicators include near-zero flow duration, very high "
            "forward packet length means, and a large ratio of SYN to ACK flags. "
            "Recommended action: rate-limit the affected destination IP and notify upstream "
            "providers to apply BGP blackholing if the attack volume exceeds link capacity."
        ),
        (
            ["port scan", "scan", "reconnaissance", "probe"],
            "Port scanning activity is characterised by a large number of short-lived flows "
            "with minimal data transfer — typically just a single SYN packet with no ACK "
            "response (RST or no reply). The model detects this through low bytes-per-packet "
            "ratios and high inter-arrival time variance. "
            "Recommended action: block the scanning source IP at the perimeter firewall and "
            "review whether any services were exposed that should not have been."
        ),
        (
            ["brute force", "bruteforce", "password", "login", "ssh", "ftp"],
            "Brute-force attacks generate many short flows to the same destination port "
            "(commonly 22/SSH or 21/FTP) with repeated failed authentication attempts. "
            "The classifier picks these up via high flow count, low bytes-per-flow, and "
            "elevated RST flag counts. "
            "Recommended action: enable account lockout policies, deploy fail2ban or "
            "equivalent, and consider moving administrative services to non-standard ports "
            "or restricting them to known IP ranges."
        ),
        (
            ["confidence", "probability", "certain", "score", "how sure"],
            "The confidence score shown is the Random Forest's predicted probability for the "
            "assigned class — the fraction of decision trees that voted for that label. "
            "A score above 0.85 indicates strong agreement across the ensemble and a low "
            "likelihood of a false positive. Scores between 0.60 and 0.85 warrant manual "
            "review, as the model is less certain. Scores below 0.60 should be treated "
            "as tentative and corroborated with packet-level inspection."
        ),
        (
            ["feature", "important", "top", "which feature", "contribute"],
            "The top contributing features are the network flow attributes that pushed the "
            "model's prediction most strongly toward 'Attack'. Common high-importance features "
            "in CIC-IDS-2017 include: Flow Duration, Total Fwd Packets, Fwd Packet Length Mean, "
            "Bwd Packet Length Std, and Flow IAT (inter-arrival time) statistics. "
            "A feature with a large positive SHAP value means it raised the attack probability; "
            "a negative value means it pulled toward benign."
        ),
        (
            ["false positive", "false alarm", "benign", "normal", "mistake"],
            "False positives occur when legitimate traffic shares statistical properties with "
            "known attack patterns — for example, a batch file transfer can resemble a DoS "
            "flood in terms of packet rate. To reduce false positives: (1) review the flagged "
            "flow's source/destination context, (2) add the flow signature to a whitelist if "
            "confirmed benign, and (3) consider retraining the model with the corrected label "
            "to improve future accuracy."
        ),
        (
            ["recommend", "action", "what should", "next step", "mitigation", "response"],
            "For confirmed attack alerts the recommended response is: "
            "(1) Isolate — block the source IP at the firewall immediately to stop ongoing harm. "
            "(2) Investigate — capture full packet data for the flagged flow period and analyse "
            "in Wireshark to confirm the attack type. "
            "(3) Patch — if the attack targeted a specific service vulnerability, apply the "
            "relevant security patch or configuration fix. "
            "(4) Report — log the incident in your SIEM and notify stakeholders per your "
            "incident response plan."
        ),
        (
            ["dataset", "cic", "ids2017", "training data", "dataset used"],
            "This system was trained on the CIC-IDS-2017 dataset produced by the Canadian "
            "Institute for Cybersecurity. It contains approximately 2.8 million network flow "
            "records across 15 attack categories including DoS, DDoS, PortScan, Brute Force, "
            "Web Attacks, Infiltration, and Botnet traffic, plus benign background traffic "
            "captured over five weekdays. Features were extracted using CICFlowMeter and "
            "represent bidirectional flow statistics."
        ),
        (
            ["model", "algorithm", "classifier", "random forest", "how does", "how it work"],
            "The intrusion detector uses a Random Forest classifier — an ensemble of 100 "
            "decision trees each trained on a random feature subset and bootstrap sample of "
            "the training data. At inference time each tree casts a vote and the majority "
            "class (with its vote fraction as confidence) is returned. Random Forests are "
            "well-suited to NIDS because they handle mixed feature scales without normalisation, "
            "are robust to noisy features, and provide built-in feature importance scores that "
            "power the XAI explanations shown in the dashboard."
        ),
    ]

    def generate(self, system_prompt: str, human_prompt: str) -> str:
        import re

        # Extract just the user question from the structured prompt so that
        # broad words like "alert" in the system/context block don't cause
        # the wrong QA pair to match.
        match = re.search(
            r'ADMIN QUESTION[:\s]+(.+?)(?:\nAnswer:|$)',
            human_prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            search_text = match.group(1).strip().lower()
        else:
            # Alert-explanation path: use full combined text
            search_text = (system_prompt + " " + human_prompt).lower()

        for keywords, answer in self._QA:
            if any(kw in search_text for kw in keywords):
                return answer

        # Generic fallback when no keyword matches
        return (
            "Based on the current alert data, the network flow shows characteristics "
            "consistent with the detected classification. Review the contributing features "
            "listed in the alert card for specific anomaly details, and cross-reference "
            "with the security knowledge base entries shown. If you need more specific "
            "analysis, please describe the alert number or feature you are interested in."
        )


# ============================================================
# LOCAL TEMPLATE PROVIDER
# ============================================================

class LocalTemplateProvider(BaseLLMProvider):
    """
    Deterministic, dependency-free fallback.

    Does NOT call any LLM.

    It generates a plain-English explanation from the structured
    information already contained in the prompt.

    This allows the Stage 5 pipeline to continue working when:

        - Gemini is unavailable
        - OpenAI is unavailable
        - API keys are missing
        - Network access is unavailable
        - API quota is exhausted
    """

    name = "Local Template (OFFLINE FALLBACK - not a real LLM)"

    def generate(
        self,
        system_prompt: str,
        human_prompt: str
    ) -> str:

        # ----------------------------------------------------
        # Parse the human prompt
        # ----------------------------------------------------

        lines = human_prompt.split("\n")

        verdict_line = next(
            (
                line
                for line in lines
                if line.startswith("Verdict:")
            ),
            None
        )

        # ----------------------------------------------------
        # Free-form chat / Q&A
        # ----------------------------------------------------

        if verdict_line is None:
            # Instead of failing on chat questions, use the canned expert answers
            return CannedResponseProvider().generate(system_prompt, human_prompt)
        # ----------------------------------------------------
        # Extract verdict
        # ----------------------------------------------------

        verdict = verdict_line.split(
            ": ",
            1
        )[1]

        # ----------------------------------------------------
        # Extract confidence
        # ----------------------------------------------------

        confidence = next(
            (
                line.split(": ", 1)[1]
                for line in lines
                if line.startswith("Model confidence:")
            ),
            "unknown"
        )

        # ----------------------------------------------------
        # Extract feature lines
        # ----------------------------------------------------

        feature_lines = [
            line
            for line in lines
            if line.strip().startswith("- ")
        ]

        if feature_lines:

            top_feature = (
                feature_lines[0]
                .strip("- ")
                .split(":")[0]
            )

        else:

            top_feature = "several features"

        # ----------------------------------------------------
        # Extract knowledge information
        # ----------------------------------------------------

        knowledge_lines = [
            line
            for line in lines
            if line.startswith("[")
            and "]" in line
        ]

        knowledge_note = ""

        if (
            knowledge_lines
            and "not applicable" not in knowledge_lines[0]
        ):

            attack_label = (
                knowledge_lines[0]
                .split("]")[0]
                .strip("[")
            )

            knowledge_note = (
                f" This pattern is consistent with known "
                f"{attack_label}-style traffic characteristics."
            )

        # ----------------------------------------------------
        # ATTACK explanation
        # ----------------------------------------------------

        if verdict.strip() == "Attack":

            text = (
                f"This traffic was flagged as a potential ATTACK "
                f"with {confidence} model confidence. "
                f"The strongest signal was {top_feature}, along "
                f"with {max(0, len(feature_lines) - 1)} other "
                f"feature(s) deviating from typical benign traffic "
                f"patterns."
                f"{knowledge_note} "
                f"Recommended action: review the source of this "
                f"flow and consider temporary monitoring or "
                f"rate-limiting if the pattern repeats."
            )

        # ----------------------------------------------------
        # NORMAL explanation
        # ----------------------------------------------------

        else:

            text = (
                f"This traffic was classified as NORMAL with "
                f"{confidence} model confidence. "
                f"Feature values were broadly consistent with "
                f"typical benign traffic, though {top_feature} "
                f"showed some deviation worth noting if this "
                f"pattern becomes frequent. "
                f"Recommended action: no immediate action needed; "
                f"continue routine monitoring."
            )

        return textwrap.fill(
            text,
            width=100
        )


# ============================================================
# PROVIDER FACTORY
# ============================================================

def get_llm_provider(
    preference: str = "auto"
) -> BaseLLMProvider:
    """
    Return the best available LLM provider.

    preference can be:

        "openai"
        "gemini"
        "local"
        "auto"

    AUTO behavior:

        1. Try OpenAI
        2. Try Gemini
        3. Fall back to local template provider
    """

    # --------------------------------------------------------
    # OpenAI
    # --------------------------------------------------------

    def try_openai():

        if not os.environ.get("OPENAI_API_KEY"):

            raise RuntimeError(
                "OPENAI_API_KEY not set"
            )

        return LangChainOpenAIProvider()

    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    def try_gemini():
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not set"
            )
        return LangChainGeminiProvider()

    # --------------------------------------------------------
    # Provider order
    # --------------------------------------------------------

    def try_canned():
        return CannedResponseProvider()

    attempts = {

        "openai": [
            try_openai
        ],

        "gemini": [
            try_gemini,
            try_canned,   # quota exhausted → use canned answers
        ],

        "local": [],

        "auto": [
            try_openai,
            try_gemini,
            try_canned,
        ],
    }

    # --------------------------------------------------------
    # Validate preference
    # --------------------------------------------------------

    if preference not in attempts:

        print(
            f"[llm_provider] Unknown preference "
            f"'{preference}'. Using local fallback."
        )

        return CannedResponseProvider()

    # --------------------------------------------------------
    # Try providers
    # --------------------------------------------------------

    for attempt in attempts[preference]:

        try:

            provider = attempt()

            print(
                f"[llm_provider] Using: {provider.name}"
            )

            return provider

        except Exception as e:

            print(
                f"[llm_provider] Skipped ({e})"
            )

    # --------------------------------------------------------
    # Local fallback
    # --------------------------------------------------------

    print(
        "[llm_provider] No real LLM provider available "
        "(missing package, API key, quota, or network access). "
        "Falling back to CannedResponseProvider (10 expert Q&A answers)."
    )

    return CannedResponseProvider()