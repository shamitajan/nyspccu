# main.py
import os
import re
import json
import requests
from typing import Optional
from flask import Request, jsonify

# For Cloud Functions we use functions-framework entry point `main`
# See requirements.txt for functions-framework dependency.

# Configuration
OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "openrouter/auto"
LLM_TEMPERATURE = 0.12
LLM_MAX_TOKENS = 150

# Knowledge base (small, editable). Replace or expand by loading from GCS.
KB_SNIPPETS = [
    "NYSP Computer Crimes Unit provides outreach, education, and assistance with digital evidence for prosecution.",
    "Before filing a complaint, collect incident date/time, screenshots, email headers, and masked transaction references. Do not include full account numbers or SSNs.",
    "Basic security: use unique passwords, enable multi-factor authentication, keep systems updated, and avoid clicking unknown links."
]

# Topic gating tokens
CYBER_KEYWORDS = [
    "cyber", "security", "malware", "ransomware", "phish", "phishing", "breach",
    "password", "mfa", "2fa", "encryption", "vulnerability", "forensic", "incident",
    "report", "ransom", "intrusion", "exploit", "ddos", "botnet", "fraud"
]
NEGATIVE_KEYWORDS = [
    "weather", "movie", "song", "music", "recipe", "colour", "color", "sky",
    "football", "cricket"
]
INTENT_WORDS = ["how", "how to", "what to do", "report", "enable", "secure", "detect", "mitigate", "protect"]

# Compile regex for efficiency
CYBER_RE = re.compile("|".join(re.escape(x) for x in CYBER_KEYWORDS), re.I)
NEG_RE = re.compile("|".join(re.escape(x) for x in NEGATIVE_KEYWORDS), re.I)

# Simple PII patterns
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{6,10}\b")
}

REFUSAL_MESSAGE = "I only answer cybersecurity-related questions. Please ask a cyber-related question."
PII_MESSAGE = "I cannot process or store sensitive personal data. Please remove personal details and try again."

def contains_pii(text: str) -> bool:
    if not text:
        return False
    for p in PII_PATTERNS.values():
        if p.search(text):
            return True
    return False

def is_on_topic(text: str):
    """
    Returns (bool on_topic, dict details)
    """
    details = {"negative": False, "keyword": False, "intent": False}
    if not text:
        return False, details
    if NEG_RE.search(text):
        details["negative"] = True
        return False, details
    if CYBER_RE.search(text):
        details["keyword"] = True
    low = text.lower()
    if any(w in low for w in INTENT_WORDS):
        details["intent"] = True
    return (details["keyword"] or details["intent"]), details

def simple_retrieve(query: str, k: int = 2):
    """
    Lightweight retrieval: returns up to k KB snippets that contain a token in common,
    otherwise returns top-k default snippets. This is intentionally simple for Cloud Functions.
    """
    query_l = (query or "").lower()
    hits = []
    for s in KB_SNIPPETS:
        # simple substring match on keywords to rank relevance
        if any(tok in s.lower() for tok in query_l.split()):
            hits.append(s)
            if len(hits) >= k:
                break
    if not hits:
        # fallback to the first k snippets
        hits = KB_SNIPPETS[:k]
    return hits

def call_openrouter(prompt: str) -> Optional[str]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a concise cybersecurity assistant. Answer in 1-3 short sentences. Do not provide legal advice. Do not request or record PII."},
            {"role": "user", "content": prompt}
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS
    }
    r = requests.post(OPENROUTER_API, headers=headers, json=body, timeout=25)
    r.raise_for_status()
    j = r.json()
    # handle typical OpenRouter shape
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        # fallback if schema differs
        return j.get("choices", [{}])[0].get("message", {}).get("content")

def trim_sentences(text: str, n: int = 3) -> str:
    # keep first n sentences
    if not text:
        return ""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return " ".join(parts[:n])

# Cloud Functions entrypoint
def main(request: Request):
    """
    Handle Dialogflow webhook requests.
    Expects Dialogflow v2 webhook format in JSON and returns JSON containing 'fulfillmentText'.
    Also serves a health endpoint at path /health when tested with GET.
    """
    # Health check
    if request.method == "GET":
        # allow a simple health check
        path = request.path or ""
        if path.endswith("/health") or path == "/":
            return ("ok", 200)
        return ("method not allowed", 405)

    try:
        req = request.get_json(force=True)
    except Exception:
        return jsonify({"fulfillmentText": "Invalid request format."})

    # Dialogflow v2 queryResult -> queryText
    query_result = req.get("queryResult", {})
    user_text = query_result.get("queryText", "") or ""
    user_text = user_text.strip()

    # PII guard
    if contains_pii(user_text):
        return jsonify({"fulfillmentText": PII_MESSAGE})

    # Topic gating
    on_topic, details = is_on_topic(user_text)
    if not on_topic:
        return jsonify({"fulfillmentText": REFUSAL_MESSAGE})

    # Retrieval
    retrieved = simple_retrieve(user_text, k=2)
    grounding = "\n\n".join(retrieved)

    # Compose prompt for LLM (if available)
    prompt = f"Context: {grounding}\n\nQuestion: {user_text}\n\nAnswer concisely in 1-3 short sentences. If user must file a report, say so and provide the NYSP link: https://troopers.ny.gov/computer-crimes"

    answer_text = None
    try:
        llm_resp = call_openrouter(prompt)
        if llm_resp:
            answer_text = trim_sentences(llm_resp, 3)
    except Exception:
        # If LLM fails, fall back to retrieval-only answer
        answer_text = None

    if not answer_text:
        # fallback: use first retrieved snippet's first sentence
        first = retrieved[0] if retrieved else "No guidance available."
        answer_text = trim_sentences(first, 1)

    # Final response
    return jsonify({"fulfillmentText": answer_text})
