"""
Fake News Detector — Flask Web App  (v2.4)

New in v2.4:
  • Extractive article summary using NLTK (no API needed)
  • Rule-based explanation of prediction using model signals
  • Raised FAKE threshold to 0.85 to reduce false positives
"""

import os
import io
import csv
import base64
import uuid
import json
import re
import math
from collections import Counter
from datetime import datetime

import numpy as np

import joblib

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

from flask import Flask, render_template, request, jsonify, session, Response
from data_utils import preprocess_text, fetch_article_from_url, extract_extra_features

app = Flask(__name__)

# ── Secret key ────────────────────────────────────────────────────────────────
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_fakenews_secret_2024")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH    = "outputs/xgb_model.pkl"
TFIDF_PATH    = "outputs/tfidf_vectorizer.pkl"
METADATA_PATH = "outputs/model_metadata.json"

MIN_CHARS = 50
MAX_CHARS = 50_000

# ── Load model & vectorizer ───────────────────────────────────────────────────
print("Loading model...")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise SystemExit(
        f"[STARTUP ERROR] Model file not found at '{MODEL_PATH}'. "
        "Run main.py to train and save the model first."
    )
try:
    tfidf = joblib.load(TFIDF_PATH)
except FileNotFoundError:
    raise SystemExit(
        f"[STARTUP ERROR] Vectorizer file not found at '{TFIDF_PATH}'. "
        "Run main.py to train and save the vectorizer first."
    )
print("Model loaded successfully.")

# SHAP disabled — to re-enable uncomment:
# import shap
# explainer = shap.TreeExplainer(model)


# ── Text signal helpers ───────────────────────────────────────────────────────

def is_fact_check(text: str) -> bool:
    t = text.lower()
    keywords = [
        "no evidence", "not supported", "experts say", "scientists say",
        "no scientific evidence", "claim is false", "misleading",
        "debunked", "false claim",
    ]
    return any(k in t for k in keywords)


def is_short_news(text: str) -> bool:
    return len(text.split()) < 20


def is_not_news(text: str) -> bool:
    t = text.lower().strip()
    if len(t.split()) < 8:
        if any(w in t for w in ["hii", "hey", "hello", "help", "please", "can you"]):
            return True
    for pattern in [r"\bhii\b", r"\bhey\b", r"\bhello\b", r"\bcan you\b", r"\bhelp me\b", r"\bpls\b", r"\bplease\b"]:
        if re.search(pattern, t):
            return True
    if any(x in t for x in ["i think", "in my opinion", "i feel"]):
        return True
    return False


# ── Extractive summarizer (NLTK, no API) ─────────────────────────────────────

def extractive_summary(text: str, num_sentences: int = 3) -> str:
    """
    Pick the top N most representative sentences using TF-IDF-style word scoring.
    Runs entirely offline with NLTK.
    """
    try:
        stop_words = set(stopwords.words("english"))
        sentences  = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Score words by frequency, ignoring stopwords
        words      = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stop_words]
        freq       = Counter(words)
        max_freq   = max(freq.values()) if freq else 1
        freq       = {w: f / max_freq for w, f in freq.items()}

        # Score each sentence by sum of word scores, normalised by length
        scores = {}
        for i, sentence in enumerate(sentences):
            words_in = [w.lower() for w in word_tokenize(sentence) if w.isalpha()]
            if len(words_in) < 3:
                continue
            scores[i] = sum(freq.get(w, 0) for w in words_in) / math.log(len(words_in) + 1)

        # Pick top N by score, return in original order
        top_indices = sorted(sorted(scores, key=scores.get, reverse=True)[:num_sentences])
        return " ".join(sentences[i] for i in top_indices)

    except Exception:
        # Fallback — just return first 3 sentences
        sentences = text.split(".")[:num_sentences]
        return ". ".join(s.strip() for s in sentences if s.strip()) + "."


# ── Rule-based explanation ────────────────────────────────────────────────────

def generate_explanation(prediction: str, prob_fake: float, prob_real: float,
                          confidence: float, stats: dict) -> str:
    """
    Generate a human-readable explanation based on model signals.
    No API needed — purely rule-based using the stats we already compute.
    """
    caps        = stats.get("caps_ratio", 0)
    exclaims    = stats.get("exclaim_count", 0)
    word_count  = stats.get("word_count", 0)
    reasons     = []

    if prediction == "FAKE":
        reasons.append(f"The model assigned a {prob_fake:.0f}% probability of this being fake news.")
        if caps > 5:
            reasons.append(f"Unusually high use of capital letters ({caps:.1f}%) — a common pattern in sensationalist writing.")
        if exclaims > 2:
            reasons.append(f"Contains {exclaims} exclamation mark(s), which is typical of emotionally manipulative content.")
        if word_count < 100:
            reasons.append("The article is very short, which is common in clickbait and misleading posts.")
        if prob_fake > 90:
            reasons.append("The language pattern strongly matches known fake news articles in the training data.")
        if not reasons[1:]:
            reasons.append("The vocabulary and writing style closely matches patterns found in fake news sources.")

    elif prediction == "REAL":
        reasons.append(f"The model assigned a {prob_real:.0f}% probability of this being real news.")
        if caps <= 2:
            reasons.append("The article uses measured, formal language with minimal capitalisation.")
        if exclaims == 0:
            reasons.append("No exclamation marks detected — consistent with professional journalism.")
        if word_count >= 100:
            reasons.append(f"Substantial article length ({word_count} words) suggests detailed, sourced reporting.")
        if prob_real > 80:
            reasons.append("The writing style closely matches legitimate news sources in the training data.")
        if not reasons[1:]:
            reasons.append("The vocabulary and structure are consistent with credible news reporting.")

    elif prediction == "UNCERTAIN":
        reasons.append(f"The model could not confidently classify this article (confidence: {confidence:.0f}%).")
        reasons.append("The writing style contains mixed signals — some patterns match real news, others match fake news.")
        if caps > 3:
            reasons.append(f"Elevated capitalisation ({caps:.1f}%) adds uncertainty.")
        reasons.append("Consider verifying this article through a trusted fact-checking source.")

    elif prediction == "NOT NEWS":
        reasons.append("The input does not appear to be a news article.")
        reasons.append("It contains conversational or opinion-based language not suitable for fake news classification.")

    else:  # REAL via fact-check path
        reasons.append("The article contains fact-checking language (e.g. 'experts say', 'debunked', 'no evidence').")
        reasons.append("Such language is typically associated with credible fact-checking journalism.")

    return " ".join(reasons)


# ── Build feature vector (TF-IDF only — matches saved model) ─────────────────

# Fixed — matches new model
import pandas as pd
import scipy.sparse as sp

def build_feature_vector(text: str):
    clean     = preprocess_text(text)
    tfidf_vec = tfidf.transform([clean])
    extra     = extract_extra_features(pd.Series([text]))
    return tfidf_vec


# ── SHAP plot (disabled) ──────────────────────────────────────────────────────

def get_shap_plot(text: str):
    if "explainer" not in globals():
        return None
    import shap
    vec           = build_feature_vector(text).toarray()
    sv            = explainer.shap_values(vec, check_additivity=False)
    feature_names = list(tfidf.get_feature_names_out())
    shap.force_plot(
        explainer.expected_value, sv[0], vec[0],
        feature_names=feature_names, matplotlib=True, show=False, text_rotation=15,
    )
    fig = plt.gcf()
    fig.set_size_inches(14, 3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Core prediction logic ─────────────────────────────────────────────────────

def predict_text(text: str, include_shap: bool = True, include_summary: bool = True) -> dict:
    vec  = build_feature_vector(text)
    prob = [float(p) for p in model.predict_proba(vec)[0]]

    # ── Rule-based layer ──────────────────────────────────────────────────────
    FAKE_KEYWORDS = [
        "5g", "covid hoax", "illuminati", "deep state", "microchip", "bill gates",
        "new world order", "chemtrail", "flat earth", "crisis actor", "false flag",
        "george soros", "antifa plot", "secret agenda", "plandemic", "scamdemic",
        "they dont want you to know", "mainstream media lies", "wake up sheeple"
    ]
    REAL_MARKERS = [
        "according to", "said in a statement", "confirmed by", "reported by",
        "official", "ministry", "government", "rbi", "sebi", "supreme court",
        "high court", "election commission", "press trust", "ani", "pti"
    ]
    text_lower = text.lower()
    has_conspiracy = any(kw in text_lower for kw in FAKE_KEYWORDS)
    has_real_marker = any(kw in text_lower for kw in REAL_MARKERS)

    # Boost FAKE if conspiracy keywords found
    if has_conspiracy:
        prob[1] = max(prob[1], 0.93)
        prob[0] = 1 - prob[1]

    # Reduce FAKE confidence for short casual text with real markers
    word_count = len(text.split())
    if word_count < 60 and has_real_marker and not has_conspiracy:
        prob[1] = min(prob[1], 0.45)
        prob[0] = 1 - prob[1]

    # Short casual text with no signals → cap at UNCERTAIN
    exclaims = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if not has_conspiracy and not has_real_marker and exclaims == 0 and caps_ratio < 0.05:
        prob[1] = min(prob[1], 0.60)
        prob[0] = 1 - prob[1]

    # Final thresholds

    # Final thresholds
    if prob[1] > 0.92:
        prediction = "FAKE"
    elif prob[0] > 0.65:
        prediction = "REAL"
    else:
        prediction = "UNCERTAIN"

    label    = 1 if prediction == "FAKE" else (0 if prediction == "REAL" else -1)
    shap_img = get_shap_plot(text) if include_shap else None

    caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim_count = text.count("!")
    word_count    = len(text.split())

    stats = {
        "word_count":    word_count,
        "char_count":    len(text),
        "caps_ratio":    round(caps_ratio * 100, 1),
        "exclaim_count": exclaim_count,
    }

    # Generate summary and explanation
    summary     = extractive_summary(text)     if include_summary else None
    explanation = generate_explanation(
        prediction,
        round(prob[1] * 100, 1),
        round(prob[0] * 100, 1),
        round(max(prob) * 100, 1),
        stats,
    )

    return {
        "id":          str(uuid.uuid4())[:8],
        "prediction":  prediction,
        "label":       label,
        "prob_real":   round(prob[0] * 100, 1),
        "prob_fake":   round(prob[1] * 100, 1),
        "confidence":  round(max(prob) * 100, 1),
        "shap_img":    shap_img,
        "snippet":     text[:120] + "…" if len(text) > 120 else text,
        "timestamp":   datetime.now().strftime("%H:%M:%S"),
        "summary":     summary,
        "explanation": explanation,
        "stats":       stats,
    }


def _session_push(result: dict):
    history = session.get("history", [])
    history.insert(0, {
        "id":         result["id"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "snippet":    result["snippet"],
        "timestamp":  result["timestamp"],
    })
    session["history"] = history[:10]
    session.modified = True


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = ""
    if "file" in request.files and request.files["file"].filename:
        text = request.files["file"].read().decode("utf-8", errors="ignore")
    elif request.form.get("text"):
        text = request.form.get("text").strip()

    if not text:
        return jsonify({"error": "No text provided."}), 400
    if len(text) < MIN_CHARS:
        return jsonify({"error": f"Article too short — need at least {MIN_CHARS} characters."}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Article too long — maximum {MAX_CHARS:,} characters accepted."}), 400

    # Step 1: NOT NEWS
    if is_not_news(text):
        stats  = {"word_count": len(text.split()), "char_count": len(text), "caps_ratio": 0, "exclaim_count": text.count("!")}
        result = {
            "id": str(uuid.uuid4())[:8], "prediction": "NOT NEWS", "label": -1,
            "prob_real": 0, "prob_fake": 0, "confidence": 100.0, "shap_img": None,
            "snippet":    text[:120] + "…" if len(text) > 120 else text,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "summary":    None,
            "explanation": "The input does not appear to be a news article. It contains conversational language not suitable for fake news classification.",
            "stats": stats,
        }
        _session_push(result)
        return jsonify(result)

    # Step 2: FACT-CHECK
    if is_fact_check(text):
        stats  = {"word_count": len(text.split()), "char_count": len(text), "caps_ratio": 0, "exclaim_count": text.count("!")}
        result = {
            "id": str(uuid.uuid4())[:8], "prediction": "REAL", "label": 0,
            "prob_real": 70.0, "prob_fake": 30.0, "confidence": 70.0, "shap_img": None,
            "snippet":   text[:120] + "…" if len(text) > 120 else text,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "summary":   extractive_summary(text),
            "explanation": "The article contains fact-checking language (e.g. 'experts say', 'debunked', 'no evidence'). Such language is typically associated with credible fact-checking journalism.",
            "stats": stats,
        }
        _session_push(result)
        return jsonify(result)

    # Step 3: SHORT TEXT
    if is_short_news(text):
        vec   = build_feature_vector(text)
        proba = [float(p) for p in model.predict_proba(vec)[0]]
        # Apply same rule-based layer
        text_lower = text.lower()
        FAKE_KEYWORDS = ["5g", "covid hoax", "illuminati", "deep state", "microchip",
            "bill gates", "new world order", "chemtrail", "flat earth", "crisis actor",
            "false flag", "plandemic", "scamdemic"]
        REAL_MARKERS = ["according to", "said in a statement", "confirmed by",
            "reported by", "official", "ministry", "government", "rbi", "sebi",
            "supreme court", "high court", "election commission", "ani", "pti"]
        has_conspiracy = any(kw in text_lower for kw in FAKE_KEYWORDS)
        has_real_marker = any(kw in text_lower for kw in REAL_MARKERS)
        if has_conspiracy:
            proba[1] = max(proba[1], 0.93)
            proba[0] = 1 - proba[1]
        elif has_real_marker and not has_conspiracy:
            proba[1] = min(proba[1], 0.45)
            proba[0] = 1 - proba[1]
            # Short casual text with no conspiracy or sensationalism → UNCERTAIN
        exclaims = text.count("!")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if not has_conspiracy and not has_real_marker and exclaims == 0 and caps_ratio < 0.05:
            proba[1] = min(proba[1], 0.60)
            proba[0] = 1 - proba[1]
        if proba[1] > 0.92:
            prediction = "FAKE"
        elif proba[0] > 0.65:
            prediction = "REAL"
        else:
            prediction = "UNCERTAIN"
        label         = 1 if prediction == "FAKE" else (0 if prediction == "REAL" else -1)
        adjusted_conf = round(max(proba) * 0.6 * 100, 1)
        stats         = {"word_count": len(text.split()), "char_count": len(text), "caps_ratio": 0, "exclaim_count": text.count("!")}
        result = {
            "id": str(uuid.uuid4())[:8], "prediction": prediction, "label": label,
            "prob_real":  round(proba[0] * 100, 1), "prob_fake": round(proba[1] * 100, 1),
            "confidence": adjusted_conf, "shap_img": None,
            "snippet":    text[:120] + "…" if len(text) > 120 else text,
            "timestamp":  datetime.now().strftime("%H:%M:%S"),
            "summary":    None,
            "explanation": generate_explanation(prediction, round(proba[1]*100,1), round(proba[0]*100,1), adjusted_conf, stats) + " Note: confidence is reduced for short texts.",
            "stats": stats,
        }
        _session_push(result)
        return jsonify(result)

    # Step 4: NORMAL
    result = predict_text(text, include_shap=True, include_summary=True)
    _session_push(result)
    return jsonify(result)


@app.route("/predict/url", methods=["POST"])
def predict_url():
    data = request.get_json(silent=True) or {}
    url  = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400
    text, fetch_error = fetch_article_from_url(url)
    if fetch_error:
        return jsonify({"error": fetch_error}), 422
    if len(text) < MIN_CHARS:
        return jsonify({"error": "Fetched article is too short to classify."}), 422
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    result = predict_text(text, include_shap=False, include_summary=True)
    result["source_url"] = url
    _session_push(result)
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    data     = request.get_json(silent=True) or {}
    articles = data.get("articles", [])
    if not articles:
        return jsonify({"error": "No articles provided."}), 400
    if len(articles) > 20:
        return jsonify({"error": "Batch limit is 20 articles per request."}), 400
    results = []
    for text in articles:
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            results.append({"error": "Too short or invalid."})
            continue
        results.append(predict_text(text[:MAX_CHARS], include_shap=False, include_summary=False))
    return jsonify({"results": results, "count": len(results)})


@app.route("/history")
def history():
    return jsonify(session.get("history", []))


@app.route("/export")
def export_history():
    history = session.get("history", [])
    if not history:
        return jsonify({"error": "No history to export."}), 404
    buf    = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "timestamp", "prediction", "confidence", "snippet"])
    writer.writeheader()
    writer.writerows(history)
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.route("/stats")
def stats():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metadata not found. Run main.py to train the model."}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)