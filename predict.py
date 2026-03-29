"""
predict.py — Run inference on new news articles using the trained model.  (v2.1)

Usage:
    python predict.py --text "Your news article text here"
    python predict.py --file article.txt
    python predict.py --url  https://example.com/article   # NEW
"""

import argparse
import joblib
import os
from data_utils import preprocess_text, fetch_article_from_url

MODEL_PATH = "outputs/xgb_model.pkl"
TFIDF_PATH = "outputs/tfidf_vectorizer.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run main.py first to train the model.")
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)
    return model, tfidf


def predict(text: str, model, tfidf) -> dict:
    clean = preprocess_text(text)
    vec   = tfidf.transform([clean])
    label = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0]

    # Style signals (printed for transparency)
    import re
    words         = text.split()
    n_words       = max(len(words), 1)
    caps_ratio    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclaim_count = text.count("!")

    return {
        "prediction":   "FAKE" if label == 1 else "REAL",
        "confidence":   f"{max(prob)*100:.1f}%",
        "prob_real":    f"{prob[0]*100:.1f}%",
        "prob_fake":    f"{prob[1]*100:.1f}%",
        "word_count":   n_words,
        "caps_ratio":   f"{caps_ratio*100:.1f}%",
        "exclamations": exclaim_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Fake News Detector v2.1")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Article text to classify")
    group.add_argument("--file", type=str, help="Path to a .txt file")
    group.add_argument("--url",  type=str, help="URL of a news article to fetch & classify")  # NEW
    args = parser.parse_args()

    model, tfidf = load_model()

    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r") as f:
            text = f.read()
    else:
        # ── NEW: URL mode ──────────────────────────────────────────────────────
        print(f"Fetching article from: {args.url}")
        text, err = fetch_article_from_url(args.url)
        if err:
            print(f"Error: {err}")
            return

    if len(text) < 50:
        print("Error: Article too short to classify (minimum 50 characters).")
        return

    result = predict(text, model, tfidf)

    verdict_icon = "🔴 FAKE" if result["prediction"] == "FAKE" else "🟢 REAL"
    print("\n" + "="*50)
    print(f"  Verdict     : {verdict_icon}")
    print(f"  Confidence  : {result['confidence']}")
    print(f"  Prob Real   : {result['prob_real']}")
    print(f"  Prob Fake   : {result['prob_fake']}")
    print(f"  Words       : {result['word_count']}")
    print(f"  CAPS ratio  : {result['caps_ratio']}")
    print(f"  Exclamations: {result['exclamations']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()