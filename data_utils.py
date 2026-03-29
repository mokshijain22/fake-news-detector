"""
data_utils.py — Data loading, preprocessing, and feature extraction
Dataset: WELFake (72K articles, 4 combined sources)
Labels : 1 = Fake, 0 = Real
"""

import re
import string
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from newspaper import Config

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Download NLTK data if not already present ─────────────────────────────────
for pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

STOPWORDS   = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()


# ── 1. Dataset loader ─────────────────────────────────────────────────────────

def load_welfake_dataset(path: str = "data/welfake/WELFake_Dataset.csv") -> pd.DataFrame:
    """
    Load and clean the WELFake dataset.

    Columns used:
      title  — article headline
      text   — article body
      label  — 1 = Fake, 0 = Real

    Returns a DataFrame with columns: text, label
    """
    df = pd.read_csv(path)

    # Drop rows with missing text or label
    df = df.dropna(subset=["text", "label"])

    # Combine title + body for richer signal
    df["title"] = df["title"].fillna("")
    df["text"]  = df["title"] + " " + df["text"]

    # Keep only what we need
    df = df[["text", "label"]].copy()
    df["label"] = df["label"].astype(int)

    # Drop duplicates and very short articles
    df = df.drop_duplicates(subset=["text"])
    df = df[df["text"].str.split().str.len() >= 10]
    df = df.reset_index(drop=True)

    print(f"  WELFake loaded : {len(df):,} articles")
    print(f"  Real (0)       : {(df['label'] == 0).sum():,}")
    print(f"  Fake (1)       : {(df['label'] == 1).sum():,}")

    return df


# ── Backwards-compatible alias (keeps main.py working if still references ISOT)
def load_isot_dataset(*args, **kwargs) -> pd.DataFrame:
    print("[INFO] load_isot_dataset() redirected to load_welfake_dataset()")
    return load_welfake_dataset()


# ── 2. Text preprocessing ─────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw article string:
      • Lowercase
      • Remove URLs, HTML tags, punctuation, digits
      • Remove stopwords
      • Lemmatize
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                    # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                 # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()              # extra whitespace

    tokens = [
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in STOPWORDS and len(w) > 2
    ]
    return " ".join(tokens)


# ── 3. Style feature extraction ───────────────────────────────────────────────

def extract_extra_features(texts: pd.Series) -> np.ndarray:
    """
    Extract 5 hand-crafted style features that capture sensationalist writing:

      0. caps_ratio       — fraction of uppercase characters
      1. exclaim_density  — exclamation marks per 100 words
      2. url_density      — URLs per 100 words
      3. word_count       — total word count (log-scaled)
      4. avg_word_len     — average word length in characters

    Returns a (n, 5) float32 numpy array.
    """
    features = []

    for text in texts:
        if not isinstance(text, str) or len(text) == 0:
            features.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue

        words      = text.split()
        word_count = max(len(words), 1)

        caps_ratio      = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclaim_density = text.count("!") / word_count * 100
        url_density     = len(re.findall(r"http\S+|www\S+", text)) / word_count * 100
        log_word_count  = np.log1p(word_count)
        avg_word_len    = np.mean([len(w) for w in words]) if words else 0.0

        features.append([
            caps_ratio,
            exclaim_density,
            url_density,
            log_word_count,
            avg_word_len,
        ])

    return np.array(features, dtype=np.float32)


# ── 4. URL article fetcher ────────────────────────────────────────────────────

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def fetch_article_from_url(url: str) -> tuple[str, str | None]:
    """
    Fetch and extract article text from a URL.

    Returns:
      (text, None)        — on success
      ("",  error_msg)    — on failure
    """
    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return "", "Request timed out — the site took too long to respond."
    except requests.exceptions.ConnectionError:
        return "", "Could not connect to the URL. Check the link and try again."
    except requests.exceptions.HTTPError as e:
        return "", f"HTTP error fetching URL: {e}"
    except Exception as e:
        return "", f"Unexpected error fetching URL: {e}"

    try:
        # Try newspaper3k with browser config first
        config = Config()
        config.browser_user_agent = BROWSER_HEADERS["User-Agent"]
        config.request_timeout = 15
        art = Article(url, config=config)
        art.download(input_html=response.text)  # reuse already-fetched HTML
        art.parse()
        text = art.text.strip()

        if len(text) < 100:
            # fallback to BeautifulSoup
            soup = BeautifulSoup(response.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            article = soup.find("article") or soup.find("main") or soup.find("body")
            if not article:
                return "", "Could not extract article content from this page."
            paragraphs = [p.get_text(separator=" ") for p in article.find_all("p") if len(p.get_text()) > 40]
            text = " ".join(paragraphs)
            text = re.sub(r"\s+", " ", text).strip()

        if len(text) < 100:
            return "", "Extracted text is too short — the page may be paywalled or JS-rendered."
        return text, None

    except Exception as e:
        return "", f"Error parsing page content: {e}"