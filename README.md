# 🗞️ Fake News Detector

A machine learning web app that classifies news articles as **Real**, **Fake**, or **Uncertain** — with an extractive summary and plain-English explanation for every prediction.

🔗 **Live Demo → [fake-news-detector-h1zk.onrender.com](https://fake-news-detector-h1zk.onrender.com/)**

---

## Screenshot

> *(Add a screenshot of the app here — drag an image into this README on GitHub)*

---

## Features

- **Paste text, enter a URL, upload a .txt file, or run batch analysis** on up to 20 articles at once
- **URL article fetcher** — paste any news URL and the app extracts and analyses the article automatically
- **Extractive summarisation** — top 3 most important sentences pulled from the article automatically
- **Plain-English verdict explanation** — tells you *why* the model made its decision, not just what it decided
- **Confidence gauge** with real vs. fake probability breakdown
- **Style signal analysis** — caps ratio, exclamation count, word count, character count
- **Session history** with CSV export
- **Smart pre-filters** — detects fact-checks, non-news input, and short texts before hitting the model
- **Rule-based layer** — conspiracy keyword detection and real-source markers override low-confidence predictions

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost + TF-IDF (10,000 features) |
| Dataset | WELFake (63K articles) + Indian News Dataset (3.7K articles) |
| URL Fetching | newspaper3k + BeautifulSoup fallback |
| Summarisation | NLTK extractive (offline, no API) |
| Backend | Flask + Gunicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render |

---

## How It Works

```
Article text / URL
     │
     ▼
Pre-filters (NOT NEWS / fact-check / short text)
     │
     ▼
TF-IDF Vectorisation (10,000 features)
     │
     ▼
XGBoost Classifier
     │
     ▼
Rule-based layer:
  conspiracy keywords  →  boost FAKE
  official source markers + short text  →  reduce FAKE
  casual text, no signals  →  cap at UNCERTAIN
     │
     ▼
Threshold logic:
  prob_fake > 0.92  →  FAKE
  prob_real > 0.65  →  REAL
  else              →  UNCERTAIN
     │
     ▼
Extractive summary + rule-based explanation
     │
     ▼
JSON response → rendered in browser
```

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 96.35% |
| ROC-AUC | 0.9943 |
| CV Score | 96.25% |
| Dataset | WELFake + Indian News (65,868 articles) |
| Features | 10,000 TF-IDF |
| Trained | 2026-03-29 |

---

## Project Structure

```
fake-news-detection/
├── app.py                  # Flask app — routes, prediction logic, summarisation
├── main.py                 # Model training script
├── data_utils.py           # Dataset loader, text preprocessing, URL fetcher
├── merge_and_retrain.py    # Script to merge datasets and retrain model
├── requirements.txt        # Python dependencies
├── build.sh                # Render build script
├── outputs/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json # Accuracy, AUC, training date
├── templates/
│   └── index.html          # Single-page frontend
└── data/
    ├── welfake/
    │   └── WELFake_Dataset.csv
    └── indian/
        └── news_dataset.csv
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/mokshijain22/fake-news-detector.git
cd fake-news-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the datasets**

- [WELFake Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) → place at `data/welfake/WELFake_Dataset.csv`
- [Indian News Dataset](https://www.kaggle.com/datasets/imbikramsaha/fake-real-news) → place at `data/indian/news_dataset.csv`

**4. Train the model**
```bash
python merge_and_retrain.py
```
This saves `xgb_model.pkl`, `tfidf_vectorizer.pkl`, and `model_metadata.json` to `outputs/`.

**5. Start the server**
```bash
python app.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Limitations

- Optimised for **formal English-language news articles** (politics, finance, world news)
- Results may be unreliable for opinion pieces, blogs, viral/trending stories, or non-Western news sources
- Some news sites (e.g. Times of India, Hindustan Times) block URL fetching — paste the article text directly as a workaround
- The model reflects biases present in its training data — always verify with a trusted fact-checking source such as [Snopes](https://www.snopes.com), [FactCheck.org](https://www.factcheck.org), or [AFP Fact Check](https://factcheck.afp.com)

---

## Roadmap

- [x] Retrain with Indian news dataset
- [x] URL article fetching with newspaper3k
- [x] Rule-based layer for conspiracy and source detection
- [ ] Re-enable SHAP word-level explanations
- [ ] Browser extension to highlight fake news inline
- [ ] Telegram / WhatsApp bot interface

---

## Author

Built by **Mokshi Jain**

---

## License

MIT