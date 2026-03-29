#!/usr/bin/env bash
# build.sh — Render build script
# Runs once at deploy time, before gunicorn starts.
# Since .pkl files are committed to the repo, no training is needed.
# This script only handles NLTK data downloads and directory setup.

set -e  # exit immediately on any error

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Downloading NLTK data..."
python - <<'EOF'
import nltk
nltk.download("stopwords",  quiet=True)
nltk.download("punkt",      quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("punkt_tab",  quiet=True)
print("NLTK data ready.")
EOF

echo "==> Verifying model files exist..."
python - <<'EOF'
import sys, os

required = [
    "outputs/xgb_model.pkl",
    "outputs/tfidf_vectorizer.pkl",
]
missing = [f for f in required if not os.path.exists(f)]

if missing:
    print("ERROR: The following model files are missing from the repo:")
    for f in missing:
        print(f"  - {f}")
    print("\nFix: commit these files and push again.")
    sys.exit(1)

print("Model files found — ready to serve.")
EOF

echo "==> Build complete."

# Add this to build.sh after pip install
echo "==> Checking dataset..."
python -c "
import os
if not os.path.exists('data/welfake/WELFake_Dataset.csv'):
    print('WARNING: WELFake dataset missing — model must be pre-trained')
"