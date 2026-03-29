"""
Visualization functions for Fake News Detection project.  (v2.1 — upgraded)

Changes vs v2.0:
  • Fixed: replaced __import__() anti-pattern with a proper top-level import
  • Added: plot_word_cloud()  — side-by-side word clouds for REAL vs FAKE
  • Added: plot_cv_scores()   — bar chart of 5-fold CV scores
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix   # ✅ fixed: proper top-level import

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
})
PALETTE = {"Real": "#1D9E75", "Fake": "#D85A30"}


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    cm  = confusion_matrix(y_true, y_pred)   # ✅ clean import — no __import__
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"],
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 2. ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_score, auc: float, output_dir: str):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#534AB7", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random baseline")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#534AB7")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 3. Top TF-IDF Features ────────────────────────────────────────────────────

def plot_top_features(model, tfidf, output_dir: str, top_n: int = 20):
    importances   = model.feature_importances_[:tfidf.max_features]  # TF-IDF cols only
    feature_names = np.array(tfidf.get_feature_names_out())
    top_idx       = np.argsort(importances)[::-1][:top_n]
    top_names     = feature_names[top_idx]
    top_scores    = importances[top_idx]

    FAKE_KEYWORDS = {"secret", "deep", "shock", "breaking", "urgent", "exclusive",
                     "exposed", "conspiracy", "globalist", "hoax"}
    colors = [
        "#D85A30" if any(kw in n for kw in FAKE_KEYWORDS) else "#1D9E75"
        for n in top_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names[::-1], top_scores[::-1], color=colors[::-1], edgecolor="none")
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title(f"Top {top_n} Most Important Words", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=10)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#D85A30", label="Fake-leaning"),
        Patch(facecolor="#1D9E75", label="Real-leaning"),
    ], loc="lower right", framealpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "top_features.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 4. SHAP Summary Plot ──────────────────────────────────────────────────────

def plot_shap_summary(shap_values, X_dense, tfidf, output_dir: str, top_n: int = 20):
    feature_names = tfidf.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values, X_dense,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Summary — Feature Impact on Predictions",
              fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 5. SHAP Force Plot ────────────────────────────────────────────────────────

def plot_shap_force(explainer, shap_values, X_dense, tfidf, idx: int, output_dir: str):
    feature_names = tfidf.get_feature_names_out()
    shap.initjs()
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx],
        X_dense[idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    fig  = plt.gcf()
    path = os.path.join(output_dir, f"shap_force_sample_{idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 6. NEW: Word Cloud (REAL vs FAKE) ─────────────────────────────────────────

def plot_word_cloud(df, output_dir: str):
    """
    Generate side-by-side word clouds for REAL and FAKE articles.
    Requires `wordcloud` to be installed (pip install wordcloud).
    Skips gracefully if not available.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except ImportError:
        print("  Skipping word cloud: install `wordcloud` package to enable this plot.")
        return

    stopwords = set(STOPWORDS) | {"said", "says", "one", "also", "would", "reuters", "ap"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, label, color, title in zip(
        axes,
        [0, 1],
        ["#1D9E75", "#D85A30"],
        ["REAL News", "FAKE News"],
    ):
        corpus = " ".join(df.loc[df["label"] == label, "clean_text"].fillna(""))
        wc = WordCloud(
            width=700, height=400,
            background_color="white",
            colormap="Greens" if label == 0 else "Reds",
            stopwords=stopwords,
            max_words=150,
            random_state=42,
        ).generate(corpus)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)

    plt.suptitle("Word Clouds — Most Common Words per Class",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "word_clouds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── 7. NEW: Cross-Validation Score Bar Chart ──────────────────────────────────

def plot_cv_scores(cv_scores: np.ndarray, output_dir: str):
    """
    Bar chart of 5-fold CV accuracy scores with mean ± std annotation.
    """
    folds  = [f"Fold {i+1}" for i in range(len(cv_scores))]
    colors = ["#534AB7" if s >= cv_scores.mean() else "#ABA6E0" for s in cv_scores]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(folds, cv_scores * 100, color=colors, edgecolor="none", width=0.5)
    ax.axhline(cv_scores.mean() * 100, color="#D85A30", lw=1.5, linestyle="--",
               label=f"Mean = {cv_scores.mean()*100:.2f}%")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5-Fold Stratified CV Accuracy", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(cv_scores.min() * 100 - 2, 101)
    ax.legend(framealpha=0.3)
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{score*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "cv_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")