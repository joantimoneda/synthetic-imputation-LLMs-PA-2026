
## Appendix E
## Author: Joan Timoneda
## Date: December 3, 2025

import os
import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_DIR   = "data"
OUT_DIR    = "results"
MODEL      = "nli-roberta-large"
THRESHOLDS = [0.85, 0.87, 0.90]

## real human-coded category sets, one per application: (file, text column)
REAL_SETS = {
    "nostalgia": ("nostalgia_data.csv", "text"),               ## 151 hand-coded nostalgic sentences
    "speeches":  ("intl_speeches_sample_150.xlsx", "text_eng"),## 150 hand-coded international speeches
}

## synthetic-text files: (application, condition, file)
CONDITIONS = [
    ("nostalgia", "n=50  (101 synth)",      "synthetic_50_101_fullsample.xlsx"),
    ("nostalgia", "n=75  (76 synth)",       "synthetic_75_76_fullsample.xlsx"),
    ("nostalgia", "n=100 (51 synth)",       "synthetic_100_51_fullsample.xlsx"),
    ("nostalgia", "n=10  (141 synth, AppB)","synthetic_10_141.xlsx"),
    ("nostalgia", "n=25  (126 synth, AppB)","synthetic_25_126.xlsx"),
    ("speeches",  "n=50  (168 synth)",      "GPT_synthetic_labeled_50_168.xlsx"),
    ("speeches",  "n=75  (143 synth)",      "GPT_synthetic_labeled_75_143.xlsx"),
    ("speeches",  "n=100 (118 synth)",      "GPT_synthetic_labeled_100_118.xlsx"),
    ("speeches",  "n=150 (68 synth)",       "GPT_synthetic_labeled_150_68.xlsx"),
]

CHECK_ONLY = "--check" in sys.argv    ## load and count the data, skip the model


def load_real(app):
    fname, col = REAL_SETS[app]
    path = os.path.join(DATA_DIR, fname)
    d = pd.read_csv(path, index_col=0) if fname.endswith(".csv") else pd.read_excel(path, index_col=0)
    if app == "nostalgia":
        d = d[d["nostalgic_at_least_3"] == 1]
    if app == "sentiment":
        d = d[d["label"] == 1]
    texts = d[col].dropna().astype(str).tolist()
    print(f"  real set, {app}: {len(texts)} human-coded texts ({fname})")
    return texts


def load_synth(fname):
    d = pd.read_excel(os.path.join(DATA_DIR, fname), index_col=0)
    if "synthetic" in d.columns:
        d = d[d["synthetic"] == 1]
    return d["text"].dropna().astype(str).tolist()


print("Loading data...")
real_texts = {app: load_real(app) for app in REAL_SETS}
synth_texts = {}
for app, cond, fname in CONDITIONS:
    synth_texts[(app, cond)] = load_synth(fname)
    print(f"  {app:10} {cond:24} {len(synth_texts[(app, cond)]):4} synthetic texts ({fname})")

if CHECK_ONLY:
    print("\n--check: data loads fine, exiting before the model.")
    sys.exit(0)


print(f"\nLoading {MODEL} ...")
model = SentenceTransformer(MODEL)

real_emb = {}
for app in REAL_SETS:
    print(f"Embedding real set: {app} ({len(real_texts[app])} texts)")
    real_emb[app] = model.encode(real_texts[app], batch_size=64,
                                 normalize_embeddings=True, show_progress_bar=True)

rows = []
for app, cond, fname in CONDITIONS:
    texts = synth_texts[(app, cond)]
    print(f"Embedding synthetic: {app} {cond} ({len(texts)} texts)")
    emb  = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    sims = emb @ real_emb[app].T                     ## cosine similarity (normalized)
    max_sim = sims.max(axis=1)
    argmax  = sims.argmax(axis=1)
    for i, t in enumerate(texts):
        rows.append({"application": app, "condition": cond, "file": fname,
                     "text": t, "max_similarity": float(max_sim[i]),
                     "closest_real_text": real_texts[app][argmax[i]][:200]})

res = pd.DataFrame(rows)

summary = []
for app, cond, fname in CONDITIONS:
    s = res[(res["application"] == app) & (res["condition"] == cond)]["max_similarity"]
    summary.append({"application": app, "condition": cond, "n": len(s),
                    "mean": s.mean(), "median": s.median(),
                    "p95": s.quantile(0.95), "max": s.max(),
                    "n_above_0.85": int((s > 0.85).sum()),
                    "n_above_0.87": int((s > 0.87).sum()),
                    "n_above_0.90": int((s > 0.90).sum())})
summary = pd.DataFrame(summary)

print("\n" + "=" * 110)
print(f"MAX COSINE SIMILARITY vs FULL HUMAN-CODED CATEGORY SET ({MODEL})")
print("=" * 110)
print(summary.round(3).to_string(index=False))
for t in THRESHOLDS:
    print(f"Total texts above {t}: {int((res['max_similarity'] > t).sum())} of {len(res)}")

os.makedirs(OUT_DIR, exist_ok=True)
res.sort_values("max_similarity", ascending=False).to_excel(
    os.path.join(OUT_DIR, "similarity_audit_per_text.xlsx"), index=False)
summary.to_excel(os.path.join(OUT_DIR, "similarity_audit_summary.xlsx"), index=False)
print(f"\nWrote {OUT_DIR}/similarity_audit_per_text.xlsx and {OUT_DIR}/similarity_audit_summary.xlsx")
