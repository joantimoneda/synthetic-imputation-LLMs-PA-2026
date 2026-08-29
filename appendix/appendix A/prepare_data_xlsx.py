"""
prepare_data_xlsx.py -- Appendix A, stage 1

Builds the seven training sets for the downstream analysis (Muller & Proksch
replication) from the archived synthetic-imputation files, written as .xlsx:

    train_50_101.xlsx        50 real + 101 synthetic positives + negatives
    train_50_baseline.xlsx   same 50 real positives + negatives, no synthetic
    train_75_76.xlsx         75 real + 76 synthetic positives + negatives
    train_75_baseline.xlsx   same 75 real positives + negatives, no synthetic
    train_100_51.xlsx        100 real + 51 synthetic positives + negatives
    train_100_baseline.xlsx  same 100 real positives + negatives, no synthetic
    train_full_151.xlsx      all 151 hand-coded positives + all negatives

Run from this folder: python3 prepare_data_xlsx.py
"""

import os
import pandas as pd

## paths (relative to this folder)
SOURCE_DIR = "0. source_data"
OUT_DIR    = "1. train_data"

CONDITIONS = [
    ("synthetic_50_101_fullsample_len.xlsx",  "train_50_101.xlsx",  "train_50_baseline.xlsx"),
    ("synthetic_75_76_fullsample_len.xlsx",   "train_75_76.xlsx",   "train_75_baseline.xlsx"),
    ("synthetic_100_51_fullsample_len.xlsx",  "train_100_51.xlsx",  "train_100_baseline.xlsx"),
]


def report(df, title):
    print(f"{title}:")
    print(f"  Total: {len(df)}")
    print(f"  Nostalgic (label=1): {(df['nostalgic'] == 1).sum()}")
    print(f"  Not nostalgic (label=0): {(df['nostalgic'] == 0).sum()}")
    if "synthetic" in df.columns:
        print(f"  Real positive: {((df['nostalgic'] == 1) & (df['synthetic'] == 0)).sum()}")
        print(f"  Synthetic positive: {((df['nostalgic'] == 1) & (df['synthetic'] == 1)).sum()}")
    print()


os.makedirs(OUT_DIR, exist_ok=True)

## synthetic-imputation conditions and their class-weighted baselines
for src, out_synth, out_base in CONDITIONS:
    df = pd.read_excel(os.path.join(SOURCE_DIR, src))
    df = df[["text", "nostalgic", "synthetic"]].dropna(subset=["text"]).reset_index(drop=True)

    report(df, f"Training set ({out_synth})")
    df.to_excel(os.path.join(OUT_DIR, out_synth), index=False)

    ## baseline: drop the synthetic rows, keep the same real positives + negatives
    baseline = df[df["synthetic"] == 0].reset_index(drop=True)
    report(baseline, f"Baseline training set ({out_base})")
    baseline.to_excel(os.path.join(OUT_DIR, out_base), index=False)

## full hand-coded data (benchmark classifier)
df = pd.read_csv(os.path.join(SOURCE_DIR, "nostalgia_data.csv"), index_col=0)
df = df[["text", "nostalgic_at_least_3"]].dropna(subset=["text"]).reset_index(drop=True)
df = df.rename(columns={"nostalgic_at_least_3": "nostalgic"})

report(df, "Full training set (train_full_151.xlsx)")
df.to_excel(os.path.join(OUT_DIR, "train_full_151.xlsx"), index=False)
print("Done.")
