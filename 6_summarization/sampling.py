# -*- coding: utf-8 -*-
"""
Stratified 1,000-sample selection for the summarization evaluation phase.

Selects 1,000 notes from the full N18 dataset reproducing the multi-label distribution.
Used to evaluate and compare the 4 generative summarization models before scaling to
the full 23,358-note dataset.

Usage:
    python 6_summarization/sampling.py \
        --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
        --output_csv data/processed/1000.csv \
        --n_samples  1000
"""

import argparse
import ast
import random

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--n_samples",  type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["icd_code"] = df["icd_code"].apply(
        lambda codes: [c for c in codes if c in N18_LABELS]
    )

    mlb = MultiLabelBinarizer(classes=N18_LABELS)
    Y = mlb.fit_transform(df["icd_code"])

    # Use iterative_train_test_split to pick n_samples stratified rows
    # We treat the sample as the "test" split
    sample_ratio = args.n_samples / len(df)
    X_idx = np.arange(len(df)).reshape(-1, 1)
    _, _, X_sample_idx, _ = iterative_train_test_split(X_idx, Y, test_size=sample_ratio)

    sample_idx = X_sample_idx.flatten()
    df_sample = df.iloc[sample_idx].reset_index(drop=True)

    df_sample.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df_sample)} stratified samples → {args.output_csv}")

    # Print label distribution comparison
    Y_full   = Y.sum(axis=0) / Y.sum()
    Y_sample = mlb.transform(df_sample["icd_code"]).sum(axis=0) / mlb.transform(df_sample["icd_code"]).sum()
    print("\nLabel distribution (full dataset vs. sample):")
    print(f"{'Code':<10} {'Full':>8} {'Sample':>8}")
    for label, pf, ps in zip(N18_LABELS, Y_full, Y_sample):
        print(f"{label:<10} {pf:>7.1%} {ps:>7.1%}")


if __name__ == "__main__":
    main()
