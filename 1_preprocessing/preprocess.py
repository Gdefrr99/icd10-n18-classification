# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for MIMIC-IV discharge summaries.

Joins diagnoses_icd and discharge tables, filters to ICD-10-CM N18 codes,
applies clinical text normalization, and produces stratified train/val/test splits.

Usage:
    python preprocess.py \
        --diagnoses_csv data/raw/diagnoses_icd.csv \
        --discharge_csv data/raw/discharge.csv \
        --output_dir    data/processed/
"""

import argparse
import ast
import re
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


# ---------------------------------------------------------------------------
# Clinical text normalization
# ---------------------------------------------------------------------------

def clean_discharge_summary(text: str) -> str:
    """Apply MIMIC-IV-specific clinical text normalization."""
    if not isinstance(text, str):
        return ""

    # 1. Remove administrative header fields
    for pattern in [
        r"Name:\s+___", r"Unit No:\s+___", r"Admission Date:\s+___",
        r"Discharge Date:\s+___", r"Date of Birth:\s+___",
    ]:
        text = re.sub(pattern, "", text)

    # Replace remaining anonymization markers
    text = text.replace("___", "[UNK]")

    # 2. Abbreviation expansion (before newline handling to avoid split patterns)
    text = re.sub(r"\bs/p\b",  "status post",   text, flags=re.IGNORECASE)
    text = re.sub(r"\bc/o\b",  "complains of",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bh/o\b",  "history of",    text, flags=re.IGNORECASE)
    text = re.sub(r"\bw/o\b",  "without",       text, flags=re.IGNORECASE)
    text = re.sub(r"\s+w/",    " with ",        text, flags=re.IGNORECASE)
    text = re.sub(r"\bpt\b",   "patient",       text, flags=re.IGNORECASE)

    # 3. Hard-wrap restoration
    text = re.sub(r'\n\s*\n', '||PARAGRAPH||', text)  # protect real paragraph breaks
    text = text.replace('\n', ' ')                      # remove MIMIC hard wraps
    text = text.replace('||PARAGRAPH||', '\n\n')       # restore paragraphs

    # 4. Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Tag clinically important sections with extra newlines
    text = re.sub(r"History of Present Illness:",
                  "\nHistory of Present Illness:\n", text, flags=re.IGNORECASE)
    text = re.sub(r"Discharge Diagnos[ei]s:",
                  "\nDischarge Diagnosis:\n", text, flags=re.IGNORECASE)

    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_n18_dataset(diagnoses_csv: str, discharge_csv: str) -> pd.DataFrame:
    """Join MIMIC-IV tables and filter to ICD-10-CM N18 notes."""
    print("Loading diagnoses_icd...")
    diag = pd.read_csv(diagnoses_csv, usecols=["subject_id", "hadm_id", "icd_version", "icd_code"])

    # Keep only ICD-10-CM records (icd_version == 10)
    diag = diag[diag["icd_version"] == 10].drop(columns=["icd_version"])

    # Group codes per admission
    diag_grouped = (
        diag.groupby(["subject_id", "hadm_id"])["icd_code"]
        .apply(list)
        .reset_index()
    )

    # Keep only admissions with at least one N18.x code
    diag_n18 = diag_grouped[
        diag_grouped["icd_code"].apply(lambda codes: any(str(c).startswith("N18") for c in codes))
    ].copy()

    # Restrict label list to N18.x codes only
    diag_n18["icd_code"] = diag_n18["icd_code"].apply(
        lambda codes: [c for c in codes if str(c).startswith("N18")]
    )

    print(f"N18 admissions found: {len(diag_n18):,}")

    print("Loading discharge notes...")
    discharge = pd.read_csv(discharge_csv, usecols=["subject_id", "hadm_id", "text"])

    # Join on (subject_id, hadm_id)
    df = diag_n18.merge(discharge, on=["subject_id", "hadm_id"], how="inner")
    print(f"Notes after join: {len(df):,}")

    # Remove duplicate N18.9 entries
    df = df[df["icd_code"].apply(lambda x: len(x) == len(set(x)))].copy()
    print(f"Notes after deduplication: {len(df):,}")

    return df.reset_index(drop=True)


def preprocess_and_split(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Applying text normalization...")
    df["text"] = df["text"].apply(clean_discharge_summary)

    # Save full N18 dataset
    full_path = output_dir / "diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv"
    df.to_csv(full_path, index=False)
    print(f"Full dataset saved → {full_path} ({len(df):,} rows)")

    # Stratified multi-label split: 70/10/20
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["icd_code"])

    splitter_tv = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_val_idx, test_idx = next(splitter_tv.split(df, Y))

    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    Y_trainval  = Y[train_val_idx]
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    val_fraction = 0.10 / 0.80
    splitter_v = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(splitter_v.split(df_trainval, Y_trainval))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = output_dir / f"ehr_icd_{name}_clean.csv"
        subset.to_csv(path, index=False)
        print(f"{name:5s} split saved → {path} ({len(subset):,} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnoses_csv", required=True)
    parser.add_argument("--discharge_csv",  required=True)
    parser.add_argument("--output_dir",     default="data/processed/")
    args = parser.parse_args()

    df = build_n18_dataset(args.diagnoses_csv, args.discharge_csv)
    preprocess_and_split(df, Path(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
