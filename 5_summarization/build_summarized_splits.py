# -*- coding: utf-8 -*-
"""
Sustituye el texto original de las particiones train/val/test de N18 por el
resumen generado con MedGemma-27B-it, conservando la misma
partición (subject_id, hadm_id) que 1_preprocessing/preprocess.py, de modo
que la comparación con la segmentación + Max Pooling (Paso 4) sea directa.

Uso:
    python 5_summarization/build_summarized_splits.py \
        --splits_dir     data/processed/ \
        --summarized_csv data/processed/ehr_n18_summarized.csv \
        --output_dir     data/processed/summarized/
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits_dir", required=True,
                         help="Directorio con ehr_icd_{train,val,test}_clean.csv")
    parser.add_argument("--summarized_csv", required=True,
                         help="Salida de summarize_medgemma.py (columna 'summary')")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sum = pd.read_csv(args.summarized_csv)
    df_sum = df_sum.dropna(subset=["summary"])
    df_sum = df_sum[["subject_id", "hadm_id", "summary"]]

    for name in ["train", "val", "test"]:
        df_split = pd.read_csv(splits_dir / f"ehr_icd_{name}_clean.csv")
        n_before = len(df_split)
        df_merged = df_split.merge(df_sum, on=["subject_id", "hadm_id"], how="inner")
        df_merged = df_merged.drop(columns=["text"]).rename(columns={"summary": "text"})

        out_path = output_dir / f"ehr_icd_{name}_clean.csv"
        df_merged.to_csv(out_path, index=False)
        print(f"{name:5s}: {n_before:,} notas -> {len(df_merged):,} con resumen válido -> {out_path}")


if __name__ == "__main__":
    main()