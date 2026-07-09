# -*- coding: utf-8 -*-
"""
Partición estratificada 70/10/20 de un conjunto de 1.000 muestras resumidas
por un modelo generativo concreto (Sección 4.5.3-4.5.4 de la memoria).

A diferencia de build_summarized_splits.py (que reutiliza la partición del
dataset N18 completo, Paso 1), aquí cada uno de los 4 modelos de resumen
genera un número distinto de resúmenes válidos (Tabla 4.5: entre 933 y 1.000
de las 1.000 muestras), por lo que cada conjunto resumido recibe su propia
partición 70/10/20 estratificada (semilla 42), igual que en
1_preprocessing/preprocess.py. Esto permite ajustar los 4 clasificadores
Transformer sobre cada uno de los 4 conjuntos y comparar su rendimiento
(Tabla 5.8) para identificar el mejor modelo de resumen.

Uso:
    python 5_summarization/build_1000_splits.py \
        --summarized_csv data/processed/muestra_1000_summarized_MedGemma-27B-it.csv \
        --output_dir     data/processed/summarized_1000/MedGemma-27B-it/
"""

import argparse
import ast
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarized_csv", required=True,
                        help="Salida de 5_summarization/summarize_1000.py (columna 'summary')")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.summarized_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    n_total = len(df)
    df = df.dropna(subset=["summary"]).reset_index(drop=True)
    print(f"Resúmenes válidos: {len(df):,} / {n_total:,}")

    df = df.drop(columns=["text"]).rename(columns={"summary": "text"})

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["icd_code"])

    splitter_tv = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_SEED
    )
    trainval_idx, test_idx = next(splitter_tv.split(df, Y))

    df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
    Y_trainval  = Y[trainval_idx]
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    splitter_v = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.10 / 0.80, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(splitter_v.split(df_trainval, Y_trainval))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = output_dir / f"ehr_icd_{name}_clean.csv"
        subset.to_csv(path, index=False)
        print(f"{name:5s}: {len(subset):,} filas -> {path}")


if __name__ == "__main__":
    main()
