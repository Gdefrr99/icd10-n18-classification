# -*- coding: utf-8 -*-
"""
Selección estratificada de 1.000 muestras para la fase de evaluación de
modelos de resumen (Sección 4.5.2 de la memoria).

Selecciona 1.000 notas del dataset N18 completo reproduciendo la distribución
multietiqueta real de cada código. Los códigos raros (frecuencia < 1 %, en la
práctica solo N18.1) se sobremuestrean hasta un mínimo de MIN_RARE muestras
para garantizar su cobertura, lo que explica que la proporción de N18.1 en la
muestra (2,3 %) sea superior a su proporción en el dataset completo (0,4 %,
véase Tabla 4.4). El resto de la muestra se completa con
`iterative_train_test_split` sobre las notas restantes.

Uso:
    python 5_summarization/sampling.py \
        --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
        --output_csv data/processed/muestra_1000.csv \
        --n_samples  1000
"""

import argparse
import ast
import random

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split

RANDOM_SEED   = 42
MIN_RARE      = 20
RARE_THRESHOLD = 0.01  # 1 %

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

    for c in N18_LABELS:
        df[c] = df["icd_code"].apply(lambda codes: int(c in codes))

    # Detectar etiquetas raras (frecuencia < 1 %) y garantizar su cobertura mínima
    label_freq = df[N18_LABELS].sum() / len(df)
    rare_labels = label_freq[label_freq < RARE_THRESHOLD].index.tolist()
    print(f"Etiquetas raras (< {RARE_THRESHOLD:.0%}): {rare_labels}")

    rare_samples = pd.DataFrame()
    for label in rare_labels:
        label_rows = df[df[label] == 1]
        if len(label_rows) <= MIN_RARE:
            rare_samples = pd.concat([rare_samples, label_rows])
        else:
            rare_samples = pd.concat(
                [rare_samples, label_rows.sample(MIN_RARE, random_state=RANDOM_SEED)]
            )
    rare_samples = rare_samples[~rare_samples.index.duplicated()]

    # Completar el resto de la muestra con particionado iterativo estratificado
    remaining_df = df.drop(rare_samples.index)
    remaining_target = args.n_samples - len(rare_samples)

    X_remaining = remaining_df.index.values.reshape(-1, 1)
    Y_remaining = remaining_df[N18_LABELS].values
    test_size = remaining_target / len(remaining_df)

    _, _, X_sample, _ = iterative_train_test_split(X_remaining, Y_remaining, test_size=test_size)
    sample_rest = remaining_df.loc[X_sample.flatten()]

    final_sample = pd.concat([rare_samples, sample_rest])
    final_sample = final_sample.sample(args.n_samples, random_state=RANDOM_SEED)
    final_sample = final_sample.drop(columns=N18_LABELS)

    final_sample.to_csv(args.output_csv, index=False)
    print(f"Guardadas {len(final_sample)} muestras estratificadas -> {args.output_csv}")

    # Comparación de la distribución de etiquetas (dataset completo vs. muestra)
    print("\nDistribución de etiquetas (dataset completo vs. muestra):")
    print(f"{'Código':<10} {'Dataset':>8} {'Muestra':>8}")
    df_full_counts = df[N18_LABELS].sum()
    sample_counts = final_sample["icd_code"].apply(
        lambda codes: pd.Series({c: int(c in codes) for c in N18_LABELS})
    ).sum()
    for label in N18_LABELS:
        pf = df_full_counts[label] / df_full_counts.sum()
        ps = sample_counts[label] / sample_counts.sum()
        print(f"{label:<10} {pf:>7.1%} {ps:>7.1%}")


if __name__ == "__main__":
    main()
