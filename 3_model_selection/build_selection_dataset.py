# -*- coding: utf-8 -*-
"""
Construcción del conjunto de datos para la selección de modelos (Sección 4.3.1).

A diferencia del resto del pipeline, este paso NO usa el subgrupo N18: parte
del dataset completo ICD-10-CM (122.288 notas, generado por
1_preprocessing/preprocess.py) y construye un subconjunto de 10.000 notas
etiquetadas con al menos 3 de los 50 códigos ICD-10-CM más frecuentes,
definiendo así un espacio de 50 etiquetas posibles. Este subconjunto
reducido permite evaluar de forma eficiente las 25 arquitecturas Transformer
candidatas antes de entrenar los 4 modelos finales sobre el N18 completo.

Uso:
    python 3_model_selection/build_selection_dataset.py \
        --input_csv  data/processed/diagnoses_icd10.csv \
        --output_csv data/processed/seleccion_10000.csv \
        --top_n_codes 50 \
        --min_matches 3 \
        --n_samples   10000
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True,
                         help="Dataset completo ICD-10-CM (salida de 1_preprocessing/preprocess.py)")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--top_n_codes", type=int, default=50,
                         help="Número de códigos ICD-10-CM más frecuentes a considerar")
    parser.add_argument("--min_matches", type=int, default=3,
                         help="Mínimo de códigos del top-N que debe contener cada nota")
    parser.add_argument("--n_samples",  type=int, default=10000)
    args = parser.parse_args()

    print(f"Cargando {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    print(f"Tamaño del dataset original: {len(df):,} notas")

    # Seleccionar los N códigos ICD-10-CM más frecuentes del dataset completo
    all_codes = df["icd_code"].explode()
    icd_counts = all_codes.value_counts()
    selected_codes = set(icd_counts.head(args.top_n_codes).index.tolist())

    # Conservar solo las notas con al menos `min_matches` códigos del top-N
    def filter_function(codes):
        return len(set(codes).intersection(selected_codes)) >= args.min_matches

    df_reduced = df[df["icd_code"].apply(filter_function)].copy()
    df_reduced["icd_code"] = df_reduced["icd_code"].apply(
        lambda codes: [c for c in codes if c in selected_codes]
    )
    df_reduced = df_reduced[df_reduced["icd_code"].map(len) > 0]
    print(f"Notas tras filtrar por >= {args.min_matches} códigos del top-{args.top_n_codes}: "
          f"{len(df_reduced):,}")

    if len(df_reduced) > args.n_samples:
        df_reduced = df_reduced.sample(n=args.n_samples, random_state=RANDOM_SEED)
        print(f"Muestreo aleatorio (semilla {RANDOM_SEED}) -> {len(df_reduced):,} notas")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_reduced.to_csv(output_path, index=False)
    print(f"Conjunto de selección guardado -> {output_path} ({len(df_reduced):,} filas, "
          f"{args.top_n_codes} etiquetas posibles)")


if __name__ == "__main__":
    main()
