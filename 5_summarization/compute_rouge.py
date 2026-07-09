# -*- coding: utf-8 -*-
"""
Métricas ROUGE de los resúmenes generados sobre las 1.000 muestras, frente a
las notas originales.

Calcula ROUGE-1, ROUGE-2 y ROUGE-L (precisión, recall y F1) promediadas sobre
las notas con resumen válido. Una precisión ROUGE elevada indica que el
vocabulario del resumen proviene mayoritariamente de la nota original
(bajo riesgo de alucinación); un recall ROUGE elevado indicaría que el
resumen reproduce literalmente grandes fragmentos de la nota, lo que aquí no
es deseable porque el objetivo es un resumen abstractivo y conciso.

Uso (una vez generado el resumen de un modelo con summarize_1000.py):
    python 5_summarization/compute_rouge.py \
        --original_csv   data/processed/muestra_1000.csv \
        --summarized_csv data/processed/muestra_1000_summarized_MedGemma-27B-it.csv \
        --model_name     MedGemma-27B-it
"""

import argparse

import pandas as pd
from rouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_csv",   required=True,
                        help="Salida de 5_summarization/sampling.py (columna 'text' = nota original)")
    parser.add_argument("--summarized_csv", required=True,
                        help="Salida de 5_summarization/summarize_1000.py (columna 'summary')")
    parser.add_argument("--model_name",     required=True,
                        help="Nombre del modelo de resumen evaluado, solo para el informe")
    parser.add_argument("--output_csv",     default=None,
                        help="Ruta opcional para guardar las puntuaciones por nota")
    args = parser.parse_args()

    df_orig = pd.read_csv(args.original_csv)[["subject_id", "hadm_id", "text"]]
    df_sum  = pd.read_csv(args.summarized_csv)[["subject_id", "hadm_id", "summary"]]

    df = df_orig.merge(df_sum, on=["subject_id", "hadm_id"], how="inner")
    n_total = len(df)
    df = df.dropna(subset=["summary"])
    n_valid = len(df)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    records = []
    for row in df.itertuples():
        scores = scorer.score(row.text, row.summary)
        records.append({
            "subject_id": row.subject_id,
            "hadm_id":    row.hadm_id,
            "rouge1_p": scores["rouge1"].precision, "rouge1_r": scores["rouge1"].recall, "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_p": scores["rouge2"].precision, "rouge2_r": scores["rouge2"].recall, "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_p": scores["rougeL"].precision, "rougeL_r": scores["rougeL"].recall, "rougeL_f1": scores["rougeL"].fmeasure,
        })

    scores_df = pd.DataFrame(records)

    if args.output_csv:
        scores_df.to_csv(args.output_csv, index=False)
        print(f"Puntuaciones por nota guardadas -> {args.output_csv}")

    means = scores_df.drop(columns=["subject_id", "hadm_id"]).mean()

    print(f"\nModelo: {args.model_name}")
    print(f"Válidas: {n_valid} / {n_total}")
    print(f"{'Métrica':<10} {'P':>7} {'R':>7} {'F1':>7}")
    for metric in ["rouge1", "rouge2", "rougeL"]:
        print(f"{metric:<10} {means[f'{metric}_p']:>7.3f} {means[f'{metric}_r']:>7.3f} {means[f'{metric}_f1']:>7.3f}")


if __name__ == "__main__":
    main()
