# -*- coding: utf-8 -*-
"""
Cálculo del índice de Jaccard sobre multiconjuntos para las respuestas de los
modelos Gemini (Sección 4.8.1 de la memoria).

Para cada nota, sea A el multiconjunto de códigos reales (a nivel de
categoría de 3 caracteres) y B el multiconjunto de códigos predichos por el
modelo:

    J(A, B) = sum_x min(mult_A(x), mult_B(x)) / sum_x max(mult_A(x), mult_B(x))

Se calculan dos variantes por nota:
  - J_real:  A se restringe a los códigos reales del grupo evaluado; B no se
             filtra (penaliza tanto omisiones como códigos fuera de rango).
  - J_ambos: tanto A como B se restringen al rango de códigos del grupo.

El resultado por grupo es la media de J_real y J_ambos sobre las notas
evaluadas (Tablas 5.1 y 5.2 de la memoria).

Espera, para cada lote generado por build_batches.py, un archivo
`lote_XX_response.json` con el JSON devuelto por Gemini (extraído
manualmente del chat) junto al `lote_XX_truth.csv` correspondiente.

Uso:
    python 2_llm_hcc_evaluation/score_jaccard.py \
        --batches_dir results/llm/batches/enfermedad_renal_cronica/specific/ \
        --group       enfermedad_renal_cronica
"""

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from hcc_groups import HCC_GROUPS


def jaccard_multiset(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 1.0
    num = sum(min(ca[k], cb[k]) for k in keys)
    den = sum(max(ca[k], cb[k]) for k in keys)
    return num / den if den > 0 else 0.0


def extract_json_block(raw_text: str) -> dict:
    """Extrae el primer bloque JSON del texto devuelto por el modelo."""
    match = re.search(r'\{.*\}', raw_text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches_dir", required=True,
                        help="Directorio generado por build_batches.py para un grupo/estrategia")
    parser.add_argument("--group", required=True, choices=list(HCC_GROUPS.keys()))
    args = parser.parse_args()

    prefijos = HCC_GROUPS[args.group]["prefijos"]
    batches_dir = Path(args.batches_dir)

    j_real_scores, j_ambos_scores = [], []

    truth_files = sorted(batches_dir.glob("lote_*_truth.csv"))
    for truth_path in truth_files:
        batch_id = truth_path.stem.replace("_truth", "")
        response_path = batches_dir / f"{batch_id}_response.json"
        if not response_path.exists():
            print(f"Aviso: falta {response_path.name}, se omite este lote.")
            continue

        truth_df = pd.read_csv(truth_path)
        truth_df["codigos_reales_grupo"] = truth_df["codigos_reales_grupo"].apply(ast.literal_eval)

        raw = response_path.read_text(encoding="utf-8")
        predictions = extract_json_block(raw)

        for row in truth_df.itertuples():
            real_codes = row.codigos_reales_grupo
            pred_codes_all = [str(c)[:3] for c in predictions.get(row.id, [])]
            pred_codes_in_range = [c for c in pred_codes_all if any(c.startswith(p) for p in prefijos)]

            j_real_scores.append(jaccard_multiset(real_codes, pred_codes_all))
            j_ambos_scores.append(jaccard_multiset(real_codes, pred_codes_in_range))

    if not j_real_scores:
        print("No se encontraron respuestas para puntuar.")
        return

    j_real = sum(j_real_scores) / len(j_real_scores)
    j_ambos = sum(j_ambos_scores) / len(j_ambos_scores)

    print(f"Grupo: {HCC_GROUPS[args.group]['nombre']}")
    print(f"Notas evaluadas: {len(j_real_scores)}")
    print(f"J_real:  {j_real:.4f}")
    print(f"J_ambos: {j_ambos:.4f}")


if __name__ == "__main__":
    main()
