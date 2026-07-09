# -*- coding: utf-8 -*-
"""
Construcción de los lotes de evaluación para los modelos Gemini.

Para cada grupo de riesgo HCC se muestrean 500 notas de alta que contienen al
menos un código del rango del grupo, organizadas en 10 lotes de 50 notas.
Cada lote se escribe como un archivo de texto con el prompt ya completado, 
listo para copiar y pegar en la interfaz web de Google
Gemini (gemini.google.com) con la opción de "chat temporal" activada.

IMPORTANTE — motivo de este flujo manual: en la memoria la
evaluación se realizó pegando manualmente el contenido de cada lote en
gemini.google.com, y NO mediante la API de Gemini. Esta elección responde a
que los chats temporales no se emplean para reentrenar los modelos y los
datos se retienen solo un tiempo limitado, evitando así que las notas de
MIMIC-IV (aunque anonimizadas) se incorporen al corpus de entrenamiento de un
modelo de terceros. El uso de la API, en cambio, generaría una relación de
"business associate" que exigiría un acuerdo formal bajo HIPAA. Por este
motivo este repositorio no automatiza las llamadas a un LLM: genera los
lotes para pegado manual y, por separado (score_jaccard.py), puntúa las
respuestas que el usuario pegue de vuelta.

Uso:
    python 2_llm_hcc_evaluation/build_batches.py \
        --data_csv   data/processed/diagnoses_icd10.csv \
        --output_dir results/llm/batches/ \
        --group      chronic_kidney_disease \
        --strategy   specific \
        --n_notes    500 \
        --batch_size 50
"""

import argparse
import ast
from pathlib import Path

import pandas as pd

from hcc_groups import HCC_GROUPS, note_codes_in_group

RANDOM_SEED = 42

PROMPT_SPECIFIC = """\
You are an expert clinical coder specialized in {NOMBRE_GRUPO_RIESGO}. Your \
task is to analyze a BATCH of 50 distinct "Discharge Summaries" and \
extract ICD-10-CM codes specific to {NOMBRE_GRUPO_RIESGO}.

CRITICAL INSTRUCTIONS:
1. INDEPENDENCE: Treat each <patient_record> as a completely separate \
entity. NEVER use information from one patient to code another.
2. RANGE OF INTEREST: Extract ONLY codes starting with prefixes: {LISTA_SUBCAPITULOS}. \
Ignore all others.
3. FORMAT: Return ONLY the first 3 characters of the code (the category, e.g., \
"{EJEMPLO_CODIGO_1}").
4. DUPLICATES: If you find multiple distinct pieces of evidence or conditions \
that map to the same 3-digit category, include the code as many times as it \
appears. Do not deduplicate.
5. OUTPUT STRUCTURE: You must return a SINGLE JSON object where keys are the \
'id' provided in the input tags, and values are lists of codes.

Input Format Example:
<patient_record id="14101416_27674789">Text...</patient_record>

Output JSON Example:
{{
"14101416_27674789": ["{EJEMPLO_CODIGO_2}", "{EJEMPLO_CODIGO_1}"],
"11258317_23346944": ["{EJEMPLO_CODIGO_2}"]
}}

BATCH DATA:
{BATCH_CONTENT}
"""

PROMPT_GENERAL = """\
You are an expert clinical coder specialized in the ICD-10-CM classification. \
Your task is to analyze a BATCH of 50 distinct "Discharge Summaries" and \
extract relevant ICD-10-CM diagnostic codes for EACH patient independently.

CRITICAL INSTRUCTIONS:
1. INDEPENDENCE: Treat each <patient_record> as a completely separate \
entity. NEVER use information from one patient to code another.
2. SCOPE: Identify ALL diagnosed conditions, diseases, and findings for each \
patient.
3. MAPPING: Map findings to ICD-10-CM codes.
4. FORMAT: Return ONLY the first 3 characters of the code (the category, e.g., \
"D67", "M53").
5. DUPLICATES: If you find multiple distinct pieces of evidence or conditions \
that map to the same 3-digit category, include the code as many times as it \
appears. Do not deduplicate.
6. OUTPUT STRUCTURE: You must return a SINGLE JSON object where keys are the \
'id' provided in the input tags, and values are lists of codes.

Input Format Example:
<patient_record id="14101416_27674789">Text...</patient_record>
<patient_record id="11258317_23346944">Text...</patient_record>

Output JSON Example:
{{
"14101416_27674789": ["R81", "I10"],
"11258317_23346944": ["L92", "H70", "H70"]
}}

BATCH DATA:
{BATCH_CONTENT}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   required=True,
                        help="Dataset completo ICD-10-CM (salida de 1_preprocessing/preprocess.py)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--group",      required=True, choices=list(HCC_GROUPS.keys()))
    parser.add_argument("--strategy",   default="specific", choices=["specific", "general"])
    parser.add_argument("--n_notes",    type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    group = HCC_GROUPS[args.group]
    prefijos = group["prefijos"]

    df = pd.read_csv(args.data_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    df = df[df["icd_code"].apply(lambda codes: len(note_codes_in_group(codes, prefijos)) > 0)]
    print(f"Notas disponibles para '{group['nombre']}': {len(df):,}")

    df_sample = df.sample(n=min(args.n_notes, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Notas muestreadas: {len(df_sample)}")

    template = PROMPT_SPECIFIC if args.strategy == "specific" else PROMPT_GENERAL
    example_codes = prefijos[:2] if len(prefijos) >= 2 else (prefijos * 2)[:2]

    output_dir = Path(args.output_dir) / args.group / args.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (len(df_sample) + args.batch_size - 1) // args.batch_size
    for b in range(n_batches):
        chunk = df_sample.iloc[b * args.batch_size: (b + 1) * args.batch_size]
        records = "\n".join(
            f'<patient_record id="{row.subject_id}_{row.hadm_id}">{row.text}</patient_record>'
            for row in chunk.itertuples()
        )
        prompt = template.format(
            NOMBRE_GRUPO_RIESGO=group["nombre"],
            LISTA_SUBCAPITULOS=", ".join(prefijos),
            EJEMPLO_CODIGO_1=example_codes[0],
            EJEMPLO_CODIGO_2=example_codes[1],
            BATCH_CONTENT=records,
        )
        batch_path = output_dir / f"lote_{b+1:02d}.txt"
        batch_path.write_text(prompt, encoding="utf-8")

        # Guardar también los códigos reales, para el scoring posterior
        truth_path = output_dir / f"lote_{b+1:02d}_truth.csv"
        chunk_truth = chunk.copy()
        chunk_truth["id"] = chunk_truth.apply(lambda r: f"{r.subject_id}_{r.hadm_id}", axis=1)
        chunk_truth["codigos_reales_grupo"] = chunk_truth["icd_code"].apply(
            lambda codes: note_codes_in_group(codes, prefijos)
        )
        chunk_truth[["id", "codigos_reales_grupo"]].to_csv(truth_path, index=False)

    print(f"{n_batches} lotes escritos en {output_dir}")
    print("Pega cada archivo lote_XX.txt en un chat temporal de gemini.google.com "
          "y guarda la respuesta JSON como lote_XX_response.json en el mismo directorio.")


if __name__ == "__main__":
    main()