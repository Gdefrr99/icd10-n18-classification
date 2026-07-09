# -*- coding: utf-8 -*-
"""
Resumen clínico de notas de alta mediante MedGemma-27B-it.

Requiere:
  - pip install -r requirements_generative.txt
  - huggingface-cli login  (MedGemma requiere aprobación de acceso)
  - ~40 GB de VRAM (bfloat16); dos A100 40 GB o una A100 80 GB.

Uso:
    python 5_summarization/summarize_medgemma.py \
        --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
        --output_csv data/processed/ehr_n18_summarized.csv
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline, set_seed

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
set_seed(42)

SYSTEM_PROMPT = (
    "You are a highly specialized clinical documentation expert with deep knowledge "
    "of ICD-10 coding guidelines.\n"
    "Your task is to produce a concise abstractive summary of a clinical note.\n"
    "The summary must:\n"
    "- Be written in clinical language, preserving all medically relevant information.\n"
    "- Explicitly mention all diagnoses (primary and secondary), significant symptoms, "
    "relevant procedures, relevant lab findings, and discharge status.\n"
    "- Prioritize information that directly supports or justifies ICD-10 code assignment "
    "(conditions, complications, comorbidities, external causes).\n"
    "- Omit administrative, demographic, or redundant information that is irrelevant for coding.\n"
    "- Be no longer than 400 words to ensure it fits within a 512-token processing window.\n"
    "Do NOT assign ICD-10 codes yourself. Only summarize."
)

GENERATION_KWARGS = dict(
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.85,
    top_k=50,
    repetition_penalty=1.15,
    do_sample=True,
)

MIN_SUMMARY_WORDS = 20


def is_valid_summary(text: str) -> bool:
    """Heurística de filtrado: rechaza respuestas vacías, demasiado cortas o
    que contienen directamente códigos ICD-10 (Sección 4.5.4)."""
    if not text or not isinstance(text, str):
        return False
    words = text.strip().split()
    if len(words) < MIN_SUMMARY_WORDS:
        return False
    # Rechazar si el modelo ha filtrado códigos ICD (p. ej. "N18.3", "E11.9")
    import re
    if re.search(r'\b[A-Z]\d{2}\.?\d*\b', text):
        return False
    return True


def summarize_note(pipe, note: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text":
            f"Please summarize the following clinical note according to your instructions:\n\n{note}"}]},
    ]
    output = pipe(text=messages, **GENERATION_KWARGS)
    return output[0]["generated_text"][-1]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model",      default="google/medgemma-27b-it")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Reintentos con semilla de generación distinta ante una respuesta inválida")
    args = parser.parse_args()

    print(f"Cargando modelo: {args.model}")
    pipe = pipeline(
        "image-text-to-text",
        model=args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        use_fast=True,
    )

    df = pd.read_csv(args.input_csv)
    if "summary" not in df.columns:
        df["summary"] = None

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Reanudación desde checkpoint si el archivo de salida ya existe
    if output_path.exists():
        df_saved = pd.read_csv(output_path)
        df["summary"] = df_saved["summary"]
        print(f"Reanudando: {df['summary'].notna().sum()} / {len(df)} notas ya procesadas.")

    pending = df[df["summary"].isna()].index.tolist()
    print(f"Resumiendo {len(pending)} notas...")

    for idx in tqdm(pending):
        note = df.at[idx, "text"]
        summary = None
        for attempt in range(args.max_retries + 1):
            try:
                candidate = summarize_note(pipe, note)
                if is_valid_summary(candidate):
                    summary = candidate
                    break
                else:
                    torch.manual_seed(42 + attempt + 1)
            except Exception as e:
                print(f"  Error en la fila {idx}, intento {attempt}: {e}")
        df.at[idx, "summary"] = summary

        # Guardado incremental cada 50 notas
        if (pending.index(idx) + 1) % 50 == 0:
            df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    valid = df["summary"].notna().sum()
    print(f"Hecho. Resúmenes válidos: {valid} / {len(df)} ({100*valid/len(df):.1f}%)")
    print(f"Salida guardada -> {output_path}")


if __name__ == "__main__":
    main()