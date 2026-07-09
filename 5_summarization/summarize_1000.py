# -*- coding: utf-8 -*-
"""
Resumen de las 1.000 muestras seleccionadas con cada uno de los 4 modelos
generativos evaluados (Sección 4.5.3 de la memoria).

Generaliza summarize_medgemma.py para poder ejecutarse con cualquiera de los
4 modelos comparados en esta fase. El prompt de sistema, los parámetros de
generación y la heurística de validación/reintento son idénticos para los 4
modelos (Sección 4.5.3-4.5.4), de forma que la única variable entre
ejecuciones es el modelo generativo.

Modelos comparados:
  - Llama3-OpenBioLLM-8B      (SaamaAIResearch 2024)
  - Bio-Medical-Llama-3-8B    (ContactDoctor 2024)
  - MedGemma-1.5-4b-it        (Google DeepMind 2025)
  - MedGemma-27B-it           (Google DeepMind 2025)

Los identificadores de HuggingFace por defecto en MODEL_REGISTRY son la
mejor referencia disponible para cada modelo: verifícalos contra la página
del modelo antes de lanzar una ejecución larga, y usa --model_id para
sobrescribir cualquiera de ellos.

Uso (una ejecución por modelo, sobre las 1.000 muestras de sampling.py):
    python 5_summarization/summarize_1000.py \
        --input_csv  data/processed/muestra_1000.csv \
        --output_csv data/processed/muestra_1000_summarized_MedGemma-27B-it.csv \
        --model      MedGemma-27B-it
"""

import argparse
import random
import re
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

# Modelo lógico -> (id de HuggingFace, tipo de pipeline)
# "image-text-to-text": familia Gemma3 (procesador multimodal, aunque aquí se
#                        use solo texto). "text-generation": familia Llama,
#                        modelo causal estándar con plantilla de chat.
MODEL_REGISTRY = {
    "Llama3-OpenBioLLM-8B":   ("aaditya/Llama3-OpenBioLLM-8B",        "text-generation"),
    "Bio-Medical-Llama-3-8B": ("ContactDoctor/Bio-Medical-Llama-3-8B", "text-generation"),
    "MedGemma-1.5-4b-it":     ("google/medgemma-4b-it",                "image-text-to-text"),
    "MedGemma-27B-it":        ("google/medgemma-27b-it",               "image-text-to-text"),
}

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
    if re.search(r'\b[A-Z]\d{2}\.?\d*\b', text):
        return False
    return True


def build_messages(note: str, pipeline_type: str) -> list:
    """Construye los mensajes de chat en el formato que espera cada tipo de
    pipeline: image-text-to-text (Gemma3) usa contenido estructurado por
    bloques {type, text}; text-generation (Llama) usa contenido como string."""
    user_text = f"Please summarize the following clinical note according to your instructions:\n\n{note}"
    if pipeline_type == "image-text-to-text":
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": user_text}]},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]


def summarize_note(pipe, pipeline_type: str, note: str) -> str:
    messages = build_messages(note, pipeline_type)
    output = pipe(text=messages, **GENERATION_KWARGS) if pipeline_type == "image-text-to-text" \
        else pipe(messages, **GENERATION_KWARGS)
    return output[0]["generated_text"][-1]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True,
                        help="Salida de 5_summarization/sampling.py (muestra_1000.csv)")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model",      required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--model_id",   default=None,
                        help="Sobrescribe el identificador de HuggingFace por defecto para --model")
    parser.add_argument("--max_retries", type=int, default=2,
                        help="Reintentos con semilla de generación distinta ante una respuesta inválida")
    args = parser.parse_args()

    default_id, pipeline_type = MODEL_REGISTRY[args.model]
    model_id = args.model_id or default_id

    print(f"Modelo: {args.model} ({model_id}), pipeline: {pipeline_type}")
    pipe = pipeline(
        pipeline_type,
        model=model_id,
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
    print(f"Resumiendo {len(pending)} de las {len(df)} notas con {args.model}...")

    for idx in tqdm(pending):
        note = df.at[idx, "text"]
        summary = None
        for attempt in range(args.max_retries + 1):
            try:
                candidate = summarize_note(pipe, pipeline_type, note)
                if is_valid_summary(candidate):
                    summary = candidate
                    break
                else:
                    torch.manual_seed(42 + attempt + 1)
            except Exception as e:
                print(f"  Error en la fila {idx}, intento {attempt}: {e}")
        df.at[idx, "summary"] = summary

        if (pending.index(idx) + 1) % 50 == 0:
            df.to_csv(output_path, index=False)

    df.to_csv(output_path, index=False)
    valid = df["summary"].notna().sum()
    print(f"Hecho ({args.model}). Resúmenes válidos: {valid} / {len(df)} ({100*valid/len(df):.1f}%)")
    print(f"Salida guardada -> {output_path}")


if __name__ == "__main__":
    main()
