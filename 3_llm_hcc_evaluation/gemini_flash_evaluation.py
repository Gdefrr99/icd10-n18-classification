# -*- coding: utf-8 -*-
"""
Zero-shot ICD-10-CM N18.x prediction with Gemini 2.5 Flash and Gemini 3 Pro.

Evaluates two prompting strategies on the N18 test set:
  - specific:  prompt explicitly names the N18.x codes to predict
  - general:   prompt asks for any ICD-10-CM codes present

Usage:
    python 3_llm_hcc_evaluation/gemini_flash_evaluation.py \
        --data_csv   data/processed/ehr_icd_test_clean.csv \
        --output_dir results/llm/ \
        --model      gemini-2.5-flash \
        --strategy   specific

Requirements:
    pip install google-generativeai pandas scikit-learn tqdm
    export GOOGLE_API_KEY="your_api_key"
"""

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import google.generativeai as genai
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]

PROMPT_SPECIFIC = """\
You are an expert clinical coder specializing in ICD-10-CM.

Read the following clinical discharge note and identify which of the following \
Chronic Kidney Disease (CKD) ICD-10-CM codes are documented in the note:

  N18.1  - Chronic kidney disease, stage 1
  N18.2  - Chronic kidney disease, stage 2 (mild)
  N18.3  - Chronic kidney disease, stage 3 (moderate)
  N18.4  - Chronic kidney disease, stage 4 (severe)
  N18.5  - Chronic kidney disease, stage 5
  N18.6  - End stage renal disease
  N18.9  - Chronic kidney disease, unspecified

Instructions:
- Assign only codes that are explicitly documented or clearly implied by the clinical content.
- You may assign more than one code if applicable.
- Respond ONLY with a JSON array of codes, e.g.: ["N18.3", "N18.6"]
- If no N18.x code applies, return an empty array: []

Clinical note:
{note}
"""

PROMPT_GENERAL = """\
You are an expert clinical coder specializing in ICD-10-CM.

Read the following clinical discharge note and identify all ICD-10-CM codes that \
are documented in the note. Focus specifically on any codes related to \
Chronic Kidney Disease (codes in the N18 category).

Instructions:
- Assign only codes that are explicitly documented or clearly implied by the clinical content.
- You may assign more than one code if applicable.
- Respond ONLY with a JSON array of ICD-10-CM codes, e.g.: ["N18.3", "E11.9"]
- If no codes apply, return an empty array: []

Clinical note:
{note}
"""

GENERATION_CONFIG = {
    "temperature":     0.0,
    "top_p":           1.0,
    "top_k":           1,
    "max_output_tokens": 256,
}


def parse_response(response_text: str, strategy: str) -> list[str]:
    """Extract N18.x codes from model response."""
    try:
        codes = json.loads(response_text.strip())
        if isinstance(codes, list):
            n18_codes = [c for c in codes if re.match(r'^N18\.[1-9]$', str(c))]
            return n18_codes
    except json.JSONDecodeError:
        pass

    # Fallback: regex extraction
    matches = re.findall(r'N18\.[1-9]', response_text)
    return list(set(matches))


def evaluate(y_true, y_pred, label_names):
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "f1_weighted":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro":       f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro":       f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro":   recall_score(y_true, y_pred, average="micro",    zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   required=True)
    parser.add_argument("--output_dir", default="results/llm/")
    parser.add_argument("--model",      default="gemini-2.5-flash",
                        choices=["gemini-2.5-flash", "gemini-3-pro"])
    parser.add_argument("--strategy",   default="specific",
                        choices=["specific", "general"])
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to N samples (for debugging)")
    parser.add_argument("--sleep",       type=float, default=1.0,
                        help="Seconds to sleep between API calls (rate limiting)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY environment variable before running.")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=args.model,
        generation_config=GENERATION_CONFIG,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["icd_code"] = df["icd_code"].apply(
        lambda codes: [c for c in codes if c in N18_LABELS]
    )

    if args.max_samples:
        df = df.head(args.max_samples)

    prompt_template = PROMPT_SPECIFIC if args.strategy == "specific" else PROMPT_GENERAL

    mlb = MultiLabelBinarizer(classes=N18_LABELS)
    mlb.fit([N18_LABELS])

    predictions = []
    errors = 0
    checkpoint_path = output_dir / f"{args.model}_{args.strategy}_checkpoint.csv"

    # Resume from checkpoint
    done_idx = set()
    if checkpoint_path.exists():
        df_ckpt = pd.read_csv(checkpoint_path)
        done_idx = set(df_ckpt.index.tolist())
        print(f"Resuming: {len(done_idx)} already processed.")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx in done_idx:
            continue
        prompt = prompt_template.format(note=row["text"][:6000])  # safety truncation
        try:
            response = model.generate_content(prompt)
            raw = response.text
        except Exception as e:
            print(f"  Error on row {idx}: {e}")
            raw = "[]"
            errors += 1

        pred_codes = parse_response(raw, args.strategy)
        predictions.append({
            "idx":      idx,
            "true":     row["icd_code"],
            "pred":     pred_codes,
            "raw":      raw,
        })

        # Checkpoint every 100 rows
        if len(predictions) % 100 == 0:
            pd.DataFrame(predictions).to_csv(checkpoint_path, index=False)

        time.sleep(args.sleep)

    pd.DataFrame(predictions).to_csv(checkpoint_path, index=False)

    Y_true = mlb.transform(df["icd_code"])
    Y_pred = mlb.transform([p["pred"] for p in predictions])

    metrics = evaluate(Y_true, Y_pred, N18_LABELS)
    metrics["model"]    = args.model
    metrics["strategy"] = args.strategy
    metrics["errors"]   = errors
    metrics["n_samples"] = len(df)

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    results_path = output_dir / f"{args.model}_{args.strategy}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
