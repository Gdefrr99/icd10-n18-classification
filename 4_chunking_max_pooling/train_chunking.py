# -*- coding: utf-8 -*-
"""
Ajuste fino de los modelos Transformer clínicos para la clasificación
multietiqueta del subgrupo N18, mediante segmentación (chunks de 512 tokens
con solapamiento de 128) y agregación por Max Pooling.

Modelos entrenados en este paso:
  - michiyasunaga/BioLinkBERT-large
  - RoBERTa-large-PM-M3-Voc-hf  (descarga manual, ver 3_model_selection/README.md)
  - bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16
  - microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract

Uso:
    python 4_chunking_max_pooling/train_chunking.py \
        --data_dir   data/processed/ \
        --output_dir models/chunking/ \
        --model_name michiyasunaga/BioLinkBERT-large \
        --thresholds 0.4 0.6 \
        --epochs     10

Para entrenar sobre los resúmenes de MedGemma-27B-it (Paso 5), pasar
--data_dir apuntando al directorio con las particiones resumidas
(ehr_icd_{train,val,test}_clean.csv generadas por
5_summarization/build_summarized_splits.py) junto con --no_chunking.
"""

import argparse
import ast
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, set_seed, DataCollatorWithPadding
)
from datasets import Dataset as HFDataset

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
set_seed(RANDOM_SEED)

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]


# ---------------------------------------------------------------------------
# Segmentación (chunking)
# ---------------------------------------------------------------------------

def chunk_text(text: str, tokenizer, chunk_size: int = 512, overlap: int = 128):
    """Divide el texto tokenizado en fragmentos solapados de `chunk_size` tokens."""
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    stride = chunk_size - 2 - overlap  # reservar espacio para [CLS] y [SEP]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size - 2, len(tokens))
        chunk_ids = [tokenizer.cls_token_id] + tokens[start:end] + [tokenizer.sep_token_id]
        attention_mask = [1] * len(chunk_ids)
        pad_len = chunk_size - len(chunk_ids)
        chunk_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        chunks.append({"input_ids": chunk_ids, "attention_mask": attention_mask})
        if end == len(tokens):
            break
        start += stride
    return chunks


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_split(path, mlb: MultiLabelBinarizer) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    df["icd_code"] = df["icd_code"].apply(
        lambda codes: [c for c in codes if c in N18_LABELS]
    )
    df["labels"] = list(mlb.transform(df["icd_code"]).astype(np.float32))
    return df


# ---------------------------------------------------------------------------
# Inferencia con Max Pooling
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_with_max_pooling(model, tokenizer, texts, device, chunk_size=512, overlap=128, batch_size=8):
    model.eval()
    all_logits = []
    for text in texts:
        chunks = chunk_text(text, tokenizer, chunk_size, overlap)
        chunk_logits = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            input_ids = torch.tensor([c["input_ids"] for c in batch]).to(device)
            attention_mask = torch.tensor([c["attention_mask"] for c in batch]).to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            chunk_logits.append(out.logits.cpu())
        # Max Pooling: el logit del documento es el máximo entre todos sus fragmentos
        doc_logits = torch.cat(chunk_logits, dim=0).max(dim=0).values
        all_logits.append(doc_logits)
    return torch.stack(all_logits)


def evaluate_at_threshold(logits: torch.Tensor, labels_true: np.ndarray, threshold: float):
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":           accuracy_score(labels_true, preds),
        "f1_weighted":        f1_score(labels_true, preds, average="weighted", zero_division=0),
        "f1_micro":           f1_score(labels_true, preds, average="micro",    zero_division=0),
        "f1_macro":           f1_score(labels_true, preds, average="macro",    zero_division=0),
        "precision_weighted": precision_score(labels_true, preds, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(labels_true, preds,    average="weighted", zero_division=0),
        "report":             classification_report(labels_true, preds, target_names=N18_LABELS, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/processed/")
    parser.add_argument("--output_dir",  default="models/chunking/")
    parser.add_argument("--model_name",  default="michiyasunaga/BioLinkBERT-large",
                        help="Nombre en HuggingFace o ruta local")
    parser.add_argument("--chunk_size",  type=int, default=512)
    parser.add_argument("--overlap",     type=int, default=128)
    parser.add_argument("--thresholds",  type=float, nargs="+", default=[0.4, 0.6])
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--weight_decay",type=float, default=0.01)
    parser.add_argument("--warmup_ratio",type=float, default=0.1)
    parser.add_argument("--no_chunking", action="store_true",
                        help="Desactiva la segmentación (para notas resumidas que caben en 512 tokens)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    model_short = args.model_name.split("/")[-1]
    model_out   = output_dir / model_short
    model_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    mlb = MultiLabelBinarizer(classes=N18_LABELS)
    mlb.fit([N18_LABELS])

    print(f"Cargando tokenizador: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Cargando particiones de datos...")
    df_train = load_split(data_dir / "ehr_icd_train_clean.csv", mlb)
    df_val   = load_split(data_dir / "ehr_icd_val_clean.csv",   mlb)
    df_test  = load_split(data_dir / "ehr_icd_test_clean.csv",  mlb)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.chunk_size,
            padding=False,
        )
        enc["labels"] = batch["labels"]
        return enc

    train_hf = HFDataset.from_pandas(df_train[["text", "labels"]])
    val_hf   = HFDataset.from_pandas(df_val[["text",   "labels"]])
    train_hf = train_hf.map(tokenize, batched=True, remove_columns=["text"])
    val_hf   = val_hf.map(tokenize,   batched=True, remove_columns=["text"])
    train_hf.set_format("torch")
    val_hf.set_format("torch")

    print(f"Cargando modelo: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(N18_LABELS),
        problem_type="multi_label_classification",
    ).to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs >= 0.5).astype(int)
        return {
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "f1_micro":    f1_score(labels, preds, average="micro",    zero_division=0),
        }

    training_args = TrainingArguments(
        output_dir=str(model_out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Entrenando...")
    trainer.train()

    best_model_path = model_out / "best_model"
    trainer.save_model(str(best_model_path))
    print(f"Mejor modelo guardado -> {best_model_path}")

    # Evaluación con Max Pooling para cada umbral
    print("Evaluando con Max Pooling...")
    logits = predict_with_max_pooling(
        model, tokenizer, df_test["text"].tolist(), device,
        chunk_size=args.chunk_size, overlap=args.overlap,
    )
    labels_true = np.array(df_test["labels"].tolist())

    for thr in args.thresholds:
        metrics = evaluate_at_threshold(logits, labels_true, thr)
        print(f"\n--- Umbral {thr} ---")
        for k, v in metrics.items():
            if k != "report":
                print(f"  {k}: {v:.4f}")
        print(metrics["report"])

        report_path = model_out / f"classification_report_thr{thr}.txt"
        report_path.write_text(metrics["report"])

    print("Hecho.")


if __name__ == "__main__":
    main()
