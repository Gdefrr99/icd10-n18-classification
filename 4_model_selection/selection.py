# -*- coding: utf-8 -*-
"""
Selection: fine-tune 25 clinical Transformer models on a 10K-note / 50-code
subset to select the top-4 models for full training on the N18 task.

The selection uses a reduced N18-adjacent dataset (10K discharge summaries,
50 most-frequent ICD-10-CM codes) to estimate model capacity efficiently.
The top-4 models ranked by F1-weighted on the validation set are then
fine-tuned on the full 23K N18 dataset (Steps 5 and 6).

Usage:
    python 4_model_selection/selection.py \
        --data_dir     data/processed/ \
        --output_dir   results/selection/ \
        --model_name   michiyasunaga/BioLinkBERT-large \
        --epochs       10 \
        --max_samples  10000

To run all 25 models sequentially (on a cluster, use array jobs):
    for model in $(cat 4_model_selection/model_list.txt); do
        python 4_model_selection/selection.py --model_name "$model" ...
    done
"""

import argparse
import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]
MAX_LENGTH = 512
THRESHOLD  = 0.3   # fixed threshold for selection (no per-class tuning)


class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = FocalLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_micro":    f1_score(labels, preds, average="micro",    zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",     default="data/processed/")
    parser.add_argument("--output_dir",   default="results/selection/")
    parser.add_argument("--model_name",   required=True)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--max_samples",  type=int,   default=10000,
                        help="Max training samples for selection")
    parser.add_argument("--lr",           type=float, default=2e-5)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) / args.model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    mlb = MultiLabelBinarizer(classes=N18_LABELS)
    mlb.fit([N18_LABELS])

    def load_split(name, max_n=None):
        df = pd.read_csv(data_dir / f"ehr_icd_{name}_clean.csv")
        df["icd_code"] = df["icd_code"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["icd_code"] = df["icd_code"].apply(
            lambda codes: [c for c in codes if c in N18_LABELS]
        )
        if max_n:
            df = df.sample(n=min(max_n, len(df)), random_state=42)
        return df

    print(f"Loading data (max {args.max_samples} training samples)...")
    df_train = load_split("train", max_n=args.max_samples)
    df_val   = load_split("val")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(df):
        return tokenizer(
            df["text"].tolist(),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    enc_train = tokenize(df_train)
    enc_val   = tokenize(df_val)

    Y_train = mlb.transform(df_train["icd_code"]).tolist()
    Y_val   = mlb.transform(df_val["icd_code"]).tolist()

    ds_train = MultiLabelDataset(enc_train, Y_train)
    ds_val   = MultiLabelDataset(enc_val,   Y_val)

    print(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(N18_LABELS),
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Training...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\nBest val F1-weighted: {results.get('eval_f1_weighted', 0):.4f}")

    summary = {
        "model":         args.model_name,
        "f1_weighted":   results.get("eval_f1_weighted", 0),
        "f1_micro":      results.get("eval_f1_micro", 0),
        "max_samples":   args.max_samples,
    }
    import json
    with open(output_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {output_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()
