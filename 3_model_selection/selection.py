# -*- coding: utf-8 -*-
"""
Selección de modelos: ajuste fino de 25 Transformers biomédicos sobre el
conjunto de 10.000 notas / 50 códigos más frecuentes.

Este subconjunto NO está restringido al subgrupo N18: se construye a partir
del dataset completo ICD-10-CM mediante
3_model_selection/build_selection_dataset.py. El objetivo es identificar qué 
4 arquitecturas ofrecen el mayor rendimiento de base
antes de ajustarlas sobre el N18 completo (Pasos 4 y 5).

Todos los modelos se entrenan con los mismos hiperparámetros: partición
70/10/20, entrada de 512 tokens, learning rate 2e-5, batch size 16, 10
épocas (guardado por época, selección del mejor modelo en validación según
F1-weighted), función de pérdida Binary Cross-Entropy estándar, weight decay
0.01, warmup ratio 0.1, semilla 42, umbral de aceptación 0.3.

Uso:
    python 3_model_selection/selection.py \
        --data_csv     data/processed/seleccion_10000.csv \
        --output_dir   results/selection/ \
        --model_name   michiyasunaga/BioLinkBERT-large

Para ejecutar los 25 modelos secuencialmente (usar array jobs en un clúster):
    while IFS= read -r model; do
        python 3_model_selection/selection.py --model_name "$model" ...
    done < 3_model_selection/model_list.txt
"""

import argparse
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
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

RANDOM_SEED = 42
MAX_LENGTH  = 512
THRESHOLD   = 0.3


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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= THRESHOLD).astype(int)
    return {
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "f1_micro":    f1_score(labels, preds, average="micro",    zero_division=0),
    }


def load_and_split(data_csv: str):
    """Carga el conjunto de selección y genera la partición 70/10/20."""
    df = pd.read_csv(data_csv)
    df["icd_code"] = df["icd_code"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["icd_code"])

    splitter_tv = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_SEED
    )
    trainval_idx, test_idx = next(splitter_tv.split(df, Y))
    df_trainval, Y_trainval = df.iloc[trainval_idx].reset_index(drop=True), Y[trainval_idx]

    splitter_v = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.10 / 0.80, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(splitter_v.split(df_trainval, Y_trainval))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, mlb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   required=True,
                         help="Salida de 3_model_selection/build_selection_dataset.py")
    parser.add_argument("--output_dir", default="results/selection/")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.model_name.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando y particionando el conjunto de selección...")
    df_train, df_val, mlb = load_and_split(args.data_csv)
    n_labels = len(mlb.classes_)
    print(f"Etiquetas posibles: {n_labels}")

    print(f"Cargando tokenizador: {args.model_name}")
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

    print(f"Cargando modelo: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=n_labels,
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
        seed=RANDOM_SEED,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Entrenando...")
    trainer.train()

    results = trainer.evaluate()
    print(f"\nMejor F1-weighted en validación: {results.get('eval_f1_weighted', 0):.4f}")

    summary = {
        "model":       args.model_name,
        "f1_weighted": results.get("eval_f1_weighted", 0),
        "f1_micro":    results.get("eval_f1_micro", 0),
        "n_labels":    n_labels,
    }
    with open(output_dir / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Resumen guardado -> {output_dir / 'selection_summary.json'}")


if __name__ == "__main__":
    main()