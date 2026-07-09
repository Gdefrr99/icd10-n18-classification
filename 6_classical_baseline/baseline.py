# -*- coding: utf-8 -*-
"""
Línea base clásica multietiqueta para la clasificación ICD-10-CM del
subgrupo N18.

Representaciones: TF-IDF, BM25, BM25+
Clasificadores:   Regresión Logística, LinearSVC (One-vs-Rest)
Vocabulario:      Varias configuraciones (36K, 50K, 53K, 103K, 930K características)

Preprocesado adicional:
  - Minúsculas + eliminación de puntuación
  - Lematización con scispaCy en_core_sci_sm
  - Eliminación de stop words (estándar + extensión clínica)
  - Eliminación de tokens de 1-2 caracteres y cadenas numéricas puras

Uso:
    python 6_classical_baseline/baseline.py \
        --data_dir   data/processed/ \
        --output_dir results/baseline/
"""

import argparse
import ast
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import parallel_backend
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

import spacy

warnings.filterwarnings("ignore")

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]

_SLURM_CORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
os.environ["OMP_NUM_THREADS"]      = str(_SLURM_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(_SLURM_CORES)
os.environ["MKL_NUM_THREADS"]      = str(_SLURM_CORES)

CLINICAL_STOPWORDS = {
    "patient", "mg", "daily", "history", "given", "noted",
    "follow", "continued", "started", "received", "also",
    "however", "without", "well", "including", "used",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def lemmatize_texts(texts, nlp, batch_size=256):
    """Lemmatize using scispaCy; remove stop words, punctuation, short tokens."""
    stop_words = nlp.Defaults.stop_words | CLINICAL_STOPWORDS
    result = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.lemma_.lower() not in stop_words
            and len(token.text) > 2
            and not token.text.isdigit()
        ]
        result.append(" ".join(tokens))
    return result


# ---------------------------------------------------------------------------
# BM25 vectorization
# ---------------------------------------------------------------------------

def bm25_transform(tokenized_corpus, tokenized_query, k1=1.5, b=0.75, delta=0.0):
    """Compute BM25 (or BM25+ if delta > 0) score matrix."""
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
    # BM25+ adds delta floor
    rows, cols, data = [], [], []
    vocab = {w: i for i, w in enumerate(bm25.idf.keys())}
    N = len(tokenized_corpus)
    avgdl = bm25.avgdl

    for doc_idx, doc_tokens in enumerate(tokenized_corpus):
        tf = {}
        for t in doc_tokens:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc_tokens)
        for term, freq in tf.items():
            if term not in vocab:
                continue
            idf_val = bm25.idf.get(term, 0.0)
            tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
            score = idf_val * (tf_norm + delta)
            rows.append(doc_idx)
            cols.append(vocab[term])
            data.append(score)

    return sp.csr_matrix((data, (rows, cols)), shape=(N, len(vocab)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true, y_pred, label_names):
    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "f1_weighted":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro":           f1_score(y_true, y_pred, average="micro",    zero_division=0),
        "f1_macro":           f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_true, y_pred, average="weighted",    zero_division=0),
        "report":             classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/processed/")
    parser.add_argument("--output_dir", default="results/baseline/")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading scispaCy model...")
    nlp = spacy.load("en_core_sci_sm", disable=["ner", "parser"])

    mlb = MultiLabelBinarizer(classes=N18_LABELS)
    mlb.fit([N18_LABELS])

    def load_split(name):
        df = pd.read_csv(data_dir / f"ehr_icd_{name}_clean.csv")
        df["icd_code"] = df["icd_code"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        df["icd_code"] = df["icd_code"].apply(
            lambda codes: [c for c in codes if c in N18_LABELS]
        )
        return df

    print("Loading data...")
    df_train = load_split("train")
    df_test  = load_split("test")

    print("Lemmatizing texts (train)...")
    train_texts = lemmatize_texts(df_train["text"].tolist(), nlp)
    print("Lemmatizing texts (test)...")
    test_texts  = lemmatize_texts(df_test["text"].tolist(),  nlp)

    Y_train = mlb.transform(df_train["icd_code"])
    Y_test  = mlb.transform(df_test["icd_code"])

    classifiers = {
        "LogReg":    LogisticRegression(max_iter=1000, n_jobs=-1),
        "LinearSVC": LinearSVC(max_iter=2000),
    }

    # TF-IDF configurations
    tfidf_configs = [
        {"ngram_range": (1, 1), "max_features": None,   "label": "TF-IDF uni full (36K)"},
        {"ngram_range": (1, 2), "max_features": None,   "label": "TF-IDF uni+bigr full (930K)"},
        {"ngram_range": (1, 1), "max_features": 50000,  "label": "TF-IDF uni 50K"},
        {"ngram_range": (1, 2), "max_features": 50000,  "label": "TF-IDF uni+bigr 50K"},
    ]

    results = []

    print("\n=== TF-IDF experiments ===")
    for cfg in tfidf_configs:
        vec = TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            max_features=cfg["max_features"],
            min_df=2, max_df=0.95,
        )
        X_train = vec.fit_transform(train_texts)
        X_test  = vec.transform(test_texts)
        n_feat  = X_train.shape[1]

        for clf_name, clf_base in classifiers.items():
            clf = OneVsRestClassifier(clf_base, n_jobs=-1)
            clf.fit(X_train, Y_train)
            preds = clf.predict(X_test)
            metrics = evaluate(Y_test, preds, N18_LABELS)
            row = {"config": cfg["label"], "clf": clf_name, "features": n_feat}
            row.update({k: v for k, v in metrics.items() if k != "report"})
            results.append(row)
            print(f"  {cfg['label']} | {clf_name} → F1-w={metrics['f1_weighted']:.4f}")

    print("\n=== BM25 / BM25+ grid search ===")
    train_tok = [t.split() for t in train_texts]
    test_tok  = [t.split() for t in test_texts]

    bm25_variants = [
        {"delta": 0.0, "label": "BM25"},
        {"delta": 0.5, "label": "BM25+"},
        {"delta": 1.0, "label": "BM25+"},
        {"delta": 1.5, "label": "BM25+"},
    ]
    k1_grid = [0.5, 1.0, 1.5, 2.0]
    b_grid  = [0.25, 0.5, 0.75, 1.0]

    best_f1w = 0.0
    for bm25_v in bm25_variants:
        for k1 in k1_grid:
            for b in b_grid:
                X_train_bm = bm25_transform(train_tok, train_tok, k1=k1, b=b, delta=bm25_v["delta"])
                X_test_bm  = bm25_transform(test_tok,  train_tok, k1=k1, b=b, delta=bm25_v["delta"])
                for clf_name, clf_base in classifiers.items():
                    clf = OneVsRestClassifier(clf_base, n_jobs=-1)
                    clf.fit(X_train_bm, Y_train)
                    preds = clf.predict(X_test_bm)
                    metrics = evaluate(Y_test, preds, N18_LABELS)
                    row = {
                        "config": f"{bm25_v['label']} k1={k1} b={b} delta={bm25_v['delta']}",
                        "clf": clf_name, "features": X_train_bm.shape[1],
                    }
                    row.update({k: v for k, v in metrics.items() if k != "report"})
                    results.append(row)
                    if metrics["f1_weighted"] > best_f1w:
                        best_f1w = metrics["f1_weighted"]
                        print(f"  New best F1-w={best_f1w:.4f} | {row['config']} | {clf_name}")

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nAll results saved → {results_path}")

    # Print best per metric
    print("\n=== Best result per metric ===")
    numeric_cols = ["accuracy", "f1_weighted", "f1_micro", "f1_macro",
                    "precision_weighted", "recall_weighted"]
    for col in numeric_cols:
        best_row = results_df.loc[results_df[col].idxmax()]
        print(f"  {col:20s}: {best_row[col]:.4f} | {best_row['config']} | {best_row['clf']}")


if __name__ == "__main__":
    main()