# Step 7 — Classical Baseline (TF-IDF / BM25 / BM25+)

Reproduces the classical multi-label baseline from section 5.4 of the thesis.

## Method

- **Representations**: TF-IDF (unigrams and bi-grams), BM25, BM25+
- **Classifiers**: Logistic Regression and LinearSVC via One-vs-Rest (OvR) decomposition
- **Vocabulary sizes explored**: 36K (unigram full), 50K, 53K, 103K, 930K (uni+bigram full)
- **Text preprocessing**: scispaCy `en_core_sci_sm` lemmatization, clinical stop word removal, removal of tokens < 3 chars or purely numeric

## Requirements

```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
```

## Usage

```bash
python 7_classical_baseline/baseline.py \
    --data_dir   data/processed/ \
    --output_dir results/baseline/
```

Results are written to `results/baseline/baseline_results.csv`.

## Results (test set, best configurations)

| Representation | Features | Classifier | Accuracy | F1-weighted | F1-micro | F1-macro | Prec-micro | Recall-micro |
|---|---|---|---|---|---|---|---|---|
| TF-IDF uni | 36K | LinearSVC | 0.571 | 0.720 | 0.756 | 0.436 | 0.766 | 0.747 |
| TF-IDF uni+bigr | 930K | LinearSVC | 0.562 | 0.703 | 0.745 | 0.412 | **0.844** | 0.666 |
| TF-IDF uni | **50K** | LinearSVC | **0.592** | **0.736** | **0.770** | **0.472** | 0.794 | **0.748** |
| TF-IDF uni+bigr | 50K | LinearSVC | 0.581 | 0.724 | 0.760 | 0.449 | 0.805 | 0.719 |
| BM25+ (best) | 53K | LogReg | 0.581 | 0.725 | 0.754 | 0.448 | 0.789 | 0.723 |

> **Key finding**: Large vocabularies (930K or 103K) only improve micro precision. All other metrics (F1-weighted, F1-micro, F1-macro, recall, accuracy) peak at 50K or 53K features, where the balance between vocabulary coverage and regularization is optimal.

## BM25 hyperparameter grid

The script runs a full grid search over:

| Parameter | Values |
|---|---|
| k1 | 0.5, 1.0, 1.5, 2.0 |
| b | 0.25, 0.5, 0.75, 1.0 |
| delta (BM25+) | 0.0 (BM25), 0.5, 1.0, 1.5 |

Best BM25 configuration: `k1=1.5, b=0.75, delta=1.0` (BM25+), `max_features=53K`.
