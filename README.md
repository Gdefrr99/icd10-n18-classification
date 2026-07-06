# ICD-10-CM Automatic Coding — CKD Subgroup N18

Replication repository for the experimental pipeline described in the Bachelor's Thesis *"Sistema de Codificación Automática ICD-10-CM en Grupos de Riesgo CMS-HCC"* (Universidad de León, 2026).

The experiments focus on multi-label classification of the **Chronic Kidney Disease (CKD) subgroup N18** of ICD-10-CM (7 codes: N18.1–N18.6, N18.9) using clinical discharge summaries from **MIMIC-IV**.

> **Deployment note.** The two best-performing models from this pipeline are integrated into the [icd10_system](https://github.com/Gdefrr99/icd10_system) clinical coding assistant:
> - **Segmentation pipeline**: `PubMedBERT_abstract`, threshold 0.6, Max Pooling.
> - **Summarization pipeline**: `BioLinkBERT-large`, threshold 0.4, trained on notes summarized by MedGemma-27B-it.

---

## Table of Contents

1. [Dataset acquisition](#1-dataset-acquisition)
2. [Building the N18 working set](#2-building-the-n18-working-set)
3. [Repository structure](#3-repository-structure)
4. [Step-by-step pipeline](#4-step-by-step-pipeline)
5. [Hardware requirements](#5-hardware-requirements)
6. [Results](#6-results)
7. [Citation](#7-citation)

---

## 1. Dataset acquisition

Access to MIMIC-IV requires a PhysioNet credentialed account. **No data is included in this repository.**

1. Create an account at [physionet.org](https://physionet.org).
2. Complete the required CITI training courses and sign the data use agreement.
3. Download MIMIC-IV (v2.2 or later) from [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/).
4. From the downloaded archive, you need exactly two files:
   - `hosp/diagnoses_icd.csv.gz` — ICD codes per admission.
   - `note/discharge.csv.gz` — Discharge summary text per admission.

Decompress `discharge.csv.gz`. Then put `diagnoses_icd.csv.gz` and `discharge.csv` files into `data/raw/`.

---

## 2. Building the N18 working set

Run the preprocessing pipeline (see [1_preprocessing/](1_preprocessing/README.md)) to:

1. Join `diagnoses_icd` and `discharge` on `(subject_id, hadm_id)`.
2. Filter to `icd_version = 10` only (ICD-10-CM records).
3. Filter to notes containing at least one N18.x code → **23,358 notes**, 7 labels.
4. Apply clinical text normalization (abbreviation expansion, hard-wrap restoration, section tagging).
5. Stratified split 70/10/20 % → train / validation / test.

```
data/
└── raw/
    ├── diagnoses_icd.csv.gz          # from MIMIC-IV hosp module
    └── discharge.csv              # from MIMIC-IV note module
```

The preprocessing script writes the following files consumed by all downstream steps:

```
data/processed/
├── diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv   # full N18 dataset (23,358 rows)
├── ehr_icd_train_clean.csv                                 # 16,351 rows
├── ehr_icd_val_clean.csv                                   #  2,337 rows
└── ehr_icd_test_clean.csv                                  #  4,670 rows
```

Each CSV has columns: `subject_id`, `hadm_id`, `text` (preprocessed), `icd_code` (list of N18.x codes as a Python literal string).

---

## 3. Repository structure

```
icd10-n18-classification/
├── README.md                        ← this file
├── requirements.txt                 ← Transformer + classical baseline deps
├── requirements_generative.txt      ← extra deps for MedGemma summarization
├── .gitignore
│
├── data/
│   └── README.md                    ← acquisition & build instructions
│
├── 1_preprocessing/
│   ├── README.md
│   └── preprocess.py                ← build processed CSVs from raw MIMIC-IV
│
├── 2_eda/
│   └── README.md                    ← EDA description & figures
│
├── 3_llm_hcc_evaluation/
│   ├── README.md
│   └── gemini_flash_evaluation.py   ← Gemini 2.5 Flash batch evaluation
│
├── 4_model_selection/
│   ├── README.md
│   └── selection.py              ← fine-tune 25 models on 10K notes / 50 codes
│
├── 5_chunking_max_pooling/
│   ├── README.md
│   └── train_chunking.py            ← focal loss + chunking + Max Pooling
│
├── 6_summarization/
│   ├── README.md
│   ├── mlsmote_sampling.py          ← stratified 1K-sample selection
│   └── summarize_medgemma.py        ← MedGemma-27B-it summarization
│
├── 7_classical_baseline/
│   ├── README.md
│   └── baseline.py                  ← TF-IDF / BM25 / BM25+ + OvR
│
└── 8_explainability/
    ├── README.md
    └── icd10_explainability.py      ← Integrated Gradients + Noise Tunnel
```

---

## 4. Step-by-step pipeline

### Step 0 — Install dependencies

```bash
# Core (transformers, sklearn, scispaCy)
pip install -r requirements.txt

# Optional: generative summarization only
pip install -r requirements_generative.txt
```

### Step 1 — Preprocess

```bash
python 1_preprocessing/preprocess.py \
    --diagnoses_csv data/raw/diagnoses_icd.csv \
    --discharge_csv  data/raw/discharge.csv \
    --output_dir     data/processed/
```

### Step 2 (optional) — Exploratory Data Analysis

See [2_eda/README.md](2_eda/README.md).

### Step 3 — LLM evaluation on HCC risk groups

```bash
# Requires a Google Gemini API key and manual copy-paste via gemini.google.com
# See 3_llm_hcc_evaluation/README.md for the prompting protocol.
python 3_llm_hcc_evaluation/gemini_flash_evaluation.py
```

### Step 4 — Model selection

Fine-tunes 25 clinical Transformers on 10,000 notes / 50 most frequent ICD-10-CM codes:

```bash
python 4_model_selection/selection.py \
    --data_dir     data/processed/ \
    --output_dir   models/selection/ \
    --n_samples    10000 \
    --n_labels     50 \
    --epochs       10 \
    --lr           2e-5 \
    --batch_size   16
```

### Step 5 — Chunking + Max Pooling (main classification)

Fine-tunes the 4 selected models on the full N18 dataset with 512-token chunks (128-token overlap), focal loss, and Max Pooling aggregation:

```bash
python 5_chunking_max_pooling/train_chunking.py \
    --data_dir     data/processed/ \
    --output_dir   models/chunking/ \
    --chunk_size   512 \
    --overlap      128 \
    --thresholds   0.4 0.6 \
    --epochs       10 \
    --lr           2e-5 \
    --batch_size   16
```

### Step 6 — Clinical summarization + classification on summaries

**6a. MLSMOTE stratified sampling (1,000 notes):**

```bash
python 6_summarization/mlsmote_sampling.py \
    --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/mlsmote_1000.csv \
    --n_samples  1000
```

**6b. MedGemma-27B-it summarization (requires ≥40 GB VRAM):**

```bash
python 6_summarization/summarize_medgemma.py \
    --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/ehr_n18_summarized.csv
```

**6c. Fine-tune classifiers on summaries:**

Use `5_chunking_max_pooling/train_chunking.py` with `--no_chunking` flag (summaries fit within 512 tokens):

```bash
python 5_chunking_max_pooling/train_chunking.py \
    --data_dir     data/processed/ \
    --data_csv     data/processed/ehr_n18_summarized.csv \
    --output_dir   models/summarized/ \
    --no_chunking \
    --thresholds   0.4 0.6 \
    --epochs       10 \
    --lr           2e-5 \
    --batch_size   16
```

### Step 7 — Classical baseline

```bash
python 7_classical_baseline/baseline.py \
    --data_dir   data/processed/ \
    --output_dir results/baseline/
```

### Step 8 — Explainability

```bash
python 8_explainability/icd10_explainability.py \
    --model_dir  models/summarized/RoBERTa-large-pubmed-mimic3-Voc-hf/ \
    --note_csv   data/processed/ehr_n18_summarized.csv \
    --hadm_id    <HADM_ID> \
    --label      N18.4 \
    --n_ig_steps 20 \
    --nt_samples 10
```

---

## 5. Hardware requirements

| Experiment | Minimum GPU VRAM | Notes |
|---|---|---|
| Selection (25 models × base/large) | 16 GB | A100 40 GB recommended |
| Chunking + Max Pooling (large models) | 24 GB | Multi-GPU supported via `CUDA_VISIBLE_DEVICES` |
| MedGemma-27B-it summarization | 40 GB (bfloat16) | Two A100 40 GB or one A100 80 GB |
| Classical baseline (TF-IDF/BM25) | CPU only | 32 cores recommended |
| Explainability (IG + Noise Tunnel) | 16 GB | |

All Transformer experiments were run on an HPC cluster (SLURM) with NVIDIA A100 GPUs.

---

## 6. Results

### 6.1 LLM evaluation — Gemini 2.5 Flash (specific prompt, 13 HCC groups)

| # | HCC group | ICD codes | N notes | J_real | J_ambos |
|---|---|---|---|---|---|
| 1 | Diabetes mellitus | E08–E13 | 34,608 | 0.456 | — |
| 2 | Congestive heart failure | I11, I42, I50 | 24,527 | 0.326 | — |
| 3 | Vascular disease | I25, I70–I73 | 31,641 | 0.293 | 0.297 |
| 4 | Chronic kidney disease | N18 | 23,358 | 0.310 | 0.351 |
| 5 | COPD & lung disorders | J41–J45, J47, J84 | 25,858 | 0.528 | — |
| 6 | Oncology | C | 22,322 | 0.266 | — |
| 7 | Major psychiatric conditions | F20–F33 | 29,337 | 0.603 | — |
| 8 | Neurological conditions | G20–G83 | 11,216 | 0.340 | 0.343 |
| 9 | Hepatic disease | K7, B18 | 13,045 | 0.353 | 0.353 |
| 10 | HIV | B20 | 825 | 0.055 | — |
| 11 | Amputations | Z89 | 2,213 | 0.485 | 0.487 |
| 12 | Severe hematological disorders | D57, D61, D63, D66, D67, D69 | 19,611 | 0.403 | 0.434 |
| 13 | Severe infections | A40, A41 | 7,493 | 0.407 | 0.409 |

### 6.2 LLM evaluation — Gemini 3 Pro vs Flash (6 selected groups, J_real)

| HCC group | Flash specific | G3 Pro specific | G3 Pro general |
|---|---|---|---|
| Diabetes mellitus | 0.456 | 0.505 | **0.729** |
| Congestive heart failure | 0.326 | 0.443 | **0.698** |
| Vascular disease | 0.293 | 0.530 | **0.648** |
| Chronic kidney disease | 0.310 | 0.432 | **0.826** |
| COPD & lung disorders | 0.528 | 0.497 | **0.750** |
| Oncology | 0.266 | **0.696** | 0.680 |

### 6.3 Model selection — Top-4 selection (10K notes, 50 labels)

| Model | F1-weighted | F1-micro | F1-macro |
|---|---|---|---|
| BioLinkBERT-large | **0.5162** | **0.5490** | **0.4396** |
| RoBERTa-large-pubmed-mimic3-Voc-hf | 0.5091 | 0.5404 | 0.4326 |
| BlueBERT-pubmed-mimic-large-uncased | 0.4936 | 0.5278 | 0.4156 |
| PubMedBERT_abstract | 0.4737 | 0.5255 | 0.3808 |

### 6.4 Chunking + Max Pooling — Global metrics (N18 test set, 4,670 notes)

*Underline: best threshold per model per metric. Bold+underline: global best.*

| Model | Thr | Acc | Prec-Mi | Prec-Ma | Prec-W | Rec-Mi | Rec-Ma | Rec-W | F1-Mi | F1-Ma | F1-W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BioLinkBERT-large | 0.4 | 0.371 | 0.542 | 0.645 | 0.590 | **0.886** | **0.769** | **0.886** | 0.673 | **0.675** | 0.686 |
| BioLinkBERT-large | 0.6 | 0.553 | 0.655 | **0.744** | 0.717 | 0.826 | 0.641 | 0.826 | 0.730 | 0.645 | 0.740 |
| BlueBERT-large | 0.4 | 0.319 | 0.526 | 0.606 | 0.563 | 0.898 | 0.740 | 0.898 | 0.664 | 0.615 | 0.677 |
| BlueBERT-large | 0.6 | 0.579 | 0.668 | 0.735 | 0.719 | 0.835 | 0.721 | 0.835 | 0.742 | 0.711 | 0.750 |
| PubMedBERT_abstract | 0.4 | 0.171 | 0.475 | 0.589 | 0.507 | **0.944** | **0.815** | **0.944** | 0.632 | 0.657 | 0.645 |
| PubMedBERT_abstract | 0.6 | **0.593** | **0.693** | 0.729 | **0.724** | 0.810 | 0.741 | 0.810 | **0.747** | **0.723** | **0.752** |
| RoBERTa-large-pubmed-mimic3-Voc-hf | 0.4 | 0.133 | 0.464 | 0.612 | 0.501 | 0.943 | 0.793 | 0.943 | 0.622 | 0.661 | 0.637 |
| RoBERTa-large-pubmed-mimic3-Voc-hf | 0.6 | 0.557 | 0.661 | 0.701 | 0.686 | 0.827 | 0.753 | 0.827 | 0.735 | 0.716 | 0.740 |

### 6.5 Chunking + Max Pooling — Per-code metrics, threshold 0.4

| Code | Prec BioL | Prec Blue | Prec PubM | Prec Rob | Rec BioL | Rec Blue | Rec PubM | Rec Rob | F1 BioL | F1 Blue | F1 PubM | F1 Rob | Support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N18.1 | **0.85** | 0.80 | 0.69 | 0.73 | **0.65** | 0.24 | **0.65** | **0.65** | **0.73** | 0.36 | 0.67 | 0.69 | 17 |
| N18.2 | **0.85** | 0.70 | 0.83 | 0.75 | 0.57 | **0.59** | 0.58 | 0.58 | **0.68** | 0.64 | **0.68** | 0.65 | 157 |
| N18.3 | **0.74** | 0.61 | 0.40 | 0.39 | 0.77 | 0.82 | 0.97 | **0.98** | **0.76** | 0.70 | 0.56 | 0.56 | 1,361 |
| N18.4 | 0.54 | **0.61** | 0.57 | 0.60 | 0.76 | 0.74 | **0.77** | 0.75 | 0.63 | **0.67** | 0.66 | 0.66 | 396 |
| N18.5 | 0.46 | 0.36 | 0.48 | **0.69** | 0.65 | **0.83** | 0.76 | 0.62 | 0.54 | 0.51 | 0.59 | **0.65** | 113 |
| N18.6 | 0.66 | **0.73** | 0.71 | 0.71 | **0.99** | 0.98 | **0.99** | **0.99** | 0.79 | **0.84** | 0.83 | 0.82 | 937 |
| N18.9 | 0.43 | 0.42 | **0.44** | 0.42 | **1.00** | 0.99 | 0.99 | **1.00** | 0.60 | 0.59 | **0.61** | 0.59 | 1,713 |

BioL = BioLinkBERT-large, Blue = BlueBERT-large, PubM = PubMedBERT_abstract, Rob = RoBERTa-large-pubmed-mimic3-Voc-hf.

### 6.6 Chunking + Max Pooling — Per-code metrics, threshold 0.6

| Code | Prec BioL | Prec Blue | Prec PubM | Prec Rob | Rec BioL | Rec Blue | Rec PubM | Rec Rob | F1 BioL | F1 Blue | F1 PubM | F1 Rob | Support |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N18.1 | 0.67 | 0.69 | **0.79** | **0.79** | 0.12 | 0.53 | **0.65** | **0.65** | 0.20 | 0.60 | **0.71** | **0.71** | 17 |
| N18.2 | 0.83 | **0.84** | 0.81 | 0.75 | 0.55 | 0.58 | 0.58 | **0.59** | 0.66 | **0.69** | 0.68 | 0.66 | 157 |
| N18.3 | **0.93** | 0.92 | 0.90 | 0.82 | 0.72 | 0.73 | 0.73 | **0.75** | **0.81** | **0.81** | **0.81** | 0.78 | 1,361 |
| N18.4 | **0.82** | 0.76 | 0.69 | 0.59 | 0.68 | 0.69 | 0.72 | **0.76** | **0.74** | 0.72 | 0.71 | 0.67 | 396 |
| N18.5 | **0.72** | 0.66 | 0.57 | 0.62 | 0.53 | 0.63 | 0.69 | **0.70** | 0.61 | 0.64 | 0.62 | **0.66** | 113 |
| N18.6 | 0.74 | 0.74 | 0.77 | **0.79** | 0.98 | 0.97 | **0.99** | 0.97 | 0.84 | 0.84 | **0.87** | **0.87** | 937 |
| N18.9 | 0.50 | 0.53 | **0.56** | 0.54 | 0.91 | **0.92** | 0.83 | 0.86 | 0.65 | **0.67** | **0.67** | 0.66 | 1,713 |

### 6.7 Summarization quality — ROUGE metrics (1,000 MLSMOTE samples)

| Model | Valid | R1-P | R1-R | R1-F1 | R2-P | R2-R | R2-F1 | RL-P | RL-R | RL-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3-OpenBioLLM-8B | 933 | 0.701 | 0.041 | 0.076 | 0.298 | 0.019 | 0.034 | 0.481 | 0.027 | 0.050 |
| Bio-Medical-Llama-3-8B | 995 | 0.803 | 0.049 | 0.088 | 0.422 | 0.022 | 0.039 | 0.567 | 0.030 | 0.054 |
| MedGemma-1.5-4b-it | 955 | 0.845 | **0.111** | **0.193** | 0.434 | **0.056** | **0.097** | 0.518 | **0.068** | **0.118** |
| MedGemma-27B-it | **1000** | **0.855** | 0.102 | 0.179 | **0.397** | 0.047 | 0.082 | **0.482** | 0.057 | 0.100 |

### 6.8 Comparison: Max Pooling vs. summarized notes (23,358 notes, N18 test set)

| Metric | Max Pooling value | Model | Thr | Summarized value | Model | Thr |
|---|---|---|---|---|---|---|
| Accuracy | 0.593 | PubM | 0.6 | **0.770** | PubM | 0.4 |
| Precision micro | 0.693 | PubM | 0.6 | **0.810** | Rob | 0.6 |
| Precision macro | 0.744 | BioL | 0.6 | **0.840** | Blue | 0.6 |
| Precision weighted | 0.724 | PubM | 0.6 | **0.844** | Rob | 0.6 |
| Recall micro | **0.944** | PubM | 0.4 | 0.787 | Blue | 0.4 |
| Recall macro | **0.815** | PubM | 0.4 | 0.637 | Blue | 0.4 |
| Recall weighted | **0.944** | PubM | 0.4 | 0.787 | Blue | 0.4 |
| F1 micro | 0.747 | PubM | 0.6 | **0.783** | BioL | 0.4 |
| F1 macro | **0.723** | PubM | 0.6 | 0.691 | BioL | 0.4 |
| F1 weighted | 0.752 | PubM | 0.6 | **0.776** | BioL | 0.4 |

### 6.9 Classical baseline — Best results per metric (N18 test set)

| Metric | Value | Representation | BM25 params | Classifier | Features |
|---|---|---|---|---|---|
| Accuracy | 0.602 | TF-IDF uni+bigr | — | LinearSVC | 50K |
| Precision micro | 0.844 | TF-IDF uni+bigr full vocab | — | LogReg | 930K |
| Precision macro | 0.747 | BM25 unigrams full vocab | k₁=2, b=0.25 | LogReg | 103K |
| Precision weighted | 0.810 | TF-IDF uni+bigr full vocab | — | LinearSVC | 930K |
| Recall micro | 0.620 | BM25+ uni+bigr | k₁=2, b=1, δ=0.5 | LogReg | 50K |
| Recall macro | 0.402 | BM25+ unigrams filtered | k₁=1.5, b=1, δ=0.5 | LinearSVC | 53K |
| Recall weighted | 0.620 | BM25+ uni+bigr | k₁=2, b=1, δ=0.5 | LogReg | 50K |
| F1 micro | 0.690 | TF-IDF uni+bigr | — | LinearSVC | 50K |
| F1 macro | 0.451 | BM25+ unigrams filtered | k₁=1.5, b=1, δ=0.5 | LinearSVC | 53K |
| F1 weighted | 0.660 | TF-IDF uni+bigr | — | LinearSVC | 50K |

---

## 7. Citation

If you use this code or results in your work, please cite:

```bibtex
@thesis{defrancisco2026icd10,
  author  = {de Francisco Rodríguez, Gonzalo},
  title   = {Sistema de Codificación Automática {ICD-10-CM} en Grupos de Riesgo {CMS-HCC}},
  school  = {Universidad de León},
  year    = {2026},
  type    = {Bachelor's Thesis}
}
```

### Key references

- Johnson et al. (2023). MIMIC-IV. *Scientific Data*, 10, 1. DOI: 10.1038/s41597-022-01899-x
- Yasunaga et al. (2022). LinkBERT. *ACL 2022*.
- Gu et al. (2021). PubMedBERT. *ACL 2021*.
- Sundararajan et al. (2017). Integrated Gradients. *ICML 2017*.
- Lin & Fox (2004). ROUGE. *ACL 2004 Workshop*.
