# Step 2 — Exploratory Data Analysis (EDA)

After preprocessing (Step 1), run this EDA to understand the dataset characteristics before model training.

## Dataset statistics

The N18 sub-cohort extracted from MIMIC-IV contains:

| Split | Notes | % |
|---|---|---|
| Train | 16,351 | 70% |
| Validation | 2,336 | 10% |
| Test | 4,671 | 20% |
| **Total** | **23,358** | 100% |

The 7 ICD-10-CM N18.x codes and their distribution in the full dataset:

| Code | Description | Count | % |
|---|---|---|---|
| N18.9 | CKD, unspecified | 8,571 | 36.7% |
| N18.3 | CKD, stage 3 (moderate) | 6,796 | 29.1% |
| N18.6 | End stage renal disease | 4,673 | 20.0% |
| N18.4 | CKD, stage 4 (severe) | 1,986 | 8.5% |
| N18.2 | CKD, stage 2 (mild) | 795 | 3.4% |
| N18.5 | CKD, stage 5 | 561 | 2.4% |
| N18.1 | CKD, stage 1 | 93 | 0.4% |

Notes are **multi-label**: a single discharge summary may contain multiple N18.x codes (e.g., a patient with CKD stage 3 admitted for ESRD-related complications).

## Note length statistics

| Metric | Value |
|---|---|
| Mean tokens (BERT tokenizer) | ~1,840 |
| Median tokens | ~1,650 |
| Notes ≤ 512 tokens | ~9% |
| Notes > 512 tokens | ~91% |

This length distribution motivates the chunking strategy (Step 5): most clinical discharge summaries exceed BERT's 512-token limit.

## EDA notebook

Run the following cell in a Jupyter notebook to reproduce the EDA figures:

```python
import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data/processed/")
df = pd.read_csv(data_dir / "diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv")
df["icd_code"] = df["icd_code"].apply(ast.literal_eval)

N18_LABELS = ["N18.1", "N18.2", "N18.3", "N18.4", "N18.5", "N18.6", "N18.9"]

# Label distribution
from collections import Counter
label_counts = Counter(code for codes in df["icd_code"] for code in codes if code in N18_LABELS)
pd.Series(label_counts).sort_index().plot(kind="bar", title="N18.x label distribution")
plt.tight_layout()
plt.savefig("2_eda/label_distribution.png", dpi=150)

# Note length distribution
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
lengths = df["text"].apply(lambda t: len(tokenizer(t, truncation=False)["input_ids"]))
lengths.hist(bins=50, title="Token length distribution")
plt.axvline(512, color="red", linestyle="--", label="512-token limit")
plt.legend()
plt.tight_layout()
plt.savefig("2_eda/length_distribution.png", dpi=150)
```

## Label co-occurrence matrix

Run to understand which N18 codes frequently appear together:

```python
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns

mlb = MultiLabelBinarizer(classes=N18_LABELS)
Y = mlb.fit_transform(df["icd_code"])
cooc = pd.DataFrame(Y.T @ Y, index=N18_LABELS, columns=N18_LABELS)
cooc_norm = cooc / cooc.values.diagonal()

sns.heatmap(cooc_norm, annot=True, fmt=".2f", cmap="Blues")
plt.title("Label co-occurrence (normalized by diagonal)")
plt.tight_layout()
plt.savefig("2_eda/cooccurrence_matrix.png", dpi=150)
```
