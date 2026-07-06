# Step 4 — Model election

Before full training (Step 5), we run a lightweight selection on 25 clinical Transformer models to identify the 4 best candidates.

## Selection setup

| Parameter | Value |
|---|---|
| Samples | 10,000 (random subset of train split) |
| Epochs | 10 (with early stopping, patience=3) |
| Threshold | 0.3 (fixed; no per-class tuning) |
| Selection metric | F1-weighted on validation |

## Model list

The 23 models evaluated in selection (file `4_model_selection/model_list.txt`):

```
michiyasunaga/BioLinkBERT-large
michiyasunaga/BioLinkBERT-base
RoBERTa-large-PM-M3-Voc-hf
bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16
bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
allenai/scibert_scivocab_uncased
emilyalsentzer/Bio_ClinicalBERT
dmis-lab/biobert-large-cased-v1.1-squad
dmis-lab/biobert-v1.1
sultan/BioM-ELECTRA-Large-SQuAD2
sultan/BioM-BERT-PubMed-PMC-Large
sultan/BioM-RoBERTa-Large
UFNLP/gatortron-base
UFNLP/gatortron-medium
medicalai/ClinicalBERT
Charangan/MedBERT
NLP4Science/medicine-llm
NLP4Science/bio-llm
google/bigbird-roberta-base
allenai/longformer-base-4096
yikuan8/Clinical-Longformer
```

## Running selection

```bash
# Single model
python 4_model_selection/selection.py \
    --data_dir   data/processed/ \
    --output_dir results/selection/ \
    --model_name michiyasunaga/BioLinkBERT-large

# All 23 models (SLURM array job recommended)
while IFS= read -r model; do
    python 4_model_selection/selection.py \
        --data_dir   data/processed/ \
        --output_dir results/selection/ \
        --model_name "$model"
done < 4_model_selection/model_list.txt
```

## Selection results (section 4.3 of the thesis)

Top-10 models by F1-weighted on validation:

| Rank | Model | F1-micro | F1-weighted |
|---|---|---|---|
| 1 | **BioLinkBERT-large** | **0.5490** | **0.5162** |
| 2 | **RoBERTa-large-pubmed-mimic3-Voc-hf** | 0,5404 | 0,5091 |
| 3 | **BlueBERT-pubmed-mimic-large-uncased** | 0,5278 | 0,4936 |
| 4 | **PubMedBERT_abstract** | 0,5255 | 0,4737 |
| 5 | SciBERT-scivocab-uncased | 0,5210 | 0,4690 |
| 6 | RoBERTa-base-pubmed-mimic3-Voc-train-longer | 0,5156 | 0,4680 |
| 7 | BioBERT-large-cased-v1.1 | 0,5083 | 0,4671 |
| 8 | BioLinkBERT-base | 0,5231 | 0,4666 |
| 9 | RoBERTa-base-pubmed-mimic3-Voc | 0,5172 | 0,4644 |
| 10 | PubMedBERT_abstract_fulltext | 0,5193 | 0,4619 |

**Selected for full training** (bold, top 4): BioLinkBERT-large, RoBERTa-large-pubmed-mimic3-Voc-hf, BlueBERT-pubmed-mimic-large-uncased, PubMedBERT_abstract.

These 4 models proceed to Steps 5 (chunking + max pooling on full notes) and 6 (training on MedGemma-27b-it summaries).

## Aggregating results

```python
import json, pandas as pd
from pathlib import Path

records = []
for p in Path("results/selection/").glob("*/selection_summary.json"):
    records.append(json.loads(p.read_text()))
df = pd.DataFrame(records).sort_values("f1_weighted", ascending=False)
print(df.to_string(index=False))
```
