# Step 4 — Model Prescreening and Selection

Before full training (Step 5), we run a lightweight prescreening on 23 clinical Transformer models to identify the 4 best candidates. This avoids training each model for 10 epochs on the full 23K-note dataset.

## Prescreening setup

| Parameter | Value |
|---|---|
| Training samples | 10,000 (random subset of train split) |
| Validation samples | full validation set (2,336 notes) |
| Epochs | 5 (with early stopping, patience=2) |
| Threshold | 0.5 (fixed; no per-class tuning) |
| Selection metric | F1-weighted on validation |

## Model list

The 23 models evaluated in prescreening (file `4_model_selection/model_list.txt`):

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

## Running prescreening

```bash
# Single model
python 4_model_selection/prescreening.py \
    --data_dir   data/processed/ \
    --output_dir results/prescreening/ \
    --model_name michiyasunaga/BioLinkBERT-large

# All 23 models (SLURM array job recommended)
while IFS= read -r model; do
    python 4_model_selection/prescreening.py \
        --data_dir   data/processed/ \
        --output_dir results/prescreening/ \
        --model_name "$model"
done < 4_model_selection/model_list.txt
```

## Prescreening results (section 4.3 of the thesis)

Top-10 models by F1-weighted on validation:

| Rank | Model | F1-weighted | F1-micro |
|---|---|---|---|
| 1 | **BioLinkBERT-large** | **0.781** | **0.793** |
| 2 | **PubMedBERT_abstract** | 0.774 | 0.785 |
| 3 | **RoBERTa-large-PM-M3-Voc-hf** | 0.769 | 0.780 |
| 4 | **BlueBERT-pubmed-mimic-large** | 0.763 | 0.774 |
| 5 | GatorTron-medium | 0.751 | 0.762 |
| 6 | BioM-ELECTRA-Large | 0.748 | 0.759 |
| 7 | BioLinkBERT-base | 0.744 | 0.756 |
| 8 | PubMedBERT_abstract-fulltext | 0.741 | 0.753 |
| 9 | BlueBERT-pubmed-mimic-base | 0.738 | 0.749 |
| 10 | Bio_ClinicalBERT | 0.729 | 0.741 |

**Selected for full training** (bold, top 4): BioLinkBERT-large, PubMedBERT_abstract, RoBERTa-large-PM-M3-Voc-hf, BlueBERT-pubmed-mimic-large-uncased.

These 4 models proceed to Steps 5 (chunking + max pooling on full notes) and 6 (training on MedGemma-27b-it summaries).

## Aggregating results

```python
import json, pandas as pd
from pathlib import Path

records = []
for p in Path("results/prescreening/").glob("*/prescreening_summary.json"):
    records.append(json.loads(p.read_text()))
df = pd.DataFrame(records).sort_values("f1_weighted", ascending=False)
print(df.to_string(index=False))
```
