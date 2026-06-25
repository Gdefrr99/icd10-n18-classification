# Step 5 — Chunking + Max Pooling

Fine-tunes the 4 selected clinical Transformer models on the full N18 dataset (23,358 notes) using:

- **Chunking**: each note is split into overlapping 512-token chunks (128-token overlap).
- **Label inheritance**: every chunk of a note inherits all labels of that note.
- **Focal loss**: `gamma=2.0`, `alpha=0.25` to mitigate class imbalance.
- **Max Pooling**: at inference, the maximum logit across all chunks of a note is used as the document-level logit.
- **Decision thresholds**: evaluated at both 0.4 (higher recall) and 0.6 (higher precision).

## Models

| Model | HuggingFace ID |
|---|---|
| BioLinkBERT-large | `michiyasunaga/BioLinkBERT-large` |
| RoBERTa-large-pubmed-mimic3-Voc-hf | `RoBERTa-large-PM-M3-Voc-hf` (local) |
| BlueBERT-pubmed-mimic-large-uncased | `bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16` |
| PubMedBERT_abstract | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` |

> **Deployment**: `PubMedBERT_abstract` at threshold 0.6 with Max Pooling is the segmentation-pipeline model used in [icd10_system](https://github.com/Gdefrr99/icd10_system).

## Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 10 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Chunk size | 512 tokens |
| Chunk overlap | 128 tokens |
| Model selection metric | F1-weighted (validation) |

## Usage

```bash
# BioLinkBERT-large
python 5_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --output_dir models/chunking/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --thresholds 0.4 0.6

# PubMedBERT_abstract
python 5_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --output_dir models/chunking/ \
    --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract \
    --thresholds 0.4 0.6

# For summarized notes (no chunking needed — summaries fit in 512 tokens)
python 5_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --data_csv   data/processed/ehr_n18_summarized.csv \
    --output_dir models/summarized/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --no_chunking \
    --thresholds 0.4 0.6
```

> **Note for summarization pipeline**: `BioLinkBERT-large` at threshold 0.4 trained on MedGemma-27b-it summaries is the summarization-pipeline model used in [icd10_system](https://github.com/Gdefrr99/icd10_system).
