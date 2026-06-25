# Step 6 — Clinical Summarization

Two-stage pipeline that (a) selects a representative 1,000-note subset for model comparison, (b) summarizes all N18 notes with MedGemma-27b-it, and (c) fine-tunes classifiers on the resulting summaries.

## 6a. MLSMOTE stratified sampling

Selects 1,000 notes from the full N18 dataset preserving the multi-label code distribution.

```bash
python 6_summarization/mlsmote_sampling.py \
    --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/mlsmote_1000.csv \
    --n_samples  1000
```

Expected distribution (from the thesis):

| Code | Dataset | Sample |
|---|---|---|
| N18.9 | 36.7 % | 35.8 % |
| N18.3 | 29.1 % | 28.6 % |
| N18.6 | 20.0 % | 19.7 % |
| N18.4 | 8.5 % | 8.3 % |
| N18.2 | 3.4 % | 3.3 % |
| N18.5 | 2.4 % | 2.4 % |
| N18.1 | 0.4 % | 2.3 % ← MLSMOTE correction for rarest code |

## 6b. MedGemma-27b-it summarization

### Why MedGemma-27b-it?

Among the 4 evaluated generative models (Llama3-OpenBioLLM-8B, Bio-Medical-Llama-3-8B, MedGemma-1.5-4b-it, MedGemma-27b-it), MedGemma-27b-it was selected because:
- It is the **only model achieving 100 % valid responses** (1,000/1,000) on the MLSMOTE subset.
- It keeps 95 % of summaries under 487 tokens (within the 512-token BERT limit).
- It achieves the **highest ROUGE-1 precision** (0.855), indicating minimal hallucination.
- It produces the **best downstream classification metrics** when used for training.

### Requirements

- GPU: ≥ 40 GB VRAM (bfloat16). Two A100 40 GB or one A100 80 GB.
- HuggingFace access: request approval at [huggingface.co/google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it).

```bash
huggingface-cli login
python 6_summarization/summarize_medgemma.py \
    --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/ehr_n18_summarized.csv
```

The script saves checkpoints every 50 notes and resumes automatically if interrupted.

### Summarization prompt

The system prompt instructs the model to produce an **abstractive** clinical summary:
- Preserves all diagnostically relevant information (diagnoses, symptoms, procedures, labs).
- Omits administrative/demographic content.
- Maximum 400 words (≈512 tokens).
- **Does not assign ICD-10 codes**.

## 6c. Fine-tuning classifiers on summaries

See [Step 5](../5_chunking_max_pooling/README.md) with the `--no_chunking` flag.

> **Deployment**: `BioLinkBERT-large`, threshold 0.4, trained on MedGemma-27b-it summaries is the summarization-pipeline model used in [icd10_system](https://github.com/Gdefrr99/icd10_system).
