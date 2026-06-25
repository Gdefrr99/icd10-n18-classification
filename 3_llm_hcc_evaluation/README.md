# Step 3 — LLM Zero-Shot Evaluation and HCC Group Analysis

This step covers two sub-experiments from chapter 3:

1. **Gemini zero-shot baseline**: evaluate Gemini 2.5 Flash and Gemini 3 Pro on the N18 test set without any fine-tuning.
2. **CMS-HCC group analysis**: validate MIMIC-IV patient coverage using CMS-HCC risk groups and Jaccard similarity.

## 3a. Gemini zero-shot ICD-10-CM prediction

### Requirements

```bash
pip install google-generativeai
export GOOGLE_API_KEY="your_api_key"
```

### Usage

```bash
# Specific prompting (names the N18.x codes explicitly)
python 3_llm_hcc_evaluation/gemini_flash_evaluation.py \
    --data_csv   data/processed/ehr_icd_test_clean.csv \
    --output_dir results/llm/ \
    --model      gemini-2.5-flash \
    --strategy   specific

# General prompting (asks for any ICD-10-CM codes)
python 3_llm_hcc_evaluation/gemini_flash_evaluation.py \
    --data_csv   data/processed/ehr_icd_test_clean.csv \
    --output_dir results/llm/ \
    --model      gemini-2.5-flash \
    --strategy   general
```

### Results (test set, section 5.1 of the thesis)

| Model | Strategy | Accuracy | F1-w | F1-micro | Prec-micro | Recall-micro |
|---|---|---|---|---|---|---|
| Gemini 2.5 Flash | specific | 0.447 | 0.618 | 0.638 | 0.617 | 0.660 |
| Gemini 2.5 Flash | general  | 0.446 | 0.591 | 0.613 | 0.637 | 0.590 |
| Gemini 3 Pro     | specific | 0.475 | 0.645 | 0.665 | 0.651 | 0.680 |
| Gemini 3 Pro     | general  | 0.457 | 0.625 | 0.643 | 0.660 | 0.627 |

**Key finding**: Specific prompting consistently outperforms general prompting across all metrics. Gemini 3 Pro improves over Gemini 2.5 Flash but remains well below the fine-tuned Transformer models (F1-micro up to 0.845).

## 3b. CMS-HCC risk group analysis

The thesis evaluates the MIMIC-IV N18 cohort using the 13 CMS-HCC (Centers for Medicare and Medicaid Services Hierarchical Condition Categories) risk groups. This analysis validates that the selected patient population covers all clinically relevant risk strata.

### Jaccard similarity metric

The dataset uses Jaccard similarity to measure patient code coverage:

```
J_real  = |codes_patient ∩ codes_universe| / |codes_patient ∪ codes_universe|
J_ambos = |predicted ∩ real| / |predicted ∪ real|
```

These metrics quantify: (a) how representative each patient's codes are of the full coding universe, and (b) how aligned model predictions are with ground truth at patient level.

### HCC group distribution (MIMIC-IV N18 cohort)

| HCC Group | Description | % of patients |
|---|---|---|
| G1 | Transplant | 1.2% |
| G2 | Complications / advanced CKD (excl. stage 5/ESRD) | 18.4% |
| G3 | CKD stage 5 | 2.4% |
| G4 | End stage renal disease (ESRD) | 20.0% |
| G5 | Vascular / ischemic disease | 8.7% |
| G6 | Diabetes | 31.5% |
| G7 | Hypertension | 42.1% |
| G8 | Heart failure | 22.3% |
| G9 | Anemia | 15.6% |
| G10 | Bone / mineral metabolism | 9.8% |
| G11 | Neurological / cerebrovascular | 11.2% |
| G12 | Pulmonary / respiratory | 14.4% |
| G13 | Infection | 19.7% |

Note: patients may belong to multiple HCC groups simultaneously.
