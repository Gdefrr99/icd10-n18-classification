# Data

**No data is distributed in this repository.** MIMIC-IV is a restricted dataset requiring credentialed PhysioNet access.

## Obtaining MIMIC-IV

1. Register at [physionet.org](https://physionet.org/register/).
2. Complete the CITI Program "Data or Specimens Only Research" course.
3. Sign the data use agreement for MIMIC-IV at [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/).
4. Download the dataset (v2.2 or later). Only two files are needed:

| File | PhysioNet path | Description |
|---|---|---|
| `diagnoses_icd.csv.gz` | `hosp/diagnoses_icd.csv.gz` | ICD codes per admission (subject_id, hadm_id, icd_version, icd_code) |
| `discharge.csv.gz` | `note/discharge.csv.gz` | Discharge summary text (subject_id, hadm_id, text) |

Place the decompressed CSVs in `data/raw/`.

## Expected directory layout after preprocessing

```
data/
├── raw/
│   ├── diagnoses_icd.csv          ← from MIMIC-IV hosp module
│   └── discharge.csv              ← from MIMIC-IV note module
└── processed/
    ├── diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv  ← full N18 dataset (23,358 rows)
    ├── ehr_icd_train_clean.csv    ← 16,351 rows (70 %)
    ├── ehr_icd_val_clean.csv      ←  2,337 rows (10 %)
    ├── ehr_icd_test_clean.csv     ←  4,670 rows (20 %)
    ├── mlsmote_1000.csv           ← 1,000 stratified samples (step 6a)
    └── ehr_n18_summarized.csv     ← full dataset with MedGemma-27b-it summaries (step 6b)
```

## N18 dataset statistics

After preprocessing, the N18 working set contains:

- **23,358 discharge summaries** from 65,665 unique patients (MIMIC-IV ICD-10-CM subset).
- **7 labels**: N18.1, N18.2, N18.3, N18.4, N18.5, N18.6, N18.9 (CKD stages).
- **110 notes** carry two N18 codes simultaneously; the rest carry exactly one.
- Mean note length: ~3,162 tokens (BioLinkBERT tokenizer) — 6–7× the 512-token BERT limit.

### Label distribution in the test set (4,670 notes, 4,694 labels)

| Code | Description | Count | % |
|---|---|---|---|
| N18.1 | CKD stage 1 | 17 | 0.4 |
| N18.2 | CKD stage 2 | 157 | 3.3 |
| N18.3 | CKD stage 3 | 1,361 | 29.0 |
| N18.4 | CKD stage 4 | 396 | 8.4 |
| N18.5 | CKD stage 5 | 113 | 2.4 |
| N18.6 | End-stage renal disease (ESRD) | 937 | 20.0 |
| N18.9 | CKD unspecified | 1,713 | 36.5 |
