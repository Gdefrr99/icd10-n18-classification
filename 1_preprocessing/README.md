# Step 1 — Preprocessing

Joins MIMIC-IV tables, filters to N18, applies clinical text normalization, and produces stratified splits.

## What the script does

1. Loads `diagnoses_icd.csv` and filters to `icd_version = 10`.
2. Groups ICD-10-CM codes per admission and keeps only admissions with ≥1 N18.x code.
3. Joins with `discharge.csv` on `(subject_id, hadm_id)`.
4. Applies the following text normalization steps in order:
   - Removes administrative headers (`Name: ___`, `Admission Date: ___`, etc.).
   - Replaces `___` anonymization markers with `[UNK]`.
   - Expands common clinical abbreviations: `s/p → status post`, `c/o → complains of`, `h/o → history of`, `w/o → without`, `w/ → with`, `pt → patient`.
   - Restores hard-wrapped paragraphs (MIMIC hard-wrap fix).
   - Normalizes multiple spaces.
   - Tags `History of Present Illness:` and `Discharge Diagnosis:` sections with extra newlines.
5. Saves the full N18 dataset.
6. Produces a stratified 70/10/20 train/val/test split using `MultilabelStratifiedShuffleSplit`.

## Usage

```bash
python 1_preprocessing/preprocess.py \
    --diagnoses_csv data/raw/diagnoses_icd.csv \
    --discharge_csv  data/raw/discharge.csv \
    --output_dir     data/processed/
```

## Output files

| File | Rows | Description |
|---|---|---|
| `diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv` | 23,358 | Full N18 dataset |
| `ehr_icd_train_clean.csv` | 16,351 | Training split (70 %) |
| `ehr_icd_val_clean.csv` | 2,337 | Validation split (10 %) |
| `ehr_icd_test_clean.csv` | 4,670 | Test split (20 %) |

Each CSV has columns: `subject_id`, `hadm_id`, `icd_code` (Python list literal), `text` (preprocessed).
