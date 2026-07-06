# Paso 1 — Preprocesado

Une las tablas de MIMIC-IV, filtra los registros ICD-10-CM, aplica la normalización de texto clínico y construye tanto el dataset completo ICD-10-CM como el subgrupo N18 con sus particiones.

## Qué hace el script

1. Carga `diagnoses_icd.csv(.gz)` y filtra a `icd_version = 10`.
2. Agrupa los códigos ICD-10-CM por ingreso y cruza con `discharge.csv(.gz)` mediante `(subject_id, hadm_id)`.
3. Aplica la normalización de texto clínico (Sección 3.2 de la memoria) a **todas** las notas:
   - Elimina cabeceras administrativas (`Name: ___`, `Admission Date: ___`, etc.).
   - Sustituye los marcadores de anonimización `___` restantes por `[UNK]`.
   - Expande abreviaturas clínicas: `s/p → status post`, `c/o → complains of`, `h/o → history of`, `w/o → without`, `w/ → with`, `pt → patient`.
   - Restaura los párrafos partidos por el hard-wrap de MIMIC-IV.
   - Normaliza espacios múltiples.
   - Marca las secciones `History of Present Illness:` y `Discharge Diagnosis:`.
4. Guarda el **dataset completo ICD-10-CM** (122.288 notas, Sección 3.1.1), que es la entrada del [Paso 3 — Selección de modelos](../3_model_selection/README.md).
5. Filtra al **subgrupo N18** (Enfermedad Renal Crónica, Sección 4.2): conserva las notas con al menos un código N18.x y elimina la única nota con un código N18 duplicado (23.359 → 23.358 notas).
6. Genera una partición estratificada multietiqueta 70/10/20 (semilla 42) mediante `MultilabelStratifiedShuffleSplit`.

## Uso

Los dos archivos de MIMIC-IV pueden pasarse tal cual se descargan de PhysioNet, en formato `.csv.gz` — pandas infiere la compresión a partir de la extensión, por lo que **no es necesario descomprimirlos manualmente**:

```bash
python 1_preprocessing/preprocess.py \
    --diagnoses_csv data/raw/diagnoses_icd.csv.gz \
    --discharge_csv  data/raw/discharge.csv.gz \
    --output_dir     data/processed/
```

## Archivos de salida

| Archivo | Filas | Descripción |
|---|---|---|
| `diagnoses_icd10.csv` | 122.288 | Dataset completo ICD-10-CM (todos los grupos, no solo N18) |
| `diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv` | 23.358 | Dataset N18 completo |
| `ehr_icd_train_clean.csv` | 16.351 | Partición de entrenamiento (70 %) |
| `ehr_icd_val_clean.csv` | 2.337 | Partición de validación (10 %) |
| `ehr_icd_test_clean.csv` | 4.670 | Partición de test (20 %) |

Cada CSV tiene las columnas: `subject_id`, `hadm_id`, `icd_code` (lista de códigos como literal Python), `text` (texto preprocesado).
