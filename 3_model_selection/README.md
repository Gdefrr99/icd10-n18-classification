# Paso 3 — Selección de modelos

Antes del entrenamiento completo (Pasos 4 y 5), se ejecuta una selección sobre 25 modelos Transformer biomédicos y clínicos para identificar los 4 mejores candidatos.

## Construcción del conjunto de selección

A diferencia del resto del pipeline, este paso **no** parte del subgrupo N18. Se construye un subconjunto de 10.000 notas etiquetadas con al menos 3 de los 50 códigos ICD-10-CM más frecuentes del **dataset completo** (122.288 notas, salida de [1_preprocessing](../1_preprocessing/README.md)), definiendo un espacio de 50 etiquetas posibles:

```bash
python 3_model_selection/build_selection_dataset.py \
    --input_csv  data/processed/diagnoses_icd10.csv \
    --output_csv data/processed/seleccion_10000.csv \
    --top_n_codes 50 \
    --min_matches 3 \
    --n_samples   10000
```

## Configuración de la selección

| Parámetro | Valor |
|---|---|
| Muestras | 10.000 (subconjunto del dataset ICD-10-CM completo, no de N18) |
| Etiquetas posibles | 50 códigos más frecuentes |
| Partición | 70/10/20 (semilla 42) |
| Entrada | primeros 512 tokens |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Épocas | 10 (con parada temprana, patience=3) |
| Función de pérdida | Binary Cross-Entropy estándar |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Umbral de aceptación | 0.3 |
| Métrica de selección | F1-weighted en validación |

## Lista de modelos

Los 25 modelos evaluados (archivo `3_model_selection/model_list.txt`) cubren las principales familias Transformer preentrenadas en dominio biomédico o clínico: BlueBERT, BioLinkBERT, PubMedBERT, BioBERT, SciBERT, BioELECTRA, Bio_ClinicalBERT, Bio_Discharge_Summary_BERT, variantes RoBERTa adaptadas al dominio clínico, biomed_RoBERTa_base, BiomedBERT_hash_nano y ClinicalBERT.

> **Descarga manual necesaria.** Los modelos de la familia `RoBERTa-*-PM-M3-Voc-hf` (RoBERTa-large-PM-M3-Voc-hf, RoBERTa-base-PM-M3-Voc-hf, RoBERTa-base-PM-M3-Voc-train-longer-hf) **no están disponibles en HuggingFace Hub**. Deben descargarse desde el repositorio oficial de los autores (Lewis et al. 2020): [facebookresearch/bio-lm](https://github.com/facebookresearch/bio-lm), y cargarse después desde una ruta local pasando esa ruta como `--model_name`.

## Ejecución

```bash
# Un solo modelo
python 3_model_selection/selection.py \
    --data_csv   data/processed/seleccion_10000.csv \
    --output_dir results/selection/ \
    --model_name michiyasunaga/BioLinkBERT-large

# Los 25 modelos (recomendado como array job en SLURM)
while IFS= read -r model; do
    python 3_model_selection/selection.py \
        --data_csv   data/processed/seleccion_10000.csv \
        --output_dir results/selection/ \
        --model_name "$model"
done < 3_model_selection/model_list.txt
```

## Resultados de la selección

Ranking completo de los 25 modelos ordenados por F1-weighted en test. Los 4 modelos en negrita son los seleccionados para el entrenamiento completo sobre N18 (Pasos 4 y 5):

| Rk | Modelo | F1-micro | F1-weighted | F1-macro |
|---|---|---|---|---|
| 1 | **BioLinkBERT-large** | **0,5490** | **0,5162** | **0,4396** |
| 2 | **RoBERTa-large-pubmed-mimic3-Voc-hf** | **0,5404** | **0,5091** | **0,4326** |
| 3 | **BlueBERT-pubmed-mimic-large-uncased** | **0,5278** | **0,4936** | **0,4156** |
| 4 | **PubMedBERT_abstract** | **0,5255** | **0,4737** | **0,3808** |
| 5 | SciBERT-scivocab-uncased | 0,5210 | 0,4690 | 0,3732 |
| 6 | RoBERTa-base-pubmed-mimic3-Voc-train-longer | 0,5156 | 0,4680 | 0,3764 |
| 7 | BioBERT-large-cased-v1.1 | 0,5083 | 0,4671 | 0,3799 |
| 8 | BioLinkBERT-base | 0,5231 | 0,4666 | 0,3702 |
| 9 | RoBERTa-base-pubmed-mimic3-Voc | 0,5172 | 0,4644 | 0,3711 |
| 10 | PubMedBERT_abstract_fulltext | 0,5193 | 0,4619 | 0,3659 |
| 11 | BlueBERT-pubmed-large-uncased | 0,5067 | 0,4609 | 0,3747 |
| 12 | SciBERT-scivocab-cased | 0,5037 | 0,4504 | 0,3550 |
| 13 | ClinicalBERT | 0,5000 | 0,4470 | 0,3530 |
| 14 | biomed-RoBERTa-base | 0,4910 | 0,4335 | 0,3387 |
| 15 | BioBERT-base-cased-v1.2 | 0,4844 | 0,4218 | 0,3251 |
| 16 | BioBERT-v1.1 | 0,4781 | 0,4107 | 0,3102 |
| 17 | BioBERT-base-cased-v1.1 | 0,4778 | 0,4100 | 0,3082 |
| 18 | Bio-Discharge-Summary-BERT | 0,4714 | 0,4084 | 0,3090 |
| 19 | BlueBERT-pubmed-mimic-base-uncased | 0,4664 | 0,3961 | 0,2964 |
| 20 | BlueBERT-pubmed-base-uncased | 0,4634 | 0,3946 | 0,2929 |
| 21 | Bio_ClinicalBERT | 0,4626 | 0,3902 | 0,2866 |
| 22 | BioELECTRA-pubmed | 0,4205 | 0,3252 | 0,2109 |
| 23 | BioELECTRA-pubmed-pmc | 0,4111 | 0,3086 | 0,1972 |
| 24 | BioELECTRA-pubmed-pmc-lt | 0,3993 | 0,2975 | 0,1856 |
| 25 | BiomedBERT-hash-nano | 0,3003 | 0,1385 | 0,0442 |

**Seleccionados para el entrenamiento completo**: BioLinkBERT-large, RoBERTa-large-pubmed-mimic3-Voc-hf, BlueBERT-pubmed-mimic-large-uncased, PubMedBERT_abstract. Estos 4 modelos pasan a los Pasos 4 (segmentación + Max Pooling sobre las notas completas) y 5 (entrenamiento sobre los resúmenes de MedGemma-27B-it).

## Agregación de resultados

```python
import json, pandas as pd
from pathlib import Path

records = []
for p in Path("results/selection/").glob("*/selection_summary.json"):
    records.append(json.loads(p.read_text()))
df = pd.DataFrame(records).sort_values("f1_weighted", ascending=False)
print(df.to_string(index=False))
```