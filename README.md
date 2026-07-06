# Codificación automática ICD-10-CM — Subgrupo ERC (N18)

Repositorio de reproducibilidad del pipeline experimental descrito en el Trabajo Fin de Grado *"Sistema de Codificación Automática ICD-10-CM en Grupos de Riesgo CMS-HCC"* (Universidad de León, 2026).

Los experimentos se centran en la clasificación multietiqueta del subgrupo **Enfermedad Renal Crónica (ERC) N18** de ICD-10-CM (7 códigos: N18.1–N18.6, N18.9) a partir de notas de alta clínicas de **MIMIC-IV**.

> **Nota de despliegue.** Los dos modelos con mejor rendimiento de este pipeline están integrados en el asistente de codificación clínica [icd10_system](https://github.com/Gdefrr99/icd10_system):
> - **Pipeline de segmentación**: `PubMedBERT_abstract`, umbral 0,6, Max Pooling.
> - **Pipeline de resumen**: `BioLinkBERT-large`, umbral 0,4, entrenado sobre notas resumidas por MedGemma-27B-it.

---

## Índice

1. [Obtención del dataset](#1-obtención-del-dataset)
2. [Construcción del conjunto de trabajo N18](#2-construcción-del-conjunto-de-trabajo-n18)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Pipeline paso a paso](#4-pipeline-paso-a-paso)
5. [Requisitos de hardware](#5-requisitos-de-hardware)
6. [Resultados](#6-resultados)
7. [Cita](#7-cita)

---

## 1. Obtención del dataset

El acceso a MIMIC-IV requiere una cuenta acreditada de PhysioNet. **Este repositorio no incluye ningún dato.**

1. Crear una cuenta en [physionet.org](https://physionet.org).
2. Completar los cursos CITI requeridos y firmar el acuerdo de uso de datos.
3. Descargar MIMIC-IV (v2.2 o posterior) desde [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/).
4. Del archivo descargado solo se necesitan dos ficheros:
   - `hosp/diagnoses_icd.csv.gz` — códigos ICD por ingreso.
   - `note/discharge.csv.gz` — texto de las notas de alta por ingreso.

Ambos archivos pueden usarse **tal cual, en formato `.csv.gz`**: el script de preprocesado infiere la compresión a partir de la extensión, por lo que no es necesario descomprimirlos manualmente. Basta con colocarlos en `data/raw/`.

---

## 2. Construcción del conjunto de trabajo N18

Ejecutar el pipeline de preprocesado (ver [1_preprocessing/](1_preprocessing/README.md)) para:

1. Cruzar `diagnoses_icd` y `discharge` mediante `(subject_id, hadm_id)`.
2. Filtrar a `icd_version = 10` (registros ICD-10-CM) → **122.288 notas** (dataset completo, usado también en el Paso 3).
3. Filtrar a notas con al menos un código N18.x → **23.358 notas**, 7 etiquetas.
4. Aplicar la normalización de texto clínico (expansión de abreviaturas, restauración de hard-wrap, marcado de secciones).
5. Partición estratificada 70/10/20 % → train / validación / test.

```
data/
└── raw/
    ├── diagnoses_icd.csv.gz          # módulo hosp de MIMIC-IV
    └── discharge.csv.gz              # módulo note de MIMIC-IV
```

Los archivos que produce el script y que consumen el resto de pasos:

```
data/processed/
├── diagnoses_icd10.csv                                     # dataset completo ICD-10-CM (122.288 filas)
├── diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv   # dataset N18 completo (23.358 filas)
├── ehr_icd_train_clean.csv                                 # 16.351 filas
├── ehr_icd_val_clean.csv                                   #  2.337 filas
└── ehr_icd_test_clean.csv                                  #  4.670 filas
```

Cada CSV tiene las columnas: `subject_id`, `hadm_id`, `text` (preprocesado), `icd_code` (lista de códigos como literal de Python).

---

## 3. Estructura del repositorio

```
icd10-n18-classification/
├── README.md                        ← este archivo
├── requirements.txt                 ← dependencias de Transformers + línea base clásica
├── requirements_generative.txt      ← dependencias extra para el resumen con MedGemma
├── .gitignore
│
├── data/
│   └── README.md                    ← instrucciones de obtención y construcción
│
├── 1_preprocessing/
│   ├── README.md
│   └── preprocess.py                ← construye los CSV procesados a partir de MIMIC-IV
│
├── 2_llm_hcc_evaluation/
│   ├── README.md
│   ├── hcc_groups.py                ← definición de los 13 grupos de riesgo CMS-HCC
│   ├── build_batches.py             ← genera los lotes para pegar en gemini.google.com
│   └── score_jaccard.py             ← puntúa las respuestas pegadas de vuelta
│
├── 3_model_selection/
│   ├── README.md
│   ├── model_list.txt               ← los 25 modelos evaluados
│   ├── build_selection_dataset.py   ← construye el subconjunto de 10.000 notas / 50 códigos
│   └── selection.py                 ← ajusta cada modelo sobre ese subconjunto
│
├── 4_chunking_max_pooling/
│   ├── README.md
│   └── train_chunking.py            ← segmentación + Max Pooling (BCE estándar)
│
├── 5_summarization/
│   ├── README.md
│   ├── sampling.py                   ← selección estratificada de 1.000 muestras
│   ├── summarize_medgemma.py         ← resumen con MedGemma-27B-it
│   └── build_summarized_splits.py    ← reconstruye train/val/test con los resúmenes
│
├── 6_classical_baseline/
│   ├── README.md
│   └── baseline.py                  ← TF-IDF / BM25 / BM25+ + OvR
│
└── 7_explainability/
    ├── README.md
    └── icd10_explainability.py      ← Gradientes Integrados + Noise Tunnel
```

---

## 4. Pipeline paso a paso

### Paso 0 — Instalar dependencias

```bash
# Núcleo (transformers, sklearn, scispaCy)
pip install -r requirements.txt

# Opcional: solo para el resumen generativo
pip install -r requirements_generative.txt
```

### Paso 1 — Preprocesado

```bash
python 1_preprocessing/preprocess.py \
    --diagnoses_csv data/raw/diagnoses_icd.csv.gz \
    --discharge_csv  data/raw/discharge.csv.gz \
    --output_dir     data/processed/
```

### Paso 2 — Evaluación con LLM sobre los 13 grupos de riesgo HCC

Ver [2_llm_hcc_evaluation/README.md](2_llm_hcc_evaluation/README.md). **Este paso no usa ninguna API de LLM**: genera lotes de texto para copiar y pegar manualmente en un chat temporal de gemini.google.com (por motivos de cumplimiento normativo, Sección 4.1.3 de la memoria), y puntúa después las respuestas pegadas de vuelta.

```bash
python 2_llm_hcc_evaluation/build_batches.py \
    --data_csv   data/processed/diagnoses_icd10.csv \
    --output_dir results/llm/batches/ \
    --group      enfermedad_renal_cronica \
    --strategy   specific
```

### Paso 3 — Selección de modelos

Construye primero el subconjunto de 10.000 notas / 50 códigos más frecuentes (a partir del dataset **completo**, no de N18) y ajusta los 25 modelos candidatos:

```bash
python 3_model_selection/build_selection_dataset.py \
    --input_csv  data/processed/diagnoses_icd10.csv \
    --output_csv data/processed/seleccion_10000.csv

python 3_model_selection/selection.py \
    --data_csv   data/processed/seleccion_10000.csv \
    --output_dir results/selection/ \
    --model_name michiyasunaga/BioLinkBERT-large
```

### Paso 4 — Segmentación + Max Pooling (clasificación principal)

Ajusta los 4 modelos seleccionados sobre el N18 completo, con fragmentos de 512 tokens (solapamiento de 128) y Max Pooling:

```bash
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --output_dir models/chunking/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --thresholds 0.4 0.6 \
    --epochs     10
```

### Paso 5 — Resumen clínico automático + clasificación sobre los resúmenes

**5a. Muestreo estratificado (1.000 notas):**

```bash
python 5_summarization/sampling.py \
    --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/muestra_1000.csv
```

**5b. Resumen con MedGemma-27B-it (requiere ≥ 40 GB de VRAM):**

```bash
python 5_summarization/summarize_medgemma.py \
    --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/ehr_n18_summarized.csv
```

**5c. Reconstruir las particiones con el texto resumido:**

```bash
python 5_summarization/build_summarized_splits.py \
    --splits_dir     data/processed/ \
    --summarized_csv data/processed/ehr_n18_summarized.csv \
    --output_dir     data/processed/summarized/
```

**5d. Ajustar los clasificadores sobre los resúmenes** (usar [Paso 4](4_chunking_max_pooling/README.md) con `--no_chunking`):

```bash
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/summarized/ \
    --output_dir models/summarized/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --no_chunking \
    --thresholds 0.4 0.6
```

### Paso 6 — Línea base clásica

```bash
python 6_classical_baseline/baseline.py \
    --data_dir   data/processed/ \
    --output_dir results/baseline/
```

### Paso 7 — Explicabilidad

```python
from icd10_explainability import load_model, analyze_note
load_model("models/summarized/RoBERTa-large-pubmed-mimic3-Voc-hf/best_model")
resultado = analyze_note(open("nota.txt").read())
```

---

## 5. Requisitos de hardware

| Experimento | VRAM mínima | Notas |
|---|---|---|
| Selección (25 modelos × base/large) | 16 GB | Se recomienda A100 40 GB |
| Segmentación + Max Pooling (modelos large) | 24 GB | Multi-GPU soportado vía `CUDA_VISIBLE_DEVICES` |
| Resumen con MedGemma-27B-it | 40 GB (bfloat16) | Dos A100 40 GB o una A100 80 GB |
| Línea base clásica (TF-IDF/BM25) | Solo CPU | Se recomiendan 32 núcleos |
| Explicabilidad (IG + Noise Tunnel) | 16 GB | |

Todos los experimentos con Transformer se ejecutaron en un clúster HPC (SLURM) con GPUs NVIDIA A100.

> **Descarga manual de modelos.** Los modelos de la familia `RoBERTa-*-PM-M3-Voc-hf` no están en HuggingFace Hub: deben descargarse desde el repositorio oficial [facebookresearch/bio-lm](https://github.com/facebookresearch/bio-lm) (Lewis et al. 2020) y cargarse desde una ruta local.

---

## 6. Resultados

Cada carpeta contiene en su propio README las tablas de resultados detalladas de la sección correspondiente de la memoria:

- [2_llm_hcc_evaluation](2_llm_hcc_evaluation/README.md) — Jaccard por grupo HCC, Gemini 2.5 Flash vs. Gemini 3 Pro (Sección 5.1).
- [3_model_selection](3_model_selection/README.md) — ranking completo de los 25 modelos (Anexo B).
- [4_chunking_max_pooling](4_chunking_max_pooling/README.md) — métricas globales y por código, segmentación + Max Pooling (Sección 5.3).
- [5_summarization](5_summarization/README.md) — calidad ROUGE de los resúmenes y comparativa Max Pooling vs. resúmenes (Secciones 5.4 y 5.5).
- [6_classical_baseline](6_classical_baseline/README.md) — mejores resultados TF-IDF/BM25/BM25+ (Sección 5.6).

### Resumen — mejor F1-weighted por enfoque (conjunto de test N18, 4.670 notas)

| Enfoque | F1-weighted | Modelo | Umbral |
|---|---|---|---|
| Segmentación + Max Pooling | 0,7516 | PubMedBERT_abstract | 0,6 |
| Resumen (MedGemma-27B-it) | 0,7758 | BioLinkBERT-large | 0,4 |
| Línea base clásica (TF-IDF) | 0,6595 | TF-IDF uni+bigr + LinearSVC | — |

---

## 7. Cita

Si utilizas este código o estos resultados en tu trabajo, cita por favor:

```bibtex
@thesis{defrancisco2026icd10,
  author  = {de Francisco Rodríguez, Gonzalo},
  title   = {Sistema de Codificación Automática {ICD-10-CM} en Grupos de Riesgo {CMS-HCC}},
  school  = {Universidad de León},
  year    = {2026},
  type    = {Trabajo Fin de Grado}
}
```

### Referencias clave

- Johnson et al. (2023). MIMIC-IV. *Scientific Data*, 10, 1. DOI: 10.1038/s41597-022-01899-x
- Yasunaga et al. (2022). LinkBERT. *ACL 2022*.
- Gu et al. (2021). PubMedBERT. *ACL 2021*.
- Lewis et al. (2020). RoBERTa-PM-M3-Voc (bio-lm). *ClinicalNLP Workshop 2020*.
- Sundararajan et al. (2017). Integrated Gradients. *ICML 2017*.
- Lin (2004). ROUGE. *ACL 2004 Workshop*.
