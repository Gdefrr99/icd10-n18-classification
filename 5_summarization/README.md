# Paso 5 — Resumen clínico automático

Pipeline en dos fases: (A) comparación de 4 modelos generativos sobre un subconjunto representativo de 1.000 notas, para elegir el mejor; (B) resumen de las 23.358 notas N18 completas con el modelo ganador y ajuste fino de los clasificadores sobre los resúmenes resultantes.

## Fase A — Comparación de modelos de resumen sobre 1.000 muestras

### 5a. Muestreo estratificado (1.000 notas)

Selecciona 1.000 notas del dataset N18 completo preservando la distribución multietiqueta real, con sobremuestreo del código raro N18.1 hasta un mínimo de 20 muestras (Sección 4.5.2):

```bash
python 5_summarization/sampling.py \
    --data_csv   data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/muestra_1000.csv \
    --n_samples  1000
```

Distribución esperada (Tabla 4.4 de la memoria):

| Código | Dataset | Muestra |
|---|---|---|
| N18.9 | 36,7 % | 35,8 % |
| N18.3 | 29,1 % | 28,6 % |
| N18.6 | 20,0 % | 19,7 % |
| N18.4 | 8,5 % | 8,3 % |
| N18.2 | 3,4 % | 3,3 % |
| N18.5 | 2,4 % | 2,4 % |
| N18.1 | 0,4 % | 2,3 % ← sobremuestreo deliberado para garantizar cobertura mínima |

### 5b. Resumen de las 1.000 muestras con cada uno de los 4 modelos generativos

Los 4 modelos comparados (Sección 4.5.3) reciben el mismo prompt de sistema y los mismos parámetros de generación; solo cambia el modelo. Ejecutar una vez por modelo:

```bash
for MODEL in Llama3-OpenBioLLM-8B Bio-Medical-Llama-3-8B MedGemma-1.5-4b-it MedGemma-27B-it; do
    python 5_summarization/summarize_1000.py \
        --input_csv  data/processed/muestra_1000.csv \
        --output_csv "data/processed/muestra_1000_summarized_${MODEL}.csv" \
        --model      "$MODEL"
done
```

> Los identificadores de HuggingFace de `Llama3-OpenBioLLM-8B`, `Bio-Medical-Llama-3-8B` y `MedGemma-1.5-4b-it` en `MODEL_REGISTRY` (dentro de `summarize_1000.py`) son la mejor referencia disponible; verifícalos contra la página del modelo o pasa `--model_id` para sobrescribirlos antes de lanzar una ejecución larga.

Cada ejecución guarda checkpoints cada 50 notas y reanuda automáticamente si se interrumpe. Una fracción de las respuestas puede no ajustarse al formato esperado (vacías, demasiado cortas, o con códigos ICD-10 filtrados pese a la prohibición explícita); estas se detectan con una heurística y se reprocesan con una semilla de generación distinta (Sección 4.5.4).

### 5c. Métricas ROUGE de cada modelo frente a las notas originales

```bash
for MODEL in Llama3-OpenBioLLM-8B Bio-Medical-Llama-3-8B MedGemma-1.5-4b-it MedGemma-27B-it; do
    python 5_summarization/compute_rouge.py \
        --original_csv   data/processed/muestra_1000.csv \
        --summarized_csv "data/processed/muestra_1000_summarized_${MODEL}.csv" \
        --model_name     "$MODEL"
done
```

### 5d. Partición 70/10/20 de cada conjunto resumido y ajuste fino de los clasificadores

Cada modelo produce un número distinto de resúmenes válidos (Tabla 4.5), por lo que cada conjunto de 1.000 muestras resumidas recibe su propia partición estratificada 70/10/20 (semilla 42):

```bash
python 5_summarization/build_1000_splits.py \
    --summarized_csv data/processed/muestra_1000_summarized_MedGemma-27B-it.csv \
    --output_dir     data/processed/summarized_1000/MedGemma-27B-it/
```

El ajuste fino de los 4 clasificadores Transformer sobre cada conjunto es análogo al Paso 4, usando `--no_chunking` (los resúmenes caben en 512 tokens) y apuntando `--data_dir` al directorio generado arriba:

```bash
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/summarized_1000/MedGemma-27B-it/ \
    --output_dir models/summarized_1000/MedGemma-27B-it/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --no_chunking \
    --thresholds 0.4 0.6
```

Repetir para los 4 modelos de resumen y los 4 clasificadores. El mejor modelo de resumen (Sección 5.5.1, Tabla 5.8) es el que, combinado con el mejor clasificador, obtiene las mejores métricas — en la memoria, **MedGemma-27B-it**.

## Fase B — Escalado al dataset completo con el modelo ganador

### 5e. Resumen de las 23.358 notas con MedGemma-27B-it

Tras seleccionar MedGemma-27B-it en la Fase A (100 % de respuestas válidas, mejor precisión ROUGE-1, mejores métricas de clasificación posteriores), se extiende el resumen a las 22.358 notas restantes del dataset N18:

- GPU: ≥ 40 GB VRAM (bfloat16). Dos A100 40 GB o una A100 80 GB.
- Acceso en HuggingFace: solicitar aprobación en [huggingface.co/google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it).

```bash
huggingface-cli login
python 5_summarization/summarize_medgemma.py \
    --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/ehr_n18_summarized.csv
```

### 5f. Reconstrucción de las particiones completas con las notas resumidas

Sustituye el texto de `ehr_icd_{train,val,test}_clean.csv` por el resumen correspondiente, conservando la misma partición que en el Paso 1 (a diferencia de 5d, aquí no se genera una partición nueva):

```bash
python 5_summarization/build_summarized_splits.py \
    --splits_dir     data/processed/ \
    --summarized_csv data/processed/ehr_n18_summarized.csv \
    --output_dir     data/processed/summarized/
```

### 5g. Ajuste fino de los clasificadores finales sobre los resúmenes completos

Ver [Paso 4](../4_chunking_max_pooling/README.md) con el flag `--no_chunking`, apuntando `--data_dir` al directorio generado en el paso anterior.

> **Despliegue**: `BioLinkBERT-large` con umbral 0,4 y entrenado sobre los resúmenes de MedGemma-27B-it es el modelo seleccionado para el pipeline de resumen integrado en [icd10_system](https://github.com/Gdefrr99/icd10_system).

## Resultados

### Calidad del resumen — métricas ROUGE (1.000 muestras)

| Modelo | Válidas | R1-P | R1-R | R1-F1 | R2-P | R2-R | R2-F1 | RL-P | RL-R | RL-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3-OpenBioLLM-8B | 933 | 0,701 | 0,041 | 0,076 | 0,298 | 0,019 | 0,034 | 0,481 | 0,027 | 0,050 |
| Bio-Medical-Llama-3-8B | 995 | 0,803 | 0,049 | 0,088 | 0,422 | 0,022 | 0,039 | 0,567 | 0,030 | 0,054 |
| MedGemma-1.5-4b-it | 955 | 0,845 | 0,111 | 0,193 | 0,434 | 0,056 | 0,097 | 0,518 | 0,068 | 0,118 |
| MedGemma-27B-it | 1.000 | 0,855 | 0,102 | 0,179 | 0,397 | 0,047 | 0,082 | 0,482 | 0,057 | 0,100 |

Todos los modelos generan resúmenes de alta precisión léxica (P-ROUGE-1 ≥ 0,70) y recall muy bajo (R-ROUGE-1 ≤ 0,11): son concisos y no reproducen el texto original de forma extensiva, el comportamiento deseable en un resumen abstractivo. MedGemma-27B-it es el único modelo con el 100 % de respuestas válidas y la mayor precisión ROUGE-1; MedGemma-1.5-4b-it tiene el F1-ROUGE-1 más alto pero con más varianza en la validez de las respuestas.

### Mejores resultados de clasificación sobre las 1.000 muestras resumidas (Tabla 5.8)

| Métrica | Valor | Modelo clasificación | Umbral | Modelo de resumen |
|---|---|---|---|---|
| Accuracy | 0,7000 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| Precisión micro | 0,7714 | PubMedBERT_abstract | 0,6 | MedGemma-27B-it |
| Precisión macro | 0,6344 | BioLinkBERT-large | 0,6 | MedGemma-27B-it |
| Precisión weighted | 0,7736 | BioLinkBERT-large | 0,6 | MedGemma-27B-it |
| Recall micro | 0,7065 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| Recall macro | 0,4683 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| Recall weighted | 0,7065 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| F1 micro | 0,7226 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| F1 macro | 0,5592 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |
| F1 weighted | 0,6966 | PubMedBERT_abstract | 0,4 | MedGemma-27B-it |

En todas las métricas, el mejor modelo de resumen resulta ser MedGemma-27B-it.

### Comparativa: Max Pooling (notas completas) vs. resúmenes (23.358 notas, test N18)

| Métrica | Max Pooling | Modelo | Umbral | Resumido (MedGemma-27B-it) | Modelo | Umbral |
|---|---|---|---|---|---|---|
| Accuracy | 0,5925 | PubM | 0,6 | 0,7704 | PubM | 0,4 |
| Precisión micro | 0,6927 | PubM | 0,6 | 0,8102 | Rob | 0,6 |
| Precisión macro | 0,7444 | BioL | 0,6 | 0,8396 | Blue | 0,6 |
| Precisión weighted | 0,7239 | PubM | 0,6 | 0,8444 | Rob | 0,6 |
| Recall micro | 0,9444 | PubM | 0,4 | 0,7865 | Blue | 0,4 |
| Recall macro | 0,8148 | PubM | 0,4 | 0,6369 | Blue | 0,4 |
| Recall weighted | 0,9444 | PubM | 0,4 | 0,7865 | Blue | 0,4 |
| F1 micro | 0,7467 | PubM | 0,6 | 0,7830 | BioL | 0,4 |
| F1 macro | 0,7233 | PubM | 0,6 | 0,6905 | BioL | 0,4 |
| F1 weighted | 0,7516 | PubM | 0,6 | 0,7758 | BioL | 0,4 |

El resumen clínico mejora la precisión y el F1-weighted; la segmentación + Max Pooling domina en recall y F1-macro, al poder localizar evidencia en cualquier fragmento de la nota completa.
