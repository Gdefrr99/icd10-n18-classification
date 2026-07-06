# Paso 5 — Resumen clínico automático

Pipeline en tres etapas: (a) selección de un subconjunto representativo de 1.000 notas para comparar modelos de resumen, (b) resumen de las 23.358 notas N18 con MedGemma-27B-it, y (c) ajuste fino de los clasificadores sobre los resúmenes resultantes.

## 5a. Muestreo estratificado (1.000 notas)

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

## 5b. Resumen con MedGemma-27B-it

### Por qué MedGemma-27B-it

Entre los 4 modelos generativos evaluados (Llama3-OpenBioLLM-8B, Bio-Medical-Llama-3-8B, MedGemma-1.5-4b-it, MedGemma-27B-it), se seleccionó MedGemma-27B-it porque:
- Es el **único modelo que produce el 100 % de respuestas válidas** (1.000/1.000) sobre las 1.000 muestras.
- Mantiene el 95 % de los resúmenes por debajo de 486 tokens (dentro del límite de 512 tokens de BERT).
- Obtiene la **mayor precisión ROUGE-1** (0,855), lo que indica una alucinación mínima.
- Produce las **mejores métricas de clasificación posteriores** al usarse para entrenamiento.

### Requisitos

- GPU: ≥ 40 GB VRAM (bfloat16). Dos A100 40 GB o una A100 80 GB.
- Acceso en HuggingFace: solicitar aprobación en [huggingface.co/google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it).

```bash
huggingface-cli login
python 5_summarization/summarize_medgemma.py \
    --input_csv  data/processed/diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv \
    --output_csv data/processed/ehr_n18_summarized.csv
```

El script guarda checkpoints cada 50 notas y reanuda automáticamente si se interrumpe.

### Prompt de resumen (Anexo A.3)

El prompt del sistema instruye al modelo para producir un resumen **abstractivo**:
- Conserva toda la información diagnósticamente relevante (diagnósticos, síntomas, procedimientos, analíticas).
- Omite contenido administrativo/demográfico.
- Máximo 400 palabras (≈ 512 tokens).
- **No asigna códigos ICD-10** por sí mismo.

## 5c. Reconstrucción de las particiones con las notas resumidas

Sustituye el texto de `ehr_icd_{train,val,test}_clean.csv` por el resumen correspondiente, conservando la misma partición que en el Paso 1:

```bash
python 5_summarization/build_summarized_splits.py \
    --splits_dir     data/processed/ \
    --summarized_csv data/processed/ehr_n18_summarized.csv \
    --output_dir     data/processed/summarized/
```

## 5d. Ajuste fino de los clasificadores sobre los resúmenes

Ver [Paso 4](../4_chunking_max_pooling/README.md) con el flag `--no_chunking`, apuntando `--data_dir` al directorio generado en el paso anterior.

> **Despliegue**: `BioLinkBERT-large`, umbral 0,4, entrenado sobre los resúmenes de MedGemma-27B-it, es el modelo de la pipeline de resumen integrado en [icd10_system](https://github.com/Gdefrr99/icd10_system).

## Resultados (Secciones 5.4 y 5.5 de la memoria)

### Calidad del resumen — métricas ROUGE (1.000 muestras)

| Modelo | Válidas | R1-P | R1-R | R1-F1 | R2-P | R2-R | R2-F1 | RL-P | RL-R | RL-F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama3-OpenBioLLM-8B | 933 | 0,701 | 0,041 | 0,076 | 0,298 | 0,019 | 0,034 | 0,481 | 0,027 | 0,050 |
| Bio-Medical-Llama-3-8B | 995 | 0,803 | 0,049 | 0,088 | 0,422 | 0,022 | 0,039 | 0,567 | 0,030 | 0,054 |
| MedGemma-1.5-4b-it | 955 | 0,845 | 0,111 | 0,193 | 0,434 | 0,056 | 0,097 | 0,518 | 0,068 | 0,118 |
| MedGemma-27B-it | 1.000 | 0,855 | 0,102 | 0,179 | 0,397 | 0,047 | 0,082 | 0,482 | 0,057 | 0,100 |

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

El resumen clínico mejora sistemáticamente la precisión y el F1-weighted; la segmentación + Max Pooling domina en recall y F1-macro, al poder localizar evidencia en cualquier fragmento de la nota completa.
