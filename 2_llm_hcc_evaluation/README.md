# Paso 2 — Evaluación con grandes modelos de lenguaje (13 grupos de riesgo HCC)

Antes de entrenar modelos especializados, la memoria evalúa hasta qué punto un LLM generalista sin ajuste fino es capaz de asignar códigos ICD-10-CM en las 13 categorías CMS-HCC de mayor impacto clínico-económico (Sección 4.1). El grupo con mejor resultado, Enfermedad Renal Crónica (N18), es el que se selecciona como foco del resto del proyecto (Sección 4.2).

## Protocolo — flujo manual, no por API

**Este paso no llama a ninguna API de LLM.** En la memoria (Sección 4.1.3), la evaluación se realizó pegando manualmente cada lote en la interfaz web de [gemini.google.com](https://gemini.google.com), con la opción de **chat temporal** activada, en lugar de usar la API de Gemini. Esta elección responde a un criterio de cumplimiento normativo: los chats temporales no se usan para reentrenar los modelos y los datos se retienen solo un tiempo limitado, lo que evita que las notas de MIMIC-IV (aunque anonimizadas conforme a HIPAA Safe Harbor) se incorporen al corpus de entrenamiento de un modelo de terceros. Usar la API, en cambio, generaría una relación de *business associate* que exigiría un acuerdo formal bajo HIPAA (Sección 9.1).

Por ello, este repositorio reproduce el protocolo real en dos scripts independientes:

1. **`build_batches.py`** — construye, para un grupo de riesgo y una estrategia de prompt, 500 notas en 10 lotes de 50, cada uno como un archivo de texto con el prompt ya completado (Anexo A.1 o A.2), listo para copiar y pegar en un chat temporal de Gemini.
2. **`score_jaccard.py`** — una vez copiada de vuelta la respuesta JSON del modelo en `lote_XX_response.json`, calcula el índice de Jaccard sobre multiconjuntos (Sección 4.8.1).

## Construcción de los conjuntos de evaluación (Sección 4.1.2)

Para cada uno de los 13 grupos de riesgo (definidos en `hcc_groups.py`) se construye un conjunto de 500 notas de alta seleccionadas aleatoriamente (semilla 42) del subconjunto ICD-10-CM completo de MIMIC-IV, con la condición de que cada nota contenga al menos un código del rango del grupo:

```bash
python 2_llm_hcc_evaluation/build_batches.py \
    --data_csv   data/processed/diagnoses_icd10.csv \
    --output_dir results/llm/batches/ \
    --group      enfermedad_renal_cronica \
    --strategy   specific \
    --n_notes    500 \
    --batch_size 50
```

Esto genera 10 archivos `lote_01.txt` … `lote_10.txt` en `results/llm/batches/enfermedad_renal_cronica/specific/`. Cada uno debe pegarse en un chat temporal nuevo de gemini.google.com; la respuesta JSON del modelo se guarda como `lote_01_response.json`, etc.

## Estrategias de prompting (Anexo A)

- **Prompt específico** (`--strategy specific`, Anexo A.1): instruye al modelo para identificar únicamente diagnósticos del rango de códigos del grupo evaluado, devolviendo solo los 3 primeros caracteres de cada código.
- **Prompt general** (`--strategy general`, Anexo A.2): instruye al modelo para identificar y mapear **todos** los diagnósticos de la nota a códigos ICD-10-CM, sin restricción de rango; los códigos del grupo de interés se filtran después.

En ambas estrategias se procesan lotes de 50 notas por interacción (10 interacciones por grupo), y el modelo debe devolver un único objeto JSON que asocie el `id` de cada `<patient_record>` con la lista de códigos de 3 caracteres predichos, sin deduplicar apariciones repetidas del mismo código.

## Modelos evaluados

- **Gemini 2.5 Flash** — prompt específico, sobre los 13 grupos de riesgo.
- **Gemini 3 Pro** — prompt específico y general, sobre 6 de los 13 grupos.

## Métrica: índice de Jaccard sobre multiconjuntos (Sección 4.8.1)

Para una nota, sean A el multiconjunto de códigos reales (a nivel de categoría de 3 caracteres) y B el multiconjunto de códigos predichos:

```
J(A, B) = Σ min(mult_A(x), mult_B(x)) / Σ max(mult_A(x), mult_B(x))
```

Se calculan dos variantes, promediadas sobre las notas del grupo:

- **J_real**: A se restringe a los códigos reales del grupo; B no se filtra. Penaliza tanto la omisión de códigos como la predicción de códigos fuera de rango.
- **J_ambos**: tanto A como B se restringen al rango de códigos del grupo. Penaliza con más suavidad las predicciones fuera de rango.

```bash
python 2_llm_hcc_evaluation/score_jaccard.py \
    --batches_dir results/llm/batches/enfermedad_renal_cronica/specific/ \
    --group       enfermedad_renal_cronica
```

## Resultados (Sección 5.1 de la memoria)

### Gemini 2.5 Flash, prompt específico — los 13 grupos de riesgo HCC

"—" indica que J_ambos coincide con J_real (el prompt específico ya restringe las predicciones al rango).

| # | Grupo HCC | Rango de códigos | N notas | J_real | J_ambos |
|---|---|---|---|---|---|
| 1 | Diabetes mellitus | E08, E09, E10, E11, E13 | 34.608 | 0,4557 | — |
| 2 | Insuficiencia cardíaca (CHF) | I11, I42, I50 | 24.527 | 0,3258 | — |
| 3 | Enfermedad vascular | I25, I70, I71, I72, I73 | 31.641 | 0,2927 | 0,2966 |
| 4 | Enfermedad renal crónica | N18 | 23.358 | 0,3104 | 0,3513 |
| 5 | EPOC y trastornos pulmonares | J41, J42, J43, J44, J45, J47, J84 | 25.858 | 0,5275 | — |
| 6 | Oncología | C | 22.322 | 0,2656 | — |
| 7 | Condiciones psiquiátricas | F20–F25, F28–F33 | 29.337 | 0,6025 | — |
| 8 | Condiciones neurológicas | G20, G30, G35, G40, G80–G83 | 11.216 | 0,3402 | 0,3431 |
| 9 | Enfermedad hepática | K7, B18 | 13.045 | 0,3530 | 0,3532 |
| 10 | VIH | B20 | 825 | 0,0554 | — |
| 11 | Amputaciones | Z89 | 2.213 | 0,4845 | 0,4868 |
| 12 | Trastornos hematológicos | D57, D61, D63, D66, D67, D69 | 19.611 | 0,4033 | 0,4340 |
| 13 | Infecciones severas | A40, A41 | 7.493 | 0,4073 | 0,4085 |

Los grupos con terminología más delimitada (condiciones psiquiátricas, EPOC, amputaciones) obtienen los valores más altos. El VIH registra la puntuación más baja: el único código relevante (B20) es poco frecuente y el modelo tiende a predecir códigos complementarios incorrectos.

### Gemini 3 Pro vs. Gemini 2.5 Flash — comparativa de estrategias (J_real, 6 grupos)

| Grupo HCC | Rango de códigos | Flash específico | G3 Pro específico | G3 Pro general |
|---|---|---|---|---|
| Diabetes mellitus | E08, E09, E10, E11, E13 | 0,4557 | 0,5052 | **0,7293** |
| Insuficiencia cardíaca (CHF) | I11, I42, I50 | 0,3258 | 0,4426 | **0,6977** |
| Enfermedad vascular | I25, I70, I71, I72, I73 | 0,2927 | 0,5300 | **0,6481** |
| Enfermedad renal crónica | N18 | 0,3104 | 0,4319 | **0,8260** |
| EPOC y trastornos pulmonares | J41, J42, J43, J44, J45, J47, J84 | 0,5275 | 0,4972 | **0,7498** |
| Oncología | C | 0,2656 | **0,6957** | 0,6796 |

Gemini 3 Pro supera sistemáticamente a Gemini 2.5 Flash con prompt específico (salvo en EPOC). El prompt general mejora de forma muy significativa sobre el específico en casi todos los grupos: el modelo razona en lenguaje clínico natural sobre la nota completa, en lugar de tener que decidir simultáneamente si un hallazgo es relevante y si pertenece al rango solicitado. El resultado más alto, Enfermedad Renal Crónica con prompt general (J_real = 0,8260), motiva la selección de N18 como subgrupo de estudio para el resto del proyecto (Sección 4.2).
