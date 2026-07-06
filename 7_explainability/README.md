# Paso 7 — Explicabilidad (Noise Tunnel + Gradientes Integrados)

Módulo de explicabilidad post-hoc para los clasificadores N18.x ajustados. Identifica los fragmentos de texto de una nota de alta que más sustentan cada código ICD-10-CM predicho (Sección 4.7 de la memoria).

## Método

**Gradientes Integrados (IG)** calcula la atribución de cada token de entrada a la salida del modelo integrando el gradiente a lo largo de un camino desde una entrada de referencia (secuencia de PAD) hasta la entrada real.

**Noise Tunnel (SmoothGrad-IG)** envuelve IG y promedia las atribuciones sobre `NT_SAMPLES` perturbaciones gaussianas de los embeddings de entrada, reduciendo la varianza intrínseca de IG y produciendo rankings de spans más estables entre ejecuciones.

**Extracción de spans en dos pasadas**:
1. **Primera pasada** — segmenta la nota en líneas/oraciones, calcula la suma de atribuciones con signo por segmento, normaliza a [0, 1] y selecciona los top-N segmentos por encima de `MIN_SPAN_SCORE`.
2. **Segunda pasada** — para cada span seleccionado, extrae sub-frases dividiendo por paréntesis, comas y conjunciones coordinantes. Puntúa cada sub-frase por **atribución positiva media por token** (suma / nº de tokens) — el promedio evita favorecer sub-frases más largas. Devuelve la sub-frase con mayor puntuación.

**Agregación por dimensión de embedding**: suma con signo (no norma L2). Esto preserva la dirección de la atribución: los tokens positivos apoyan el código, los negativos actúan como evidencia contraria.

## Configuración

Editar las constantes al inicio de `icd10_explainability.py` antes de ejecutar:

| Constante | Valor por defecto | Significado |
|---|---|---|
| `DEFAULT_MODEL_DIR` | `./trained_RoBERTa-large-pubmed-mimic3-Voc-hf` | Ruta al modelo ajustado |
| `LABELS` | `["N181", ..., "N189"]` | Debe coincidir con `mlb.classes_` del entrenamiento |
| `THRESHOLDS` | `{cls: 0.6}` | Umbrales de decisión por clase |
| `N_IG_STEPS` | `20` | Pasos de integración de IG por muestra de ruido |
| `NT_SAMPLES` | `10` | Número de perturbaciones gaussianas |
| `NT_STDEVS` | `0.01` | Desviación estándar del ruido gaussiano |
| `MAX_SPANS` | `5` | Máximo de spans en confianza alta |
| `MIN_SPAN_SCORE` | `0.05` | Puntuación normalizada mínima para incluir un span |
| `CONF_HIGH` | `0.95` | Umbral de confianza alta |
| `CONF_MED` | `0.85` | Umbral de confianza media |

## Selección de N_IG_STEPS (Sección 4.7.3 y 5.7)

La memoria evalúa la estabilidad de los spans variando N_IG_STEPS ∈ {10, 20, 50, 75, 100}. Con evidencia diagnóstica concentrada y explícita, el span de mayor atribución es idéntico para cualquier valor evaluado. Con evidencia distribuida y confianza moderada del modelo, los top-5 spans cambian entre configuraciones, y las configuraciones con menor error de aproximación (20 y 75) son las que capturan más spans clínicamente relevantes. El valor por defecto de este script, `N_IG_STEPS=20`, corresponde a la configuración usada para los resultados finales.

## Requisitos de hardware

| Modelo | VRAM | Notas |
|---|---|---|
| BioLinkBERT-large | ≥ 16 GB | 10×20 = 200 pasadas forward por etiqueta |
| RoBERTa-large-PM-M3-Voc-hf | ≥ 16 GB | Mismo coste |
| PubMedBERT_abstract | ≥ 8 GB | Modelo más pequeño, más rápido |

La inferencia en CPU es posible pero lenta (~5-10 min por nota en modelos grandes).

## Uso

```python
from icd10_explainability import load_model, analyze_note

# Cargar el modelo ajustado (salida del Paso 4 o 5)
load_model("models/summarized/trained_RoBERTa-large-pubmed-mimic3-Voc-hf")

# Analizar una nota de alta
results = analyze_note(open("nota.txt").read())

# Estructura de salida:
# {
#   "codes": [
#     {
#       "id":        "N183",
#       "desc":      "Chronic kidney disease, stage 3 (moderate)",
#       "conf":      0.982,
#       "conf_tier": "high",
#       "cat":       "Diseases of the genitourinary system",
#       "spans": [
#         {"text": "stage III CKD (baseline Cr 2.0-2.1)", "score": 1.0,  "subphrase": True,  "span_text": "..."},
#         {"text": "His creatinine increased from 2.1 to 2.7",  "score": 0.74, "subphrase": False, "span_text": "..."}
#       ]
#     }
#   ]
# }
print(results)
```

## Importante: verificación del orden de las etiquetas

Antes de usar el módulo, verificar que `LABELS` en `icd10_explainability.py` coincide exactamente con el orden de `mlb.classes_` del entrenamiento:

```python
import joblib
mlb = joblib.load("mlb.pkl")  # guardado durante el preprocesado
print(list(mlb.classes_))     # debe coincidir con la lista LABELS
```

`MultiLabelBinarizer` ordena las clases alfabéticamente por defecto, por lo que el orden esperado es:
`["N181", "N182", "N183", "N184", "N185", "N186", "N189"]`
(los puntos se eliminan, en el mismo formato usado durante el entrenamiento).
