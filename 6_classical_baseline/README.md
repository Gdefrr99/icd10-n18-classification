# Paso 6 — Línea base clásica (TF-IDF / BM25 / BM25+)

Reproduce la línea base clásica multietiqueta de la Sección 4.6 de la memoria, para contextualizar el rendimiento de los modelos Transformer y cuantificar la ventaja del modelado semántico contextual.

## Método

- **Representaciones**: TF-IDF (unigramas y unigramas+bigramas), BM25, BM25+
- **Clasificadores**: Regresión Logística y LinearSVC mediante descomposición One-vs-Rest (OvR)
- **Tamaños de vocabulario explorados**: 36K (unigramas completo), 50K, 53K, 103K, 930K (uni+bigramas completo)
- **Preprocesado adicional de texto** (Sección 4.6.2): conversión a minúsculas, lematización con scispaCy `en_core_sci_sm`, eliminación de stop words estándar y clínicas, eliminación de tokens de 1-2 caracteres y cadenas numéricas puras

## Requisitos

```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_sm
```

## Uso

```bash
python 6_classical_baseline/baseline.py \
    --data_dir   data/processed/ \
    --output_dir results/baseline/
```

Los resultados se guardan en `results/baseline/baseline_results.csv`.

## Resultados (Tabla 5.10 de la memoria, conjunto de test N18)

| Métrica | Valor | Representación | Parámetros BM25/BM25+ | Clasificador | Vocabulario | Nº características |
|---|---|---|---|---|---|---|
| Accuracy | 0,6019 | TF-IDF, uni+bigr | — | LinearSVC | — | 50K |
| Precisión micro | 0,8439 | TF-IDF, uni+bigr total | — | LogReg | — | 930K |
| Precisión macro | 0,7468 | BM25, unigramas total | k1=2, b=0,25 | LogReg | Distinto a TF-IDF | 103K |
| Precisión weighted | 0,8101 | TF-IDF, uni+bigr total | — | LinearSVC | — | 930K |
| Recall micro | 0,6204 | BM25+, uni+bigr | k1=2, b=1, δ=0,5 | LogReg | Igual que TF-IDF | 50K |
| Recall macro | 0,4022 | BM25+, unigramas reducidos | k1=1,5, b=1, δ=0,5 | LinearSVC | Distinto a TF-IDF | 53K |
| Recall weighted | 0,6204 | BM25+, uni+bigr | k1=2, b=1, δ=0,5 | LogReg | Igual que TF-IDF | 50K |
| F1 micro | 0,6899 | TF-IDF, uni+bigr | — | LinearSVC | — | 50K |
| F1 macro | 0,4507 | BM25+, unigramas reducidos | k1=1,5, b=1, δ=0,5 | LinearSVC | Distinto a TF-IDF | 53K |
| F1 weighted | 0,6595 | TF-IDF, uni+bigr | — | LinearSVC | — | 50K |

TF-IDF con unigramas y bigramas (50K características) domina F1-weighted, F1-micro y accuracy. BM25+ obtiene los mejores recall-macro y F1-macro. Los vocabularios más grandes (930K, 103K) solo mejoran la precisión; para el resto de métricas los vocabularios intermedios (50K–53K) son óptimos.

## Rejilla de hiperparámetros BM25

| Parámetro | Valores |
|---|---|
| k1 | 0,5, 1,0, 1,5, 2,0 |
| b | 0,25, 0,5, 0,75, 1,0 |
| delta (BM25+) | 0,0 (BM25), 0,5, 1,0, 1,5 |
