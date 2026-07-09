# Paso 4 — Segmentación + Max Pooling

Ajusta los 4 modelos Transformer seleccionados en el paso 3 sobre el dataset N18 completo (23.358 notas) mediante:

- **Segmentación (chunking)**: cada nota se divide en fragmentos solapados de 512 tokens (128 de solapamiento).
- **Herencia de etiquetas**: cada fragmento hereda todas las etiquetas de la nota a la que pertenece.
- **Max Pooling**: en inferencia, el logit máximo entre todos los fragmentos de una nota se usa como logit a nivel de documento.
- **Umbrales de decisión**: se evalúan 0,4 (mayor recall) y 0,6 (mayor precisión).

## Modelos

| Modelo | Identificador HuggingFace |
|---|---|
| BioLinkBERT-large | `michiyasunaga/BioLinkBERT-large` |
| RoBERTa-large-pubmed-mimic3-Voc-hf | `RoBERTa-large-PM-M3-Voc-hf` (descarga manual, ver [3_model_selection](../3_model_selection/README.md)) |
| BlueBERT-pubmed-mimic-large-uncased | `bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16` |
| PubMedBERT_abstract | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` |

> **Despliegue**: `PubMedBERT_abstract` con umbral 0,6 y Max Pooling es el modelo seleccionado para el pipeline de segmentación integrado en [icd10_system](https://github.com/Gdefrr99/icd10_system).

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Épocas | 10 |
| Weight decay | 0,01 |
| Warmup ratio | 0,1 |
| Tamaño de fragmento | 512 tokens |
| Solapamiento | 128 tokens |
| Métrica de selección | F1-weighted (validación) |
| Semilla | 42 |

## Uso

```bash
# BioLinkBERT-large
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --output_dir models/chunking/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --thresholds 0.4 0.6

# PubMedBERT_abstract
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/ \
    --output_dir models/chunking/ \
    --model_name microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract \
    --thresholds 0.4 0.6

# Para las notas resumidas (Paso 5): sin segmentación, los resúmenes caben en 512 tokens
python 4_chunking_max_pooling/train_chunking.py \
    --data_dir   data/processed/summarized/ \
    --output_dir models/summarized/ \
    --model_name michiyasunaga/BioLinkBERT-large \
    --no_chunking \
    --thresholds 0.4 0.6
```

## Resultados (Sección 5.3 de la memoria)

### Métricas globales por umbral (conjunto de test, 4.670 notas)

| Modelo | Umbral | Acc. | Prec-Mi | Prec-Ma | Prec-W | Rec-Mi | Rec-Ma | Rec-W | F1-Mi | F1-Ma | F1-W |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BioLinkBERT-large | 0,4 | 0,3713 | 0,5421 | 0,6453 | 0,5900 | 0,8858 | 0,7694 | 0,8858 | 0,6726 | 0,6752 | 0,6864 |
| BioLinkBERT-large | 0,6 | 0,5529 | 0,6548 | 0,7444 | 0,7174 | 0,8255 | 0,6408 | 0,8255 | 0,7303 | 0,6450 | 0,7398 |
| BlueBERT-large | 0,4 | 0,3188 | 0,5264 | 0,6057 | 0,5628 | 0,8982 | 0,7403 | 0,8982 | 0,6638 | 0,6150 | 0,6765 |
| BlueBERT-large | 0,6 | 0,5792 | 0,6675 | 0,7354 | 0,7193 | 0,8347 | 0,7207 | 0,8347 | 0,7418 | 0,7113 | 0,7502 |
| PubMedBERT_abstract | 0,4 | 0,1713 | 0,4754 | 0,5890 | 0,5069 | 0,9444 | 0,8148 | 0,9444 | 0,6324 | 0,6565 | 0,6449 |
| PubMedBERT_abstract | 0,6 | 0,5925 | 0,6927 | 0,7287 | 0,7239 | 0,8100 | 0,7409 | 0,8100 | 0,7467 | 0,7233 | 0,7516 |
| RoBERTa-large-pubmed-mimic3-Voc-hf | 0,4 | 0,1334 | 0,4638 | 0,6120 | 0,5012 | 0,9433 | 0,7932 | 0,9433 | 0,6219 | 0,6609 | 0,6366 |
| RoBERTa-large-pubmed-mimic3-Voc-hf | 0,6 | 0,5572 | 0,6606 | 0,7007 | 0,6862 | 0,8272 | 0,7532 | 0,8272 | 0,7346 | 0,7160 | 0,7400 |

El umbral 0,6 domina en accuracy, F1 y precisión para los cuatro modelos; el umbral 0,4 domina en recall.

### Métricas por código (umbral 0,4)

| Código | Prec BioL | Prec Blue | Prec PubM | Prec Rob | Rec BioL | Rec Blue | Rec PubM | Rec Rob | F1 BioL | F1 Blue | F1 PubM | F1 Rob | Soporte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N18.1 | 0,85 | 0,80 | 0,69 | 0,73 | 0,65 | 0,24 | 0,65 | 0,65 | 0,73 | 0,36 | 0,67 | 0,69 | 17 |
| N18.2 | 0,85 | 0,70 | 0,83 | 0,75 | 0,57 | 0,59 | 0,58 | 0,58 | 0,68 | 0,64 | 0,68 | 0,65 | 157 |
| N18.3 | 0,74 | 0,61 | 0,40 | 0,39 | 0,77 | 0,82 | 0,97 | 0,98 | 0,76 | 0,70 | 0,56 | 0,56 | 1.361 |
| N18.4 | 0,54 | 0,61 | 0,57 | 0,60 | 0,76 | 0,74 | 0,77 | 0,75 | 0,63 | 0,67 | 0,66 | 0,66 | 396 |
| N18.5 | 0,46 | 0,36 | 0,48 | 0,69 | 0,65 | 0,83 | 0,76 | 0,62 | 0,54 | 0,51 | 0,59 | 0,65 | 113 |
| N18.6 | 0,66 | 0,73 | 0,71 | 0,71 | 0,99 | 0,98 | 0,99 | 0,99 | 0,79 | 0,84 | 0,83 | 0,82 | 937 |
| N18.9 | 0,43 | 0,42 | 0,44 | 0,42 | 1,00 | 0,99 | 0,99 | 1,00 | 0,60 | 0,59 | 0,61 | 0,59 | 1.713 |

### Métricas por código (umbral 0,6)

| Código | Prec BioL | Prec Blue | Prec PubM | Prec Rob | Rec BioL | Rec Blue | Rec PubM | Rec Rob | F1 BioL | F1 Blue | F1 PubM | F1 Rob | Soporte |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N18.1 | 0,67 | 0,69 | 0,79 | 0,79 | 0,12 | 0,53 | 0,65 | 0,65 | 0,20 | 0,60 | 0,71 | 0,71 | 17 |
| N18.2 | 0,83 | 0,84 | 0,81 | 0,75 | 0,55 | 0,58 | 0,58 | 0,59 | 0,66 | 0,69 | 0,68 | 0,66 | 157 |
| N18.3 | 0,93 | 0,92 | 0,90 | 0,82 | 0,72 | 0,73 | 0,73 | 0,75 | 0,81 | 0,81 | 0,81 | 0,78 | 1.361 |
| N18.4 | 0,82 | 0,76 | 0,69 | 0,59 | 0,68 | 0,69 | 0,72 | 0,76 | 0,74 | 0,72 | 0,71 | 0,67 | 396 |
| N18.5 | 0,72 | 0,66 | 0,57 | 0,62 | 0,53 | 0,63 | 0,69 | 0,70 | 0,61 | 0,64 | 0,62 | 0,66 | 113 |
| N18.6 | 0,74 | 0,74 | 0,77 | 0,79 | 0,98 | 0,97 | 0,99 | 0,97 | 0,84 | 0,84 | 0,87 | 0,87 | 937 |
| N18.9 | 0,50 | 0,53 | 0,56 | 0,54 | 0,91 | 0,92 | 0,83 | 0,86 | 0,65 | 0,67 | 0,67 | 0,66 | 1.713 |

BioL = BioLinkBERT-large, Blue = BlueBERT-large, PubM = PubMedBERT_abstract, Rob = RoBERTa-large-pubmed-mimic3-Voc-hf.
