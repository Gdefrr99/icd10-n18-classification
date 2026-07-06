# Datos

**Este repositorio no distribuye ningún dato.** MIMIC-IV es un dataset restringido que requiere acceso acreditado en PhysioNet.

## Obtención de MIMIC-IV

1. Registrarse en [physionet.org](https://physionet.org/register/).
2. Completar los cursos CITI "Data or Specimens Only Research" y "Conflicts of Interest".
3. Firmar el acuerdo de uso de datos de MIMIC-IV en [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/).
4. Descargar el dataset (v2.2 o posterior). Solo se necesitan dos archivos:

| Archivo | Ruta en PhysioNet | Descripción |
|---|---|---|
| `diagnoses_icd.csv.gz` | `hosp/diagnoses_icd.csv.gz` | Códigos ICD por ingreso (subject_id, hadm_id, icd_version, icd_code) |
| `discharge.csv.gz` | `note/discharge.csv.gz` | Texto de las notas de alta (subject_id, hadm_id, text) |

Colocar ambos archivos, **en su formato `.csv.gz` original, sin descomprimir**, en `data/raw/`. El script de preprocesado infiere la compresión a partir de la extensión.

## Estructura esperada tras el preprocesado

```
data/
├── raw/
│   ├── diagnoses_icd.csv.gz         ← módulo hosp de MIMIC-IV
│   └── discharge.csv.gz             ← módulo note de MIMIC-IV
└── processed/
    ├── diagnoses_icd10.csv                                     ← dataset completo ICD-10-CM (122.288 filas)
    ├── diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv  ← dataset N18 completo (23.358 filas)
    ├── ehr_icd_train_clean.csv    ← 16.351 filas (70 %)
    ├── ehr_icd_val_clean.csv      ←  2.337 filas (10 %)
    ├── ehr_icd_test_clean.csv     ←  4.670 filas (20 %)
    ├── seleccion_10000.csv        ← 10.000 notas / 50 códigos más frecuentes (paso 3)
    ├── muestra_1000.csv           ← 1.000 muestras estratificadas (paso 5a)
    └── ehr_n18_summarized.csv     ← dataset completo con resúmenes de MedGemma-27B-it (paso 5b)
```

## Estadísticas del dataset N18

Tras el preprocesado, el conjunto de trabajo N18 contiene:

- **23.358 notas de alta**, procedentes de un total de 122.288 notas ICD-10-CM de MIMIC-IV.
- **7 etiquetas**: N18.1, N18.2, N18.3, N18.4, N18.5, N18.6, N18.9 (estadios de la ERC).
- **110 notas** presentan dos códigos N18 simultáneamente; el resto presenta uno solo.
- Se elimina 1 nota adicional por contener un código N18 duplicado (23.359 → 23.358).
- Longitud media: ~3.162 tokens (tokenizador BioLinkBERT-large) — de 6 a 7 veces el límite de 512 tokens de BERT.

### Distribución de etiquetas en el conjunto de test (4.670 notas, 4.694 etiquetas)

| Código | Descripción | Frec. absoluta | Frec. relativa (%) |
|---|---|---|---|
| N18.1 | ERC estadio 1 | 17 | 0,4 |
| N18.2 | ERC estadio 2 | 157 | 3,3 |
| N18.3 | ERC estadio 3 | 1.361 | 29,0 |
| N18.4 | ERC estadio 4 | 396 | 8,4 |
| N18.5 | ERC estadio 5 | 113 | 2,4 |
| N18.6 | ERC estadio terminal (ESRD) | 937 | 20,0 |
| N18.9 | ERC sin especificar | 1.713 | 36,5 |
