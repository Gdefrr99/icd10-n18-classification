# -*- coding: utf-8 -*-
"""
Pipeline de preprocesado de las notas de alta de MIMIC-IV.

Une las tablas diagnoses_icd y discharge, filtra los registros ICD-10-CM,
aplica la normalización de texto clínico descrita en la Sección 3.2 de la
memoria y construye:

  1. El dataset completo ICD-10-CM (122.288 notas) usado en la Sección 3.1.1
     y en la construcción del conjunto de selección de modelos (Paso 3).
  2. El subgrupo N18 — Enfermedad Renal Crónica (Sección 4.2), con su
     partición estratificada 70/10/20 en train/val/test.

Acepta tanto archivos .csv como .csv.gz de forma indistinta: pandas infiere
la compresión a partir de la extensión del archivo, por lo que no es
necesario descomprimir manualmente discharge.csv.gz ni diagnoses_icd.csv.gz.

Uso:
    python preprocess.py \
        --diagnoses_csv data/raw/diagnoses_icd.csv.gz \
        --discharge_csv data/raw/discharge.csv.gz \
        --output_dir    data/processed/
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Normalización de texto clínico (Sección 3.2 de la memoria)
# ---------------------------------------------------------------------------

def clean_discharge_summary(text: str) -> str:
    """Aplica la normalización de texto clínico específica de MIMIC-IV."""
    if not isinstance(text, str):
        return ""

    # 1. Eliminación de cabeceras administrativas
    for pattern in [
        r"Name:\s+___", r"Unit No:\s+___", r"Admission Date:\s+___",
        r"Discharge Date:\s+___", r"Date of Birth:\s+___",
    ]:
        text = re.sub(pattern, "", text)

    # Sustitución de los marcadores de anonimización residuales
    text = text.replace("___", "[UNK]")

    # 2. Expansión de abreviaturas clínicas (antes de tratar los saltos de
    #    línea, para no partir los patrones entre líneas)
    text = re.sub(r"\bs/p\b",  "status post",   text, flags=re.IGNORECASE)
    text = re.sub(r"\bc/o\b",  "complains of",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bh/o\b",  "history of",    text, flags=re.IGNORECASE)
    text = re.sub(r"\bw/o\b",  "without",       text, flags=re.IGNORECASE)
    text = re.sub(r"\s+w/",    " with ",        text, flags=re.IGNORECASE)
    text = re.sub(r"\bpt\b",   "patient",       text, flags=re.IGNORECASE)

    # 3. Restauración de la estructura de párrafos (hard-wrap de MIMIC-IV)
    text = re.sub(r'\n\s*\n', '||PARAGRAPH||', text)  # proteger párrafos reales
    text = text.replace('\n', ' ')                      # eliminar saltos forzados
    text = text.replace('||PARAGRAPH||', '\n\n')       # restaurar párrafos

    # 4. Normalización de espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Marcado de las secciones clínicas clave
    text = re.sub(r"History of Present Illness:",
                  "\nHistory of Present Illness:\n", text, flags=re.IGNORECASE)
    text = re.sub(r"Discharge Diagnos[ei]s:",
                  "\nDischarge Diagnosis:\n", text, flags=re.IGNORECASE)

    return text


# ---------------------------------------------------------------------------
# Construcción del dataset completo ICD-10-CM (Sección 3.1.1)
# ---------------------------------------------------------------------------

def build_full_icd10_dataset(diagnoses_csv: str, discharge_csv: str) -> pd.DataFrame:
    """Une diagnoses_icd y discharge y filtra a los registros ICD-10-CM.

    Acepta tanto rutas .csv como .csv.gz: pandas descomprime automáticamente
    según la extensión del archivo.
    """
    print("Cargando diagnoses_icd...")
    diag = pd.read_csv(
        diagnoses_csv,
        usecols=["subject_id", "hadm_id", "icd_version", "icd_code"],
    )

    # Conservar únicamente los registros ICD-10-CM (icd_version == 10)
    diag = diag[diag["icd_version"] == 10].drop(columns=["icd_version"])

    diag_grouped = (
        diag.groupby(["subject_id", "hadm_id"])["icd_code"]
        .apply(list)
        .reset_index()
    )

    print("Cargando notas de alta (discharge)...")
    discharge = pd.read_csv(
        discharge_csv,
        usecols=["subject_id", "hadm_id", "text"],
    )

    df = diag_grouped.merge(discharge, on=["subject_id", "hadm_id"], how="inner")
    print(f"Notas ICD-10-CM tras el cruce: {len(df):,} "
          f"({df['subject_id'].nunique():,} pacientes únicos)")

    return df.reset_index(drop=True)


def build_n18_dataset(df_full: pd.DataFrame) -> pd.DataFrame:
    """Filtra el dataset completo al subgrupo N18 (Sección 4.2)."""
    df = df_full.copy()

    # Conservar solo los ingresos con al menos un código N18.x
    df = df[
        df["icd_code"].apply(lambda codes: any(str(c).startswith("N18") for c in codes))
    ].copy()

    # Restringir la lista de etiquetas a los códigos N18.x
    df["icd_code"] = df["icd_code"].apply(
        lambda codes: [c for c in codes if str(c).startswith("N18")]
    )

    n_before = len(df)

    # Eliminar las notas con un código N18 duplicado (p. ej. dos veces N18.9):
    # la memoria reporta 23.359 candidatas menos 1 nota eliminada = 23.358.
    df = df[df["icd_code"].apply(lambda x: len(x) == len(set(x)))].copy()

    print(f"Notas N18 candidatas: {n_before:,}")
    print(f"Notas eliminadas por contener un código N18 duplicado: {n_before - len(df):,}")
    print(f"Notas N18 finales: {len(df):,}")

    return df.reset_index(drop=True)


def preprocess_and_split(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Aplicando normalización de texto...")
    df["text"] = df["text"].apply(clean_discharge_summary)

    # Guardar el dataset completo de N18
    full_path = output_dir / "diagnoses_icd10_filtrado_enfermedad_renal_cronica.csv"
    df.to_csv(full_path, index=False)
    print(f"Dataset N18 completo guardado -> {full_path} ({len(df):,} filas)")

    # Partición estratificada multietiqueta 70/10/20
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["icd_code"])

    splitter_tv = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=RANDOM_SEED
    )
    train_val_idx, test_idx = next(splitter_tv.split(df, Y))

    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    Y_trainval  = Y[train_val_idx]
    df_test     = df.iloc[test_idx].reset_index(drop=True)

    val_fraction = 0.10 / 0.80
    splitter_v = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=RANDOM_SEED
    )
    train_idx, val_idx = next(splitter_v.split(df_trainval, Y_trainval))

    df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
    df_val   = df_trainval.iloc[val_idx].reset_index(drop=True)

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = output_dir / f"ehr_icd_{name}_clean.csv"
        subset.to_csv(path, index=False)
        print(f"Partición {name:5s} guardada -> {path} ({len(subset):,} filas)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnoses_csv", required=True,
                         help="Ruta a diagnoses_icd.csv o diagnoses_icd.csv.gz")
    parser.add_argument("--discharge_csv", required=True,
                         help="Ruta a discharge.csv o discharge.csv.gz")
    parser.add_argument("--output_dir", default="data/processed/")
    args = parser.parse_args()

    df_full = build_full_icd10_dataset(args.diagnoses_csv, args.discharge_csv)

    print("Aplicando normalización de texto al dataset completo...")
    df_full_clean = df_full.copy()
    df_full_clean["text"] = df_full_clean["text"].apply(clean_discharge_summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_icd10_path = output_dir / "diagnoses_icd10.csv"
    df_full_clean.to_csv(full_icd10_path, index=False)
    print(f"Dataset completo ICD-10-CM guardado -> {full_icd10_path} ({len(df_full_clean):,} filas)")
    print("Este archivo es la entrada del Paso 3 (construcción del conjunto de "
          "selección de modelos, 10.000 notas / 50 códigos más frecuentes).")

    df_n18 = build_n18_dataset(df_full)
    preprocess_and_split(df_n18, output_dir)
    print("Hecho.")


if __name__ == "__main__":
    main()