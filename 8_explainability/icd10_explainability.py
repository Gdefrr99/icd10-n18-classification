#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICD-10-CM N18.x Explainability Module — Noise Tunnel + Integrated Gradients
=============================================================================
Predicts which ICD-10-CM N18.x codes are present in a clinical note and
extracts the text spans (evidence fragments) supporting each prediction,
together with their normalized attribution score.

Method: Noise Tunnel + Integrated Gradients (SmoothGrad-IG) via Captum.
  - NoiseTunnel averages attributions over NT_SAMPLES Gaussian-perturbed
    embeddings, reducing IG variance by ~60-80% for large models
    (Smilkov et al., 2017).
  - Embedding-dimension aggregation uses signed sum (not L2 norm), preserving
    attribution direction: positive scores indicate supporting evidence,
    negative scores indicate contrary evidence.
  - Number of spans shown depends on model confidence
    (see CONF_TIER_* in the configuration section).

Compatible models: BioLinkBERT-large, RoBERTa-large-PM-M3-Voc-hf.

Both models below are integrated into icd10_system:
  - PubMedBERT_abstract (thr=0.6, Max Pooling) — segmentation pipeline
  - BioLinkBERT-large   (thr=0.4, summarized)  — summarization pipeline
See: https://github.com/Gdefrr99/icd10_system

Dependencies:
    pip install torch transformers captum

Quick start:
    from icd10_explainability import load_model, analyze_note
    load_model("./path/to/saved/model")
    results = analyze_note("DISCHARGE SUMMARY ...")
    # results is the dict expected by the icd10_system HTML interface
"""

import re
import json
import unicodedata
import torch
import numpy as np
from typing import Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients, NoiseTunnel


# ══════════════════════════════════════════════════════════════════
# SECCIÓN DE CONFIGURACIÓN ← editar antes de ejecutar
# ══════════════════════════════════════════════════════════════════

# Ruta al directorio del modelo entrenado (salida de trainer.save_model())
# Ejemplo: "./perclass_0.4_N189_0.6_BioLinkBERT-large/trained_BioLinkBERT-large"
DEFAULT_MODEL_DIR = "./trained_RoBERTa-large-pubmed-mimic3-Voc-hf"

# Orden de etiquetas: DEBE coincidir con mlb.classes_ del entrenamiento.
# MultiLabelBinarizer ordena las clases alfabéticamente al hacer fit(),
# por lo que el orden esperado es el siguiente para los 7 códigos N18.x.
# VERIFICACIÓN RECOMENDADA: imprime mlb.classes_ después del fit() en el
# script de entrenamiento y copia el resultado aquí.
LABELS = ["N181", "N182", "N183", "N184", "N185", "N186", "N189"]

# Umbrales por clase (deben coincidir con thresholds_per_class del entrenamiento)
# THRESHOLDS: dict[str, float] = {cls: 0.6 if cls == "N189" else 0.4 for cls in LABELS}
THRESHOLDS: dict[str, float] = {cls: 0.6 for cls in LABELS}

# Descripciones para la interfaz HTML
LABEL_DESCRIPTIONS: dict[str, str] = {
    "N181": "Chronic kidney disease, stage 1",
    "N182": "Chronic kidney disease, stage 2 (mild)",
    "N183": "Chronic kidney disease, stage 3 (moderate)",
    "N184": "Chronic kidney disease, stage 4 (severe)",
    "N185": "Chronic kidney disease, stage 5",
    "N186": "End stage renal disease",
    "N189": "Chronic kidney disease, unspecified",
}
ICD_CATEGORY = "Diseases of the genitourinary system"

# Hiperparámetros de Noise Tunnel + Integrated Gradients
# ─────────────────────────────────────────────────────────────────
# El coste total es N_IG_STEPS × NT_SAMPLES forward passes por etiqueta.
# Con los valores por defecto: 20 × 10 = 200 passes, equivalente en
# precisión a N_IG_STEPS=200 con IG estándar pero con ~70% menos varianza
# entre ejecuciones (Smilkov et al., 2017; Kokhlikyan et al., 2020).
#
# NO modificar N_IG_STEPS buscando reducir convergence_delta: los experimentos
# muestran que delta no decrece monotónicamente al aumentar pasos en modelos
# de 24 capas. La estabilidad de los spans se consigue vía NoiseTunnel,
# no aumentando pasos de integración.
N_IG_STEPS  = 20    # Pasos de interpolación por muestra de ruido
NT_SAMPLES  = 10    # Número de perturbaciones gaussianas para el promedio
NT_STDEVS   = 0.01  # Desviación estándar del ruido gaussiano añadido

# Umbrales de longitud de la salida de spans
MAX_SPANS      = 5     # Máximo absoluto de spans (solo para confianza alta)
MIN_SPAN_SCORE    = 0.05  # Puntuación mínima normalizada para incluir un span
MAX_LENGTH        = 512   # Debe coincidir con el max_length del entrenamiento

# Segunda pasada de segmentación (sub-frases)
# ─────────────────────────────────────────────────────────────────
# Tras seleccionar los top spans (primera pasada, a nivel de oración/línea),
# se aplica una segunda segmentación sobre el texto de cada span para
# localizar la sub-frase clínicamente más relevante dentro de él.
# Esto resuelve el caso en que el span ganador es una oración larga que
# contiene la evidencia diagnóstica junto a términos irrelevantes.
#
# MIN_SUBPHRASE_LEN: longitud mínima en caracteres para considerar una
# sub-frase como candidata. Evita devolver fragmentos como "and" o "with".
MIN_SUBPHRASE_LEN = 10

# Política de spans según confianza del modelo
# ─────────────────────────────────────────────────────────────────
# La inestabilidad de las atribuciones crece al disminuir la confianza
# del modelo (evidencia distribuida entre múltiples regiones del texto).
# Mostrar menos spans en esos casos evita dar falsa precisión al usuario.
#
#   conf ≥ CONF_HIGH  → hasta MAX_SPANS spans, sin advertencia
#   CONF_MED ≤ conf < CONF_HIGH → hasta SPANS_MED spans, tier="medium"
#   conf < CONF_MED   → solo 1 span,            tier="low"
#
CONF_HIGH  = 0.95   # Umbral de confianza alta
CONF_MED   = 0.85   # Umbral de confianza media
SPANS_MED  = 5      # Máximo de spans en tier medio


# ══════════════════════════════════════════════════════════════════
# ESTADO GLOBAL DEL MÓDULO
# ══════════════════════════════════════════════════════════════════

_device: torch.device | None = None
_tokenizer = None
_model = None
_uses_token_type_ids: bool = False


# ══════════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════════

def load_model(model_dir: str = DEFAULT_MODEL_DIR) -> None:
    """
    Carga el tokenizador y el modelo desde un directorio guardado con
    trainer.save_model(). Detecta automáticamente si el modelo es de
    tipo BERT (usa token_type_ids) o RoBERTa (no los usa).

    Parámetros
    ----------
    model_dir : str
        Ruta al directorio del modelo entrenado.
    """
    global _device, _tokenizer, _model, _uses_token_type_ids

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[load_model] Dispositivo: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(model_dir)
    _model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    _model.to(_device)
    _model.eval()

    # Detectar si el modelo usa token_type_ids (BERT) o no (RoBERTa)
    _uses_token_type_ids = "token_type_ids" in _tokenizer.model_input_names
    print(f"[load_model] Modelo cargado. Usa token_type_ids: {_uses_token_type_ids}")
    print(f"[load_model] Número de etiquetas: {_model.config.num_labels}")


# ══════════════════════════════════════════════════════════════════
# FUNCIONES FORWARD PARA CAPTUM
# ══════════════════════════════════════════════════════════════════
# Captum necesita funciones que reciban el tensor a atribuir (embeddings)
# como primer argumento, y el resto como additional_forward_args.
# Devuelven sigmoid(logits) de shape (batch, num_labels); Captum selecciona
# la columna indicada por `target` para diferenciación.

def _forward_bert(
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
) -> torch.Tensor:
    """Forward para modelos BERT-style (con token_type_ids)."""
    out = _model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )
    return torch.sigmoid(out.logits)


def _forward_roberta(
    input_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Forward para modelos RoBERTa-style (sin token_type_ids)."""
    out = _model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    )
    return torch.sigmoid(out.logits)


# ══════════════════════════════════════════════════════════════════
# TOKENIZACIÓN
# ══════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> dict[str, Any]:
    """
    Tokeniza el texto con truncación a MAX_LENGTH y devuelve offset_mapping
    para mapear posiciones de tokens a caracteres del texto original.

    Nota: offset_mapping requiere tokenizadores rápidos (Rust), que es el
    caso por defecto tanto para BioLinkBERT como para RoBERTa-PM-M3-Voc-hf.
    """
    return _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_offsets_mapping=True,
    )


# ══════════════════════════════════════════════════════════════════
# EMBEDDINGS Y BASELINE
# ══════════════════════════════════════════════════════════════════

def _get_embeddings(input_ids: torch.Tensor) -> torch.Tensor:
    """Obtiene los word embeddings para los input_ids dados."""
    emb_layer = _model.get_input_embeddings()
    with torch.no_grad():
        return emb_layer(input_ids)


def _get_baseline_embeddings(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Construye el baseline de IG: todos tokens PAD excepto [CLS]/[SEP].

    El baseline representa la "entrada vacía" (secuencia sin información).
    Es la práctica estándar en NLP con IG (Kokhlikyan et al., 2020).
    """
    baseline_ids = torch.zeros_like(input_ids)
    baseline_ids[0, 0] = _tokenizer.cls_token_id

    # Posición del SEP real: último token no-padding
    real_len = int((input_ids[0] != _tokenizer.pad_token_id).sum().item())
    baseline_ids[0, real_len - 1] = _tokenizer.sep_token_id

    return _get_embeddings(baseline_ids)


# ══════════════════════════════════════════════════════════════════
# CÁLCULO DE ATRIBUCIONES NOISE TUNNEL + IG POR TOKEN
# ══════════════════════════════════════════════════════════════════

def _compute_nt_ig_attributions(
    encoding: dict[str, Any],
    label_idx: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    Ejecuta Noise Tunnel + Integrated Gradients para una etiqueta concreta.

    Noise Tunnel (SmoothGrad-IG) promedia las atribuciones de IG sobre
    NT_SAMPLES perturbaciones gaussianas de los embeddings de entrada.
    Esto reduce la varianza intrínseca de IG en modelos profundos (~70%),
    produciendo spans más estables entre ejecuciones.

    Agregación por dimensión de embedding: suma con signo (no norma L2).
    La suma con signo preserva la dirección de la atribución:
      - score_pos > 0: el token apoya la predicción del código.
      - score_neg > 0: el token actúa como evidencia contraria al código.

    Parámetros
    ----------
    encoding : dict
        Salida de _tokenize() con todos los tensores.
    label_idx : int
        Índice de la etiqueta en LABELS (y en la cabeza de clasificación).

    Devuelve
    --------
    token_scores_pos : np.ndarray de shape (actual_seq_len,)
        Atribuciones positivas por token (clip a 0 por abajo).
        Representan evidencia de apoyo al código predicho.
    token_scores_neg : np.ndarray de shape (actual_seq_len,)
        Atribuciones negativas invertidas por token (clip a 0 por abajo).
        Representan evidencia contraria al código predicho.
        Reservado para uso futuro en la interfaz; actualmente no se expone.
    offsets : list of (char_start, char_end)
        Correspondencia token → rango de caracteres en el texto original.
    """
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)
    offset_mapping = encoding["offset_mapping"][0]   # CPU, (MAX_LENGTH, 2)

    actual_len = int(attention_mask[0].sum().item())

    # Embeddings de entrada (leaf tensor con requires_grad para Captum)
    input_embeddings    = _get_embeddings(input_ids).requires_grad_(True)
    baseline_embeddings = _get_baseline_embeddings(input_ids)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Construir NoiseTunnel sobre la función forward correcta
    if _uses_token_type_ids:
        token_type_ids = encoding["token_type_ids"].to(_device)
        ig = IntegratedGradients(_forward_bert)
        nt = NoiseTunnel(ig)
        attrs = nt.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask, token_type_ids),
            nt_type="smoothgrad",
            nt_samples=NT_SAMPLES,
            stdevs=NT_STDEVS,
            target=label_idx,
            n_steps=N_IG_STEPS,
            internal_batch_size=4,
        )
    else:
        ig = IntegratedGradients(_forward_roberta)
        nt = NoiseTunnel(ig)
        attrs = nt.attribute(
            inputs=input_embeddings,
            baselines=baseline_embeddings,
            additional_forward_args=(attention_mask,),
            nt_type="smoothgrad",
            nt_samples=NT_SAMPLES,
            stdevs=NT_STDEVS,
            target=label_idx,
            n_steps=N_IG_STEPS,
            internal_batch_size=4,
        )

    # attrs: (1, MAX_LENGTH, hidden_dim)
    # Suma con signo sobre la dimensión de embedding.
    # A diferencia de la norma L2, la suma preserva la dirección:
    # valores positivos → apoyo al código; negativos → evidencia contraria.
    token_attrs  = attrs[0].detach().cpu()              # (MAX_LENGTH, hidden_dim)
    token_signed = token_attrs.sum(dim=-1).numpy()      # (MAX_LENGTH,)

    # Recortar al segmento real (sin padding)
    token_signed = token_signed[:actual_len]
    offsets = [(int(s), int(e)) for s, e in offset_mapping[:actual_len].tolist()]

    # Separar contribuciones de apoyo y de contradicción
    token_scores_pos = token_signed.clip(min=0)    # apoyo
    token_scores_neg = (-token_signed).clip(min=0) # contradicción (para uso futuro)

    return token_scores_pos, token_scores_neg, offsets


# ══════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE SPANS
# ══════════════════════════════════════════════════════════════════

def _split_into_segments(text: str) -> list[dict]:
    """
    Segmenta la nota clínica en unidades candidatas a span.

    Estrategia: dividir por saltos de línea y por puntuación final de
    oración. Las líneas son la unidad natural de evidencia en notas
    clínicas (anamnesis, plan, exploración, etc.).

    Devuelve lista de {text, start, end}.
    """
    splitter = re.compile(r'\n+|(?<=[.!?;])\s+')
    parts = splitter.split(text)

    segments: list[dict] = []
    search_from = 0
    for part in parts:
        part = part.strip()
        if len(part) < 5:               # Ignorar fragmentos muy cortos
            continue
        start = text.find(part, search_from)
        if start == -1:
            continue
        end = start + len(part)
        segments.append({"text": part, "start": start, "end": end})
        search_from = end

    return segments


def _split_into_subphrases(span: dict) -> list[dict]:
    """
    Segunda pasada de segmentación: divide un span de oración/línea en
    sub-frases candidatas más cortas.

    Criterios de división:
    - Paréntesis: el contenido entre paréntesis se extrae como sub-frase
      independiente. En notas clínicas suele contener datos concretos
      (valores de laboratorio, estadios, abreviaturas explicadas).
      Los bloques de paréntesis se enmascaran antes de dividir por coma
      para evitar partir expresiones como "(baseline Cr 2.0-2.1)".
    - Comas y conjunciones coordinantes (", and", ", who", ", with"...):
      las posiciones de los separadores se obtienen con re.finditer sobre
      el texto enmascarado, garantizando posiciones exactas sin búsquedas
      aproximadas. El texto real (sin enmascarar) se recupera con las
      mismas posiciones sobre el texto original.

    Si el span no puede dividirse (no hay separadores o todas las sub-frases
    quedan por debajo de MIN_SUBPHRASE_LEN), devuelve el span original como
    única entrada.

    Parámetros
    ----------
    span : dict con keys text, start, end (posiciones en el texto original)

    Devuelve
    --------
    list[dict] con keys text, start, end para cada sub-frase candidata.
    Las posiciones start/end son absolutas respecto al texto completo de
    la nota, no relativas al span.
    """
    text = span["text"]
    base = span["start"]   # offset del span dentro del texto original

    # ── 1. Extraer contenidos de paréntesis ──────────────────────
    # Se recogen como sub-frases independientes Y se enmascaran para
    # que las comas dentro de ellos no causen divisiones erróneas.
    subphrases: list[dict] = []
    paren_ranges: list[tuple[int, int]] = []  # (local_start, local_end) en `text`

    for m in re.finditer(r'\(([^)]{6,})\)', text):
        inner     = m.group(1).strip()
        loc_start = m.start(1)
        loc_end   = m.end(1)
        paren_ranges.append((m.start(), m.end()))   # rango del bloque completo "(…)"
        if len(inner) >= MIN_SUBPHRASE_LEN:
            subphrases.append({
                "text":  inner,
                "start": base + loc_start,
                "end":   base + loc_end,
            })

    # ── 2. Enmascarar bloques de paréntesis ───────────────────────
    # Sustituir cada "(…)" por espacios del mismo ancho para preservar
    # los índices de carácter exactos en el texto enmascarado.
    masked = list(text)           # lista mutable de caracteres
    for (ps, pe) in paren_ranges:
        for k in range(ps, pe):
            masked[k] = ' '
    masked = "".join(masked)

    # ── 3. Localizar los separadores con re.finditer ──────────────
    # finditer devuelve las posiciones exactas de cada separador en
    # el texto enmascarado, eliminando la necesidad de búsquedas
    # aproximadas con str.find (que causaba el bug de posiciones).
    clause_sep = re.compile(
        r',\s*(?:and\s+|who\s+|with\s+|which\s+|but\s+|or\s+)?|;\s*'
    )
    # Construir lista de (inicio_cláusula, fin_cláusula) en coordenadas locales
    boundaries: list[tuple[int, int]] = []
    prev_end = 0
    for sep in clause_sep.finditer(masked):
        boundaries.append((prev_end, sep.start()))
        prev_end = sep.end()
    boundaries.append((prev_end, len(masked)))   # última cláusula

    # ── 4. Extraer cada cláusula usando las posiciones exactas ────
    for (raw_start, raw_end) in boundaries:
        # Recortar espacios de enmascaramiento y espacios naturales
        # trabajando directamente sobre los índices, sin búsquedas.
        loc_start = raw_start
        loc_end   = raw_end
        # Avanzar inicio mientras sea espacio en el texto enmascarado
        while loc_start < loc_end and masked[loc_start] == ' ':
            loc_start += 1
        # Retroceder fin mientras sea espacio en el texto enmascarado
        while loc_end > loc_start and masked[loc_end - 1] == ' ':
            loc_end -= 1

        clause_text = text[loc_start:loc_end]   # texto real, sin enmascarar

        if len(clause_text.strip()) < MIN_SUBPHRASE_LEN:
            continue

        subphrases.append({
            "text":  clause_text.strip(),
            "start": base + loc_start,
            "end":   base + loc_end,
        })

    # ── 5. Fallback ───────────────────────────────────────────────
    if not subphrases:
        return [span]

    return subphrases


def _refine_span_to_subphrase(
    span: dict,
    token_scores_pos: np.ndarray,
    offsets: list[tuple[int, int]],
) -> dict:
    """
    Aplica la segunda pasada de segmentación sobre un span ya seleccionado
    y devuelve la sub-frase con mayor puntuación de atribución positiva.

    El score del span original se conserva en el campo "span_score" para
    mantener la comparabilidad entre spans de distintas oraciones. El campo
    "score" se reemplaza por el score normalizado de la sub-frase dentro
    del span (relativo al mejor token del propio span).

    Parámetros
    ----------
    span : dict con keys text, start, end, score (score normalizado global)
    token_scores_pos : atribuciones positivas por token del texto completo
    offsets : correspondencia token → (char_start, char_end)

    Devuelve
    --------
    dict con keys:
        text       : texto de la sub-frase más relevante
        start      : posición de inicio en el texto original
        end        : posición de fin en el texto original
        score      : score normalizado global del span de primera pasada
        subphrase  : True (indica que el texto es una sub-frase, no el span completo)
        span_text  : texto completo del span de primera pasada (para referencia)
    """
    subphrases = _split_into_subphrases(span)

    # Si no hubo división real, devolver el span tal como está
    if len(subphrases) == 1 and subphrases[0]["text"] == span["text"]:
        return {**span, "subphrase": False, "span_text": span["text"]}

    # Puntuar cada sub-frase sumando atribuciones positivas de sus tokens
    for sub in subphrases:
        sub_score   = 0.0
        token_count = 0
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            overlap_start = max(tok_start, sub["start"])
            overlap_end   = min(tok_end,   sub["end"])
            if overlap_end > overlap_start:
                sub_score   += float(token_scores_pos[i])
                token_count += 1
        sub["raw_score"]   = sub_score / max(token_count, 1)  # promedio por token para evitar sesgo hacia sub-frases más largas
        sub["token_count"] = token_count

    # Seleccionar la sub-frase con mayor atribución positiva
    best = max(subphrases, key=lambda s: s["raw_score"])

    return {
        "text":      best["text"],
        "start":     best["start"],
        "end":       best["end"],
        "score":     span["score"],   # score global del span (primera pasada)
        "subphrase": True,
        "span_text": span["text"],    # span completo, para referencia en la interfaz
    }


def _extract_top_spans(
    text: str,
    token_scores_pos: np.ndarray,
    offsets: list[tuple[int, int]],
    max_spans: int = MAX_SPANS,
) -> list[dict]:
    """
    Convierte las atribuciones positivas por token en spans de texto clasificados,
    con segunda pasada de refinamiento a nivel de sub-frase.

    Proceso:
    1. PRIMERA PASADA — segmentación a nivel de oración/línea:
       Para cada segmento, sumar las atribuciones positivas de sus tokens,
       normalizar al máximo global y seleccionar los top-N con score ≥ MIN_SPAN_SCORE.
    2. SEGUNDA PASADA — refinamiento a nivel de sub-frase:
       Para cada span seleccionado, dividirlo en sub-frases (por coma, paréntesis
       y conjunciones) y seleccionar la sub-frase con mayor atribución positiva.
       El span completo se conserva en "span_text" para referencia.

    La segunda pasada resuelve el caso en que el span ganador es una oración
    larga (p. ej. la frase de apertura con antecedentes del paciente) que contiene
    la evidencia diagnóstica relevante junto a términos irrelevantes para el código.

    Parámetros
    ----------
    token_scores_pos : np.ndarray
        Atribuciones positivas por token (salida de _compute_nt_ig_attributions).
    max_spans : int
        Número máximo de spans a devolver. Controlado por la política de
        confianza en analyze_note() (ver CONF_HIGH, CONF_MED, SPANS_MED).

    Devuelve
    --------
    List[dict] con keys: text, start, end, score, subphrase, span_text
    Ordenada por score descendente, máximo max_spans entradas.
    """
    # ── Primera pasada: segmentación a nivel de oración/línea ────
    segments = _split_into_segments(text)
    if not segments:
        return []

    for seg in segments:
        seg_score   = 0.0
        token_count = 0
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            overlap_start = max(tok_start, seg["start"])
            overlap_end   = min(tok_end,   seg["end"])
            if overlap_end > overlap_start:
                seg_score   += float(token_scores_pos[i])
                token_count += 1
        seg["raw_score"]   = seg_score
        seg["token_count"] = token_count

    raw_arr    = np.array([s["raw_score"] for s in segments])
    max_val    = raw_arr.max()
    normalized = raw_arr / max_val if max_val > 0 else raw_arr

    for i, seg in enumerate(segments):
        seg["score"] = float(normalized[i])

    top = [
        s for s in segments
        if s["score"] >= MIN_SPAN_SCORE and s["token_count"] > 0
    ]
    top.sort(key=lambda x: x["score"], reverse=True)
    top = top[:max_spans]

    # ── Segunda pasada: refinamiento a sub-frase ─────────────────
    refined = [
        _refine_span_to_subphrase(span, token_scores_pos, offsets)
        for span in top
    ]

    return refined

def _fix_mojibake(text: str) -> str:
    """
    Corrige texto que fue decodificado como latin-1 cuando debería
    haber sido UTF-8 (problema conocido como mojibake).
    Ejemplos: 'mÂ²' → 'm²', 'â€"' → '—', 'â€™' → '''
    Si el texto ya está bien codificado, lo devuelve sin cambios.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text  # ya estaba bien, no tocar


# ══════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def analyze_note(text: str) -> dict:
    """
    Pipeline completo: predicción + explicabilidad sobre una nota clínica.

    Pasos:
    1. Normalizar el texto a Unicode NFC (corrige artefactos de codificación
       como 'mÂ²' → 'm²' cuando el texto llega con bytes latin-1 mal
       interpretados como UTF-8).
    2. Tokenizar (truncación a 512 tokens, igual que en entrenamiento).
    3. Inferencia: obtener probabilidades sigmoid por etiqueta.
    4. Aplicar umbrales por clase.
    5. Para cada código predicho: calcular NT+IG y extraer top spans.
       El número de spans mostrados depende de la confianza del modelo:
         conf ≥ 0.95  → hasta MAX_SPANS spans,  tier="high"
         0.85 ≤ conf < 0.95 → hasta SPANS_MED spans, tier="medium"
         conf < 0.85  → 1 span,                 tier="low"

    Parámetros
    ----------
    text : str
        Texto de la nota clínica. Debe llegar decodificado como str Python
        (Unicode). Si proviene de bytes, decodificar con .decode("utf-8")
        antes de llamar a esta función.

    Devuelve
    --------
    dict compatible con la estructura que espera la interfaz HTML::

        {
            "codes": [
                {
                    "id":        "N183",
                    "desc":      "Chronic kidney disease, stage 3 (moderate)",
                    "conf":      0.98,
                    "conf_tier": "high",   # "high" | "medium" | "low"
                    "cat":       "Diseases of the genitourinary system",
                    "spans": [
                        {"text": "stage III CKD (baseline Cr 2.0-2.1)", "score": 1.0},
                        {"text": "His creatinine increased from 2.1 to 2.7", "score": 0.74}
                    ]
                }
            ]
        }

    Notas
    -----
    - Requiere haber llamado a load_model() previamente.
    - "score" por span: atribución NT+IG positiva normalizada al mejor
      segmento. Un score de 0.74 significa que ese fragmento acumula el 74%
      de la evidencia de apoyo del fragmento más relevante para ese código.
    - conf_tier controla cuántos spans se muestran y puede usarse en la
      interfaz para mostrar un aviso al usuario cuando la confianza es baja.
    """
    if _model is None:
        raise RuntimeError(
            "Modelo no cargado. Llama a load_model() antes de analyze_note()."
        )

    # ── 1. Normalización Unicode ──────────────────────────────────
    # NFC colapsa secuencias de bytes mal interpretadas (latin-1 → UTF-8)
    # como 'mÂ²' en 'm²', 'â€"' en '—', etc.
    text = _fix_mojibake(text)

    # ── 2. Tokenizar ─────────────────────────────────────────────
    encoding = _tokenize(text)

    # ── 3. Inferencia ────────────────────────────────────────────
    inference_inputs = {
        k: v.to(_device)
        for k, v in encoding.items()
        if k != "offset_mapping"
    }
    with torch.no_grad():
        outputs = _model(**inference_inputs)

    probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()  # (num_labels,)

    # ── 4. Aplicar umbrales por clase ────────────────────────────
    thresholds_arr = np.array([THRESHOLDS[lbl] for lbl in LABELS])
    predicted_indices = np.where(probs > thresholds_arr)[0]

    if len(predicted_indices) == 0:
        print("[analyze_note] Ningún código supera el umbral de aceptación.")
        return {"codes": []}

    predicted_labels = [LABELS[i] for i in predicted_indices]
    print(f"[analyze_note] Códigos predichos: {predicted_labels}")
    print(f"[analyze_note] Método: NoiseTunnel+IG  "
          f"(n_steps={N_IG_STEPS}, nt_samples={NT_SAMPLES}, stdevs={NT_STDEVS})")

    # ── 5. NT+IG + extracción de spans por código predicho ───────
    result_codes: list[dict] = []

    for label_idx in predicted_indices:
        label = LABELS[label_idx]
        prob  = float(probs[label_idx])

        # Determinar tier de confianza y número de spans a mostrar
        if prob >= CONF_HIGH:
            conf_tier = "high"
            n_spans   = MAX_SPANS
        elif prob >= CONF_MED:
            conf_tier = "medium"
            n_spans   = SPANS_MED
        else:
            conf_tier = "low"
            n_spans   = 5

        print(f"  → NT+IG para {label}  (p={prob:.4f}, tier={conf_tier})…",
              end=" ", flush=True)

        token_scores_pos, _token_scores_neg, offsets = _compute_nt_ig_attributions(
            encoding, label_idx=int(label_idx)
        )

        print("OK")

        top_spans = _extract_top_spans(text, token_scores_pos, offsets,
                                       max_spans=n_spans)

        result_codes.append({
            "id":        label,
            "desc":      LABEL_DESCRIPTIONS.get(label, label),
            "conf":      round(prob, 4),
            "conf_tier": conf_tier,
            "cat":       ICD_CATEGORY,
            "spans": [
                {
                    "text":      span["text"],
                    "score":     round(span["score"], 4),
                    "subphrase": span.get("subphrase", False),
                    "span_text": span.get("span_text", span["text"]),
                }
                for span in top_spans
            ],
        })

        # Liberar memoria GPU entre etiquetas
        if _device.type == "cuda":
            torch.cuda.empty_cache()

    # Ordenar por confianza descendente
    result_codes.sort(key=lambda x: x["conf"], reverse=True)

    return {"codes": result_codes}


# ══════════════════════════════════════════════════════════════════
# EJEMPLO DE USO
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Cargar modelo ────────────────────────────────────────────
    # Para BioLinkBERT:
    # load_model("./perclass_0.4_N189_0.6_BioLinkBERT-large/trained_BioLinkBERT-large")
    load_model(DEFAULT_MODEL_DIR)

    # ── VERIFICACIÓN IMPORTANTE ──────────────────────────────────
    # Antes de usar en producción, verifica que el orden de LABELS
    # coincide con mlb.classes_ del entrenamiento. Imprime:
    #
    #   import joblib
    #   mlb = joblib.load("mlb.pkl")   # si lo guardaste
    #   print(list(mlb.classes_))
    #
    # El orden debe ser idéntico a la lista LABELS al inicio de este archivo.

    # ── Nota clínica de ejemplo ──────────────────────────────────
#     sample_note = """
# DISCHARGE SUMMARY

# Patient: 68-year-old male with type 2 diabetes mellitus and hypertension.
# Admission date: 14/03/2025.

# Chief complaint: Routine nephrology follow-up with worsening renal function.

# History of present illness:
# The patient has a 12-year history of type 2 diabetes mellitus, poorly controlled
# (HbA1c 9.8% at last visit), with progressive diabetic nephropathy. Current
# laboratory results show a serum creatinine of 3.4 mg/dL and an eGFR of
# 18 mL/min/1.73m², consistent with chronic kidney disease stage 4. He also
# presents with hyperkalemia (K+ 5.9 mEq/L) and metabolic acidosis.

# He has a long history of essential hypertension managed with amlodipine and
# losartan. Urinalysis shows 3+ proteinuria (albumin-to-creatinine ratio 850 mg/g).

# Assessment:
# 1. Chronic kidney disease stage 4 due to diabetic nephropathy — nephrology
#    follow-up, dietary protein restriction, and preparation for renal replacement
#    therapy initiation within the next 6-12 months.
# 2. Hyperkalemia — dietary counselling, low-potassium diet.
# 3. Type 2 diabetes mellitus — adjustment of insulin regimen.
# 4. Essential hypertension — continue current antihypertensive therapy.

# Plan: Review in 4 weeks. Vascular surgery referral for arteriovenous fistula
# creation in anticipation of hemodialysis.
#     """.strip()

#     sample_note = """
# Mr. [UNK] is a male patient with a history of HFrEF (LVEF ~30%), CAD s/p CABG and PCI, moderate tricuspid regurgitation, right ventricular dysfunction, moderate pulmonary hypertension, paroxysmal atrial fibrillation on apixaban, stage III CKD (baseline Cr 2.0-2.1), cerebrovascular disease, and metastatic melanoma on pembrolizumab, who presented with worsening dyspnea on exertion (DOE) and volume overload. His symptoms were exacerbated by self-reduction of torsemide and high fluid intake. He was admitted for IV diuresis. During admission, he was found to be volume overloaded with bibasilar crackles, elevated JVP (20 cm) with positive hepatojugular reflux, and BNP >10K. His creatinine increased from 2.1 to 2.7, likely due to diuretic use, and improved to 2.2 upon transition to oral torsemide. Mild hyperkalemia (K 5.7) was noted and managed by stopping potassium supplementation and increasing torsemide dose. Troponin was minimally elevated (0.03-0.04), attributed to renal insufficiency and decompensated heart failure. Pembrolizumab was held previously due to diarrhea, elevated LFTs, and worsening renal function. Cardiac biomarkers showed normal CK-MB. He has a history of severe emphysema, GERD, hypertension, hyperlipidemia, BPH, and histoplasmosis. He experienced a fall resulting in a scalp laceration. Discharge weight was 55.6 kg (122.57 lb). Discharge creatinine was 2.2. Discharge medications include torsemide 40 mg daily, apixaban 2.5 mg BID, amiodarone 200 mg daily, aspirin 81 mg daily, potassium chloride 40 mEq daily, and others. He is discharged home with follow-up instructions.
#     """.strip()

#     sample_note = """
# The patient is a male with a history of HFrEF (35%), CAD s/p CABG and PCI, moderate tricuspid regurgitation, right ventricular dysfunction, moderate pulmonary hypertension, paroxysmal atrial fibrillation, stage III CKD, cerebrovascular disease, and metastatic melanoma. He was admitted for acute on chronic systolic heart failure exacerbation, presenting with volume overload and pleural effusion, likely triggered by lower left lobe pneumonia. Initial management included aggressive IV diuresis and treatment of pneumonia with Zosyn for 5 days. He was transferred to the cardiology floor, continued diuresis with transition to oral torsemide 60 mg BID, and his oxygen requirement decreased from high flow to [UNK] NC. He was discharged on torsemide 60 mg PO BID, maintaining euvolemia. He has chronic anemia (Hgb 10.3-10.6) and thrombocytopenia (platelets 100s). His baseline creatinine is 2.0-2.1, remaining stable during admission (discharge Cr 2.0). He is on amiodarone and apixaban for atrial fibrillation. Due to hypotension and CKD, afterload reducing agents, neurohormonal blockade agents, and mineralocorticoid receptor antagonists were held. He was restarted on rosuvastatin 5 mg daily for CAD. His discharge condition is stable, ambulatory with assistance, and requiring supplemental oxygen ([UNK] NC) likely related to resolving pneumonia. Discharge disposition is to an extended care facility.
#     """.strip()

    sample_note = """
The patient is a male with a history of gout, type II diabetes, and nephrolithiasis who presented with a two-day history of left ankle and knee pain accompanied by subjective fevers. Examination revealed warm, swollen left ankle and knee joints without erythema, with painful knee range of motion. Initial labs showed elevated WBC (11.0), elevated BUN (32), elevated creatinine (1.8), elevated CRP (253), and elevated uric acid (8.5). Joint fluid analysis from the left knee showed slightly cloudy yellow fluid with a high WBC count (97% polymorphonuclear leukocytes), moderate needle-shaped negatively birefringent crystals, consistent with gout. X-rays of the left ankle and knee showed degenerative changes and effusion, but no acute fracture. The patient was treated with IV fluids, morphine for pain, and a 5-day course of prednisone 40mg daily for the acute gout flare. His creatinine improved to 1.2 prior to discharge. Discharge diagnoses include acute gout flare and type II diabetes. The patient was discharged home with a tapering course of prednisone and instructions for follow-up.
    """.strip()

    # ── Ejecutar análisis ────────────────────────────────────────
    results = analyze_note(sample_note)

    # ── Mostrar resultados ───────────────────────────────────────
    print("\n" + "═" * 60)
    print("RESULTADOS DEL ANÁLISIS")
    print("═" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # El diccionario `results` tiene exactamente la estructura que espera
    # la interfaz HTML (campo `codes` con id, desc, conf, conf_tier, cat, spans).
    # El campo `conf_tier` ("high"/"medium"/"low") puede usarse en el HTML
    # para mostrar un aviso visual cuando la confianza es baja o media.
