# Step 8 — Explainability (Noise Tunnel + Integrated Gradients)

Post-hoc explainability module for the fine-tuned N18.x classifiers.
Identifies the text spans in a discharge note that most support each predicted ICD-10-CM code.

## Method

**Integrated Gradients (IG)** computes the attribution of each input token to the model's output by integrating gradients along a path from a baseline (all-PAD sequence) to the actual input.

**Noise Tunnel (SmoothGrad-IG)** wraps IG and averages attributions over `NT_SAMPLES` Gaussian perturbations of the input embeddings. This reduces variance inherent to IG in large models (~70% reduction) and produces more stable span rankings across runs.

**Two-pass span extraction**:
1. **First pass** — segment the note into lines/sentences, compute signed attribution sum per segment, normalize to [0, 1], and select the top-N segments above `MIN_SPAN_SCORE`.
2. **Second pass** — for each selected span, extract sub-phrases by splitting on parentheses, commas, and coordinating conjunctions. Score each sub-phrase by **average positive attribution per token** (sum / token_count) — the average avoids length bias toward longer sub-phrases. Return the highest-scoring sub-phrase.

**Aggregation over embedding dimension**: signed sum (not L2 norm). This preserves attribution direction: positive tokens support the code, negative tokens act as contrary evidence.

## Configuration

Edit the constants at the top of `icd10_explainability.py` before running:

| Constant | Default | Meaning |
|---|---|---|
| `DEFAULT_MODEL_DIR` | `./trained_RoBERTa-large-pubmed-mimic3-Voc-hf` | Path to fine-tuned model |
| `LABELS` | `["N181", ..., "N189"]` | Must match `mlb.classes_` from training |
| `THRESHOLDS` | `{cls: 0.6}` | Per-class decision thresholds |
| `N_IG_STEPS` | `20` | IG integration steps per noise sample |
| `NT_SAMPLES` | `10` | Number of Gaussian perturbations |
| `NT_STDEVS` | `0.01` | Standard deviation of Gaussian noise |
| `MAX_SPANS` | `5` | Max spans at high confidence |
| `MIN_SPAN_SCORE` | `0.05` | Minimum normalized score to include a span |
| `CONF_HIGH` | `0.95` | Threshold for high-confidence tier |
| `CONF_MED` | `0.85` | Threshold for medium-confidence tier |

## Hardware requirements

| Model | VRAM | Notes |
|---|---|---|
| BioLinkBERT-large | ≥ 16 GB | 10×20 = 200 forward passes per label |
| RoBERTa-large-PM-M3-Voc-hf | ≥ 16 GB | Same cost |
| PubMedBERT_abstract | ≥ 8 GB | Smaller model, faster |

CPU inference is supported but slow (~5-10 min per note for large models).

## Usage

```python
from icd10_explainability import load_model, analyze_note

# Load fine-tuned model (output of Step 5)
load_model("models/chunking/trained_BioLinkBERT-large")

# Analyze a discharge note
results = analyze_note(open("note.txt").read())

# Output structure:
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

## Important: label order verification

Before deploying, verify that `LABELS` in `icd10_explainability.py` matches exactly the order of `mlb.classes_` from training:

```python
import joblib
mlb = joblib.load("mlb.pkl")  # saved during preprocessing
print(list(mlb.classes_))     # must match LABELS list
```

`MultiLabelBinarizer` sorts classes alphabetically by default, so the expected order is:
`["N181", "N182", "N183", "N184", "N185", "N186", "N189"]`
(note: dots are removed, matching the format used during training).
