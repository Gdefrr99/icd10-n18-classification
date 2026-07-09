# -*- coding: utf-8 -*-
"""
Definición de los 13 grupos de riesgo CMS-HCC evaluados (Sección 3.4 y 4.1).

Cada grupo se define por una lista de prefijos de código ICD-10-CM (a nivel
de categoría de 3 caracteres). Una nota pertenece a un grupo si al menos uno
de sus códigos empieza por alguno de los prefijos del grupo.
"""

HCC_GROUPS = {
    "diabetes_mellitus":            {
        "nombre": "Diabetes mellitus",
        "prefijos": ["E08", "E09", "E10", "E11", "E13"],
    },
    "congestive_heart_failure":       {
        "nombre": "Congestive heart failure (CHF)",
        "prefijos": ["I11", "I42", "I50"],
    },
    "vascular_disease":          {
        "nombre": "Vascular disease",
        "prefijos": ["I25", "I70", "I71", "I72", "I73"],
    },
    "chronic_kidney_disease":     {
        "nombre": "Chronic kidney disease (CKD)",
        "prefijos": ["N18"],
    },
    "copd_and_lung_disorders":   {
        "nombre": "COPD and lung disorders",
        "prefijos": ["J41", "J42", "J43", "J44", "J45", "J47", "J84"],
    },
    "oncology":                    {
        "nombre": "Oncology",
        "prefijos": ["C"],
    },
    "major_psychiatric_conditions":    {
        "nombre": "Major psychiatric conditions",
        "prefijos": ["F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29", "F30", "F31", "F32", "F33"],
    },
    "neurological_conditions":     {
        "nombre": "Neurological conditions",
        "prefijos": ["G20", "G30", "G35", "G40", "G80", "G81", "G82", "G83"],
    },
    "hepatic_disease":          {
        "nombre": "Hepatic disease",
        "prefijos": ["K7", "B18"],
    },
    "vih":                          {
        "nombre": "VIH",
        "prefijos": ["B20"],
    },
    "amputations":                 {
        "nombre": "Amputations",
        "prefijos": ["Z89"],
    },
    "hematological_disorders":     {
        "nombre": "Severe hematological disorders",
        "prefijos": ["D57", "D61", "D63", "D66", "D67", "D69"],
    },
    "severe_infections":          {
        "nombre": "Severe infections",
        "prefijos": ["A40", "A41"],
    },
}


def matches_group(code: str, prefijos: list) -> bool:
    return any(str(code).startswith(p) for p in prefijos)


def note_codes_in_group(codes: list, prefijos: list) -> list:
    """Devuelve la sublista (a nivel de 3 caracteres) de códigos de la nota
    que pertenecen al grupo de riesgo."""
    return [str(c)[:3] for c in codes if matches_group(c, prefijos)]