import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drugsafety.data import build_examples  # noqa: E402

# Real sentences from ADE Corpus V2.
RELATION_ROWS = [
    {"text": "Intravenous azithromycin-induced ototoxicity.", "drug": "azithromycin", "effect": "ototoxicity"},
    {"text": "Methotrexate-induced pneumonitis in a patient with rheumatoid arthritis.", "drug": "Methotrexate",
     "effect": "pneumonitis"},
    {"text": "Lithium toxicity and hypothyroidism after starting lithium carbonate.", "drug": "lithium carbonate",
     "effect": "Lithium toxicity"},
    {"text": "Lithium toxicity and hypothyroidism after starting lithium carbonate.", "drug": "lithium carbonate",
     "effect": "hypothyroidism"},
]
CLASS_ROWS = [
    {"text": "Methotrexate-induced pneumonitis in a patient with rheumatoid arthritis.", "label": 1},
    {"text": "We report a case of a 54-year-old woman treated with methotrexate for psoriasis.", "label": 0},
    {"text": "The patient recovered after supportive care.", "label": 0},
]


@pytest.fixture
def examples():
    return build_examples(RELATION_ROWS, CLASS_ROWS)
