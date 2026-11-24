"""Load the ADE Corpus V2 and turn it into extraction examples.

ADE Corpus V2 (Gurulingappa et al., 2012) has sentences from PubMed case reports.
Experts marked each sentence as describing an adverse drug event or not, and for the
positive sentences they marked the drug and the effect spans.

One example = one sentence and the list of (drug, effect) pairs in it. Sentences with
no adverse event get an empty list.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

HF_DATASET = "ade-benchmark-corpus/ade_corpus_v2"
RELATION_CONFIG = "Ade_corpus_v2_drug_ade_relation"
CLASSIFICATION_CONFIG = "Ade_corpus_v2_classification"


@dataclass
class Example:
    id: str
    text: str
    pairs: list[dict] = field(default_factory=list)  # [{"drug": ..., "effect": ...}]
    split: str = ""

    @property
    def has_ade(self) -> bool:
        return bool(self.pairs)


def sentence_id(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:12]


def split_of(text: str, test_share: float = 0.15, val_share: float = 0.05) -> str:
    """Stable split from a hash of the sentence, so the same sentence is always in the same split."""
    h = int(hashlib.md5(text.strip().encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < test_share:
        return "test"
    if h < test_share + val_share:
        return "val"
    return "train"


def build_examples(relation_rows: list[dict], classification_rows: list[dict]) -> list[Example]:
    """Group relation rows by sentence, then add the sentences that have no adverse event."""
    by_text: dict[str, Example] = {}
    for r in relation_rows:
        text = r["text"].strip()
        ex = by_text.setdefault(text, Example(id=sentence_id(text), text=text, split=split_of(text)))
        pair = {"drug": r["drug"].strip(), "effect": r["effect"].strip()}
        if pair not in ex.pairs:
            ex.pairs.append(pair)
    for r in classification_rows:
        text = r["text"].strip()
        if int(r["label"]) == 0 and text not in by_text:
            by_text[text] = Example(id=sentence_id(text), text=text, split=split_of(text))
    return sorted(by_text.values(), key=lambda e: e.id)


def _load_config(config: str) -> list[dict]:
    try:
        from datasets import load_dataset

        return list(load_dataset(HF_DATASET, config, split="train"))
    except Exception as err:  # fall back to the Parquet export of the Hugging Face dataset viewer
        import pandas as pd
        import requests

        print(f"load_dataset failed ({err}); reading the Parquet export", flush=True)
        files = requests.get("https://datasets-server.huggingface.co/parquet", params={"dataset": HF_DATASET},
                             timeout=60).json()["parquet_files"]
        urls = [f["url"] for f in files if f["config"] == config and f["split"] == "train"]
        return pd.concat([pd.read_parquet(u) for u in urls]).to_dict("records")


def load_from_hub() -> list[Example]:
    return build_examples(_load_config(RELATION_CONFIG), _load_config(CLASSIFICATION_CONFIG))


def sample_balanced(examples: list[Example], n_pos: int, n_neg: int, seed: int = 0) -> list[Example]:
    rng = random.Random(seed)
    pos = [e for e in examples if e.has_ade]
    neg = [e for e in examples if not e.has_ade]
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(out)
    return out


def save_jsonl(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for e in examples:
            f.write(json.dumps(asdict(e)) + "\n")


def load_jsonl(path: Path) -> list[Example]:
    with path.open() as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]
