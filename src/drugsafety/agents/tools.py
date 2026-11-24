"""Tools the agents use: search the case report sentences, and read a drug's FDA label."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from ..data import Example
from ..extract import norm

LABEL_SECTIONS = ["boxed_warning", "warnings_and_cautions", "warnings", "precautions", "adverse_reactions",
                  "contraindications"]


def drug_matches(name: str, target: str) -> bool:
    a, b = norm(name), norm(target)
    return bool(a) and bool(b) and (a == b or re.search(rf"\b{re.escape(b)}\b", a) is not None
                                    or re.search(rf"\b{re.escape(a)}\b", b) is not None)


class SentenceCorpus:
    def __init__(self, examples: list[Example]):
        self.examples = examples
        self._norm = [norm(e.text) for e in examples]

    def search(self, drug: str, limit: int = 60) -> list[Example]:
        """Sentences that mention the drug as a whole word."""
        pat = re.compile(rf"\b{re.escape(norm(drug))}\b")
        hits = [e for e, t in zip(self.examples, self._norm) if pat.search(t)]
        return hits[:limit]

    def known_drugs(self) -> set[str]:
        return {norm(p["drug"]) for e in self.examples for p in e.pairs}


class FDALabels:
    """Drug labels from openFDA (api.fda.gov), cached as JSON files."""

    URL = "https://api.fda.gov/drug/label.json"

    def __init__(self, cache_dir: Path, offline: bool = False):
        self.cache_dir = cache_dir
        self.offline = offline
        cache_dir.mkdir(parents=True, exist_ok=True)

    def _fetch(self, drug: str) -> dict | None:
        for field in ("openfda.generic_name", "openfda.brand_name", "openfda.substance_name"):
            q = f'{field}:"{drug}"'
            for attempt in range(3):
                try:
                    r = requests.get(self.URL, params={"search": q, "limit": 1}, timeout=30)
                except requests.RequestException:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 404:
                    break
                if r.ok:
                    res = r.json().get("results") or []
                    if res:
                        return res[0]
                    break
                time.sleep(2 ** attempt)
        return None

    def get(self, drug: str) -> dict:
        """{"found": bool, "set_id": ..., "text": all safety sections joined}"""
        path = self.cache_dir / f"{norm(drug).replace(' ', '_')}.json"
        if path.exists():
            return json.loads(path.read_text())
        label = None if self.offline else self._fetch(drug)
        if label is None:
            out = {"drug": drug, "found": False, "set_id": None, "text": ""}
        else:
            text = " ".join(" ".join(label.get(s, [])) for s in LABEL_SECTIONS)
            out = {"drug": drug, "found": True, "set_id": label.get("set_id"),
                   "generic_name": (label.get("openfda", {}).get("generic_name") or [None])[0],
                   "sections": [s for s in LABEL_SECTIONS if label.get(s)], "text": text}
        if not self.offline:
            path.write_text(json.dumps(out))
        return out


_STOP = {"of", "the", "and", "with", "in", "a", "an", "to", "induced", "related", "associated", "severe", "acute",
         "chronic", "drug", "toxicity", "on", "due", "after", "for"}


def _stem(w: str) -> str:
    return w[:6]


def effect_on_label(effect: str, label_text: str) -> bool:
    """True if the effect, or all of its content words (first 6 letters), appear in the label."""
    e, lab = norm(effect), norm(label_text)
    if not e or not lab:
        return False
    if e in lab:
        return True
    words = [w for w in e.split() if w not in _STOP and len(w) > 2]
    if not words:
        return False
    lab_stems = {_stem(w) for w in lab.split()}
    return all(_stem(w) in lab_stems for w in words)
