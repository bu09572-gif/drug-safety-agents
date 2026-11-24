"""Prompts for adverse event extraction, and parsing of the model's answer."""

from __future__ import annotations

import json
import re

INSTRUCTION = (
    "You read one sentence from a medical case report. List every adverse drug event in it: "
    "a drug and the harmful effect the sentence says that drug caused. Copy the drug and the effect "
    "exactly as they are written in the sentence. Answer with JSON only, as a list like "
    '[{"drug": "...", "effect": "..."}]. If the sentence has no adverse drug event, answer [].'
)


def user_message(text: str) -> str:
    return f"Sentence: {text}"


def answer_json(pairs: list[dict]) -> str:
    return json.dumps([{"drug": p["drug"], "effect": p["effect"]} for p in pairs], ensure_ascii=False)


def build_messages(text: str, shots: list | None = None) -> list[dict]:
    """Chat messages for one sentence. `shots` are worked examples (RAG few-shot)."""
    msgs = [{"role": "system", "content": INSTRUCTION}]
    for s in shots or []:
        msgs.append({"role": "user", "content": user_message(s.text)})
        msgs.append({"role": "assistant", "content": answer_json(s.pairs)})
    msgs.append({"role": "user", "content": user_message(text)})
    return msgs


_LIST_RE = re.compile(r"\[.*\]", re.S)


def parse_answer(raw: str) -> tuple[list[dict], bool]:
    """Return (pairs, valid_json). Invalid output counts as no pairs."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    m = _LIST_RE.search(raw)
    if not m:
        return [], False
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [], False
    if not isinstance(data, list):
        return [], False
    pairs, ok = [], True
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("drug"), str) and isinstance(item.get("effect"), str):
            p = {"drug": item["drug"].strip(), "effect": item["effect"].strip()}
            if p["drug"] and p["effect"] and p not in pairs:
                pairs.append(p)
        else:
            ok = False
    return pairs, ok


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())
