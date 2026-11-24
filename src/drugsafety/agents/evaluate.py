"""Score the agent workflow against the expert labels of the sentences it read."""

from __future__ import annotations

from collections import Counter

from ..data import Example
from ..extract import norm
from ..metrics import prf
from .tools import drug_matches


def top_drugs(examples: list[Example], k: int) -> list[str]:
    c = Counter(norm(p["drug"]) for e in examples for p in e.pairs)
    return [d for d, _ in c.most_common(k)]


def gold_for(drug: str, sentence_ids: set[str], by_id: dict[str, Example]) -> set[tuple[str, str]]:
    return {(sid, norm(p["effect"])) for sid in sentence_ids for p in by_id[sid].pairs
            if drug_matches(p["drug"], drug)}


def _match(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> int:
    """Same sentence, and one effect span contains the other."""
    used, tp = set(), 0
    for sid, e in pred:
        for g in gold:
            if g not in used and g[0] == sid and (e in g[1] or g[1] in e):
                used.add(g)
                tp += 1
                break
    return tp


def score_drug(drug: str, state: dict, by_id: dict[str, Example]) -> dict:
    sids = {s["id"] for s in state["sentences"].get(drug, [])}
    gold = gold_for(drug, sids, by_id)
    first = {(v["sentence_id"], norm(p["effect"])) for v in state["first_pass"].values() if v["drug"] == drug
             for p in v["pairs"] if drug_matches(p["drug"], drug)}
    final = {(a["sentence_id"], norm(a["effect"])) for a in state.get("accepted", {}).get(drug, [])}
    return {"drug": drug, "sentences": len(sids), "gold_pairs": len(gold),
            "first_pass": {"tp": _match(first, gold), "pred": len(first)},
            "final": {"tp": _match(final, gold), "pred": len(final)}}


def summarize(per_drug: list[dict]) -> dict:
    out = {}
    g = sum(d["gold_pairs"] for d in per_drug)
    for stage in ("first_pass", "final"):
        tp = sum(d[stage]["tp"] for d in per_drug)
        pr = sum(d[stage]["pred"] for d in per_drug)
        out[stage] = {**prf(tp, pr, g), "tp": tp, "pred": pr, "gold": g}
    return out
