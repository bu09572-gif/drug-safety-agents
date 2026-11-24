"""Scores for extracted (drug, effect) pairs."""

from __future__ import annotations

from .extract import norm


def _key(p: dict) -> tuple[str, str]:
    return norm(p["drug"]), norm(p["effect"])


def _overlap(a: str, b: str) -> bool:
    """Lenient match: one span contains the other (after normalizing)."""
    return bool(a) and bool(b) and (a in b or b in a)


def match_counts(pred: list[dict], gold: list[dict], lenient: bool = False) -> tuple[int, int, int]:
    """Return (true positives, number predicted, number gold) for one sentence."""
    pk = list(dict.fromkeys(_key(p) for p in pred))
    gk = list(dict.fromkeys(_key(g) for g in gold))
    used, tp = set(), 0
    for pd, pe in pk:
        for i, (gd, ge) in enumerate(gk):
            if i in used:
                continue
            ok = (pd == gd and pe == ge) if not lenient else (_overlap(pd, gd) and _overlap(pe, ge))
            if ok:
                used.add(i)
                tp += 1
                break
    return tp, len(pk), len(gk)


def prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def grounded(pair: dict, text: str) -> bool:
    """Both spans appear in the sentence (after normalizing)."""
    t = norm(text)
    d, e = norm(pair["drug"]), norm(pair["effect"])
    return bool(d) and bool(e) and d in t and e in t


def score(rows: list[dict]) -> dict:
    """rows: dicts with text, gold (pairs), pred (pairs), valid_json, seconds."""
    out: dict = {"n_sentences": len(rows)}
    for name, lenient in (("strict", False), ("lenient", True)):
        tp = npred = ngold = 0
        for r in rows:
            a, b, c = match_counts(r["pred"], r["gold"], lenient)
            tp, npred, ngold = tp + a, npred + b, ngold + c
        out[name] = prf(tp, npred, ngold)
    # Sentence level: does the sentence describe an adverse event at all?
    tp = sum(1 for r in rows if r["pred"] and r["gold"])
    fp = sum(1 for r in rows if r["pred"] and not r["gold"])
    fn = sum(1 for r in rows if not r["pred"] and r["gold"])
    out["sentence_detection"] = prf(tp, tp + fp, tp + fn)
    n_pairs = sum(len(r["pred"]) for r in rows)
    out["valid_json_rate"] = round(sum(r["valid_json"] for r in rows) / max(len(rows), 1), 4)
    out["grounded_rate"] = round(
        sum(grounded(p, r["text"]) for r in rows for p in r["pred"]) / n_pairs, 4) if n_pairs else None
    secs = sorted(r.get("seconds", 0.0) for r in rows)
    out["median_seconds"] = round(secs[len(secs) // 2], 3) if secs else None
    out["mean_prompt_tokens"] = round(sum(r.get("prompt_tokens", 0) for r in rows) / max(len(rows), 1), 1)
    return out
