"""Drug safety review as a multi-agent workflow (LangGraph).

    planner -> search -> worker -> reviewer --(rejected items, once)--> worker
                                        \\-> label_check -> report

- planner:  a general chat model turns the question into a plan (which drugs to review).
            The plan is checked in code; a drug that is not in the question is dropped.
- search:   finds case report sentences that mention each drug.
- worker:   the fine-tuned extractor lists (drug, effect) pairs for each sentence.
- reviewer: checks every pair in code: valid JSON, both spans copied from the sentence,
            the drug is the one under review. Rejected sentences go back to a worker once,
            this time with retrieved worked examples (RAG few-shot) as a second opinion.
- label_check: looks up each drug's FDA label and marks each effect as on or not on the label.
- report:   groups effects with the sentences that support them.

Every node call is written to a trace (JSON lines) with its inputs, outputs and time.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from ..extract import build_messages, norm, parse_answer
from ..metrics import grounded
from .tools import FDALabels, SentenceCorpus, drug_matches, effect_on_label

PLANNER_PROMPT = (
    "You plan a drug safety review. Read the question and answer with JSON only, like "
    '{"drugs": ["drug one", "drug two"], "check_label": true}. "drugs" lists every drug name the question '
    'asks about, written as in the question. "check_label" is true if the question asks whether the effects '
    "are on the drug's FDA label."
)


class Tracer:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[dict] = []
        self.run_id = ""

    def log(self, node: str, started: float, inputs: Any, outputs: Any, **extra) -> None:
        ev = {"run": self.run_id, "node": node, "ms": round((time.perf_counter() - started) * 1000, 1),
              "inputs": inputs, "outputs": outputs, **extra}
        self.events.append(ev)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")


class State(TypedDict, total=False):
    question: str
    plan: dict
    planner_valid: bool
    sentences: dict          # drug -> list of {id, text}
    extractions: dict        # sentence id -> {"drug": target, "pairs": [...], "valid": bool, "by": str}
    first_pass: dict         # copy of extractions before review
    retry_queue: list        # [(drug, sentence id, reason)]
    retries: int
    accepted: dict           # drug -> list of {"effect", "sentence_id", "text"}
    rejected: list
    labels: dict             # drug -> label info
    report: dict


def _parse_plan(raw: str, question: str, known: set[str]) -> tuple[dict, bool]:
    m = re.search(r"\{.*\}", raw, re.S)
    plan, valid = None, False
    if m:
        try:
            plan = json.loads(m.group(0))
        except json.JSONDecodeError:
            plan = None
    q = norm(question)
    if isinstance(plan, dict) and isinstance(plan.get("drugs"), list):
        drugs = [d for d in plan["drugs"] if isinstance(d, str) and norm(d) and
                 re.search(rf"\b{re.escape(norm(d))}\b", q)]
        if drugs:
            valid = len(drugs) == len(plan["drugs"])
            return {"drugs": list(dict.fromkeys(drugs)), "check_label": bool(plan.get("check_label", True))}, valid
    # Fallback: known drug names found in the question.
    drugs = sorted((d for d in known if re.search(rf"\b{re.escape(d)}\b", q)), key=lambda d: q.find(d))
    return {"drugs": drugs, "check_label": True, "fallback": True}, False


def build_graph(planner_llm, extractor_llm, corpus: SentenceCorpus, labels: FDALabels,
                tracer: Tracer, retriever=None, second_opinion_llm=None, max_sentences: int = 60):
    known = corpus.known_drugs()

    def planner(state: State) -> State:
        t = time.perf_counter()
        gen = planner_llm.chat([{"role": "system", "content": PLANNER_PROMPT},
                                {"role": "user", "content": state["question"]}], max_new_tokens=80)
        plan, valid = _parse_plan(gen.text, state["question"], known)
        tracer.log("planner", t, state["question"], {"raw": gen.text, "plan": plan, "valid": valid},
                   model=getattr(planner_llm, "name", "planner"))
        return {"plan": plan, "planner_valid": valid, "retries": 0}

    def search(state: State) -> State:
        t = time.perf_counter()
        found = {d: [{"id": e.id, "text": e.text} for e in corpus.search(d, max_sentences)]
                 for d in state["plan"]["drugs"]}
        tracer.log("search", t, state["plan"]["drugs"], {d: len(v) for d, v in found.items()})
        return {"sentences": found}

    def _extract(llm, text: str, shots=None):
        gen = llm.chat(build_messages(text, shots))
        pairs, valid = parse_answer(gen.text)
        return pairs, valid, gen

    def worker(state: State) -> State:
        t = time.perf_counter()
        ex = dict(state.get("extractions") or {})
        queue = state.get("retry_queue") or []
        if not queue:  # first pass: every sentence
            jobs = [(d, s["id"], s["text"]) for d, ss in state["sentences"].items() for s in ss]
            llm, by = extractor_llm, "fine-tuned"
        else:          # second opinion on the rejected sentences only
            texts = {s["id"]: s["text"] for ss in state["sentences"].values() for s in ss}
            jobs = [(d, sid, texts[sid]) for d, sid, _ in queue]
            llm, by = second_opinion_llm or extractor_llm, "second-opinion"
        for drug, sid, text in jobs:
            shots = retriever.top_k(text) if (queue and retriever) else None
            pairs, valid, gen = _extract(llm, text, shots)
            ex[f"{drug}|{sid}"] = {"drug": drug, "sentence_id": sid, "text": text, "pairs": pairs,
                                   "valid": valid, "by": by}
            tracer.log("worker", t, {"drug": drug, "sentence_id": sid}, {"raw": gen.text, "pairs": pairs},
                       model=getattr(llm, "name", by), pass_=by, prompt_tokens=gen.prompt_tokens)
            t = time.perf_counter()
        out: State = {"extractions": ex, "retry_queue": []}
        if not queue:
            out["first_pass"] = json.loads(json.dumps(ex))
        else:
            out["retries"] = state.get("retries", 0) + 1
        return out

    def reviewer(state: State) -> State:
        t = time.perf_counter()
        accepted: dict = defaultdict(list)
        rejected, retry = [], []
        for key, item in state["extractions"].items():
            reasons = []
            if not item["valid"]:
                reasons.append("answer was not valid JSON")
            for p in item["pairs"]:
                if not grounded(p, item["text"]):
                    reasons.append(f"'{p['drug']}' / '{p['effect']}' is not copied from the sentence")
                elif not drug_matches(p["drug"], item["drug"]):
                    continue  # an event for another drug in the same sentence: not part of this review
                elif norm(p["effect"]) == norm(p["drug"]):
                    reasons.append("effect is the drug name")
                else:
                    accepted[item["drug"]].append({"effect": p["effect"], "sentence_id": item["sentence_id"],
                                                   "text": item["text"], "by": item["by"]})
            if reasons:
                rejected.append({"key": key, "reasons": reasons, "by": item["by"]})
                if item["by"] == "fine-tuned":
                    retry.append((item["drug"], item["sentence_id"], "; ".join(reasons)))
        do_retry = bool(retry) and state.get("retries", 0) < 1
        if do_retry:  # remove the rejected first-pass answers; the second opinion replaces them
            keep = {k: v for k, v in state["extractions"].items() if (v["drug"], v["sentence_id"]) not in
                    {(d, s) for d, s, _ in retry}}
        tracer.log("reviewer", t, {"items": len(state["extractions"])},
                   {"accepted": sum(len(v) for v in accepted.values()), "rejected": len(rejected),
                    "sent_back": len(retry) if do_retry else 0, "rejections": rejected[:20]})
        out: State = {"accepted": dict(accepted), "rejected": rejected, "retry_queue": retry if do_retry else []}
        if do_retry:
            out["extractions"] = keep
        return out

    def route_after_review(state: State) -> str:
        return "worker" if state.get("retry_queue") else "label_check"

    def label_check(state: State) -> State:
        t = time.perf_counter()
        info = {}
        if state["plan"].get("check_label", True):
            for d in state["plan"]["drugs"]:
                lab = labels.get(d)
                info[d] = {"found": lab["found"], "set_id": lab.get("set_id"), "text": lab.get("text", "")}
        tracer.log("label_check", t, state["plan"]["drugs"],
                   {d: {"found": v["found"], "set_id": v["set_id"]} for d, v in info.items()})
        return {"labels": info}

    def report(state: State) -> State:
        t = time.perf_counter()
        rep = {}
        for d in state["plan"]["drugs"]:
            groups: dict = defaultdict(list)
            for a in state.get("accepted", {}).get(d, []):
                groups[norm(a["effect"])].append(a)
            lab = state.get("labels", {}).get(d)
            effects = []
            for _key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
                on = effect_on_label(items[0]["effect"], lab["text"]) if lab and lab["found"] else None
                effects.append({"effect": items[0]["effect"], "sentences": len({i["sentence_id"] for i in items}),
                                "on_label": on, "evidence": [{"id": i["sentence_id"], "text": i["text"]}
                                                             for i in items[:3]]})
            rep[d] = {"sentences_read": len(state["sentences"].get(d, [])),
                      "label_found": lab["found"] if lab else None, "label_set_id": lab["set_id"] if lab else None,
                      "effects": effects,
                      "possible_new_signals": [e["effect"] for e in effects if e["on_label"] is False]}
        tracer.log("report", t, None, {d: {"effects": len(v["effects"]),
                                           "not_on_label": len(v["possible_new_signals"])} for d, v in rep.items()})
        return {"report": rep}

    g = StateGraph(State)
    for name, fn in [("planner", planner), ("search", search), ("worker", worker), ("reviewer", reviewer),
                     ("label_check", label_check), ("report", report)]:
        g.add_node(name, fn)
    g.set_entry_point("planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "worker")
    g.add_edge("worker", "reviewer")
    g.add_conditional_edges("reviewer", route_after_review, {"worker": "worker", "label_check": "label_check"})
    g.add_edge("label_check", "report")
    g.add_edge("report", END)
    return g.compile()


def render_report(question: str, rep: dict) -> str:
    lines = [f"**Question:** {question}", ""]
    for drug, r in rep.items():
        lab = "found" if r["label_found"] else ("not found" if r["label_found"] is False else "not checked")
        lines.append(f"### {drug}  ({r['sentences_read']} sentences read, FDA label {lab})")
        if not r["effects"]:
            lines.append("No adverse events found.")
        for e in r["effects"]:
            tag = {True: "on label", False: "**NOT on label: review**", None: "label not checked"}[e["on_label"]]
            ev = e["evidence"][0]
            lines.append(f"- {e['effect']} ({e['sentences']} sentence(s), {tag}). Evidence [{ev['id']}]: "
                         f"\"{ev['text']}\"")
        lines.append("")
    return "\n".join(lines)
