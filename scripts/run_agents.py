"""Run the drug safety review workflow on questions about the most reported drugs in the test sentences.

Writes results/agent_eval.json, results/sample_reports.md and traces/agent_trace.jsonl.

    python scripts/run_agents.py --adapter adapters/qwen0.5b-ade --drugs 15
"""

import argparse
import json
import time
from pathlib import Path

from drugsafety.agents.evaluate import score_drug, summarize, top_drugs
from drugsafety.agents.graph import Tracer, build_graph, render_report
from drugsafety.agents.tools import FDALabels, SentenceCorpus
from drugsafety.extract import norm
from drugsafety.llm import LocalChatModel
from drugsafety.retrieve import ExampleRetriever
from drugsafety.splits import BASE_MODEL, BIG_MODEL, splits

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="adapters/qwen0.5b-ade")
ap.add_argument("--drugs", type=int, default=15)
ap.add_argument("--pairs", type=int, default=5, help="extra questions that ask about two drugs at once")
ap.add_argument("--max-sentences", type=int, default=60)
a = ap.parse_args()

sp = splits()
corpus = SentenceCorpus(sp["test_all"])
by_id = {e.id: e for e in sp["test_all"]}
drugs = top_drugs(sp["test_all"], a.drugs)
questions = [f"What adverse events are reported for {d} in these case reports, and are they on its FDA label?"
             for d in drugs]
questions += [f"Compare the side effects reported for {drugs[i]} and {drugs[i + 1]}. Which ones are not on "
              f"their FDA labels?" for i in range(0, min(2 * a.pairs, len(drugs) - 1), 2)]

trace_path = Path("traces/agent_trace.jsonl")
trace_path.unlink(missing_ok=True)
tracer = Tracer(trace_path)
planner = LocalChatModel(BIG_MODEL, max_new_tokens=80)
extractor = LocalChatModel(BASE_MODEL, adapter=a.adapter)
second = LocalChatModel(BASE_MODEL)
graph = build_graph(planner, extractor, corpus, FDALabels(Path("data/fda_labels")), tracer,
                    retriever=ExampleRetriever(sp["train_pool"]), second_opinion_llm=second,
                    max_sentences=a.max_sentences)
Path("results").mkdir(exist_ok=True)
Path("results/graph.mmd").write_text(graph.get_graph().draw_mermaid())

runs, per_drug, reports = [], [], []
for qi, q in enumerate(questions):
    tracer.run_id = f"q{qi}"
    t = time.perf_counter()
    state = graph.invoke({"question": q}, {"recursion_limit": 20})
    asked = [d for d in drugs if norm(d) in norm(q)]
    got = [norm(d) for d in state["plan"]["drugs"]]
    run = {"id": f"q{qi}", "question": q, "plan": state["plan"], "planner_valid_json": state["planner_valid"],
           "planner_correct": sorted(got) == sorted(asked), "retries": state.get("retries", 0),
           "rejected": len(state.get("rejected", [])), "seconds": round(time.perf_counter() - t, 1),
           "report": state["report"]}
    runs.append(run)
    if qi < len(drugs):  # score the single-drug questions; pairs repeat the same drugs
        per_drug.append(score_drug(drugs[qi], state, by_id))
    if qi < 3 or qi == len(drugs):
        reports.append(render_report(q, state["report"]))
    print(json.dumps({k: v for k, v in run.items() if k != "report"}), flush=True)

signals = sum(len(r["possible_new_signals"]) for run in runs[: len(drugs)] for r in run["report"].values())
effects = sum(len(r["effects"]) for run in runs[: len(drugs)] for r in run["report"].values())
summary = {
    "questions": len(runs), "drugs": drugs,
    "planner_correct": sum(r["planner_correct"] for r in runs), "planner_valid_json": sum(r["planner_valid_json"]
                                                                                       for r in runs),
    "labels_found": sum(1 for run in runs[: len(drugs)] for r in run["report"].values() if r["label_found"]),
    "extraction": summarize(per_drug),
    "rejected_items": sum(r["rejected"] for r in runs), "questions_with_retry": sum(r["retries"] > 0 for r in runs),
    "distinct_effects": effects, "not_on_label": signals,
    "median_seconds_per_question": sorted(r["seconds"] for r in runs)[len(runs) // 2],
    "trace_events": len(tracer.events), "per_drug": per_drug, "runs": runs,
}
Path("results/agent_eval.json").write_text(json.dumps(summary, indent=2))
Path("results/sample_reports.md").write_text("# Sample drug safety reports\n\n" + "\n\n".join(reports))
print(json.dumps({k: v for k, v in summary.items() if k not in ("runs", "per_drug")}, indent=2))
