"""Merge the shard files and score every method. Writes results/extraction_eval.json and .md."""

import json
from collections import defaultdict
from pathlib import Path

from drugsafety.metrics import score

NAMES = {
    "zero_shot": "Prompting (0.5B, instruction only)",
    "rag": "RAG few-shot (0.5B + 4 retrieved examples)",
    "finetuned": "LoRA fine-tuned (0.5B, instruction only)",
    "rag_1_5b": "RAG few-shot, 3x larger model (1.5B + 4 retrieved examples)",
}

rows = defaultdict(dict)
for p in sorted(Path("results/rows").glob("*_shard*.jsonl")):
    for line in p.open():
        r = json.loads(line)
        rows[r["method"]][r["id"]] = r

res = {m: score(list(rows[m].values())) for m in NAMES if m in rows}
Path("results/extraction_eval.json").write_text(json.dumps(res, indent=2))

lines = ["| Method | Pair F1 (exact) | Pair F1 (overlap) | Precision | Recall | ADE sentence F1 | Valid JSON | "
         "Prompt tokens | Median s/sentence |", "|---|---|---|---|---|---|---|---|---|"]
for m, s in res.items():
    lines.append(f"| {NAMES[m]} | {s['strict']['f1']:.3f} | {s['lenient']['f1']:.3f} | {s['strict']['precision']:.3f}"
                 f" | {s['strict']['recall']:.3f} | {s['sentence_detection']['f1']:.3f} | "
                 f"{s['valid_json_rate']:.1%} | {s['mean_prompt_tokens']:.0f} | {s['median_seconds']} |")
counts = ", ".join(f"{m} {s['n_sentences']}" for m, s in res.items())
lines.append(f"\nTest sentences per method: {counts}")
Path("results/extraction_eval.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
