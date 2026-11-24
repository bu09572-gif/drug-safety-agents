"""Run one extraction method on one shard of the test set.

Methods (same instruction and JSON format for all):
  zero_shot      base 0.5B model, instruction only
  rag            base 0.5B model + the 4 most similar labeled training sentences as worked examples
  finetuned      0.5B model + LoRA adapter, instruction only
  rag_1_5b       base 1.5B model (3x larger) + the same 4 retrieved examples

    python scripts/run_eval.py --method finetuned --adapter adapters/qwen0.5b-ade --shard 0/4
"""

import argparse
import json
from pathlib import Path

from drugsafety.extract import build_messages, parse_answer
from drugsafety.llm import LocalChatModel
from drugsafety.retrieve import ExampleRetriever
from drugsafety.splits import BASE_MODEL, BIG_MODEL, splits

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["zero_shot", "rag", "finetuned", "rag_1_5b"])
ap.add_argument("--adapter", default="adapters/qwen0.5b-ade")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

i, n = map(int, a.shard.split("/"))
sp = splits()
test = sp["test"][: a.limit] if a.limit else sp["test"]
test = test[i::n]
model_name = BIG_MODEL if a.method == "rag_1_5b" else BASE_MODEL
llm = LocalChatModel(model_name, adapter=a.adapter if a.method == "finetuned" else None)
retriever = ExampleRetriever(sp["train_pool"]) if a.method.startswith("rag") else None

out = Path(f"results/rows/{a.method}_shard{i}.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    done = {json.loads(line)["id"] for line in out.open()}
with out.open("a") as f:
    for k, ex in enumerate(test):
        if ex.id in done:
            continue
        shots = retriever.top_k(ex.text) if retriever else None
        gen = llm.chat(build_messages(ex.text, shots))
        pred, valid = parse_answer(gen.text)
        row = {"id": ex.id, "method": a.method, "model": llm.name, "text": ex.text, "gold": ex.pairs,
               "pred": pred, "valid_json": valid, "raw": gen.text, "seconds": round(gen.seconds, 3),
               "prompt_tokens": gen.prompt_tokens, "output_tokens": gen.output_tokens}
        f.write(json.dumps(row) + "\n")
        f.flush()
        if k % 25 == 0:
            print(k, len(test), round(gen.seconds, 2), gen.text[:100], flush=True)
