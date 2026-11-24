"""The fixed train / validation / test sets used by every script."""

from __future__ import annotations

import json
from pathlib import Path

from .data import Example, load_from_hub, load_jsonl, sample_balanced, save_jsonl

CACHE = Path("data/cache/examples.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BIG_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_POS, TRAIN_NEG = 2000, 1000
VAL_POS, VAL_NEG = 100, 100
TEST_POS, TEST_NEG = 300, 300


def all_examples() -> list[Example]:
    if CACHE.exists():
        return load_jsonl(CACHE)
    ex = load_from_hub()
    save_jsonl(ex, CACHE)
    return ex


def splits() -> dict[str, list[Example]]:
    ex = all_examples()
    by = {s: [e for e in ex if e.split == s] for s in ("train", "val", "test")}
    return {
        "train_pool": by["train"],
        "train": sample_balanced(by["train"], TRAIN_POS, TRAIN_NEG, seed=1),
        "val": sample_balanced(by["val"], VAL_POS, VAL_NEG, seed=2),
        "test": sample_balanced(by["test"], TEST_POS, TEST_NEG, seed=3),
        "test_all": by["test"],
    }


def stats(sp: dict[str, list[Example]]) -> dict:
    def c(xs):
        return {"sentences": len(xs), "with_ade": sum(e.has_ade for e in xs),
                "pairs": sum(len(e.pairs) for e in xs)}

    return {k: c(v) for k, v in sp.items()}


if __name__ == "__main__":
    sp = splits()
    s = stats(sp)
    Path("results").mkdir(exist_ok=True)
    Path("results/data_stats.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
