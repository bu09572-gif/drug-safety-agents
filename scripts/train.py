"""Fine-tune the base model with LoRA on the training sentences.

    python scripts/train.py --out adapters/qwen0.5b-ade --time-budget-min 270
    python scripts/train.py --qlora --model Qwen/Qwen2.5-1.5B-Instruct   # on a CUDA GPU (Colab/Kaggle)
"""

import argparse
import json
from pathlib import Path

from drugsafety.finetune import train
from drugsafety.splits import BASE_MODEL, splits

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=BASE_MODEL)
ap.add_argument("--out", default="adapters/qwen0.5b-ade")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--time-budget-min", type=float, default=1e9)
ap.add_argument("--limit", type=int, default=0, help="use only the first N training examples (smoke test)")
ap.add_argument("--qlora", action="store_true")
a = ap.parse_args()

sp = splits()
tr = sp["train"][: a.limit] if a.limit else sp["train"]
va = sp["val"][: max(8, a.limit // 4)] if a.limit else sp["val"]
info = train(tr, va, a.model, Path(a.out), epochs=a.epochs, rank=a.rank, lr=a.lr,
             time_budget_min=a.time_budget_min, qlora=a.qlora)
print(json.dumps({k: v for k, v in info.items() if k != "loss_log"}, indent=2))
