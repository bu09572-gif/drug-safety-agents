"""LoRA fine-tuning of a small chat model for adverse event extraction.

Runs on CPU (slow but free, e.g. a GitHub Actions runner) or on a GPU. With --qlora on a CUDA
GPU the base model is loaded in 4-bit (QLoRA), which lets a 1.5B-7B model fit on a free
Colab or Kaggle T4.

The loss is computed only on the answer tokens, not on the instruction or the sentence.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from .data import Example
from .extract import answer_json, build_messages

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, ex: Example, max_len: int = 256) -> dict:
    """Token ids and labels for one example; labels are -100 on the prompt part."""
    msgs = build_messages(ex.text)
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full = prompt + answer_json(ex.pairs) + (tok.eos_token or "")
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int):
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        labels[i, :k] = torch.tensor(b["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train(
    train_examples: list[Example],
    val_examples: list[Example],
    model_name: str,
    out_dir: Path,
    epochs: float = 1.0,
    batch_size: int = 8,
    micro_batch: int = 4,
    lr: float = 2e-4,
    rank: int = 16,
    alpha: int = 32,
    max_len: int = 256,
    time_budget_min: float = 1e9,
    qlora: bool = False,
    seed: int = 0,
) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    random.seed(seed)
    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(model_name)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if qlora:
        if not cuda:
            raise SystemExit("--qlora needs a CUDA GPU (bitsandbytes 4-bit)")
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if cuda else torch.float32)
        model = model.to("cuda" if cuda else "cpu")
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    data = [encode(tok, e, max_len) for e in train_examples]
    val = [encode(tok, e, max_len) for e in val_examples]
    steps_per_epoch = math.ceil(len(data) / batch_size)
    total_steps = max(1, int(steps_per_epoch * epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * s / total_steps))))
    accum = max(1, batch_size // micro_batch)
    device = next(model.parameters()).device

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), micro_batch):
                b = {k: v.to(device) for k, v in collate(val[i:i + micro_batch], pad_id).items()}
                tot += model(**b).loss.item()
                n += 1
        model.train()
        return tot / max(n, 1)

    log, t0, step = [], time.time(), 0
    first_val = val_loss() if val else None
    model.train()
    order: list[int] = []
    stopped_early = False
    while step < total_steps:
        if not order:
            order = list(range(len(data)))
            random.shuffle(order)
        loss_sum = 0.0
        for _ in range(accum):
            idx, order = order[:micro_batch], order[micro_batch:]
            if not idx:
                break
            b = {k: v.to(device) for k, v in collate([data[i] for i in idx], pad_id).items()}
            loss = model(**b).loss / accum
            loss.backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        elapsed = time.time() - t0
        if step % 10 == 0 or step == total_steps:
            log.append({"step": step, "loss": round(loss_sum, 4), "minutes": round(elapsed / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
        if elapsed / 60 > time_budget_min:
            stopped_early = True
            print(f"time budget reached at step {step}/{total_steps}", flush=True)
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    info = {
        "base_model": model_name, "method": "QLoRA (4-bit)" if qlora else "LoRA",
        "device": "cuda" if cuda else "cpu", "rank": rank, "alpha": alpha, "lr": lr,
        "batch_size": batch_size, "train_examples": len(data), "steps_done": step, "steps_planned": total_steps,
        "stopped_early": stopped_early, "trainable_params": trainable, "total_params": total,
        "trainable_share": round(trainable / total, 5), "train_minutes": round((time.time() - t0) / 60, 1),
        "val_loss_before": round(first_val, 4) if first_val is not None else None,
        "val_loss_after": round(val_loss(), 4) if val else None, "loss_log": log,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    return info
