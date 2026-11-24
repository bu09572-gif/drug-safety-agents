"""Small local chat model on Hugging Face transformers, with an optional LoRA adapter."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


class LocalChatModel:
    def __init__(self, model_name: str, adapter: str | None = None, max_new_tokens: int = 128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name + (f" + LoRA({adapter})" if adapter else "")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()  # same speed as the base model
        self.model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.max_new_tokens = max_new_tokens
        torch.set_num_threads(max(torch.get_num_threads(), 1))

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> Generation:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
        new = out[0, ids["input_ids"].shape[1]:]
        return Generation(
            text=self.tok.decode(new, skip_special_tokens=True),
            prompt_tokens=int(ids["input_ids"].shape[1]),
            output_tokens=int(new.shape[0]),
            seconds=time.perf_counter() - t0,
        )
