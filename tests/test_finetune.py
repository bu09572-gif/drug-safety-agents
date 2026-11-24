"""Train a LoRA adapter on a tiny random model, so the training and loading code is tested
without downloading anything."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")


@pytest.fixture
def tiny_model(tmp_path, examples):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from drugsafety.extract import INSTRUCTION, answer_json

    words = set((INSTRUCTION + " Sentence: system user assistant").split())
    for e in examples:
        words |= set(e.text.split()) | set(answer_json(e.pairs).split())
    special = ["<pad>", "<|im_start|>", "<|im_end|>", "<unk>"]
    vocab = {w: i for i, w in enumerate(special + sorted(words))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="<|im_end|>", unk_token="<unk>")
    tok.chat_template = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
                         "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}")
    d = tmp_path / "tiny"
    tok.save_pretrained(d)
    cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      pad_token_id=0, eos_token_id=vocab["<|im_end|>"])
    Qwen2ForCausalLM(cfg).save_pretrained(d)
    return str(d), tok


def test_encode_masks_prompt(tiny_model, examples):
    from drugsafety.finetune import encode

    _, tok = tiny_model
    ex = next(e for e in examples if e.pairs)
    enc = encode(tok, ex)
    n_answer = sum(1 for x in enc["labels"] if x != -100)
    assert 0 < n_answer < len(enc["input_ids"])
    assert enc["labels"][-1] == tok.eos_token_id


def test_train_and_load_adapter(tiny_model, examples, tmp_path):
    from drugsafety.finetune import train
    from drugsafety.llm import LocalChatModel

    path, _ = tiny_model
    out = tmp_path / "adapter"
    info = train(examples, examples[:2], path, out, epochs=3, batch_size=2, micro_batch=2, lr=1e-2)
    assert info["steps_done"] == info["steps_planned"] > 0
    assert 0 < info["trainable_share"] < 1
    assert json.loads((out / "train_info.json").read_text())["method"] == "LoRA"
    llm = LocalChatModel(path, adapter=str(out), max_new_tokens=5)
    gen = llm.chat([{"role": "user", "content": "Sentence: x"}])
    assert gen.output_tokens <= 5
