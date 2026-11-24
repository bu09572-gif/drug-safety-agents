# Drug Safety Review with Multi-Agent LLMs and a LoRA Fine-Tuned Extractor

I fine-tune a small open model (Qwen2.5-0.5B) with LoRA to pull adverse drug events out of real PubMed case reports, compare it fairly against prompting and RAG, and then use it inside a planner / worker / reviewer workflow built with LangGraph that checks every finding against the drug's FDA label.

Results are filled in after the first full training and evaluation run.

## Data

- **ADE Corpus V2** (Gurulingappa et al., 2012): sentences from PubMed case reports, labeled by experts with the drug and the adverse effect it caused. Loaded from Hugging Face (`ade-benchmark-corpus/ade_corpus_v2`).
- **openFDA drug labels** (api.fda.gov): boxed warning, warnings, precautions and adverse reactions sections.

Train / validation / test splits come from a hash of each sentence, so no sentence is in two splits.

## How to run

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
python scripts/train.py --time-budget-min 270           # LoRA on CPU
python scripts/run_eval.py --method finetuned           # or zero_shot, rag, rag_1_5b
python scripts/merge_eval.py
python scripts/run_agents.py
```

On a free Colab or Kaggle GPU: `pip install -e ".[gpu]"` and `python scripts/train.py --qlora --model Qwen/Qwen2.5-1.5B-Instruct`.

The full run happens in GitHub Actions (`.github/workflows/train-eval.yml`), on free CPU runners.
