| Method | Pair F1 (exact) | Pair F1 (overlap) | Precision | Recall | ADE sentence F1 | Valid JSON | Prompt tokens | Median s/sentence |
|---|---|---|---|---|---|---|---|---|
| Prompting (0.5B, instruction only) | 0.293 | 0.463 | 0.269 | 0.322 | 0.725 | 77.7% | 121 | 3.12 |
| RAG few-shot (0.5B + 4 retrieved examples) | 0.362 | 0.495 | 0.310 | 0.434 | 0.721 | 76.5% | 298 | 3.572 |
| LoRA fine-tuned (0.5B, instruction only) | 0.667 | 0.759 | 0.643 | 0.693 | 0.901 | 99.5% | 121 | 1.753 |
| RAG few-shot, 3x larger model (1.5B + 4 retrieved examples) | 0.467 | 0.579 | 0.384 | 0.594 | 0.780 | 80.2% | 298 | 13.356 |

Test sentences per method: zero_shot 600, rag 600, finetuned 600, rag_1_5b 600
