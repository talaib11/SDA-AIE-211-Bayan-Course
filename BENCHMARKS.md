# BENCHMARKS

> Fill these tables from **your own runs**. Do not copy course reference numbers.

## Lab 1 — Tokenizer audit
| Tokenizer | AR fertility | EN fertility | AR p95 len | EN p95 len | AR UNK rate |
|---|---:|---:|---:|---:|---:|
| mBERT | 2.153 | 1.510 | 25.0 | 23.0 | 0.0045 |
| XLM-R | 1.672 | 1.434 | 19.0 | 21.0 | 0.0000 |
| CAMeLBERT | 1.405 | 2.705 | 18.0 | 36.0 | 0.0080 |
| DistilBERT | 4.527 | 1.298 | 45.0 | 19.0 | 0.0022 |

- Golden preprocessing: 25 / 25 passed
- PII masking recall: 60 / 60 = 100%

## Lab 2 — Attention

| Check | Result |
|---|---|
| Scaled dot-product attention vs PyTorch | Passed (`atol=1e-6`) |
| Attention tests | 2 / 2 passed |
| Multi-head input shape | `[2, 5, 64]` |
| Multi-head output shape | `[2, 5, 64]` |
| Multi-head configuration | 4 heads, 16 dimensions per head |
| Multi-head parameters | 16,640 |
| Causal attention | Lower-triangular verified (`True`) |
| Future-attention mass | 0.0 |
| Attention family | Decoder-style causal attention |
| Average `[SEP]` attention | 0.25995659828186035 |
| `[PAD]` mass without mask | 0.3397676348686218 (~33.98%) |
| `[PAD]` mass with mask | 0.0 (0%) |
| Pad-attention leakage fix | Verified (`True`) |
## Lab 3A — Classification
| Model | Validation macro-F1 | Frozen test macro-F1 |
|---|---:|---:|
| TF-IDF + LinearSVC | 1.0000 | 1.0000 |
| XLM-R topic classifier | 1.0000 | 1.0000 |

- Grouped split integrity test: 1 / 1 passed
- Fine-tuning checkpoint: `xlm-roberta-base`
- Training epochs: 3
- Learning rate: `2e-5`
- Seed: 42
- Classifier artifact: `artifacts/topic_classifier`
## Lab 3B — NER + QA
| QA smoke-set file | 12 answerable / 0 unanswerable |
| README QA target | 9 answerable / 3 unanswerable |
| QA smoke-set status | Dataset/README mismatch detected |

- `best_span()` supports honest null/no-answer handling.
- The supplied `qa_smoke_set.json` does not contain the 3 unanswerable cases stated in the Lab 3B README.
## Lab 4 — Arabic normalization / dialect

| Check | Result |
|---|---|
| Arabic normalization golden pairs | 30 / 30 passed |
| Arabic slice size | 7,200 |
| Gulf | 4,800 (66.7%) |
| MSA | 2,400 (33.3%) |
| Original NER LOCATION recall | 1.0000 |
| Segmented NER LOCATION recall | 1.0000 |
| LOCATION recall delta | +0.0000 |

- The original Day-2 NER already achieved 100% LOCATION recall on this split, so segmentation could not improve recall further.

## Lab 5 — Semantic search

## Lab 6 — Evaluation / model cards

## Lab 7 — ONNX / INT8 + FastAPI

## Capstone