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
| Causal mask | Lower-triangular mask verified |
| Causal output shape | `[1, 1, 4, 8]` |
| Pad-attention leakage | Output changed after padding mask (`True`) |

## Lab 3A — Classification

## Lab 3B — NER + QA

## Lab 4 — Arabic normalization / dialect

## Lab 5 — Semantic search

## Lab 6 — Evaluation / model cards

## Lab 7 — ONNX / INT8 + FastAPI

## Capstone