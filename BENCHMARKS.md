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

### Arabic model bake-off

| Model | All macro-F1 | Gulf macro-F1 | MSA macro-F1 |
|---|---:|---:|---:|
| Day-2 XLM-R | 1.0000 | 1.0000 | 1.0000 |
| CAMeLBERT-mix | 1.0000 | 1.0000 | 1.0000 |
| CAMeLBERT-DA | 1.0000 | 1.0000 | 1.0000 |

- Bake-off evaluation split: 1,200 held-out Arabic examples (960 Gulf, 240 MSA), created deterministically from the supplied Arabic training data with joint dialect/topic stratification because the supplied validation split contains no Gulf examples.
- Winner by Gulf slice: CAMeLBERT-mix (tie with CAMeLBERT-DA).
- Best Gulf macro-F1 delta vs Day-2 XLM-R: +0.0000.
- Lab 4 Gulf target (+0.04 macro-F1 over Day-2): NOT MET.
- Decision: retain Day-2 XLM-R because the Arabic-centric candidates showed no measured Gulf improvement.
## Lab 5 — Semantic search

| Metric | Measured result | Target |
|---|---:|---:|
| Recall@10 without reranking | 0.0103 | ≥ 0.80 |
| MRR@10 without reranking | 0.0056 | ≥ 0.70 |
| Recall@10 with reranking | 0.0026 | ≥ 0.80 |
| MRR@10 with reranking | 0.0015 | ≥ 0.70 |
| Arabic reranked Recall@10 | 0.0056 | — |
| Arabic reranked MRR@10 | 0.0033 | — |
| English reranked Recall@10 | 0.0000 | — |
| English reranked MRR@10 | 0.0000 | — |
| Cross-lingual MRR gap | 0.0033 | — |
| No-answer correctness | 20 / 20 | ≥ 17 / 20 |
| Bi-encoder mean latency | 18.78 ms/query | — |
| Two-stage mean latency | 58.15 ms/query | — |

- Evaluated queries: 150 total (130 answerable, 20 no-answer).
- Tuned `min_score`: `-2.092221`.
- No-answer target passed: 20 / 20.
- Retrieval quality targets were not met with the evaluated multilingual bi-encoder / cross-encoder configuration.
- Diagnosis: L2 normalisation and FAISS indexing were verified. All 390 judged relevant case IDs were present in the 20k corpus. A diagnostic topic-filtered retrieval run still produced Recall@10 = 0.0051 and MRR@10 = 0.0038. The corpus contains highly similar and duplicate synthetic cases that are not included in each query's judged `relevant_case_ids`, so plausible-looking nearest neighbours do not necessarily match the labelled relevance set.
- Planted-bug lesson: retrieval quality must be approved using the labelled query set and measured metrics rather than by visually inspecting plausible results.

## Lab 6 — Evaluation / model cards
| Check | Result |
|---|---:|
| Validation examples | 2,400 |
| Overall accuracy | 0.8750 |
| 95% bootstrap CI | [0.8617, 0.8888] |
| Validation errors | 300 |
| Arabic accuracy | 0.7500 |
| English accuracy | 1.0000 |
| MSA accuracy | 0.7500 |
| Parks accuracy | 0.0000 |
| Invariance behavioural tests | 8 / 8 = 100% |
| Directional behavioural tests | 5 / 5 = 100% |
| MFT behavioural tests | 8 / 8 = 100% |
| Manually reviewed errors | 120 |
| Class-overlap / semantic ambiguity | 116 / 120 = 96.7% |
| Code-switching | 4 / 120 = 3.3% |
| Model cards generated | 3 |

- Invariance target (>=95%): MET
- MFT target (>=90%): MET
- Main observed weakness: parks class accuracy = 0.0000.
- Primary reviewed error category: class-overlap / semantic ambiguity.
## Lab 7 — Part 1: CPU Inference Benchmark

### Benchmark configuration

- Device: CPU
- OMP_NUM_THREADS: 4
- Production length mix: `data/serving/bench_mix.npy`
- Production mix examples: 2000
- Warm-up iterations are excluded from latency measurements.
- Latency is reported using p50 and p99.
- Dynamic padding is evaluated with `max_length=128`.

### Measured results

| Configuration | p50 (ms) | p99 (ms) |
|---|---:|---:|
| PyTorch FP32, dynamic padding, max_length=128 | 198.36 | 359.99 |

For the fixed `max_length=512` configuration, CPU inference was prohibitively slow in the available runtime. Three repeated single-inference measurements were:

- Run 1: 12815.34 ms
- Run 2: 12870.50 ms
- Run 3: 12811.06 ms
- Median: 12815.34 ms

The dynamic-padding configuration therefore provides a substantial latency reduction compared with fixed 512-token padding in the measured CPU environment.
## Capstone