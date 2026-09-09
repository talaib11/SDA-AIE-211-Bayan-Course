Bayan Project — Steps Completed
Project Setup
Connected the project to the provided GitHub repository.
Cloned the repository and worked on the same shared codebase throughout the labs.
Set up the Python environment and installed the required dependencies.
Used Git to track changes and push completed work to the repository.
Used Google Colab when GPU resources were required for model training.

The course specifically requires keeping the same repository throughout the labs because each lab builds on previous components and evidence.

Lab 1 — Bilingual Preprocessing and Tokenization
Inspected the raw Arabic and English data and identified common text defects.
Implemented text normalization and cleaning.
Implemented PII masking for sensitive information.
Implemented sentence segmentation.
Compared four tokenizer candidates using fertility and sequence-length metrics.
Selected XLM-R as the tokenizer/model checkpoint based on the measured bilingual results.
Recorded the benchmark results and tokenizer decision.
Lab 2 — Transformer Attention
Implemented scaled dot-product attention from scratch.
Implemented Multi-Head Attention.
Compared our implementation with the PyTorch reference.
Audited model parameter distributions.
Implemented and verified causal masking.
Inspected attention maps and diagnosed padding-mask leakage.
Lab 3A — Topic Classification
Built a TF-IDF + LinearSVC baseline classifier.
Created grouped train, validation, and test splits to prevent citizen-level data leakage.
Tokenized the dataset using the selected XLM-R checkpoint.
Fine-tuned the topic classification model.
Evaluated the model using macro-F1.
Compared the Transformer classifier against the baseline.
Saved the trained classifier artifact.
Lab 3B — NER and Extractive QA
Implemented BIO label alignment between words and tokenizer subwords.
Fine-tuned the Named Entity Recognition model.
Evaluated NER using entity-level metrics.
Implemented extractive QA span selection.
Added honest no-answer handling.
Evaluated the QA pipeline using the provided smoke-test questions.
Lab 4 — Arabic Pipeline and Dialect Awareness
Implemented Arabic-specific normalization profiles.
Audited the dialect distribution of the Arabic dataset.
Integrated Arabic clitic segmentation into the NER pipeline.
Evaluated the effect of segmentation on entity recognition.
Compared XLM-R with Arabic-centric CAMeLBERT candidates.
Evaluated models across All, Gulf, and MSA slices.
Retained XLM-R based on the measured slice-level evidence.
Lab 5 — Bilingual Semantic Search
Encoded historical cases into dense vector representations.
Applied L2 normalization to the embeddings.
Built and persisted a FAISS search index with its metadata and manifest.
Implemented bi-encoder candidate retrieval.
Added cross-encoder re-ranking.
Evaluated retrieval using Recall@10 and MRR@10.
Evaluated cross-lingual retrieval behaviour.
Added a minimum-score threshold for honest no-result responses.
Lab 6 — Evaluation and Model Reporting
Implemented bootstrap confidence intervals.
Implemented paired model comparison.
Evaluated performance across language, dialect, class, and length slices.
Implemented behavioural tests for invariance and minimum functionality.
Reviewed and categorized model errors using an error taxonomy.
Generated the final evaluation report.
Produced model cards containing performance evidence and known limitations.
Lab 7 — Part 1: CPU Inference Benchmark
Configured the benchmark to use a fixed CPU thread count.
Implemented the inference benchmark harness.
Loaded the production sequence-length mix from bench_mix.npy.
Added warm-up iterations before latency measurement.
Measured inference latency using p50 and p99.
Evaluated the FP32 padded baseline with max_length=512.
Evaluated the free optimization using dynamic padding and max_length=128.
Recorded the measured benchmark results in BENCHMARKS.md.

These are specifically the requirements of Lab 7 Part 1; ONNX, INT8, FastAPI, canaries, and HTTP load testing belong to the later Lab 7 steps.

Overall workflow:
Repository Setup → Preprocessing → Transformer Attention → Classification/NER/QA → Arabic Optimization → Semantic Search → Evaluation → Inference Optimization

That follows the intended Bayan progression from raw bilingual text toward a complete NLP service.
