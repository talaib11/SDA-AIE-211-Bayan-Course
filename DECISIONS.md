# Decision Records

## tokenizer

- Chosen checkpoint(s): xlm-roberta-base (XLM-R)

- Arabic fertility evidence: XLM-R achieved 1.672, with AR p95 length of 19 and AR UNK rate of 0.0000.

- English fertility evidence: XLM-R achieved 1.434, with EN p95 length of 21.

- p95 length evidence: XLM-R had AR p95 = 19 and EN p95 = 21, giving a balanced sequence-length profile across both languages.

- Operational trade-off / rationale: XLM-R provides the best overall bilingual balance, with zero observed Arabic UNK tokens and strong fertility and p95 results for both Arabic and English.

## arabic-model

- Incumbent: xlm-roberta-base (Day-2 XLM-R)

- Candidate: CAMeL-Lab/bert-base-arabic-camelbert-mix and CAMeL-Lab/bert-base-arabic-camelbert-da

- All/Gulf/MSA evidence: On the deterministic Arabic bake-off evaluation split (1,200 examples: 960 Gulf and 240 MSA), Day-2 XLM-R, CAMeLBERT-mix, and CAMeLBERT-DA all achieved 1.0000 macro-F1 on All, Gulf, and MSA slices.

- CI-backed verdict: Retain xlm-roberta-base as the incumbent. The best Arabic-centric candidate did not improve Gulf macro-F1 over Day-2 XLM-R (delta = +0.0000), so the Lab 4 target of at least +0.04 Gulf macro-F1 was NOT MET. CAMeLBERT-mix and CAMeLBERT-DA tied on the measured slices, so there is no measured quality justification for replacing the incumbent.

- Segmentation contract: Arabic clitic segmentation remains a separate preprocessing/NER experiment and is not used to claim an Arabic model quality improvement in this bake-off.

## search-min-score

- Threshold:

- No-answer evidence:

- False-positive / false-negative trade-off:

## quantisation-split

- Topic artefact:

- NER artefact:

- Latency evidence:

- Paired quality-tax evidence:

- Rollback artefact retained:

## architecture

- Encoder/decoder rationale by task:

- Multilingual vs Arabic-centric rationale:

- Evidence used: