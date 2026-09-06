# Decision Records

## tokenizer

- Chosen checkpoint(s): xlm-roberta-base (XLM-R)

- Arabic fertility evidence: XLM-R achieved 1.672, with AR p95 length of 19 and AR UNK rate of 0.0000.

- English fertility evidence: XLM-R achieved 1.434, with EN p95 length of 21.

- p95 length evidence: XLM-R had AR p95 = 19 and EN p95 = 21, giving a balanced sequence-length profile across both languages.

- Operational trade-off / rationale: XLM-R provides the best overall bilingual balance, with zero observed Arabic UNK tokens and strong fertility and p95 results for both Arabic and English.

## arabic-model

- Incumbent:

- Candidate:

- All/Gulf/MSA evidence:

- CI-backed verdict:

- Segmentation contract:

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