# Bayan Evaluation Report

## Manager headline

Bayan achieved 87.5% validation accuracy (95% bootstrap CI 86.2%–88.9%), but the aggregate hides a material weakness in the parks class. The reviewed error sample shows that semantic class overlap is the primary remediation priority before relying on the aggregate score alone.

## Aggregate result

- Validation examples: 2400
- Accuracy: 0.8750
- 95% bootstrap CI: [0.8617, 0.8888]
- Errors: 300

## Sliced evaluation

| Slice | Value | N | Accuracy | Small |
| --- | --- | --- | --- | --- |
| language | ar | 1200 | 0.7500 | False |
| language | en | 1200 | 1.0000 | False |
| dialect | MSA | 1200 | 0.7500 | False |
| dialect | NA | 1200 | 1.0000 | False |
| class | billing | 300 | 1.0000 | False |
| class | digital_services | 300 | 1.0000 | False |
| class | licensing | 300 | 1.0000 | False |
| class | lighting | 300 | 1.0000 | False |
| class | parks | 300 | 0.0000 | False |
| class | roads | 300 | 1.0000 | False |
| class | waste | 300 | 1.0000 | False |
| class | water | 300 | 1.0000 | False |
| length | short | 754 | 0.8435 | False |
| length | medium | 1646 | 0.8894 | False |

Small slices are explicitly flagged and should not be interpreted
with the same confidence as large slices.

## Behavioural tests

| Family | Passed | Total | Rate |
| --- | --- | --- | --- |
| invariance | 8 | 8 | 100.0% |
| directional | 5 | 5 | 100.0% |
| mft | 8 | 8 | 100.0% |

Course targets:
- Invariance: approximately >= 95%
- MFT: approximately >= 90%

## Manual error review

A manually reviewed sample of 120 validation errors was
classified using the Lab 6 error taxonomy.

| Category | Count | Share |
| --- | --- | --- |
| class-overlap/semantic ambiguity | 116 | 96.7% |
| code-switching | 4 | 3.3% |

## Top three prioritised fixes

| Priority | Fix | Predicted delta |
| --- | --- | --- |
| 1 | Add targeted training examples and behavioural regression tests for parks-vs-roads semantic overlap. | +0.04 to +0.08 overall accuracy if the dominant confusion is substantially reduced. |
| 2 | Add more bilingual/code-switched training examples and preserve mixed-language tokens in preprocessing. | +0.005 to +0.015 overall accuracy; larger expected benefit on the code-switching slice. |
| 3 | Add class-contrast examples and confidence-based error review for semantically neighbouring topics. | +0.01 to +0.03 overall accuracy, pending validation. |

The predicted deltas are hypotheses for the next validation run, not
measured improvements.

## Known limitations

- Validation performance is not uniform across classes; aggregate
  accuracy should not be used without slice results.
- The manual review identified substantial semantic overlap between
  neighbouring topic classes.
- Code-switched messages remain a smaller but visible failure mode.
- Predicted fix deltas have not yet been confirmed by retraining.
- Lab 5 semantic-search quality targets were not met in the measured
  retrieval run and remain a separate system limitation.

## Evidence

- Bootstrap confidence intervals: `src/bayan/evaluation/bootstrap.py`
- Sliced reporting: `src/bayan/evaluation/slices.py`
- Behavioural suite: `src/bayan/evaluation/behavioural.py`
- Validation predictions: `data/eval/validation_predictions.csv`
- Human-reviewed error sample: `data/eval/Lab6_120_errors_with_suggestions.csv`
