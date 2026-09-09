## Lab 6 — Manual Error Review

A sample of 120 validation errors was manually reviewed and tagged using
the taxonomy above. Where the existing categories were too broad, the
review used the more specific label `class-overlap / semantic ambiguity`
to capture errors caused by overlapping topic semantics.

### Error-category histogram

| Error category | Count | Share |
|---|---:|---:|
| Class-overlap / semantic ambiguity | 116 | 96.7% |
| Dialect or code-switching | 4 | 3.3% |
| **Total** | **120** | **100.0%** |

The dominant failure mode was class-overlap / semantic ambiguity.
Code-switching represented a much smaller portion of the reviewed errors.

### Top 3 prioritised fixes

1. **Target parks-vs-roads semantic overlap**
   - Add targeted contrastive training examples and behavioural regression tests.
   - Predicted metric delta: **+0.04 to +0.08 overall accuracy** if the dominant confusion is substantially reduced.

2. **Improve bilingual and code-switched coverage**
   - Add more Arabic-English code-switched examples and preserve mixed-language tokens during preprocessing.
   - Predicted metric delta: **+0.005 to +0.015 overall accuracy**, with a larger expected benefit on the code-switching slice.

3. **Add class-contrast examples for neighbouring topics**
   - Add difficult examples that explicitly distinguish semantically similar topic classes and review low-confidence predictions.
   - Predicted metric delta: **+0.01 to +0.03 overall accuracy**, pending validation.

The predicted metric deltas are hypotheses for the next evaluation cycle
and have not yet been confirmed by retraining.