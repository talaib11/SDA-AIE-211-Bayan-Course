# Model Card — Bayan Topic Classifier

## Intended use
Bilingual Arabic/English citizen-feedback topic classification.

## Artefact / data versions
- Model/checkpoint: artifacts/topic_classifier
- Preprocessing version: 1.2.0
- Data version/snapshot: Lab 6 validation snapshot

## Metrics
| Metric | Value |
| --- | --- |
| Validation accuracy | 0.8750 |
| 95% CI lower | 0.8617 |
| 95% CI upper | 0.8888 |

## Slice metrics
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

## Behavioural tests
| Family | Passed | Total | Rate |
| --- | --- | --- | --- |
| invariance | 8 | 8 | 100.0% |
| directional | 5 | 5 | 100.0% |
| mft | 8 | 8 | 100.0% |

## Known limitations
<!-- Lab 6: write this section by hand. Do not auto-generate it. -->
- Performance differs by slice and class.
- Semantic class overlap remains a known failure mode.
- Code-switching requires continued monitoring.
- Use only within the documented Bayan task scope.

## Contact / owner
Bayan NLP project team