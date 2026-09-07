# Lab Notes

## Lab 1 — Defect Safari

Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.

For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1

- Class: Leading/trailing whitespace
- Example: "  ألعاب الأطفال في حديقة حي العليا تحتاج صيانة   "
- Why it matters: Extra whitespace can create inconsistent text representations and affect tokenization and matching.
- Decision: Clean leading/trailing whitespace, while preserving meaningful internal spaces.

### Defect 2

- Class: HTML remnants
- Example: "الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة <br>"
- Why it matters: HTML tags are not part of the citizen's message and can become unwanted tokens.
- Decision: Remove HTML remnants such as <br> during preprocessing.

### Defect 3

- Class: Phone-number PII
- Example: "تطبيق خدمات المياه يتوقف عند تسجيل الدخول 0551234567"
- Why it matters: Phone numbers are personally identifiable information and should not be exposed to downstream NLP models.
- Decision: Mask phone numbers as <PHONE>.

### Defect 4

- Class: National-ID PII
- Example: "إنارة الممر لا تعمل عند طريق الملك فهد 1023456789"
- Why it matters: National IDs are sensitive personally identifiable information and must be protected.
- Decision: Mask national-ID numbers as <NATIONAL_ID>.

### Defect 5

- Class: Character elongation / repeated characters
- Example: "لووووسمحت ألطريق المؤدي إلى حي الياسمين يحتاج صيانة عاجلة"
- Why it matters: Repeated characters are common in informal Arabic and can create unnecessary token variation.
- Decision: Normalize excessive character repetition when it is clearly non-semantic, while preserving meaningful characters.

### Defect 6

- Class: Duplicated words
- Example: "My My Licence licence request has been under review for 15 days"
- Why it matters: Accidental repeated words increase noise and can negatively affect tokenization and downstream NLP performance.
- Decision: Remove obvious accidental consecutive word duplication when it does not change the intended meaning.

## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes |
|---|---:|---:|---|
| mBERT | 177,853,440 | 51.84% | Larger embedding parameter count; attention and FFN counts are the same as CAMeLBERT. |
| CAMeLBERT | 109,081,344 | 21.49% | Smaller embedding parameter count; attention and FFN counts are the same as mBERT. |
### Lab 2 — Attention diagnostics

- mBERT has a larger embedding share than CAMeLBERT because its multilingual vocabulary requires a larger embedding table (multilingual vocabulary tax).
- The causal attention matrix was verified to be lower-triangular.
- Future-attention mass: 0.0.
- This demonstrates decoder-style causal attention: each position can attend only to itself and previous positions.
- Average [SEP] attention in the diagnostic example: 0.25995659828186035.
- Pad-attention mass without a padding mask: 0.3397676348686218 (~33.98%).
- Pad-attention mass with a padding mask: 0.0 (0%).
- The padding mask successfully eliminated attention leakage to [PAD] tokens.
## Lab 4 — Dialect audit

- Distribution:
- One-sentence implication for MSA-only evaluation: