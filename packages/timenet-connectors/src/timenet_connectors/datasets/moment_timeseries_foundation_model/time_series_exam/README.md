# TimeSeriesExam

Imports the full **763-question round-three exam**, pinned to upstream commit
`384cf50864860c65e962b441eaa4c201857a06f8`. Earlier rounds are developmental
revisions, not train/validation splits. All tasks are `Split.TEST`.

Each `AnswerTask` retains the original question and option order, the upstream
format instruction and worked response example, and one or two signals in source
order. Optional question hints, concept definitions, and example signals are off,
as in the upstream CLI defaults. These remain source metadata, not model inputs.
Channel names match the upstream plots: `Time Series`, or `Time Series 1/2`.
Values are float64, without rounding or normalization; axes are ordinal because
the JSON provides no sampling rate. Original question ID, template ID, category,
subcategory, difficulty, options, and reference remain in task metadata.

TimeBench scores generated text. The released scorer uses case-insensitive
substring matching of `LETTER) option text`, anywhere in the output. Numeric
references retain their original numeric type for option lookup before the
letter and content are formatted as text. This connector preserves the same
behavior without changing source labels or observations. Copying all options can satisfy the upstream
substring rule; interpret accuracy accordingly.

The prompt wording is adapted from upstream MIT-licensed code; its license is
included in `LICENSE.upstream`.

## Distinct upstream releases

This integration uses round 3 from the requested GitHub repository (763 questions).
The authors also publish a later
[Hugging Face v1.1 release](https://huggingface.co/datasets/AutonLab/TimeSeriesExam1)
with 746 test cases, revised templates, and longer signals. Its identities and
measurements differ; it is not interchangeable with this frozen GitHub exam.
Scores must always name the release.
