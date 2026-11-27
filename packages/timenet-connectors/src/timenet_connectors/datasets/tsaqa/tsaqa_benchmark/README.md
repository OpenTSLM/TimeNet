# TSAQA with raw input series

`TSAQAConnector` builds `tsaqa/tsaqa-benchmark`, version `1.0.0`, from the
[TSAQA release](https://huggingface.co/datasets/TSAQA/TSAQA-Benchmark) at commit
`883dcb0f96e0d1c8fd3806f35548ff1f2058a9cd`. It preserves the original train, validation and
test files, with 147,000, 21,000 and 42,000 questions respectively.

This is a **raw-input variant**. It is not a reproduction of the
normalized-input evaluation in [the paper](https://arxiv.org/abs/2601.23204).

## Values, questions and provenance

Each series in `raw_ts` becomes one Record with one materialized float64 Signal. Values, precision,
lengths and source order are preserved. No series is rescaled, padded, resampled or deduplicated.
The ordinal axis means sample position, not elapsed time. Units and timestamps are unspecified:
the context sometimes describes them, but there is no verified per-series axis/unit mapping.

| Source task | Task inputs | Additional raw Records |
| --- | --- | --- |
| Anomaly detection, classification, characterization | The single raw series | None |
| Comparison | Both raw series, in source order | None |
| Data transformation | The first raw series | Preserved as provenance only |
| Temporal relationship | All raw series, labelled to match the question | None |

Temporal tasks include the given raw series and every raw candidate as numerical inputs.
Signal-scoped `series_label` annotations identify "Given series" and "Time series A" through
"Time series D". True-or-false questions use "Candidate series" for the single proposed continuation;
T and F remain answer labels. These signal annotations are exposed through `input_annotations`.
The prompt replaces each normalized candidate array with its series label. The original question
is retained in `original_question` metadata, and `question_representation` is `raw_candidate_references`.
The published target letter or patch order is unchanged and never included in input annotations.

Candidate order was audited across all 60,000 temporal samples at the pinned revision: each of
195,051 raw/text candidate pairs has the same length and a curve consistent with positive affine
rescaling, within 0.0002 in question-text units. This includes 5,901 constant pairs, whose identities
cannot be distinguished by shape. Preserve source order rather than matching or deduplicating
candidates. The connector checks candidate counts and lengths when constructing the labelled view.

Other tasks retain the published prompt unchanged. For transformation tasks, candidates cannot
generally be reconstructed directly from the additional raw arrays, so these remain provenance
records. This mapping is unchanged. The raw-input variant does not reproduce normalized-input
benchmark scores; the original question and target are preserved for traceability.

Each task has one input record containing its input signals. Extra raw series used only for
provenance stay in separate records. Record metadata identifies `file`, zero-based `row`, and
`is_task_input`; signal metadata identifies `raw_series_index`. These fields preserve the source
identity of every series without exposing provenance-only series as model inputs.
`input_ts` is not read or stored.

The connector parses JSON lists in `meta_info` and uses `Signal.annotate()` to attach each
known description to its signal. Tasks reference the annotations on their input signals.
Comparison rows associate two descriptions with two signals; ordering patches share the single
source description. Temporal tasks also expose their signal-scoped series labels.

Multiple-choice candidate rows publish four descriptions for five raw series without an explicit
association. Those rows retain the original context as an annotation on the task's input record
and carry `ambiguous_series_context` in `source_issues`; the connector does not guess signal
associations or use the answer to assign descriptions.

Task metadata retains `meta_info`, `domain`, `dataset`, `task`, and `question_type` for provenance.
The first three retain the exact source **string cells**, including JSON-encoded lists where
published. Parsed descriptions preserve their text after JSON decoding. The other metadata fields are not supplied as input annotations.
Questions, metadata, and numerical series are parsed once and retained in memory.

## Tasks and source limitations

Each source row becomes one task. True-or-false questions use `ClassificationTask` with labels
`T` and `F`. Multiple-choice and ordering questions use `AnswerTask`, with the published letter
or comma-separated patch order as the target. The actual question determines the task type.

Task metadata preserves the source `task` and `question_type`, plus the actual `question_format`,
`file`, `row`, and `series_representation: raw_ts`. `source_issues` flags
`question_type_mismatch` for a documented exception in the pinned release: temporal relationship
rows labelled `ordering` that contain a `Choices:` block with T/F or consecutive letter choices.
These rows use the choice format while retaining the published label. All other rows use the
`question_type` column directly; conflicting choices or answers fail validation.

Source duplicates, conflicting answers, identical patches, and train/test overlaps remain in
place. This connector does not audit or filter those issues; an empty `source_issues` list means
only that the question-type label agrees. Do not interpret it as a clean-sample certificate.
The previous connector's `SourceIssue` and `without_source_issues` API is removed.

## Ownership and storage

### From a source row to a TimeNet task

This example uses test row 18005, a continuation question with one given series and four
candidate series. The source answer is `C`. Each concrete TSAQA task owns the validation and
conversion for its task family; the connector chooses that class from the row's `task` field.

```python
from datasets import Dataset

from timenet.dataset import TimeFDataset
from timenet.types import Split
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark import (
    TSAQAConnector,
    TSAQARegistry,
)
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.source import TSAQARow
from timenet_connectors.datasets.tsaqa.tsaqa_benchmark.tasks import (
    TemporalRelationshipTask,
)

source = Dataset.from_parquet(
    "test.parquet", columns=list(TSAQARow.model_fields)
)
row = TSAQARow.model_validate(source[18005])
sample = TemporalRelationshipTask(
    row, 18005, split=Split.TEST, filename="test.parquet"
)
task = sample.to_timenet_task()

assert len(task.inputs) == 1
assert len(task.inputs[0].signals) == 5
assert task.targets[0].read() == "C"
print(task.prompt.read())  # A: Time series A, ..., D: Time series D

# Register this example in an in-memory dataset.
registry = TSAQARegistry()
registry.store(sample)
dataset = TimeFDataset(metadata=TSAQAConnector().metadata())
registry.register_in_timenet_dataset(dataset)
```

The full build uses `TSAQAConnector.read_tasks(files)` to apply this flow to every row and
`TSAQARegistry.register_in_timenet_dataset(dataset)` to register their records and already-built TimeNet tasks.
Use the build command below to publish the complete release, rather than this one-row example.

### Responsibilities

- `source.py` contains source rows and typed metadata. `TSAQARow` validates
  the published fields and question format.
- `tasks.py` contains `TSAQATask` and its six concrete task classes:
  `AnomalyDetectionTask`, `ClassificationTask`, `CharacterizationTask`, `ComparisonTask`,
  `DataTransformationTask`, and `TemporalRelationshipTask` each own their sample structure and
  supported question formats. Each constructor validates its row and builds its concrete TimeNet
  task immediately. `to_timenet_task()` returns that stored task without rebuilding it.
  `TSAQATask` shares record construction; it does not duplicate the question, answer, or split
  already held by the TimeNet task. No TimeNet task types are added.
- `TSAQAMetadata` validates published context and provenance. It serializes to TimeNet's existing
  JSON metadata contract; `TSAQAMetadata.model_validate(task.metadata)` restores the typed view
  after reading a stored task. Published source context remains in metadata for provenance;
  decoded signal descriptions and temporal series labels are also selected as input annotations.
- `QuestionFormat` owns the permitted choices. Validation uses named operations rather than
  reconstructing the format's rules inside compound conditions.
- The local `TSAQARawSeries` model validates TSAQA’s raw numerical arrays and owns
  their Arrow conversion. Text fields reuse TimeF's `NonEmptyString` contract.
- Hugging Face `Dataset.from_parquet(..., keep_in_memory=False)` loads the pinned source files
  through its Arrow cache. Only declared source columns are loaded, excluding `input_ts`.
- Each task's constructor materializes all its raw series as float64 Arrow arrays. Conversion
  constructs TimeNet signals directly from those arrays, without lazy loaders or source references.
- `TSAQARegistry` stores these materialized tasks in source order, preserving duplicates. Text
  and numerical values remain in memory until the build completes.
- `TSAQAConnector.read_tasks(files)` reads each row, selects its concrete task class explicitly,
  constructs the validated task, and stores it in the registry. The registry registers all raw
  records and the stored TimeNet tasks. `BaseHuggingFaceConnector` owns pinned downloads:

  ```python
  registry = self.read_tasks(files)
  dataset = TimeFDataset(metadata=self.metadata())
  registry.register_in_timenet_dataset(dataset)
  ```

  The existing engine and TimeF writer own persistence. After writing, normal TimeF readers own
  lazy access; no connector code is needed to consume the stored dataset.

Missing splits or columns, unknown task names, invalid answers, and malformed/nonfinite raw values
fail explicitly. No normalization fallback to `input_ts` is permitted. Once parsing completes,
TimeNet conversion and writing no longer read the source dataset or its Arrow cache.

```bash
uv run timenet-build build tsaqa/tsaqa-benchmark \
  --local-registry /tmp/timenet-tsaqa-registry
```

```python
from timenet import TimeNet

with TimeNet(sources=["/tmp/timenet-tsaqa-registry"]).open(
    "tsaqa/tsaqa-benchmark"
) as reader:
    dataset = reader.read()
    task = next(dataset.iter_tasks())
    raw_values = task.inputs[0].signals[0].to_numpy()
    question = task.prompt.read()
```
