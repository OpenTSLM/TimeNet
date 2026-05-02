# TimeFDataset

The **TimeFDataset** is the in-memory model a connector populates during `convert_to_timef`. The engine constructs it from the connector's `DatasetDescriptor`, hands it to the connector, and on return passes the populated model to the writer for serialization.

The dataset owns no I/O, no file paths, no shard layout. It holds samples, signals, and annotations as Python objects. Persistence is the writer's concern (see `TimeFWriter`TODO).

---

## At a glance

```python
import pandas as pd

from timenet.timef.builder import TimeFDataset, Sample, Annotation
from timenet.timef.schemas import get_valid_labels
from timenet.tasks import Task
from timenet.domains import Domain


class TimeFDataset:

    def __init__(self, descriptor: DatasetDescriptor) -> None: ...

    def add_sample(
        self,
        signals: dict[str, pd.DataFrame],
    ) -> Sample: ...

    def _add_annotation(
        self,
        samples: list[Sample],
        task: Task,
        domains: list[Domain],
        signals: list[str] | None,
        schema_: str | None,
        label: str | None,
        answer: str | None,
        question: str | None,
        rationale: str | None,
        windows: list[tuple[float, float]] | None,
    ) -> Annotation: ...


class Sample:

    signals: dict[str, pd.DataFrame]
    _dataset: TimeFDataset                # back-reference set by add_sample

    def annotate(
        self,
        task: Task,
        domains: list[Domain],
        signals: list[str] | None = None,
        schema_: str | None = None,
        label: str | None = None,
        answer: str | None = None,
        question: str | None = None,
        rationale: str | None = None,
        windows: list[tuple[float, float]] | None = None,
    ) -> Annotation: ...


class Annotation:

    id: str                               # auto-generated, f"ann_{counter}"
    samples: list[Sample]
    task: Task
    domains: list[Domain]
    signals: list[str] | None
    schema_: str | None
    label: str | None
    answer: str | None
    question: str | None
    rationale: str | None
    windows: list[tuple[float, float]] | None
```

---

## Conceptual model

### Dataset, samples, signals, annotations

A `TimeFDataset` holds:

- A `DatasetDescriptor`, declared at construction. Immutable for the dataset's life.
- A list of `Sample` objects, each created by `add_sample`.
- A list of `Annotation` objects, each created by `Sample.annotate`.

Each `Sample` holds:

- `signals: dict[str, pd.DataFrame]` , one DataFrame per signal. Each frame carries its own `time_s` column at the signal's native sampling rate.

Each `Annotation` holds:

- An auto-generated `id` (`f"ann_{counter}"`, dataset-wide running index).
- A reference to one or more samples.
- A `task` and `domains`.
- Optional `signals` (which signals on the targeted samples the annotation pertains to) and `windows` (time spans on those signals).
- An optional `schema_` plus exactly one of `label` or `answer`, and optional `question` / `rationale`.

## The contract

### `TimeFDataset(descriptor)` { data-toc-label='init()' }

Constructed once per connector run by the engine. Connector authors do not instantiate it directly.

| Argument     | Type                | Purpose                                                    |
| ------------ | ------------------- | ---------------------------------------------------------- |
| `descriptor` | `DatasetDescriptor` | Dataset identity (id, version, signals, domains, license). |

```python
def __init__(self, descriptor: DatasetDescriptor) -> None:
    self._descriptor = descriptor
    self._samples: list[Sample] = []
    self._annotations: list[Annotation] = []
```

`_descriptor` is the source of truth for which signal ids may appear in `add_sample`. `_samples` preserves insertion order. `_annotations` is a flat list,every `Sample.annotate` call appends one entry.

### `dataset.add_sample(signals) -> Sample` { data-toc-label='add_sample()' }

Adds one sample to the dataset. Returns the newly created `Sample`.

| Argument  | Type                      | Purpose                                                                                                                                                                   |
| --------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signals` | `dict[str, pd.DataFrame]` | Per-signal frames. Each key must match a `SignalDescriptor.signal_id` declared on the descriptor. Each frame must include a `time_s` column plus the signal's `channels`. |

```python
def add_sample(self, signals: dict[str, pd.DataFrame]) -> Sample:
    declared = {s.signal_id: s for s in self._descriptor.signals}
    unknown = set(signals) - declared.keys()
    if unknown:
        raise ValueError(f"unknown signal ids: {unknown}")

    for sid, frame in signals.items():
        if "time_s" not in frame.columns:
            raise ValueError(f"signal {sid!r} missing 'time_s' column")
        expected = set(declared[sid].channels)
        actual = set(frame.columns) - {"time_s"}
        if expected != actual:
            raise ValueError(
                f"signal {sid!r} channel mismatch: "
                f"expected {expected}, got {actual}"
            )

    sample = Sample(signals=signals, _dataset=self)
    self._samples.append(sample)
    return sample
```

### `sample.annotate(...) -> Annotation` { data-toc-label='annotate()' }

Adds one annotation to the sample. Returns the newly created `Annotation`.

| Argument    | Type                                | Default | Purpose                                                                                                           |
| ----------- | ----------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------- |
| `task`      | `Task`                              | —       | Annotation task type.                                                                                             |
| `domains`   | `list[Domain]`                      | —       | Domains the annotation belongs to.                                                                                |
| `signals`   | `list[str] \| None`                 | `None`  | Specific signals on the sample the annotation targets. `None` = whole-sample annotation.                          |
| `schema_`   | `str \| None`                       | `None`  | Label schema name (e.g. `"aasm"`). `None` for free-form annotations.                                              |
| `label`     | `str \| None`                       | `None`  | Discrete label. Validated against `schema_` if both are provided. XOR with `answer`.                              |
| `answer`    | `str \| None`                       | `None`  | Free-form answer. XOR with `label`.                                                                               |
| `question`  | `str \| None`                       | `None`  | Question text (QA tasks).                                                                                         |
| `rationale` | `str \| None`                       | `None`  | Optional rationale.                                                                                               |
| `windows`   | `list[tuple[float, float]] \| None` | `None`  | Time spans on the targeted signals. Validated against signal durations. `None` = annotation covers the full span. |

```python
def annotate(
    self,
    task: Task,
    domains: list[Domain],
    signals: list[str] | None = None,
    schema_: str | None = None,
    label: str | None = None,
    answer: str | None = None,
    question: str | None = None,
    rationale: str | None = None,
    windows: list[tuple[float, float]] | None = None,
) -> Annotation:
    return self._dataset._add_annotation(
        samples=[self],
        task=task,
        domains=domains,
        signals=signals,
        schema_=schema_,
        label=label,
        answer=answer,
        question=question,
        rationale=rationale,
        windows=windows,
    )
```

### `_add_annotation(...) -> Annotation` { data-toc-label='\_add_annotation()' }

Private helper used by `Sample.annotate`. Validates the label/answer XOR, label-schema conformance, target signal presence, and window range against signal duration; constructs the `Annotation` and appends it to `_annotations`.

```python
def _add_annotation(
    self,
    samples: list[Sample],
    task: Task,
    domains: list[Domain],
    signals: list[str] | None,
    schema_: str | None,
    label: str | None,
    answer: str | None,
    question: str | None,
    rationale: str | None,
    windows: list[tuple[float, float]] | None,
) -> Annotation:
    if (label is None) == (answer is None):
        raise ValueError("exactly one of 'label' or 'answer' must be provided")

    if label is not None and schema_ is not None:
        valid = get_valid_labels(schema_)
        if valid and label not in valid:
            raise ValueError(f"label {label!r} not valid for schema {schema_!r}")

    for sample in samples:
        present = set(sample.signals)
        target = set(signals) if signals is not None else present
        unknown = target - present
        if unknown:
            raise ValueError(f"signals {unknown} not present on sample")

        if windows is not None:
            for sid in target:
                duration = sample.signals[sid]["time_s"].iloc[-1]
                for w_start, w_end in windows:
                    if w_start < 0 or w_end > duration:
                        raise ValueError(
                            f"window ({w_start}, {w_end}) out of range for "
                            f"signal {sid!r} (duration {duration})"
                        )

    annotation = Annotation(
        id=f"ann_{len(self._annotations)}",
        samples=samples,
        task=task,
        domains=domains,
        signals=signals,
        schema_=schema_,
        label=label,
        answer=answer,
        question=question,
        rationale=rationale,
        windows=windows,
    )
    self._annotations.append(annotation)
    return annotation
```

## Worked example

```python
import wfdb
from timenet.timef.builder import TimeFDataset
from timenet.domains import Domain
from timenet.tasks import Task


def convert_to_timef(self, raw_ref, dataset: TimeFDataset, ctx) -> None:
    for record_id in iter_record_ids(raw_ref.local_path):
        record = wfdb.rdrecord(str(raw_ref.local_path / record_id))
        ann    = wfdb.rdann(str(raw_ref.local_path / record_id), "atr")

        frame = build_frame(record)   # columns: time_s, MLII, V1
        sample = dataset.add_sample(signals={"ecg": frame})

        fs = record.fs
        for t_sample, symbol in zip(ann.sample[1:-1], ann.symbol[1:-1], strict=True):
            t_s = t_sample / fs
            sample.annotate(
                task=Task.QUESTION_AND_ANSWER,
                domains=[Domain.CARDIOLOGY],
                signals=["ecg"],
                question="What type of beat occurs in this window?",
                answer=symbol,
                windows=[(t_s - 0.21, t_s + 0.21)],
            )
```

The resulting dataset has 48 samples (one per record), each with one signal at native rate, and ~110 000 annotations carrying the time window of each beat.
