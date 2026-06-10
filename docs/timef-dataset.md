# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their annotations as Python objects. No I/O, no file paths: persistence is `TimeFWriter` concern.

---

## At a glance

```python
from dataclasses import dataclass

from timenet.tasks import (
    Task,
    ClassificationTask,
    LabelingTask,
    CaptioningTask,
    QATask,
    ForecastingTask,
    ReasoningTask,
)


class TimeFDataset:

    def __init__(self, *, dataset_id: str, version: str) -> None: ...

    def add_sample(
        self,
        *,
        sample_id: str,
        subject_ids: tuple[str, ...],
        source_ids: tuple[str, ...],
        signals: tuple[SignalRef, ...],
        view: str,
    ) -> Sample: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...


@dataclass(frozen=True)
class SignalRef:
    spec_id: str
    channels: tuple[str, ...] = ()  # empty = all channels of the spec


@dataclass
class Sample:
    sample_id: str
    subject_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    signals: tuple[SignalRef, ...]
    view: str
    annotations: list[Annotation]

    def annotate(
        self,
        task: Task,
        *,
        spec_id: str | None = None,
        annotation_id: str | None = None,
        from_annotations: tuple[Annotation, ...] = (),
    ) -> Annotation: ...


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    task: Task
    sample_ids: tuple[str, ...]
    spec_id: str | None = None
    from_annotations: tuple[Annotation, ...] = ()

    @property
    def from_annotation_ids(self) -> tuple[str, ...]: ...


# Composition helper
def annotate_samples(
    samples: list[Sample],
    task: Task,
    *,
    spec_id: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

---

## `TimeFDataset`

### `__init__()`

```python
def __init__(self, *, dataset_id: str, version: str) -> None: ...
```

**Parameters**

| Name         | Type  | Description                                                             |
| ------------ | ----- | ----------------------------------------------------------------------- |
| `dataset_id` | `str` | Must match `metadata().dataset_id` of the connector that constructs it. |
| `version`    | `str` | Must match `metadata().version`.                                        |

---

### `add_sample()`

```python
def add_sample(
    self,
    *,
    sample_id: str,
    subject_ids: tuple[str, ...],
    source_ids: tuple[str, ...],
    signals: tuple[SignalRef, ...],
    view: str,
) -> Sample: ...
```

Creates a `Sample`, registers it, and returns it.

**Parameters**

| Name          | Type                    | Description                                                                                                                                                                                                 |
| ------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_id`   | `str`                   | Dataset-unique identifier for this sample.                                                                                                                                                                  |
| `subject_ids` | `tuple[str, ...]`       | Subjects this sample belongs to (participants, devices, instruments, locations). Single-element tuple for the single-subject case; multi-element when the sample spans multiple subjects.                   |
| `source_ids`  | `tuple[str, ...]`       | Source recordings this sample was derived from. Single-element tuple for the common case; multi-element when one sample is derived from multiple source recordings.                                         |
| `signals`     | `tuple[SignalRef, ...]` | One `SignalRef` per `SignalSpec` the sample uses, naming the subset of channels actually present. Multi-element for multi-modal samples. Each `spec_id` must be declared in `DatasetMetadata.signal_specs`. |
| `view`        | `str`                   | Name of a `ViewSpec` declared in `DatasetMetadata.view_specs`. Identifies which slice of the source this sample represents (e.g. `"full"`, `"single_channel"`, `"subset"`, `"window"`).                     |

**Returns:** The newly created `Sample`.

**Raises**

| Exception    | Condition                                                                                       |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `ValueError` | `sample_id` is already registered, or any of `subject_ids` / `source_ids` / `signals` is empty. |

**Examples**

Single-subject ECG recording:

```python
dataset = TimeFDataset(dataset_id="ecg_dataset", version="1.0.0")

full = dataset.add_sample(
    sample_id="rec_001::full",
    subject_ids=("patient_42",),
    source_ids=("rec_001",),
    signals=(
        SignalRef(spec_id="ecg_12lead", channels=("I", "II", "V1", "V2")),
    ),
    view="full",
)
```

Multi-sensor wearables session:

```python
session = dataset.add_sample(
    sample_id="participant_07::session_3",
    subject_ids=("participant_07",),
    source_ids=("session_3",),
    signals=(
        SignalRef(spec_id="ppg"),         # empty channels = all channels of the spec
        SignalRef(spec_id="accel"),       # all 3 axes
        SignalRef(spec_id="skin_temp"),   # the spec's single channel
    ),
    view="full",
)
```

Multi-source sample (one continuous overnight recording stored in two files):

```python
night = dataset.add_sample(
    sample_id="night_001::full",
    subject_ids=("participant_07",),
    source_ids=("night_001a.edf", "night_001b.edf"),
    signals=(
        SignalRef(spec_id="eeg", channels=("f3", "c4")),
        SignalRef(spec_id="eog", channels=("l",)),
        SignalRef(spec_id="emg", channels=("emg",)),
    ),
    view="full",
)
```

---

### `samples`

```python
@property
def samples(self) -> tuple[Sample, ...]: ...
```

**Returns:** All samples in insertion order. Read-only, use `add_sample()` to extend.

---

## `Sample`

One logical unit of time-series data: a recording, a session, a sensor bundle, a market window. Samples are created exclusively by `TimeFDataset.add_sample()`.

**Fields**

| Name          | Type                    | Description                                                              |
| ------------- | ----------------------- | ------------------------------------------------------------------------ |
| `sample_id`   | `str`                   | Dataset-unique identifier.                                               |
| `subject_ids` | `tuple[str, ...]`       | Subjects this sample belongs to.                                         |
| `source_ids`  | `tuple[str, ...]`       | Source recordings this sample was derived from.                          |
| `signals`     | `tuple[SignalRef, ...]` | One entry per `SignalSpec` used, paired with the channel subset present. |
| `view`        | `str`                   | Name of a `ViewSpec` declared in the dataset's metadata.                 |
| `annotations` | `list[Annotation]`      | Annotations attached to this sample. Append via `annotate()`.            |

---

### `SignalRef`

Pairs a `SignalSpec.spec_id` with the subset of its channels present in a sample.

```python
@dataclass(frozen=True)
class SignalRef:
    spec_id: str
    channels: tuple[str, ...] = ()  # empty = all channels of the spec
```

| Field      | Type              | Description                                                                                                                          |
| ---------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `spec_id`  | `str`             | Must match a `SignalSpec.spec_id` declared in `DatasetMetadata.signal_specs`.                                                        |
| `channels` | `tuple[str, ...]` | Subset of the referenced `SignalSpec.channels` present in this sample. Empty tuple is the convention for "all channels of the spec". |

---

### `annotate()`

```python
def annotate(
    self,
    task: Task,
    *,
    spec_id: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

Attaches one annotation to this sample and returns it. The `task` argument is an instance of one of the [`Task`](enums-and-spec.md#tasks) subclasses (`ClassificationTask`, `LabelingTask`, `CaptioningTask`, `QATask`, `ForecastingTask`, `ReasoningTask`). Its type determines what data the annotation carries.

**Parameters**

| Name               | Type                     | Default  | Description                                                                                                                                                                                                                       |
| ------------------ | ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task`             | `Task`                   | required | Instance of a `Task` subclass. Carries all task-specific data (label, question/answer, windows, etc.).                                                                                                                            |
| `spec_id`          | `str \| None`            | `None`   | Optional reference to an `AnnotationSpec.spec_id` declared in the dataset metadata. `None` for free-form annotations that don't follow a declared spec.                                                                           |
| `annotation_id`    | `str \| None`            | `None`   | Explicit ID. Auto-generated as `f"ann::{sample_id}::{task.task_id}"` if omitted. For multi-sample annotations created via `annotate_samples()`, the `sample_id` segment is replaced with a hash of the sorted `sample_ids` tuple. |
| `from_annotations` | `tuple[Annotation, ...]` | `()`     | Source annotations this annotation was derived from. Used to build composition chains (e.g. a `ReasoningTask` built from prior `LabelingTask` annotations).                                                                       |

**Returns:** The newly created `Annotation`, already appended to `self.annotations`.

**Examples**

Whole-sample classification (ECG rhythm):

```python
ecg_sample.annotate(ClassificationTask(label="afib"))
```

Windowed labeling on specific channels (wearables activity bout):

```python
session.annotate(LabelingTask(
    label="walking",
    channels=("accel_x", "accel_y", "accel_z"),
    windows_s=((120.0, 480.0),),
))
```

Captioning over a whole sample (finance market window):

```python
market_window.annotate(CaptioningTask(
    answer="AAPL traded sideways with low volume; SPY drifted down 0.4%.",
))
```

QA composed from prior annotations:

```python
sample.annotate(
    QATask(
        question="Is this rhythm consistent with atrial fibrillation?",
        answer="No",
    ),
    from_annotations=tuple(prior_annotations),
)
```

Reasoning composed from a chain of lower-level annotations:

```python
sample.annotate(
    ReasoningTask(
        question="Is there progression across recordings?",
        answer="Yes — ST changes worsen across the three follow-ups.",
    ),
    from_annotations=tuple(lead_level_annotations),
)
```

---

## `Annotation`

Frozen dataclass. Represents one labeled annotation over one or more samples. The annotation's payload lives on `task`; its subtype determines what fields are present.

**Fields**

| Name               | Type                     | Description                                                                                                                    |
| ------------------ | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| `annotation_id`    | `str`                    | Dataset-unique identifier. Auto-generated if not supplied to `annotate()`.                                                     |
| `task`             | `Task`                   | Instance of a `Task` subclass carrying all task-specific data.                                                                 |
| `sample_ids`       | `tuple[str, ...]`        | IDs of the samples this annotation is attached to. Length 1 for `sample.annotate()`; >1 when created via `annotate_samples()`. |
| `spec_id`          | `str \| None`            | Optional reference to an `AnnotationSpec.spec_id`. `None` for free-form annotations.                                           |
| `from_annotations` | `tuple[Annotation, ...]` | Source annotations in a composition chain. Empty if this annotation was created independently.                                 |

**Property**

| Name                  | Returns           | Description                                                                |
| --------------------- | ----------------- | -------------------------------------------------------------------------- |
| `from_annotation_ids` | `tuple[str, ...]` | IDs of all annotations in `from_annotations`. Shorthand for serialization. |

**Reading the payload**

Use `isinstance` (or `match`) on `annotation.task` to recover the task type and its data:

```python
match annotation.task:
    case ClassificationTask(label=label):
        ...
    case LabelingTask(label=label, channels=channels, windows_s=windows):
        ...
    case QATask(question=question, answer=answer):
        ...
```

---

## Composition helper

Use this when one annotation needs to span multiple samples. For single-sample annotations, use `sample.annotate()` directly. For QA or reasoning chains built from prior annotations, pass them through `from_annotations` on either entry point.

### `annotate_samples()`

```python
def annotate_samples(
    samples: list[Sample],
    task: Task,
    *,
    spec_id: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

Attaches one shared `Annotation` to every sample in `samples`. The annotation's `sample_ids` contains all of their IDs.

**Parameters:** same semantics as `Sample.annotate()` except `samples` replaces `self`.

**Returns:** The single `Annotation` appended to every sample in the list.

**Raises:** `ValueError` if `samples` is empty.

**Example**

A reasoning annotation built from per-lead labelings, attached to the corresponding full recordings:

```python
annotate_samples(
    full_samples,
    ReasoningTask(
        question="Is there progression across recordings?",
        answer="possible_progression",
    ),
    from_annotations=tuple(lead_level_annotations),
)
```
