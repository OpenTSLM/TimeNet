# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their annotations as Python objects. No I/O, no file paths: persistence is `TimeFWriter` concern.

---

## At a glance

```python
from dataclasses import dataclass
from timenet.tasks import Task

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
        *,
        task: Task,
        spec_id: str | None = None,
        label: str | None = None,
        variables: tuple[str, ...] | None = None,
        windows_s: tuple[tuple[float, float], ...] | None = None,
        question: str | None = None,
        answer: str | None = None,
        annotation_id: str | None = None,
        from_annotations: tuple[Annotation, ...] = (),
    ) -> Annotation: ...

    def annotate_qa(
        self,
        *,
        question: str,
        answer: str,
        from_annotations: list[Annotation] | None = None,
        from_samples: list[Sample] | None = None,
        task: Task = Task.QUESTION_AND_ANSWER,
        spec_id: str | None = None,
        annotation_id: str | None = None,
        include_tasks: tuple[Task, ...] | None = None,
    ) -> Annotation: ...


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    task: Task
    sample_ids: tuple[str, ...]
    spec_id: str | None = None
    label: str | None = None
    variables: tuple[str, ...] | None = None
    windows_s: tuple[tuple[float, float], ...] | None = None
    question: str | None = None
    answer: str | None = None
    from_annotations: tuple[Annotation, ...] = ()

    @property
    def from_annotation_ids(self) -> tuple[str, ...]: ...


# Composition helpers
def annotate_samples(
    samples: list[Sample],
    *,
    task: Task,
    spec_id: str | None = None,
    label: str | None = None,
    variables: tuple[str, ...] | None = None,
    windows_s: tuple[tuple[float, float], ...] | None = None,
    question: str | None = None,
    answer: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...

def qa_task(
    *,
    from_annotations: list[Annotation],
    question: str,
    answer: str,
    target_samples: list[Sample],
    spec_id: str | None = None,
    annotation_id: str | None = None,
    task: Task = Task.QUESTION_AND_ANSWER,
) -> Annotation: ...

def qa_pair(
    ann1: Annotation,
    ann2: Annotation,
    question: str,
    *,
    answer: str,
    target_samples: list[Sample],
    spec_id: str | None = None,
    annotation_id: str | None = None,
) -> Annotation: ...
```

---

## `TimeFDataset`

### `__init__()`

```python
def __init__(self, *, dataset_id: str, version: str) -> None: ...
```

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `dataset_id` | `str` | Must match `metadata().dataset_id` of the connector that constructs it. |
| `version` | `str` | Must match `metadata().version`. |

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

| Name | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Dataset-unique identifier for this sample. |
| `subject_ids` | `tuple[str, ...]` | Subjects this sample belongs to (participants, devices, instruments, locations). Single-element tuple for the single-subject case; multi-element when the sample spans multiple subjects. |
| `source_ids` | `tuple[str, ...]` | Source recordings this sample was derived from. Single-element tuple for the common case; multi-element when one sample is derived from multiple source recordings. |
| `signals` | `tuple[SignalRef, ...]` | One `SignalRef` per `SignalSpec` the sample uses, naming the subset of channels actually present.  Multi-element for multi-modal samples. Each `spec_id` must be declared in `DatasetMetadata.signal_specs`. |
| `view` | `str` | Name of a `ViewSpec` declared in `DatasetMetadata.view_specs`. Identifies which slice of the source this sample represents (e.g. `"full"`, `"single_channel"`, `"subset"`, `"window"`). |

**Returns:** The newly created `Sample`.

**Raises**

| Exception | Condition |
| --- | --- |
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

| Name | Type | Description |
| --- | --- | --- |
| `sample_id` | `str` | Dataset-unique identifier. |
| `subject_ids` | `tuple[str, ...]` | Subjects this sample belongs to. |
| `source_ids` | `tuple[str, ...]` | Source recordings this sample was derived from. |
| `signals` | `tuple[SignalRef, ...]` | One entry per `SignalSpec` used, paired with the channel subset present. |
| `view` | `str` | Name of a `ViewSpec` declared in the dataset's metadata. |
| `annotations` | `list[Annotation]` | Annotations attached to this sample. Append via `annotate()` or `annotate_qa()`. |

---

### `SignalRef`

Pairs a `SignalSpec.spec_id` with the subset of its channels present in a sample.

```python
@dataclass(frozen=True)
class SignalRef:
    spec_id: str
    channels: tuple[str, ...] = ()  # empty = all channels of the spec
```

| Field | Type | Description |
| --- | --- | --- |
| `spec_id` | `str` | Must match a `SignalSpec.spec_id` declared in `DatasetMetadata.signal_specs`. |
| `channels` | `tuple[str, ...]` | Subset of the referenced `SignalSpec.channels` present in this sample. Empty tuple is the convention for "all channels of the spec". |

---

### `annotate()`

```python
def annotate(
    self,
    *,
    task: Task,
    spec_id: str | None = None,
    label: str | None = None,
    variables: tuple[str, ...] | None = None,
    windows_s: tuple[tuple[float, float], ...] | None = None,
    question: str | None = None,
    answer: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

Attaches one annotation to this sample and returns it.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | `Task` | required | Annotation task type. |
| `spec_id` | `str \| None` | `None` | Optional reference to an `AnnotationSpec.spec_id` declared in the dataset metadata. `None` for free-form annotations that don't follow a declared spec. |
| `label` | `str \| None` | `None` | Discrete class label (e.g. `"normal_sinus_rhythm"`). Use for classification and labeling tasks. |
| `variables` | `tuple[str, ...] \| None` | `None` | Channels of the parent sample the annotation targets. Must be drawn from `self.signals[*].channels`. `None` = whole sample. |
| `windows_s` | `tuple[tuple[float, float], ...] \| None` | `None` | Time spans in seconds the annotation covers. `None` = full duration. |
| `question` | `str \| None` | `None` | Question text. Use for `Task.QUESTION_AND_ANSWER`. |
| `answer` | `str \| None` | `None` | Answer text. Use for `Task.QUESTION_AND_ANSWER`. |
| `annotation_id` | `str \| None` | `None` | Explicit ID. Auto-generated as `f"ann::{sample_id}::{task}"` if omitted. |
| `from_annotations` | `tuple[Annotation, ...]` | `()` | Source annotations this annotation was derived from. Used to build QA composition chains. |

**Returns:** The newly created `Annotation`, already appended to `self.annotations`.

**Examples**

Whole-sample classification (ECG rhythm):

```python
ecg_sample.annotate(task=Task.CLASSIFICATION, label="afib")
```

Windowed labeling on a specific channel (wearables activity bout):

```python
session.annotate(
    task=Task.LABELING,
    label="walking",
    variables=("accel_x", "accel_y", "accel_z"),
    windows_s=((120.0, 480.0),),
)
```

Captioning over a whole sample (finance market window):

```python
market_window.annotate(
    task=Task.CAPTIONING,
    answer="AAPL traded sideways with low volume; SPY drifted down 0.4%.",
)
```

QA with a composition chain:

```python
sample.annotate(
    task=Task.QUESTION_AND_ANSWER,
    question="Is this rhythm consistent with atrial fibrillation?",
    answer="No",
    from_annotations=tuple(prior_annotations),
)
```

---

### `annotate_qa()`

```python
def annotate_qa(
    self,
    *,
    question: str,
    answer: str,
    from_annotations: list[Annotation] | None = None,
    from_samples: list[Sample] | None = None,
    task: Task = Task.QUESTION_AND_ANSWER,
    spec_id: str | None = None,
    annotation_id: str | None = None,
    include_tasks: tuple[Task, ...] | None = None,
) -> Annotation: ...
```

Convenience wrapper around `annotate()` for QA annotations that compose existing annotations. Exactly one of `from_annotations` or `from_samples` must be provided.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `question` | `str` | required | Question text. |
| `answer` | `str` | required | Answer text. |
| `from_annotations` | `list[Annotation] \| None` | `None` | Explicit list of source annotations. Use when you already hold the annotation objects. |
| `from_samples` | `list[Sample] \| None` | `None` | Collect source annotations from these samples. Filter by `include_tasks` if provided. |
| `task` | `Task` | `Task.QUESTION_AND_ANSWER` | Task type for the created annotation. |
| `spec_id` | `str \| None` | `None` | Optional `AnnotationSpec.spec_id`. `None` for free-form QA. |
| `annotation_id` | `str \| None` | `None` | Explicit ID. Auto-generated if omitted. |
| `include_tasks` | `tuple[Task, ...] \| None` | `None` | When using `from_samples`, only collect annotations whose `task` is in this tuple. Ignored when using `from_annotations`. |

**Returns:** The newly created `Annotation`.

**Raises:** `ValueError` if both or neither of `from_annotations` / `from_samples` are provided, or if the resolved source list is empty.

**Example**

```python
# From explicit annotations
full.annotate_qa(
    question="Is there ST elevation across any lead?",
    answer="Yes",
    from_annotations=lead_annotations,
)

# From samples: collect all LABELING annotations
full.annotate_qa(
    question="Summarize findings across all leads.",
    answer="ST changes in V2, V3.",
    from_samples=lead_samples,
    include_tasks=(Task.LABELING,),
)
```

---

## `Annotation`

Frozen dataclass. Represents one label, QA pair, or detection event over one or more samples.

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `annotation_id` | `str` | Dataset-unique identifier. Auto-generated if not supplied to `annotate()`. |
| `task` | `Task` | Task type. |
| `sample_ids` | `tuple[str, ...]` | IDs of the samples this annotation is attached to. Length 1 for `sample.annotate()`; >1 for composition helpers. |
| `spec_id` | `str \| None` | Optional reference to an `AnnotationSpec.spec_id`. `None` for free-form annotations. |
| `label` | `str \| None` | Discrete class label. |
| `variables` | `tuple[str, ...] \| None` | Targeted variables. `None` = whole sample. |
| `windows_s` | `tuple[tuple[float, float], ...] \| None` | Time spans in seconds. `None` = full duration. |
| `question` | `str \| None` | Question text (QA tasks). |
| `answer` | `str \| None` | Answer text (QA tasks). |
| `from_annotations` | `tuple[Annotation, ...]` | Source annotations in a composition chain. Empty if this annotation was created independently. |

**Property**

| Name | Returns | Description |
| --- | --- | --- |
| `from_annotation_ids` | `tuple[str, ...]` | IDs of all annotations in `from_annotations`. Shorthand for serialization. |

---

## Composition helpers

Use these when one annotation needs to span multiple samples or be built from existing annotations across samples. For single-sample annotations, use `sample.annotate()` directly.

---

### `annotate_samples()`

```python
def annotate_samples(
    samples: list[Sample],
    *,
    task: Task,
    spec_id: str | None = None,
    label: str | None = None,
    variables: tuple[str, ...] | None = None,
    windows_s: tuple[tuple[float, float], ...] | None = None,
    question: str | None = None,
    answer: str | None = None,
    annotation_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

Attaches one shared `Annotation` to every sample in `samples`. The annotation's `sample_ids` contains all of their IDs.

**Parameters:** same semantics as `Sample.annotate()` except `samples` replaces `self`. `spec_id` optionally references an `AnnotationSpec`.

**Returns:** The single `Annotation` appended to every sample in the list.

**Raises:** `ValueError` if `samples` is empty.

---

### `qa_task()`

```python
def qa_task(
    *,
    from_annotations: list[Annotation],
    question: str,
    answer: str,
    target_samples: list[Sample],
    spec_id: str | None = None,
    annotation_id: str | None = None,
    task: Task = Task.QUESTION_AND_ANSWER,
) -> Annotation: ...
```

Builds a QA annotation from a list of source annotations and attaches it to `target_samples`.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `from_annotations` | `list[Annotation]` | required | Source annotations the QA is derived from. |
| `question` | `str` | required | Question text. |
| `answer` | `str` | required | Answer text. |
| `target_samples` | `list[Sample]` | required | Samples the resulting annotation is attached to. |
| `spec_id` | `str \| None` | `None` | Optional `AnnotationSpec.spec_id` reference. |
| `annotation_id` | `str \| None` | `None` | Explicit ID. Auto-generated if omitted. |
| `task` | `Task` | `Task.QUESTION_AND_ANSWER` | Task type for the created annotation. |

**Returns:** The created `Annotation`.

**Raises:** `ValueError` if `from_annotations` or `target_samples` is empty.

**Example**

```python
qa_task(
    from_annotations=lead_level_annotations,
    question="Is there progression across recordings?",
    answer="possible_progression",
    target_samples=full_samples,
    task=Task.REASONING,
)
```

---

### `qa_pair()`

```python
def qa_pair(
    ann1: Annotation,
    ann2: Annotation,
    question: str,
    *,
    answer: str,
    target_samples: list[Sample],
    spec_id: str | None = None,
    annotation_id: str | None = None,
) -> Annotation: ...
```

Convenience wrapper around `qa_task()` for exactly two source annotations.

**Parameters**

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `ann1` | `Annotation` | required | First source annotation. |
| `ann2` | `Annotation` | required | Second source annotation. |
| `question` | `str` | required | Question text. |
| `answer` | `str` | required | Answer text. |
| `target_samples` | `list[Sample]` | required | Samples the resulting annotation is attached to. |
| `spec_id` | `str \| None` | `None` | Optional `AnnotationSpec.spec_id` reference. |
| `annotation_id` | `str \| None` | `None` | Explicit ID. Auto-generated if omitted. |

**Returns:** The created `Annotation`.
