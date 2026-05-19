# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their annotations as Python objects. No I/O, no file paths: persistence is `TimeFWriter` concern.

---

## `TimeFDataset`

```python
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from timenet.tasks import (
    Task,
    ClassificationTask,
    LabelingTask,
    CaptioningTask,
    QATask,
    ForecastingTask,
    ReasoningTask,
)
from timenet.views import View


class TimeFDataset:

    def __init__(self, *, dataset_id: str, version: str) -> None: ...

    def add_sample(
        self,
        *,
        time_series: tuple[TimeSeries, ...],
        view: View,
        subject_ids: tuple[str, ...] = (),
    ) -> Sample: ...

    def add_annotation(
        self,
        samples: Sample | tuple[Sample, ...],
        task: Task,
        *,
        spec_id: str | None = None,
        from_annotations: tuple[Annotation, ...] = (),
    ) -> Annotation: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...

    @property
    def annotations(self) -> tuple[Annotation, ...]: ...
```

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
    time_series: tuple[TimeSeries, ...],
    view: View,
    subject_ids: tuple[str, ...] = (),
) -> Sample: ...
```

Creates a `Sample` with an auto-generated `sample_id`, registers it, and returns it.

**Parameters**

| Name          | Type                     | Required | Description                                                                                                                                                                                                           |
| ------------- | ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time_series` | `tuple[TimeSeries, ...]` | yes      | One `TimeSeries` per channel the sample uses.                                                                                                                                                                         |
| `view`        | `View`                   | yes      | A [`View`](types.md#view) enum member identifying which slice of the source this sample represents (e.g. `View.FULL`, `View.SINGLE_CHANNEL`, `View.SUBSET`, `View.WINDOW`).                                           |
| `subject_ids` | `tuple[str, ...]`        | no       | Subjects this sample belongs to (participants, devices, instruments). Empty by default, leave unset for subject-less domains (finance, seismology, synthetic). Multi-element when the sample spans multiple subjects. |

**Returns:** The newly created `Sample`.

**Raises**

| Exception    | Condition               |
| ------------ | ----------------------- |
| `ValueError` | `time_series` is empty. |

**Examples**

Single-subject ECG recording:

```python
from timenet.timef.dataset import TimeSeries
from timenet.timef.metadata import TimeSeriesSpec
from timenet.units import Frequency, SamplingRateUnit, TimestampUnit, ValueUnit

dataset = TimeFDataset(dataset_id="ecg_dataset", version="1.0.0")

# One modality, subclassed once. Reused by every ECG example below.
class ECGLeadSpec(TimeSeriesSpec):
    spec_id = "ecg_lead"
    name = "ECG Lead"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MILLIVOLT

def ecg_reader(path: Path, channel: str) -> np.ndarray:
    # parse the EDF and return that channel as a float32 1-D array
    ...

full = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=tuple(
        TimeSeries(
            spec=ECGLeadSpec(channel=ch),
            source_id="rec_001",
            sampling_rate=Frequency.Hz(500.0),
            reader=lambda p=path, c=ch: ecg_reader(p, c),
        )
        for ch in ("I", "II", "V1", "V2")
    ),
    view=View.FULL,
)
# full.source_ids == ("rec_001",)
```

Multi-sensor wearables session:

```python
class PPGSpec(TimeSeriesSpec):
    spec_id = "ppg"
    name = "PPG"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.DIMENSIONLESS

class AccelSpec(TimeSeriesSpec):
    spec_id = "accel"
    name = "Acceleration"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.G

class SkinTempSpec(TimeSeriesSpec):
    spec_id = "skin_temp"
    name = "Skin Temperature"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.CELSIUS

session = dataset.add_sample(
    subject_ids=("participant_07",),
    time_series=(
        TimeSeries(spec=PPGSpec(channel="ppg"), source_id="session_3",
                   sampling_rate=Frequency.Hz(64.0), reader=lambda: load_ppg(session_path)),
        TimeSeries(spec=AccelSpec(channel="x"), source_id="session_3",
                   sampling_rate=Frequency.Hz(100.0), reader=lambda: load_accel(session_path, "x")),
        TimeSeries(spec=AccelSpec(channel="y"), source_id="session_3",
                   sampling_rate=Frequency.Hz(100.0), reader=lambda: load_accel(session_path, "y")),
        TimeSeries(spec=AccelSpec(channel="z"), source_id="session_3",
                   sampling_rate=Frequency.Hz(100.0), reader=lambda: load_accel(session_path, "z")),
        TimeSeries(spec=SkinTempSpec(channel="skin_temp"), source_id="session_3",
                   sampling_rate=Frequency.Hz(1.0), reader=lambda: load_temp(session_path)),
    ),
    view=View.FULL,
)
```

Multi-source sample (one logical overnight recording split across two files on disk, the connector concatenates inside each `reader` and exposes a single `source_id`):

```python
class EEGSpec(TimeSeriesSpec):
    spec_id = "eeg"
    name = "EEG"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MICROVOLT

class EOGSpec(TimeSeriesSpec):
    spec_id = "eog"
    name = "EOG"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MICROVOLT

class EMGSpec(TimeSeriesSpec):
    spec_id = "emg"
    name = "EMG"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MICROVOLT

night = dataset.add_sample(
    subject_ids=("participant_07",),
    time_series=(
        TimeSeries(spec=EEGSpec(channel="f3"), source_id="night_001",
                   sampling_rate=Frequency.Hz(256.0),
                   reader=lambda: concat_channel(("night_001a.edf", "night_001b.edf"), "f3")),
        TimeSeries(spec=EEGSpec(channel="c4"), source_id="night_001",
                   sampling_rate=Frequency.Hz(256.0),
                   reader=lambda: concat_channel(("night_001a.edf", "night_001b.edf"), "c4")),
        TimeSeries(spec=EOGSpec(channel="l"), source_id="night_001",
                   sampling_rate=Frequency.Hz(256.0),
                   reader=lambda: concat_channel(("night_001a.edf", "night_001b.edf"), "l")),
        TimeSeries(spec=EMGSpec(channel="emg"), source_id="night_001",
                   sampling_rate=Frequency.Hz(256.0),
                   reader=lambda: concat_channel(("night_001a.edf", "night_001b.edf"), "emg")),
    ),
    view=View.FULL,
)
```

Subject-less domain. Omit `subject_ids`:

```python
class EquityTickSpec(TimeSeriesSpec):
    spec_id = "equity_tick"
    name = "Equity Tick"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.DIMENSIONLESS

tick = dataset.add_sample(
    time_series=(
        TimeSeries(spec=EquityTickSpec(channel="price"), source_id="AAPL::2026-05-13",
                   sampling_rate=Frequency.Hz(1.0), reader=lambda: load_ticks("AAPL", "2026-05-13", "price")),
        TimeSeries(spec=EquityTickSpec(channel="volume"), source_id="AAPL::2026-05-13",
                   sampling_rate=Frequency.Hz(1.0), reader=lambda: load_ticks("AAPL", "2026-05-13", "volume")),
    ),
    view=View.FULL,
)
# tick.subject_ids == ()
```

Windowed samples, two samples that slice disjoint time ranges out of the same recording. Each `TimeSeries` declares its own window, and `reader()` returns just the windowed values:

```python
def windowed(path: Path, channel: str, t_start: float, t_end: float, sr: float) -> np.ndarray:
    full = ecg_reader(path, channel)
    return full[int(t_start * sr) : int(t_end * sr)]

first = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=(
        TimeSeries(spec=ECGLeadSpec(channel="I"), source_id="rec_001",
                   sampling_rate=Frequency.Hz(500.0), t_start_s=0.0, t_end_s=10.0,
                   reader=lambda: windowed(path, "I", 0.0, 10.0, 500.0)),
    ),
    view=View.WINDOW,
)

last = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=(
        TimeSeries(spec=ECGLeadSpec(channel="I"), source_id="rec_001",
                   sampling_rate=Frequency.Hz(500.0), t_start_s=50.0, t_end_s=60.0,
                   reader=lambda: windowed(path, "I", 50.0, 60.0, 500.0)),
    ),
    view=View.WINDOW,
)
```

More examples:

```python
lead_ii = TimeSeries(
    spec=ECGLeadSpec(channel="II"), source_id="rec_001",
    sampling_rate=Frequency.Hz(500.0),
    reader=lambda: ecg_reader(path, "II"),
)

full = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=(
        TimeSeries(spec=ECGLeadSpec(channel="I"), source_id="rec_001",
                   sampling_rate=Frequency.Hz(500.0), reader=lambda: ecg_reader(path, "I")),
        lead_ii,                                  # shared object
        TimeSeries(spec=ECGLeadSpec(channel="V1"), source_id="rec_001",
                   sampling_rate=Frequency.Hz(500.0), reader=lambda: ecg_reader(path, "V1")),
    ),
    view=View.FULL,
)

lead_only = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=(lead_ii,),                       # same object → shared chunk on disk
    view=View.SINGLE_CHANNEL,
)
```

---

### `add_annotation()`

```python
def add_annotation(
    self,
    samples: Sample | tuple[Sample, ...],
    task: Task,
    *,
    spec_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation: ...
```

Creates one `Annotation`, registers it on the dataset, and links it to every
target sample by appending its `annotation_id` to each sample's
`annotation_ids`. Pass a single `Sample` to annotate one sample, or a tuple of
samples to attach the same annotation across all of them.

**Parameters**

| Name               | Type                           | Default  | Description                                                                                                                                                 |
| ------------------ | ------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `samples`          | `Sample \| tuple[Sample, ...]` | required | The sample, or tuple of samples, this annotation is attached to.                                                                                            |
| `task`             | `Task`                         | required | Instance of a `Task` subclass. Carries all task-specific data (label, question/answer, windows, etc.).                                                      |
| `spec_id`          | `str \| None`                  | `None`   | Optional reference to an `AnnotationSpec.spec_id` declared in the dataset metadata. `None` for free-form annotations that don't follow a declared spec.     |
| `from_annotations` | `tuple[Annotation, ...]`       | `()`     | Source annotations this annotation was derived from. Used to build composition chains (e.g. a `ReasoningTask` built from prior `LabelingTask` annotations). |

**Returns:** The newly created `Annotation`, already registered on the dataset and linked to every target sample.

**Raises**

| Exception    | Condition                    |
| ------------ | ---------------------------- |
| `ValueError` | `samples` is an empty tuple. |

**Implementation**

```python
def add_annotation(
    self,
    samples: Sample | tuple[Sample, ...],
    task: Task,
    *,
    spec_id: str | None = None,
    from_annotations: tuple[Annotation, ...] = (),
) -> Annotation:
    targets = (samples,) if isinstance(samples, Sample) else tuple(samples)
    if not targets:
        raise ValueError("add_annotation() requires at least one sample")

    annotation = Annotation(
        task=task,
        sample_ids=tuple(s.sample_id for s in targets),
        spec_id=spec_id,
        from_annotations=from_annotations,
    )
    self._annotations.append(annotation)
    for sample in targets:
        sample.annotation_ids += (annotation.annotation_id,)
    return annotation
```

**Examples**

Whole-sample classification (ECG rhythm):

```python
dataset.add_annotation(ecg_sample, ClassificationTask(label="afib"))
```

Captioning over a whole sample (finance market window):

```python
dataset.add_annotation(market_window, CaptioningTask(
    answer="AAPL traded sideways with low volume; SPY drifted down 0.4%.",
))
```

Composition chain, build low-level annotations first, then derive
higher-level ones from them. Each `add_annotation()` returns the `Annotation`,
which is fed into the next call via `from_annotations`:

```python
# 1. Per-lead labelings on the raw recording. `LabelingTask.channels` is
#    unchanged: each name resolves against `TimeSeries.spec.channel` on one of
#    the sample's `time_series` (here, the `ECGLeadSpec(channel=...)` instances).
lead_labels = tuple(
    dataset.add_annotation(
        ecg_sample,
        LabelingTask(label="st_elevation", channels=(lead,), windows_s=((4.0, 9.0),)),
        spec_id="st_segment",
    )
    for lead in ("V1", "V2", "V3")
)

# 2. A QA conclusion reasoned from those labelings.
qa = dataset.add_annotation(
    ecg_sample,
    QATask(
        question="Which territory shows ST elevation?",
        answer="Anteroseptal (V1–V3).",
    ),
    from_annotations=lead_labels,
)

# 3. A top-level reasoning annotation built on the QA conclusion.
dataset.add_annotation(
    ecg_sample,
    ReasoningTask(
        question="Is this consistent with an acute anterior STEMI?",
        answer="Yes — contiguous anteroseptal ST elevation across V1–V3.",
    ),
    from_annotations=(qa,),
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

### `annotations`

```python
@property
def annotations(self) -> tuple[Annotation, ...]: ...
```

**Returns:** All annotations in insertion order. Read-only, use `add_annotation()` to extend.

---

## `Sample`

One logical unit of time-series data: a recording, a session, a sensor bundle, a market window. Samples are created exclusively by `TimeFDataset.add_sample()`.

```python
@dataclass(kw_only=True)
class Sample:
    sample_id: field(default_factory=lambda: str(uuid.uuid4()))
    time_series: tuple[TimeSeries, ...]
    view: View
    subject_ids: tuple[str, ...] = ()
    annotation_ids: tuple[str, ...] = ()

    @property
    def source_ids(self) -> tuple[str, ...]: ...   # derived: distinct sources across time_series
```

**Fields**

| Name             | Type                     | Description                                                                                                                               |
| ---------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_id`      | `str`                    | Auto-generated unique identifier (uuid4-based).                                                                                           |
| `time_series`    | `tuple[TimeSeries, ...]` | One `TimeSeries` per channel present in the sample. Reuse a `TimeSeries` instance across samples to share its bytes on disk.              |
| `view`           | `View`                   | A [`View`](types.md#view) enum member identifying the slice of the source this sample represents.                                         |
| `subject_ids`    | `tuple[str, ...]`        | Subjects this sample belongs to. Empty tuple for subject-less domains (finance, seismology, synthetic).                                   |
| `annotation_ids` | `tuple[str, ...]`        | IDs of the annotations attached to this sample. Populated by `TimeFDataset.add_annotation()`; resolve against `TimeFDataset.annotations`. |

---

### `TimeSeries`

Reference to one channel of time-series data, with optional windowing and a lazy reader callable.

```python
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from timenet.timef.metadata import TimeSeriesSpec
from timenet.units import Frequency


@dataclass(frozen=True, eq=False)
class TimeSeries:
    spec: TimeSeriesSpec
    source_id: str
    sampling_rate: Frequency
    reader: Callable[[], np.ndarray]
    t_start_s: float = 0.0
    t_end_s: float | None = None
    timestamps: Callable[[], np.ndarray] | None = None
```

| Field              | Type                               | Required | Description                                                                                                                                            |
| ------------------ | ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `spec`             | `TimeSeriesSpec`                   | yes      | An instance of a `TimeSeriesSpec` subclass                                                                                                             |
| `source_id`        | `str`                              | yes      | Identifier of the raw recording this series was extracted from.                                                                                        |
| `sampling_rate`    | [`Frequency`](types.md#frequency)  | yes      | Sampling rate of the values returned by `reader()`. Build with `Frequency.Hz(...)`, `Frequency.kHz(...)`, or `Frequency.MHz(...)`.                      |
| `reader`           | `Callable[[], np.ndarray]`         | yes      | Lazy loader. Returns a 1-D `float32` array of exactly the values for this series' window. Invoked by `TimeFWriter` during `store()`.                   |
| `t_start_s`        | `float`                            | no       | Time offset of the first returned value within the original recording timeline. Default `0.0`.                                                         |
| `t_end_s`          | `float \| None`                    | no       | End of the window within the recording. `None` means "to end of source". When set, `len(reader()) == round((t_end_s - t_start_s) * sampling_rate.hz)`. |
| `timestamps`       | `Callable[[], np.ndarray] \| None` | no       | Lazy loader for explicit per-sample timestamps (non-uniform sampling). Same length as `reader()`. `None` for uniform sampling.                         |

**Identity sharing.** `TimeSeries` is `eq=False`, so the writer dedupes by Python object identity: attach the _same_ `TimeSeries` instance to two samples to share one chunk of bytes on disk.

**Windowing.** A windowed `TimeSeries` is just a `TimeSeries` with `t_start_s` / `t_end_s` set and a `reader` that returns the windowed slice.

---

## `Annotation`

Frozen dataclass. Represents one labeled annotation over one or more samples. The annotation's payload lives on `task`; its subtype determines what fields are present.

```python
@dataclass(frozen=True)
class Annotation:
    annotation_id: field(default_factory=lambda: str(uuid.uuid4()))
    task: Task
    sample_ids: tuple[str, ...]
    spec_id: str | None = None
    from_annotations: tuple[Annotation, ...] = ()

    @property
    def from_annotation_ids(self) -> tuple[str, ...]: ...
```

**Fields**

| Name               | Type                     | Description                                                                                                                                               |
| ------------------ | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `annotation_id`    | `str`                    | Auto-generated unique identifier (uuid4-based)                                                                                                            |
| `task`             | `Task`                   | Instance of a `Task` subclass carrying all task-specific data.                                                                                            |
| `sample_ids`       | `tuple[str, ...]`        | IDs of the samples this annotation is attached to. Length 1 when a single `Sample` is passed to `add_annotation()`; >1 when a tuple of samples is passed. |
| `spec_id`          | `str \| None`            | Optional reference to an `AnnotationSpec.spec_id`. `None` for free-form annotations.                                                                      |
| `from_annotations` | `tuple[Annotation, ...]` | Source annotations in a composition chain. Empty if this annotation was created independently.                                                            |

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
