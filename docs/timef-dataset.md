# TimeFDataset

The in-memory model a connector populates during `convert()`. Holds samples and their tasks as Python objects. No I/O, no file paths: persistence is `TimeFWriter` concern.

---

## `TimeFDataset`

```python
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from timenet.events import Event
from timenet.tasks import (
    Task,
    ClassificationTask,
    LabelingTask,
    CaptioningTask,
    QATask,
    ForecastingTask,
    ReasoningTask,
)
from timenet.timef.metadata import DatasetMetadata
from timenet.timef.types import Annotation
from timenet.views import View


class TimeFDataset:

    def __init__(self, *, metadata: DatasetMetadata) -> None: ...

    def add_sample(
        self,
        *,
        time_series: tuple[TimeSeries, ...],
        view: View,
        subject_ids: tuple[str, ...] = (),
        events: tuple[Event, ...] = (),
        annotations: tuple[Annotation, ...] = (),
    ) -> Sample: ...

    def add_task(
        self,
        samples: Sample | tuple[Sample, ...],
        task: Task,
        *,
        from_tasks: tuple[Task, ...] = (),
    ) -> Task: ...

    @property
    def metadata(self) -> DatasetMetadata: ...

    @property
    def samples(self) -> tuple[Sample, ...]: ...

    @property
    def tasks(self) -> tuple[Task, ...]: ...
```

### `__init__()`

```python
def __init__(self, *, metadata: DatasetMetadata) -> None: ...
```

**Parameters**

| Name       | Type              | Description                                                                                                                                     |
| ---------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `metadata` | `DatasetMetadata` | The dataset's self-description. Carries `dataset_id`, `version`, and the type catalogs used to validate `add_sample()` and `add_task()` inputs. |

---

### `add_sample()`

```python
def add_sample(
    self,
    *,
    time_series: tuple[TimeSeries, ...],
    view: View,
    subject_ids: tuple[str, ...] = (),
    events: tuple[Event, ...] = (),
    annotations: tuple[Annotation, ...] = (),
) -> Sample: ...
```

Creates a `Sample` with an auto-generated `sample_id`, registers it, and returns it.

**Parameters**

| Name          | Type                     | Required | Description                                                                                                                                                                                                           |
| ------------- | ------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time_series` | `tuple[TimeSeries, ...]` | yes      | One `TimeSeries` per channel the sample uses.                                                                                                                                                                         |
| `view`        | `View`                   | yes      | A [`View`](types.md#view) enum member identifying which slice of the source this sample represents (e.g. `View.FULL`, `View.SINGLE_CHANNEL`, `View.SUBSET`, `View.WINDOW`).                                           |
| `subject_ids` | `tuple[str, ...]`        | no       | Subjects this sample belongs to (participants, devices, instruments). Empty by default, leave unset for subject-less domains (finance, seismology, synthetic). Multi-element when the sample spans multiple subjects. |
| `events`      | `tuple[Event, ...]`      | no       | [`Event`](types.md#events) instances attached to this sample.                                                                                                                                                         |
| `annotations` | `tuple[Annotation, ...]` | no       | [`Annotation`](types.md#annotations) instances attached to this sample.                                                                                                                                               |

**Returns:** The newly created `Sample`.

**Raises**

`add_sample()` enforces every check that is intrinsic to the sample being added (the sample's own series, events, annotations, and their resolution against the dataset's metadata). Cross-sample checks (same `event_id` / annotation `id` across samples must be field-equal) and per-series array-contract checks are deferred to [`TimeFWriter.write()`](timef-writer.md#validation).

| Exception    | Condition                                                                                                                           |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | `time_series` is empty.                                                                                                             |
| `ValueError` | `type(ts.spec)` is not in `metadata.time_series_specs` for some `ts` in `time_series`.                                              |
| `ValueError` | `type(ts.spec).device is not None` and `type(ts.spec).device` is not in `metadata.devices`.                                         |
| `ValueError` | `ts.spec.channel` is empty.                                                                                                         |
| `ValueError` | `ts.t_start_s < 0`, or `ts.t_end_s is not None and ts.t_end_s <= ts.t_start_s`.                                                     |
| `ValueError` | An emitted `Event`'s type is not declared in `metadata.events`.                                                                     |
| `ValueError` | `event.kind == POINT` and `event.end_time_s is not None`.                                                                           |
| `ValueError` | `event.kind == INTERVAL` and `event.end_time_s is None or event.end_time_s <= event.start_time_s`.                                  |
| `ValueError` | The `Event` subclass pins `kind` as a class-level default and the emitted instance's `kind` differs from it.                        |
| `ValueError` | `event.time_series_ids` is non-`None` and any id does not match a `TimeSeries.series_id` in `time_series`.                          |
| `ValueError` | `event.time_series_ids is None` (trial-level) but the sample's `TimeSeries` instances do not share a common `(t_start_s, t_end_s)`. |
| `ValueError` | Event time bounds fall outside the referenced `TimeSeries` window (channel-level) or outside the common trial span (trial-level).   |
| `ValueError` | An emitted `Annotation`'s type is not declared in `metadata.annotations`.                                                           |

**Examples**

Single-subject ECG recording:

```python
from timenet.domains import Domain
from timenet.licenses import License
from timenet.timef.dataset import TimeFDataset, TimeSeries
from timenet.timef.metadata import DatasetMetadata
from timenet.timef.types import TimeSeriesSpec
from timenet.units import Frequency, SamplingRateUnit, TimestampUnit, ValueUnit
from timenet.version import Version

# One modality, subclassed once. Reused by every ECG example below.
class ECGLeadSpec(TimeSeriesSpec):
    spec_id = "ecg_lead"
    name = "ECG Lead"
    unit_sampling_rate = SamplingRateUnit.HZ
    unit_timestamp = TimestampUnit.SECONDS
    unit_value = ValueUnit.MILLIVOLT

dataset = TimeFDataset(metadata=DatasetMetadata(
    dataset_id="ecg_dataset",
    version=Version(1, 0, 0),
    description="100 patients, one 12-lead ECG recording each.",
    license=License.CC_BY_4,
    domains=(Domain.CARDIOLOGY,),
    time_series_specs=(ECGLeadSpec,),
))

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

Sample with one trial-level point event and one channel-level interval event. `StimulusLight` and `Artifact` are subclasses of [`Event`](types.md#events) declared by the connector and listed in `DatasetMetadata.events`:

```python
from timenet.events import EventKind

lead_v1 = TimeSeries(
    spec=ECGLeadSpec(channel="V1"), source_id="rec_001",
    sampling_rate=Frequency.Hz(500.0), t_start_s=0.0, t_end_s=30.0,
    reader=lambda: ecg_reader(path, "V1"),
)
lead_v2 = TimeSeries(
    spec=ECGLeadSpec(channel="V2"), source_id="rec_001",
    sampling_rate=Frequency.Hz(500.0), t_start_s=0.0, t_end_s=30.0,
    reader=lambda: ecg_reader(path, "V2"),
)

trial = dataset.add_sample(
    subject_ids=("patient_42",),
    time_series=(lead_v1, lead_v2),
    view=View.SUBSET,
    events=(
        StimulusLight(start_time_s=4.0),
        Artifact(
            kind=EventKind.INTERVAL,
            start_time_s=10.0,
            end_time_s=12.0,
            time_series_ids=(lead_v1.series_id,),
        ),
    ),
)
```

Demographics shared across every sample of one subject. `Age` and `Sex` are [`Annotation`](types.md#annotations) subclasses declared by the connector and listed in `DatasetMetadata.annotations`:

```python
age_64 = Age(value=64)
sex_m  = Sex(value="M")
for rec in subject_42_recordings:
    dataset.add_sample(
        time_series=(...),
        view=View.FULL,
        annotations=(age_64, sex_m),     # shared across all samples of subject 42
    )
```

---

### `add_task()`

```python
def add_task(
    self,
    samples: Sample | tuple[Sample, ...],
    task: Task,
    *,
    from_tasks: tuple[Task, ...] = (),
) -> Task: ...
```

Registers `task` on the dataset and links it to every target sample by appending `task.id` to each sample's `task_ids`. Pass a single `Sample` for a sample-scoped task, or a tuple of samples when the task spans several (e.g. a `ForecastingTask` over a context window plus a target window).

The `task` instance is registered as-is: its `sample_ids` field is populated by `add_task()` from the targets, and `from_tasks` is set from the keyword argument. Each task's `type(task)` must be in `DatasetMetadata.tasks`.

**Parameters**

| Name         | Type                           | Default  | Description                                                                                                                                   |
| ------------ | ------------------------------ | -------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `samples`    | `Sample \| tuple[Sample, ...]` | required | The sample, or tuple of samples, this task is attached to.                                                                                    |
| `task`       | `Task`                         | required | Instance of a `Task` subclass. Carries all task-specific data (label, question/answer, windows, …).                                           |
| `from_tasks` | `tuple[Task, ...]`             | `()`     | Source tasks this task was derived from. Used to build composition chains (e.g. a `ReasoningTask` built from prior `LabelingTask` instances). |

**Returns:** The registered `Task` (the same instance with `sample_ids` / `from_tasks` populated).

**Raises**

| Exception    | Condition                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| `ValueError` | `samples` is an empty tuple.                                                                                                           |
| `ValueError` | `type(task)` is not declared in `metadata.tasks`.                                                                                      |
| `ValueError` | `task` is a `LabelingTask` with `time_series_ids` set, and any id does not resolve to a `TimeSeries.series_id` on every target sample. |

**Examples**

Whole-sample classification (ECG rhythm):

```python
dataset.add_task(ecg_sample, ClassificationTask(label="afib"))
```

Captioning over a whole sample (finance market window):

```python
dataset.add_task(market_window, CaptioningTask(
    answer="AAPL traded sideways with low volume; SPY drifted down 0.4%.",
))
```

Composition chain, build low-level tasks first, then derive higher-level ones from them. Each `add_task()` returns the `Task`, which is fed into the next call via `from_tasks`:

```python
# 1. Per-lead labelings on the raw recording.
lead_labels = tuple(
    dataset.add_task(
        ecg_sample,
        LabelingTask(
            label="st_elevation",
            schema="st_segment",
            time_series_ids=(lead_ts.series_id,),
            windows_s=((4.0, 9.0),),
        ),
    )
    for lead_ts in (lead_v1_ts, lead_v2_ts, lead_v3_ts)
)

# 2. A QA conclusion reasoned from those labelings.
qa = dataset.add_task(
    ecg_sample,
    QATask(
        question="Which territory shows ST elevation?",
        answer="Anteroseptal (V1–V3).",
    ),
    from_tasks=lead_labels,
)

# 3. A top-level reasoning task built on the QA conclusion.
dataset.add_task(
    ecg_sample,
    ReasoningTask(
        question="Is this consistent with an acute anterior STEMI?",
        answer="Yes — contiguous anteroseptal ST elevation across V1–V3.",
    ),
    from_tasks=(qa,),
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

### `tasks`

```python
@property
def tasks(self) -> tuple[Task, ...]: ...
```

**Returns:** All tasks in insertion order. Read-only, use `add_task()` to extend.

Annotations live on samples, not on the dataset — reach them via `Sample.annotations`. The dataset has no top-level `annotations` collection.

---

## `Sample`

One logical unit of time-series data: a recording, a session, a sensor bundle, a market window. Samples are created exclusively by `TimeFDataset.add_sample()`.

```python
@dataclass(kw_only=True)
class Sample:
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_series: tuple[TimeSeries, ...]
    view: View
    subject_ids: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    events: tuple[Event, ...] = ()
    annotations: tuple[Annotation, ...] = ()

    @property
    def source_ids(self) -> tuple[str, ...]: ...   # derived: distinct sources across time_series
```

**Fields**

| Name          | Type                     | Description                                                                                                                           |
| ------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_id`   | `str`                    | Auto-generated unique identifier (uuid4-based).                                                                                       |
| `time_series` | `tuple[TimeSeries, ...]` | One `TimeSeries` per channel present in the sample. Two `TimeSeries` with the same `series_id` share one chunk on disk.               |
| `view`        | `View`                   | A [`View`](types.md#view) enum member identifying the slice of the source this sample represents.                                     |
| `subject_ids` | `tuple[str, ...]`        | Subjects this sample belongs to. Empty tuple for subject-less domains (finance, seismology, synthetic).                               |
| `task_ids`    | `tuple[str, ...]`        | IDs of the dataset-level tasks attached to this sample. Populated by `TimeFDataset.add_task()`; resolve against `TimeFDataset.tasks`. |
| `events`      | `tuple[Event, ...]`      | [`Event`](types.md#events) instances attached to this sample. Empty by default.                                                       |
| `annotations` | `tuple[Annotation, ...]` | [`Annotation`](types.md#annotations) instances attached to this sample (static context such as demographics). Empty by default.       |

---

### `TimeSeries`

Reference to one channel of time-series data, with optional windowing and a lazy reader callable.

```python
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from timenet.timef.metadata import TimeSeriesSpec
from timenet.units import Frequency


@dataclass(frozen=True, eq=False, kw_only=True)
class TimeSeries:
    series_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    spec: TimeSeriesSpec
    source_id: str
    sampling_rate: Frequency
    reader: Callable[[], np.ndarray]
    t_start_s: float = 0.0
    t_end_s: float | None = None
    timestamps: Callable[[], np.ndarray] | None = None
```

| Field           | Type                               | Required | Description                                                                                                                                            |
| --------------- | ---------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `series_id`     | `str`                              | no       | Persistent handle. Auto-generated uuid4 by default                                                                                                     |
| `spec`          | `TimeSeriesSpec`                   | yes      | An instance of a `TimeSeriesSpec` subclass                                                                                                             |
| `source_id`     | `str`                              | yes      | Identifier of the raw recording this series was extracted from.                                                                                        |
| `sampling_rate` | [`Frequency`](types.md#frequency)  | yes      | Sampling rate of the values returned by `reader()`. Build with `Frequency.Hz(...)`, `Frequency.kHz(...)`, or `Frequency.MHz(...)`.                     |
| `reader`        | `Callable[[], np.ndarray]`         | yes      | Lazy loader. Returns a 1-D `float32` array of exactly the values for this series' window. Invoked by `TimeFWriter` during `store()`.                   |
| `t_start_s`     | `float`                            | no       | Time offset of the first returned value within the original recording timeline. Default `0.0`.                                                         |
| `t_end_s`       | `float \| None`                    | no       | End of the window within the recording. `None` means "to end of source". When set, `len(reader()) == round((t_end_s - t_start_s) * sampling_rate.hz)`. |
| `timestamps`    | `Callable[[], np.ndarray] \| None` | no       | Lazy loader for explicit per-sample timestamps (non-uniform sampling). Same length as `reader()`. `None` for uniform sampling.                         |

**Sharing.** `TimeSeries` keeps identity-based equality (`eq=False`); the writer dedupes by `series_id`. Two `TimeSeries` with the same `series_id` (whether the same Python instance reused across samples, or separately constructed instances passed the same explicit `series_id`) collapse to one chunk of bytes on disk. `series_id` is also the persistent handle that `LabelingTask.time_series_ids` and `Event.time_series_ids` reference.

**Windowing.** A windowed `TimeSeries` is just a `TimeSeries` with `t_start_s` / `t_end_s` set and a `reader` that returns the windowed slice.

---

The `Task` (dataset-level) and `Annotation` (per-sample static context) dataclasses themselves are defined in [`types.md`](types.md#tasks). `TimeFDataset` only exposes them through `add_task()`, `tasks`, and `Sample.annotations` — there is no separate `Annotation` dataclass on this page.

**Reading task payloads**

`TimeFDataset.tasks` returns the task instances directly. Use `isinstance` (or `match`) to recover the subclass and its fields:

```python
for task in dataset.tasks:
    match task:
        case ClassificationTask(label=label):
            ...
        case LabelingTask(label=label, time_series_ids=channels, windows_s=windows):
            ...
        case QATask(question=question, answer=answer):
            ...
```

---
