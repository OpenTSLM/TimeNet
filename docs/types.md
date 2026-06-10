# Types

## Specs

### `TimeSeriesSpec`

The contract for a measurement modality. Every `TimeSeries` carries exactly one `TimeSeriesSpec` instance, and that spec describes exactly one channel.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from timenet.units import SamplingRateUnit, TimestampUnit, ValueUnit

from timenet.timef.metadata import Device


@dataclass(frozen=True)
class TimeSeriesSpec:
    channel: str                                       # per-series, e.g. "II", "f4", "V1", "APPL"
    spec_id: ClassVar[str]                              # modality tag (machine), e.g. "ecg_lead"
    name: ClassVar[str]                                 # modality tag (readable), e.g. "ECG Lead"
    unit_sampling_rate: ClassVar[SamplingRateUnit]
    unit_timestamp: ClassVar[TimestampUnit]
    unit_value: ClassVar[ValueUnit]
    device: ClassVar[type[Device] | None] = None        # typed back-ref (set by subclass)
```

| Field                | Type                             | Scope    | Required | Description                                                                                                                   |
| -------------------- | -------------------------------- | -------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `channel`            | `str`                            | instance | yes      | The single channel this series carries (e.g. `"II"`, `"f4"`, `"V1", "APPL"`).                                                 |
| `spec_id`            | `ClassVar[str]`                  | modality | yes      | Dataset-unique modality tag (machine), e.g. `"ecg_lead"`.                                                                     |
| `name`               | `ClassVar[str]`                  | modality | yes      | Human-readable modality label.                                                                                                |
| `unit_sampling_rate` | `ClassVar[SamplingRateUnit]`     | modality | yes      | Unit for sampling-rate values. See [SamplingRateUnit](#samplingrateunit).                                                     |
| `unit_timestamp`     | `ClassVar[TimestampUnit]`        | modality | yes      | Unit for timestamp values. See [TimestampUnit](#timestampunit).                                                               |
| `unit_value`         | `ClassVar[ValueUnit]`            | modality | yes      | Unit for channel values. See [ValueUnit](#valueunit).                                                                         |
| `device`             | `ClassVar[type[Device] \| None]` | modality | no       | Typed back-ref to the [`Device`](#device) subclass that produced this modality. `None` if hardware is unknown or not modeled. |

The base `TimeSeriesSpec` is never instantiated directly. It is used through a subclass

---

### `Device`

The contract for the hardware used to collect timeseries. Connectors subclass it per device. The base class is never instantiated directly — subclasses declare every field as a `ClassVar`.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Device:
    device_id: ClassVar[str]
    name: ClassVar[str]
    manufacturer: ClassVar[str | None] = None
    model: ClassVar[str | None] = None
```

| Field          | Type                    | Required | Description                                                                       |
| -------------- | ----------------------- | -------- | --------------------------------------------------------------------------------- |
| `device_id`    | `ClassVar[str]`         | yes      | Dataset-unique identifier. Referenced by `TimeSeriesSpec.device`.                 |
| `name`         | `ClassVar[str]`         | yes      | Human-readable device name (e.g. `"Apple Watch Series 9"`, `"Holter Monitor X"`). |
| `manufacturer` | `ClassVar[str \| None]` | no       | Vendor.                                                                           |
| `model`        | `ClassVar[str \| None]` | no       | Model identifier.                                                                 |

---

## Tasks

One labeled training segment that references one or more samples. The class is the type tag, it can be used as a filter argument (`query(tasks=[ClassificationTask])`),and the instance carries the payload.

Tasks are attached to samples via `TimeFDataset.add_task(samples, task)`.

### `Task`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(kw_only=True)
class Task:
    task_id: ClassVar[str]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sample_ids: tuple[str, ...] = ()
    from_tasks: tuple[Task, ...] = ()

    @property
    def from_task_ids(self) -> tuple[str, ...]: ...
```

| Field        | Type               | Scope    | Description                                                                                                                        |
| ------------ | ------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `task_id`    | `ClassVar[str]`    | class    | Stable string identifier for this task class. Drives both the disk partition name and the [Task registry](#task-registry) key.     |
| `id`         | `str`              | instance | Auto-generated unique identifier (uuid4-based).                                                                                    |
| `sample_ids` | `tuple[str, ...]`  | instance | IDs of the samples this task is attached to. Populated by `TimeFDataset.add_task()`; resolve against `TimeFDataset.samples`.       |
| `from_tasks` | `tuple[Task, ...]` | instance | Source tasks this task was derived from. Used to build composition chains (e.g. a `ReasoningTask` built on prior `LabelingTask`s). |

**Property**

| Name            | Returns           | Description                                                    |
| --------------- | ----------------- | -------------------------------------------------------------- |
| `from_task_ids` | `tuple[str, ...]` | IDs of all tasks in `from_tasks`. Shorthand for serialization. |

### `ClassificationTask`

One discrete label applied to the whole sample (e.g. rhythm type for an ECG recording).

```python
@dataclass(kw_only=True)
class ClassificationTask(Task):
    task_id: ClassVar[str] = "classification"
    label: str
    schema: str | None = None
```

| Field    | Type          | Description                                                                                                                                 |
| -------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `label`  | `str`         | Class label (e.g. `"afib"`).                                                                                                                |
| `schema` | `str \| None` | Optional named label schema (e.g. `"Willets2018"`). Set when there is a fixed label vocabulary; leave `None` for free-form discrete labels. |

### `LabelingTask`

Time-localized labels within a sample (e.g. beat type at a specific window). The difference vs `ClassificationTask` is granularity: classification is whole-sample, labeling is windowed and may target specific series.

```python
@dataclass(kw_only=True)
class LabelingTask(Task):
    task_id: ClassVar[str] = "labeling"
    label: str
    schema: str | None = None
    time_series_ids: tuple[str, ...] | None = None
    windows_s: tuple[tuple[float, float], ...] | None = None
```

| Field             | Type                                      | Description                                                                                                                                      |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `label`           | `str`                                     | Discrete class label.                                                                                                                            |
| `schema`          | `str \| None`                             | Optional named label schema. Same convention as `ClassificationTask.schema`.                                                                     |
| `time_series_ids` | `tuple[str, ...] \| None`                 | Series of the parent sample the task targets. Each id must match a `TimeSeries.series_id` on one of `Sample.time_series`. `None` = whole sample. |
| `windows_s`       | `tuple[tuple[float, float], ...] \| None` | Time spans in seconds the task covers. `None` = full duration.                                                                                   |

### `CaptioningTask`

Free-form text describing the sample. No question, just an answer.

```python
@dataclass(kw_only=True)
class CaptioningTask(Task):
    task_id: ClassVar[str] = "captioning"
    answer: str
```

| Field    | Type  | Description                           |
| -------- | ----- | ------------------------------------- |
| `answer` | `str` | Free-form text describing the sample. |

### `QATask`

Question and answer pair.

```python
@dataclass(kw_only=True)
class QATask(Task):
    task_id: ClassVar[str] = "question_and_answer"
    question: str
    answer: str
```

| Field      | Type  | Description    |
| ---------- | ----- | -------------- |
| `question` | `str` | Question text. |
| `answer`   | `str` | Answer text.   |

### `ForecastingTask`

Predict future values of a time series. Carries the context/target relation as a field.

```python
@dataclass(kw_only=True)
class ForecastingTask(Task):
    task_id: ClassVar[str] = "forecasting"
    context_sample_ids: tuple[str, ...]
    target_sample_id: str
```

| Field                | Type              | Description                                                      |
| -------------------- | ----------------- | ---------------------------------------------------------------- |
| `context_sample_ids` | `tuple[str, ...]` | Historical samples the forecast is conditioned on. At least one. |
| `target_sample_id`   | `str`             | Sample whose future values the model should predict.             |

### `ReasoningTask`

Higher-level analytical inference, often composed from multiple lower-level tasks (e.g. a longitudinal trend across recordings). Carries a question/answer pair like `QATask` but indicates that the conclusion was reached by reasoning over a chain (use `Task.from_tasks` to record that chain).

```python
@dataclass(kw_only=True)
class ReasoningTask(Task):
    task_id: ClassVar[str] = "reasoning"
    question: str
    answer: str
```

| Field      | Type  | Description    |
| ---------- | ----- | -------------- |
| `question` | `str` | Question text. |
| `answer`   | `str` | Answer text.   |

### Task registry

Mapping from on-disk `task_id` to `Task` subclass. `TimeFReader` uses this table to rebuild `Task` instances when reading the task partitions. An unknown `task_id` raises `ValueError`.

```python
from timenet.tasks import (
    CaptioningTask,
    ClassificationTask,
    ForecastingTask,
    LabelingTask,
    QATask,
    ReasoningTask,
    Task,
)


TASKS: dict[str, type[Task]] = {
    "classification":      ClassificationTask,
    "labeling":            LabelingTask,
    "captioning":          CaptioningTask,
    "question_and_answer": QATask,
    "forecasting":         ForecastingTask,
    "reasoning":           ReasoningTask,
}
```

Each key matches the `task_id` `ClassVar` declared on the corresponding subclass. The directory name `tasks/task=<task_id>/` written by `TimeFWriter` uses the same string.

### Examples

```python
from timenet.tasks import (
    ClassificationTask, LabelingTask, QATask, ForecastingTask,
)

dataset.add_task(sample, ClassificationTask(label="afib"))

dataset.add_task(sample, LabelingTask(
    label="walking",
    time_series_ids=(accel_x.series_id, accel_y.series_id, accel_z.series_id),
    windows_s=((120.0, 480.0),),
))

dataset.add_task(sample, QATask(
    question="What is happening between 12s and 18s?",
    answer="ST elevation in V2.",
))

dataset.add_task(target, ForecastingTask(
    context_sample_ids=("rec_001::history",),
    target_sample_id="rec_001::future",
))
```

---

## Events

A temporal marker attached to a `Sample`. Encodes either a point in time (`kind=POINT`, no end) or a bounded interval (`kind=INTERVAL`, `end_time_s > start_time_s`). The base `Event` should not be directly instantiable — connectors declare one subclass per event name and list those subclasses in `DatasetMetadata.events`.

### `Event`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class Event:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: EventKind
    start_time_s: float
    end_time_s: float | None = None
    time_series_ids: tuple[str, ...] | None = None
    name: ClassVar[str]                              # subclass sets
```

| Field             | Type                      | Scope    | Required | Description                                                                                                                                                                      |
| ----------------- | ------------------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`            | `ClassVar[str]`           | class    | yes      | Non-empty event name (e.g. `"stimulus_light"`, `"artifact"`). Discriminator in `manifest.events`.                                                                                |
| `event_id`        | `str`                     | instance | no       | Auto-generated unique identifier (uuid4-based).                                                                                                                                  |
| `kind`            | [`EventKind`](#eventkind) | both     | yes      | `POINT` or `INTERVAL`. May be pinned as a class-level default on a subclass (in which case the writer enforces that every emitted instance matches); otherwise set per instance. |
| `start_time_s`    | `float`                   | instance | yes      | Event start in the **original recording timeline**.                                                                                                                              |
| `end_time_s`      | `float \| None`           | instance | no       | End of the interval. `None` iff `kind == POINT`; required and strictly greater than `start_time_s` when `kind == INTERVAL`.                                                      |
| `time_series_ids` | `tuple[str, ...] \| None` | instance | no       | Series of the parent sample the event applies to. Each id must match a `TimeSeries.series_id` on one of `Sample.time_series`. `None` = trial-level (the whole sample).           |

**Attachment.** Events are passed via the `events=` kwarg on `TimeFDataset.add_sample()`. The same `Event` instance (or two instances sharing the same `event_id`) may be attached to multiple samples to declare reuse; the writer dedupes by `event_id`.

A connector declares the subclasses it emits.

```python
@dataclass(frozen=True, kw_only=True)
class StimulusLight(Event):
    name: ClassVar[str] = "stimulus_light"
    kind: EventKind = EventKind.POINT                # narrowed default


@dataclass(frozen=True, kw_only=True)
class Artifact(Event):
    name: ClassVar[str] = "artifact"
    # kind stays per-instance — artifacts can be POINT or INTERVAL


stimulus_onset = StimulusLight(start_time_s=4.0)

artifact = Artifact(
    kind=EventKind.INTERVAL,
    start_time_s=10.0,
    end_time_s=12.0,
    time_series_ids=(lead_v1.series_id,),
)
```

---

## Annotations

Static contextual metadata attached to a `Sample` (demographics, device serial, recording conditions, ticker symbol, …). Annotations are strictly sample-scoped and time-independent: there are no time fields on `Annotation`. The base `Annotation` should not be directly instantiable — connectors declare one subclass per `key` and list those subclasses in `DatasetMetadata.annotations`.

### `Annotation`

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True, kw_only=True)
class Annotation:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    value: Any
    key: ClassVar[str]                               # subclass sets
    unit: ClassVar[str | None] = None
    description: ClassVar[str | None] = None
```

| Field         | Type                         | Scope    | Required | Description                                                                                         |
| ------------- | ---------------------------- | -------- | -------- | --------------------------------------------------------------------------------------------------- |
| `key`         | `ClassVar[str]`              | class    | yes      | Stable string identifier for this annotation class.                                                 |
| `unit`        | `ClassVar[str \| None]`      | class    | no       | Physical unit for the value (e.g. `"years"`).                                                       |
| `description` | `ClassVar[str \| None]`      | class    | no       | Human-readable description of what this annotation represents.                                      |
| `id`          | `str`                        | instance | no       | Auto-generated unique identifier (uuid4-based). The writer dedupes annotation instances by `id`.    |
| `value`       | concrete (subclass-narrowed) | instance | yes      | The annotation's value. Narrowed per subclass to a concrete type (`int`, `str`, `Literal[...]`, …). |

**Attachment.** Annotations are passed via the `annotations=` kwarg on `TimeFDataset.add_sample()`. Passing the same `Annotation` instance to multiple `add_sample()` calls (or two instances sharing the same `id`) declares reuse.

A connector declares the subclasses it emits.

```python
from typing import ClassVar, Literal


@dataclass(frozen=True, kw_only=True)
class Age(Annotation):
    key: ClassVar[str] = "age"
    unit: ClassVar[str | None] = "years"
    value: int                                       # narrowed type


@dataclass(frozen=True, kw_only=True)
class Sex(Annotation):
    key: ClassVar[str] = "sex"
    value: Literal["M", "F", "O"]


@dataclass(frozen=True, kw_only=True)
class DeviceSerial(Annotation):
    key: ClassVar[str] = "device_serial"
    value: str


@dataclass(frozen=True, kw_only=True)
class Ticker(Annotation):
    key: ClassVar[str] = "ticker"
    value: str


@dataclass(frozen=True, kw_only=True)
class RecordingSite(Annotation):
    key: ClassVar[str] = "recording_site"
    description: ClassVar[str | None] = "Seismometer station code (FDSN convention)."
    value: str


age_64 = Age(value=64)
sex_m  = Sex(value="M")
serial = DeviceSerial(value="HX-2024-0117")

# Same instance attached to every sample of subject 42 — declares reuse.
for rec in subject_42_recordings:
    dataset.add_sample(
        time_series=(...),
        view=View.FULL,
        annotations=(age_64, sex_m, serial),
    )
```

---

## Version

A semantic version value type. Enforces `major.minor.patch` with non-negative integer components. Set on `DatasetMetadata.version`. Frozen and ordered, so versions compare with the usual precedence (`Version(1, 2, 0) > Version(1, 1, 9)`).

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Version.{name} must be a non-negative int, got {value!r}"
                )

    @classmethod
    def parse(cls, s: str) -> Version:
        parts = s.split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must be 'major.minor.patch', got {s!r}")
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError:
            raise ValueError(
                f"Version components must be integers, got {s!r}"
            ) from None
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
```

| Field   | Type  | Required | Description                    |
| ------- | ----- | -------- | ------------------------------ |
| `major` | `int` | yes      | Incompatible API changes.      |
| `minor` | `int` | yes      | Backward-compatible additions. |
| `patch` | `int` | yes      | Backward-compatible fixes.     |

Construct directly or parse from a string:

```python
from timenet.version import Version

Version(1, 0, 0)
Version.parse("1.0.0")          # equivalent
str(Version(1, 2, 3))           # "1.2.3"
```

---

## Frequency

A sampling rate as a validated value object.

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Frequency:
    hz: float                                    # canonical value, samples per second

    def __post_init__(self) -> None:
        if not math.isfinite(self.hz) or self.hz <= 0:
            raise ValueError(
                f"frequency must be a positive, finite number of Hz, got {self.hz!r}"
            )

    @classmethod
    def Hz(cls, value: float) -> "Frequency":
        return cls(float(value))

    @classmethod
    def kHz(cls, value: float) -> "Frequency":
        return cls(float(value) * 1_000.0)

    @classmethod
    def MHz(cls, value: float) -> "Frequency":
        return cls(float(value) * 1_000_000.0)
```

| Member          | Signature                     | Description                                                  |
| --------------- | ----------------------------- | ------------------------------------------------------------ |
| `hz`            | `float`                       | Canonical rate in samples per second. The only stored field. |
| `Frequency.Hz`  | `(value: float) -> Frequency` | Build from a value already in hertz.                         |
| `Frequency.kHz` | `(value: float) -> Frequency` | Build from kilohertz; `Frequency.kHz(0.5).hz == 500.0`.      |
| `Frequency.MHz` | `(value: float) -> Frequency` | Build from megahertz; `Frequency.MHz(1).hz == 1_000_000.0`.  |

```python
from timenet.units import Frequency

Frequency.Hz(500.0)            # ECG at 500 Hz
Frequency.kHz(0.5)             # same rate, written in kHz
Frequency.MHz(2.0)             # high-rate sensor, 2 MHz
Frequency.Hz(-1.0)             # ValueError
```

---

## Enums

### `View`

Identifies which slice of the source a sample represents. Set on `Sample.view`.

```python
from enum import StrEnum


class View(StrEnum):
    FULL = "full"
    SINGLE_CHANNEL = "single_channel"
    SUBSET = "subset"
    WINDOW = "window"
```

| Value            | Identifier         | Meaning                                                                         |
| ---------------- | ------------------ | ------------------------------------------------------------------------------- |
| `FULL`           | `"full"`           | The whole recording with every available channel.                               |
| `SINGLE_CHANNEL` | `"single_channel"` | One channel isolated out of a multi-channel recording.                          |
| `SUBSET`         | `"subset"`         | A subset of channels (more than one, fewer than all).                           |
| `WINDOW`         | `"window"`         | A bounded time range sliced out of the source (`TimeSeries.t_start_s/t_end_s`). |

```python
from timenet.views import View

sample = dataset.add_sample(time_series=(...), view=View.FULL)
```

---

### `EventKind`

Identifies whether an [`Event`](#event) is a point in time or a bounded interval. Set on `Event.kind`.

```python
from enum import StrEnum


class EventKind(StrEnum):
    POINT = "point"
    INTERVAL = "interval"
```

| Value      | Identifier   | Meaning                                                                      |
| ---------- | ------------ | ---------------------------------------------------------------------------- |
| `POINT`    | `"point"`    | Instantaneous event. `Event.end_time_s` must be `None`.                      |
| `INTERVAL` | `"interval"` | Bounded interval. `Event.end_time_s` is required and `> Event.start_time_s`. |

```python
StimulusLight(start_time_s=4.0)                                  # kind pinned on the subclass
Artifact(kind=EventKind.INTERVAL, start_time_s=10.0, end_time_s=12.0)
```

---

### `Domain`

Describes what kind of data a dataset contains. A dataset can declare more than one domain.

Set on `DatasetMetadata.domains`. Used as filter input on `query()` / `filter()`.

```python
from enum import StrEnum


class Domain(StrEnum):
    HEALTH = "health"
    CARDIOLOGY = "cardiology"
    SLEEP = "sleep"
    ACTIVITY = "activity"
    ECONOMICS = "economics"
    FINANCE = "finance"
    GENERAL = "general"
```

| Value        | Identifier     | Meaning                                                                           |
| ------------ | -------------- | --------------------------------------------------------------------------------- |
| `HEALTH`     | `"health"`     | Any medical or physiological data. Broad parent of more specific medical domains. |
| `CARDIOLOGY` | `"cardiology"` | Heart-specific (ECG, PPG, hemodynamics).                                          |
| `SLEEP`      | `"sleep"`      | Sleep recordings (polysomnography, actigraphy).                                   |
| `ACTIVITY`   | `"activity"`   | Human activity recognition (accelerometer, IMU).                                  |
| `ECONOMICS`  | `"economics"`  | Macroeconomic indicators.                                                         |
| `FINANCE`    | `"finance"`    | Market data.                                                                      |
| `GENERAL`    | `"general"`    | Catch-all for cross-domain or synthetic datasets.                                 |

```python
from timenet.domains import Domain

# A heart-rate-during-sleep dataset would declare three domains:
domains = (Domain.HEALTH, Domain.SLEEP, Domain.CARDIOLOGY)
```

---

### `License`

The legal license of the source data. Required on `DatasetMetadata.license`. Used as filter input on `query()` / `filter()`.

```python
from enum import StrEnum


class License(StrEnum):
    MIT = "mit"
    APACHE_2 = "apache-2.0"
    CC_BY_4 = "CC BY 4.0"
    ODC_BY_1 = "ODC-By v1.0"
```

| Value      | Identifier      | Notes                                                      |
| ---------- | --------------- | ---------------------------------------------------------- |
| `MIT`      | `"mit"`         | Permissive, attribution not required.                      |
| `APACHE_2` | `"apache-2.0"`  | Permissive with explicit patent grant.                     |
| `CC_BY_4`  | `"CC BY 4.0"`   | Attribution required. Common for academic/health datasets. |
| `ODC_BY_1` | `"ODC-By v1.0"` | Attribution required. Common for open data corpora.        |

```python
from timenet.licenses import License

metadata = DatasetMetadata(
    ...,
    license=License.CC_BY_4,
)
```

---

### `SamplingRateUnit`

Unit for a time series' sampling-rate values. Set on `TimeSeriesSpec.unit_sampling_rate`. This is the modality's declared display unit, not the actual rate of any one series — that is [`Frequency`](#frequency) on `TimeSeries.sampling_rate`.

```python
from enum import StrEnum


class SamplingRateUnit(StrEnum):
    HZ = "Hz"
    KHZ = "kHz"
```

| Value | Identifier | Meaning                      |
| ----- | ---------- | ---------------------------- |
| `HZ`  | `"Hz"`     | Samples per second.          |
| `KHZ` | `"kHz"`    | Thousand samples per second. |

```python
from timenet.units import SamplingRateUnit

class ECGLeadSpec(TimeSeriesSpec):
    unit_sampling_rate = SamplingRateUnit.HZ
    # ... other modality ClassVars (spec_id, name, unit_timestamp, unit_value)
```

---

### `TimestampUnit`

Unit for a time series' timestamp axis. Set on `TimeSeriesSpec.unit_timestamp`.

```python
from enum import StrEnum


class TimestampUnit(StrEnum):
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"
```

| Value          | Identifier | Meaning       |
| -------------- | ---------- | ------------- |
| `SECONDS`      | `"s"`      | Seconds.      |
| `MILLISECONDS` | `"ms"`     | Milliseconds. |
| `MICROSECONDS` | `"us"`     | Microseconds. |
| `NANOSECONDS`  | `"ns"`     | Nanoseconds.  |

```python
from timenet.units import TimestampUnit

class ECGLeadSpec(TimeSeriesSpec):
    unit_timestamp = TimestampUnit.SECONDS
    # ... other modality ClassVars (spec_id, name, unit_sampling_rate, unit_value)
```

---

### `ValueUnit`

Physical unit of a time series channel's values. Set on `TimeSeriesSpec.unit_value`.

```python
from enum import StrEnum


class ValueUnit(StrEnum):
    VOLT = "V"
    MILLIVOLT = "mV"
    MICROVOLT = "uV"
    METER_PER_SECOND_SQUARED = "m/s^2"
    G = "g"
    BEATS_PER_MINUTE = "bpm"
    CELSIUS = "degC"
    PERCENT = "%"
    DIMENSIONLESS = ""
```

| Value                      | Identifier | Meaning                             |
| -------------------------- | ---------- | ----------------------------------- |
| `VOLT`                     | `"V"`      | Volts.                              |
| `MILLIVOLT`                | `"mV"`     | Millivolts (e.g. ECG).              |
| `MICROVOLT`                | `"uV"`     | Microvolts (e.g. EEG).              |
| `METER_PER_SECOND_SQUARED` | `"m/s^2"`  | Acceleration in SI units.           |
| `G`                        | `"g"`      | Acceleration in standard gravities. |
| `BEATS_PER_MINUTE`         | `"bpm"`    | Heart / pulse rate.                 |
| `CELSIUS`                  | `"degC"`   | Temperature.                        |
| `PERCENT`                  | `"%"`      | Ratio as a percentage (e.g. SpO₂).  |
| `DIMENSIONLESS`            | `""`       | Unitless / normalized values.       |

```python
from timenet.units import ValueUnit

class ECGLeadSpec(TimeSeriesSpec):
    unit_value = ValueUnit.MILLIVOLT
    # ... other modality ClassVars (spec_id, name, unit_sampling_rate, unit_timestamp)
```
