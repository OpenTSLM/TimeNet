# Types

## Specs

### `SignalSpec`

Declares a measurement modality. One spec is shared by every signal of that type (e.g. all 12-lead ECG recordings share one `SignalSpec`). Connectors declare their signal specs in `DatasetMetadata.signal_specs`. Samples reference them through `Sample.signals` (a tuple of `Signal` instances, see [TimeFDataset](timef-dataset.md#signal)).

```python
from dataclasses import dataclass

from timenet.units import SamplingRateUnit, TimestampUnit, ValueUnit


@dataclass(frozen=True)
class SignalSpec:
    spec_id: str
    name: str
    channels: tuple[str, ...]
    unit_sampling_rate: SamplingRateUnit
    unit_timestamp: TimestampUnit
    unit_value: ValueUnit
    sensor_id: str | None = None
```

| Field                | Type               | Required | Description                                                                       |
| -------------------- | ------------------ | -------- | --------------------------------------------------------------------------------- |
| `spec_id`            | `str`              | yes      | Dataset-unique identifier. Referenced by `SignalRef.spec_id` in `Sample.signals`. |
| `name`               | `str`              | yes      | Human-readable modality name (e.g. `"ECG"`, `"Acceleration"`).                    |
| `channels`           | `tuple[str, ...]`  | yes      | Ordered list of channel names.                                                    |
| `unit_sampling_rate` | `SamplingRateUnit` | yes      | Unit for sampling-rate values. See [SamplingRateUnit](#samplingrateunit).         |
| `unit_timestamp`     | `TimestampUnit`    | yes      | Unit for timestamp values. See [TimestampUnit](#timestampunit).                   |
| `unit_value`         | `ValueUnit`        | yes      | Unit for channel values. See [ValueUnit](#valueunit).                             |
| `sensor_id`          | `str \| None`      | no       | References a `SensorSpec.id`. `None` if hardware is unknown or not declared.      |

---

### `AnnotationSpec`

Declares a task type and optional label schema. Connectors declare them in `DatasetMetadata.annotation_specs`. Annotations _optionally_ reference one via `Annotation.spec_id`.

```python
from dataclasses import dataclass

from timenet.tasks import Task


@dataclass(frozen=True)
class AnnotationSpec:
    spec_id: str
    task: type[Task]
    schema: str | None = None
```

| Field     | Type          | Required | Description                                                                                                                                                                                                               |
| --------- | ------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec_id` | `str`         | yes      | Dataset-unique identifier. Referenced by `Annotation.spec_id` (when set).                                                                                                                                                 |
| `task`    | `type[Task]`  | yes      | The task **class** this spec describes (e.g. `ClassificationTask`). The class is the type tag — actual annotation data lives on instances.                                                                                |
| `schema`  | `str \| None` | no       | Named label schema (e.g. `"Willets2018"`). Recommended for `LabelingTask` and `ClassificationTask` when there is a fixed label vocabulary. Omit for free-form discrete labels or for tasks where a schema does not apply. |

```python
from timenet.tasks import ClassificationTask

AnnotationSpec(spec_id="rhythm_cls", task=ClassificationTask, schema="Willets2018")
```

---

### `SensorSpec`

Declares the hardware that produced a signal. Optional metadata, connectors that don't know or care about hardware can leave `SignalSpec.sensor_id` as `None`. Connectors declare sensor specs in `DatasetMetadata.sensor_specs`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorSpec:
    id: str
    name: str
    manufacturer: str | None = None
    model: str | None = None
```

| Field          | Type          | Required | Description                                                                       |
| -------------- | ------------- | -------- | --------------------------------------------------------------------------------- |
| `id`           | `str`         | yes      | Dataset-unique identifier. Referenced by `SignalSpec.sensor_id`.                  |
| `name`         | `str`         | yes      | Human-readable device name (e.g. `"Apple Watch Series 9"`, `"Holter Monitor X"`). |
| `manufacturer` | `str \| None` | no       | Vendor.                                                                           |
| `model`        | `str \| None` | no       | Model identifier.                                                                 |

---

## Tasks

Two roles:

- The **class** is a type tag. Used in `AnnotationSpec.task` and in filter parameters (`query(tasks=[ClassificationTask])`)
- An **instance** carries the data for one annotation. `TimeFDataset.add_annotation(sample, ClassificationTask(label="afib"))` creates a concrete labeled annotation.

Adding a new task type means adding one subclass.

### `Task`

```python
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Task:
    task_id: ClassVar[str]
```

| Field     | Type            | Description                                   |
| --------- | --------------- | --------------------------------------------- |
| `task_id` | `ClassVar[str]` | Stable string identifier for this task class. |

### `ClassificationTask`

One discrete label applied to the whole sample (e.g. rhythm type for an ECG recording).

```python
@dataclass(frozen=True)
class ClassificationTask(Task):
    task_id: ClassVar[str] = "classification"
    label: str
```

| Field   | Type  | Description                  |
| ------- | ----- | ---------------------------- |
| `label` | `str` | class label (e.g. `"afib"`). |

### `LabelingTask`

Time-localized labels within a sample (e.g. beat type at a specific window). The difference vs `ClassificationTask` is granularity: classification is whole-sample, labeling is windowed and may target specific channels.

```python
@dataclass(frozen=True)
class LabelingTask(Task):
    task_id: ClassVar[str] = "labeling"
    label: str
    channels: tuple[str, ...] | None = None
    windows_s: tuple[tuple[float, float], ...] | None = None
```

| Field       | Type                                      | Description                                                                                                                                      |
| ----------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `label`     | `str`                                     | Discrete class label.                                                                                                                            |
| `channels`  | `tuple[str, ...] \| None`                 | Channels of the parent sample the annotation targets. Each name must match a `Signal.channel` on one of `Sample.signals`. `None` = whole sample. |
| `windows_s` | `tuple[tuple[float, float], ...] \| None` | Time spans in seconds the annotation covers. `None` = full duration.                                                                             |

### `CaptioningTask`

Free-form text describing the sample. No question, just an answer.

```python
@dataclass(frozen=True)
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
@dataclass(frozen=True)
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

Predict future values of a signal. Carries the context/target relation as a field.

```python
@dataclass(frozen=True)
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

Higher-level analytical inference, often composed from multiple lower-level annotations (e.g. a longitudinal trend across recordings). Carries a question/answer pair like `QATask` but signals that the conclusion was reached by reasoning over a chain (use `Annotation.from_annotations` to record that chain).

```python
@dataclass(frozen=True)
class ReasoningTask(Task):
    task_id: ClassVar[str] = "reasoning"
    question: str
    answer: str
```

| Field      | Type  | Description    |
| ---------- | ----- | -------------- |
| `question` | `str` | Question text. |
| `answer`   | `str` | Answer text.   |

### Examples

```python
from timenet.tasks import (
    ClassificationTask, LabelingTask, QATask, ForecastingTask,
)

dataset.add_annotation(sample, ClassificationTask(label="afib"))

dataset.add_annotation(sample, LabelingTask(
    label="walking",
    channels=("accel_x", "accel_y", "accel_z"),
    windows_s=((120.0, 480.0),),
))

dataset.add_annotation(sample, QATask(
    question="What is happening between 12s and 18s?",
    answer="ST elevation in V2.",
))

dataset.add_annotation(target, ForecastingTask(
    context_sample_ids=("rec_001::history",),
    target_sample_id="rec_001::future",
))
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

| Value            | Identifier         | Meaning                                                                     |
| ---------------- | ------------------ | --------------------------------------------------------------------------- |
| `FULL`           | `"full"`           | The whole recording with every available channel.                           |
| `SINGLE_CHANNEL` | `"single_channel"` | One channel isolated out of a multi-channel recording.                      |
| `SUBSET`         | `"subset"`         | A subset of channels (more than one, fewer than all).                       |
| `WINDOW`         | `"window"`         | A bounded time range sliced out of the source (`Signal.t_start_s/t_end_s`). |

```python
from timenet.views import View

sample = dataset.add_sample(signals=(...), view=View.FULL)
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

Unit for a signal's sampling-rate values. Set on `SignalSpec.unit_sampling_rate`.

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

SignalSpec(..., unit_sampling_rate=SamplingRateUnit.HZ)
```

---

### `TimestampUnit`

Unit for a signal's timestamp axis. Set on `SignalSpec.unit_timestamp`.

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

SignalSpec(..., unit_timestamp=TimestampUnit.SECONDS)
```

---

### `ValueUnit`

Physical unit of a signal channel's values. Set on `SignalSpec.unit_value`.

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

SignalSpec(..., unit_value=ValueUnit.MILLIVOLT)
```
