# Specs & Enums

Two kinds of vocabulary appear across the rest of the spec:

- **Specs**: frozen dataclasses that connectors _declare_ in `DatasetMetadata`. Samples and annotations reference them by ID. They define what kinds of signals, annotations, and views exist for a dataset.
- **Enums**: closed `StrEnum`s that supply shared vocabularies (`Domain`, `Task`, `License`).

Specs are dataset-specific (each connector ships its own). Enums are global (the same set of values applies everywhere).

---

## Specs

### `SignalSpec`

Declares a measurement modality. One spec is shared by every signal of that type (e.g. all 12-lead ECG recordings share one `SignalSpec`). Connectors declare their signal specs in `DatasetMetadata.signal_specs`. Samples reference them through `Sample.signals` (a tuple of `SignalRef` entries — see [TimeFDataset](timef-builder.md#signalref)).

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalSpec:
    spec_id: str
    name: str
    channels: tuple[str, ...]
    unit_sampling_rate: str
    unit_timestamp: str
    unit_value: str
    sensor_id: str | None = None
```

| Field                | Type              | Required | Description                                                                       |
| -------------------- | ----------------- | -------- | --------------------------------------------------------------------------------- |
| `spec_id`            | `str`             | yes      | Dataset-unique identifier. Referenced by `SignalRef.spec_id` in `Sample.signals`. |
| `name`               | `str`             | yes      | Human-readable modality name (e.g. `"ECG"`, `"Acceleration"`).                    |
| `channels`           | `tuple[str, ...]` | yes      | Ordered list of channel names.                                                    |
| `unit_sampling_rate` | `str`             | yes      | Unit for sampling rate values (e.g. `"Hz"`).                                      |
| `unit_timestamp`     | `str`             | yes      | Unit for timestamp values (e.g. `"s"`, `"us"`).                                   |
| `unit_value`         | `str`             | yes      | Unit for channel values (e.g. `"mV"`, `"m/s^2"`).                                 |
| `sensor_id`          | `str \| None`     | no       | References a `SensorSpec.id`. `None` if hardware is unknown or not declared.      |

---

### `AnnotationSpec`

Declares a task type and optional label schema. Connectors declare them in `DatasetMetadata.annotation_specs`. Annotations _optionally_ reference one via `Annotation.spec_id`.

```python
from dataclasses import dataclass

from timenet.tasks import Task


@dataclass(frozen=True)
class AnnotationSpec:
    spec_id: str
    task: Task
    schema: str | None = None
```

| Field     | Type          | Required | Description                                                                                                                                     |
| --------- | ------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `spec_id` | `str`         | yes      | Dataset-unique identifier. Referenced by `Annotation.spec_id` (when set).                                                                       |
| `task`    | `Task`        | yes      | Task type this spec describes.                                                                                                                  |
| `schema`  | `str \| None` | no       | Named label schema. Required when `task` is `Task.LABELING` or `Task.CLASSIFICATION` — must be a registered schema name (e.g. `"Willets2018"`). |

---

### `ViewSpec`

Declares a named sample view. Every `Sample.view` references one of these by `name`. Connectors declare them in `DatasetMetadata.view_specs`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSpec:
    name: str
    description: str | None = None
```

| Field         | Type          | Required | Description                                                       |
| ------------- | ------------- | -------- | ----------------------------------------------------------------- |
| `name`        | `str`         | yes      | Unique view name within the dataset. Referenced by `Sample.view`. |
| `description` | `str \| None` | no       | Human-readable explanation of what this view represents.          |

---

## Enums

### `Domain`

Describes what kind of data a dataset contains. A dataset can declare more than one domain.

Set on `DatasetMetadata.domains`. Used as filter input on `query()` / `filter()`.

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

### `Task`

Describes what kind of labeled work an annotation supports. Set on `Annotation.task` and on `AnnotationSpec.task`. A single sample can carry annotations of many different task types.

| Value                 | Identifier              | Meaning                                                                                                                                                                               |
| --------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `CLASSIFICATION`      | `"classification"`      | One discrete label per annotation, applied to the whole sample (e.g. rhythm type for an ECG recording).                                                                               |
| `LABELING`            | `"labeling"`            | Time-localized labels within a sample (e.g. beat type at a specific window). The difference vs `CLASSIFICATION` is granularity: classification is whole-sample, labeling is windowed. |
| `FORECASTING`         | `"forecasting"`         | Predict future values of a signal. A forecasting annotation links one or more historical _context_ samples to a _target_ sample that the model should predict.                        |
| `CAPTIONING`          | `"captioning"`          | Free-form text describing the sample. No question, just an answer.                                                                                                                    |
| `QUESTION_AND_ANSWER` | `"question and answer"` | Question and answer pair. The most flexible task; multi-sample QA uses references like `[ref:Person A]` to refer to specific samples in the question or answer text.                  |
| `REASONING`           | `"reasoning"`           | Higher-level analytical inference, often composed from multiple lower-level annotations (e.g. a longitudinal trend across recordings).                                                |

```python
from timenet.tasks import Task

sample.annotate(task=Task.CLASSIFICATION, label="afib")

sample.annotate(
    task=Task.QUESTION_AND_ANSWER,
    question="What is happening between 12s and 18s?",
    answer="ST elevation in V2.",
)
```

---

### `License`

The legal license of the source data. Required on `DatasetMetadata.license`. Used as filter input on `query()` / `filter()`.

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
