# Types

The TimeF value types live in `timenet.types` (one module per concept, re-exported from the package).
Every schema-carrying type is a **plain frozen dataclass** so it pickles and round-trips through
[`TimeFReader`](timef-reader.md) without runtime class synthesis. Errors live in `timenet.errors`.

---

## Version

A semantic `major.minor.patch` version. Frozen and ordered, so versions compare with the usual
precedence (`Version(1, 2, 0) > Version(1, 1, 9)`). Components must be non-negative integers.

```python
from timenet.types import Version

Version(1, 0, 0)
Version.parse("1.0.0")   # equivalent
str(Version(1, 2, 3))    # "1.2.3"
```

---

## Units

TimeNet uses [pint](https://pint.readthedocs.io) for all physical units. One process-wide registry,
`ureg`, owns every definition and conversion, plus two custom units (`beat`, `bpm`) pint does not ship.
Reference units through `ureg` (`ureg.hertz`, `ureg.millivolt`, `ureg.standard_gravity`,
`ureg.dimensionless`), never a second registry, or comparisons and conversions fail.

```python
from timenet.types import ureg

(5.0 * ureg.millivolt).to(ureg.volt).magnitude   # 0.005
```

---

## DataSource

The origin that produced a modality: a device, an API feed, a model, an institution. A flat frozen
dataclass built directly.

```python
from timenet.types import DataSource

DataSource(data_source_type="holter_x", name="Holter Monitor X", provider="Acme")
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `data_source_type` | `str` | yes | Dataset-unique type tag, referenced by `TimeSeriesSpec.data_source`. |
| `name` | `str` | yes | Human-readable name. |
| `provider` | `str \| None` | no | Vendor / originator. |

---

## TimeSeriesSpec

The contract for a measurement **modality**: its type tag, display name, and the units of its three
axes. One spec is shared across every channel of a modality; the per-channel identifier lives on
[`TimeSeries.channel`](timef-dataset.md), not here.

```python
from timenet.types import TimeSeriesSpec, ureg

ecg_lead = TimeSeriesSpec(
    spec_type="ecg_lead",
    name="ECG Lead",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.millivolt,
)
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `spec_type` | `str` | yes | Dataset-unique modality tag (e.g. `"ecg_lead"`). |
| `name` | `str` | yes | Human-readable modality label. |
| `unit_sampling_rate` | `pint.Unit` | yes | Must be a frequency, else `ValueError`. |
| `unit_timestamp` | `pint.Unit` | yes | Must be a time, else `ValueError`. |
| `unit_value` | `pint.Unit` | yes | Any unit (mV, g, bpm, dimensionless, ...). |
| `data_source` | `DataSource \| None` | no | The source that produced this modality. |

Connectors that reuse a modality can subclass with field defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ECGLead(TimeSeriesSpec):
    spec_type: str = "ecg_lead"
    name: str = "ECG Lead"
    unit_sampling_rate: object = ureg.hertz
    unit_timestamp: object = ureg.second
    unit_value: object = ureg.millivolt
```

---

## Annotations

Contextual metadata attached to a [`Sample`](timef-dataset.md). Three shapes, one flat frozen dataclass
each. `key` / `unit` / `description` are instance fields (so they round-trip without synthesis); the
per-key metadata is hoisted into the manifest at write time.

| Class | Extra fields | Meaning |
| --- | --- | --- |
| `StaticAnnotation` | `value` (required) | Sample-scoped, time-independent context (age, sex, ticker). |
| `PointAnnotation` | `start_time_s`, `time_series_ids` | Anchored to one instant in the recording timeline. |
| `IntervalAnnotation` | `start_time_s`, `end_time_s`, `time_series_ids` | Anchored to a bounded interval (`end > start`). |

Shared fields on every annotation: `key: str`, `value: Any = None`, `unit: str | None = None`,
`description: str | None = None`, `id: str` (auto uuid4). `time_series_ids` is `None` for trial-level
(whole-sample) temporal annotations.

```python
from timenet.types import StaticAnnotation, PointAnnotation, IntervalAnnotation

StaticAnnotation(key="age", value=64, unit="years")
PointAnnotation(key="stimulus_light", start_time_s=4.0)
IntervalAnnotation(key="artifact", start_time_s=10.0, end_time_s=12.0, time_series_ids=("s1",))
```

`annotation_type_of(ann)` returns the `AnnotationType` (`STATIC` / `POINT` / `INTERVAL`);
`ANNOTATION_BASES` maps each `AnnotationType` back to its class. `AnnotationDescriptor` is the
type-level projection (`key`, `annotation_type`, `value_type`, `unit`, `description`) stored in the
schema and manifest.

---

## Tasks

One labeled training target referencing one or more samples. The class is the type tag (usable as a
search filter); the instance carries the payload. Tasks are mutable so
[`add_task`](timef-dataset.md) can populate `sample_ids` after construction.

| Class | `task_type` | Payload |
| --- | --- | --- |
| `ClassificationTask` | `classification` | `label`, `label_schema` |
| `LabelingTask` | `labeling` | `label`, `label_schema`, `time_series_ids`, `windows_s` |
| `CaptioningTask` | `captioning` | `answer` |
| `QATask` | `question_and_answer` | `question`, `answer` |
| `ForecastingTask` | `forecasting` | `context_sample_ids`, `target_sample_id` |
| `ReasoningTask` | `reasoning` | `question`, `answer` |

Every task carries `id` (auto uuid4), `sample_ids`, `from_tasks`, and a `from_task_ids` property.
`TaskType` is the enum of type tags; `TASKS` maps each `TaskType` to its class and is **derived** from
`Task.__subclasses__()`, so it can never drift. Unlike specs and annotations, task payloads are fixed
in code and resolved on read against `TASKS`, not reconstructed from the manifest.

---

## DatasetMetadata

A dataset's descriptive identity (authored in the card).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | `str` | yes | Snake-cased unique id; matches the card / connector filename. |
| `dataset_version` | `Version` | yes | The upstream source's semantic version. |
| `name` | `str` | yes | Display name. |
| `description` | `str` | yes | One-sentence description. |
| `license` | `License` | yes | SPDX-style license id. |
| `domains` | `tuple[Domain, ...]` | no | Application/clinical domains. |
| `tags` | `tuple[str, ...]` | no | Free-form labels. |
| `source_url` | `str \| None` | no | Canonical source URL. |
| `yaml_schema_version` | `int` | no | The card's field-schema version (default `1`). |

---

## DatasetSchema

A dataset's type declaration, **derived** from its data (never hand-authored), then serialized into the
manifest. Holds flat descriptors for specs / data sources / annotations and the real built-in `Task`
subclasses.

```python
DatasetSchema(
    time_series_specs: tuple[TimeSeriesSpec, ...] = (),
    data_sources:      tuple[DataSource, ...] = (),
    annotations:       tuple[AnnotationDescriptor, ...] = (),
    tasks:             tuple[type[Task], ...] = (),
)
```

---

## Enums

- **`View`** — which slice of a source a sample is: `FULL`, `SINGLE_CHANNEL`, `SUBSET`, `WINDOW`.
- **`Domain`** — `HEALTH`, `CARDIOLOGY`, `SLEEP`, `ACTIVITY`, `ECONOMICS`, `FINANCE`, `GENERAL`.
- **`License`** — SPDX-style identifiers (`MIT`, `Apache-2.0`, `CC-BY-4.0`, `CC0-1.0`, ...).

All are `StrEnum`, so members compare equal to their string values.

---

## Errors

`timenet.errors` defines the exception hierarchy. `TimeNetError` is the base; validation and manifest
errors also derive from `ValueError` so existing handlers keep working.

| Exception | Base(s) | Raised when |
| --- | --- | --- |
| `TimeNetError` | `Exception` | base for all TimeNet errors |
| `RegistryError` | `TimeNetError` | a registry can't be loaded/reached/served |
| `DatasetNotFoundError` | `TimeNetError` | an unknown dataset id/version |
| `TimeFValidationError` | `TimeNetError`, `ValueError` | a dataset/array violates a TimeF invariant |
| `TimeFFormatError` | `TimeNetError` | a corrupt or unsupported on-disk artifact |
| `InvalidManifestError` | `TimeFFormatError`, `ValueError` | a malformed `manifest.json` |
