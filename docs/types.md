---
icon: lucide/shapes
description: "TimeF value types: versions, units, specs, tasks, annotations, and metadata."
tags:
  - reference
  - types
---

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

TimeNet uses [pint](https://pint.readthedocs.io) for all physical units. One registry, `ureg`, owns
every definition and conversion, plus two custom units (`beat`, `bpm`) pint does not ship. Reference
units through `ureg` (`ureg.hertz`, `ureg.millivolt`, `ureg.standard_gravity`, `ureg.dimensionless`),
never a second registry, or comparisons and conversions fail.

```python
from timenet.types import ureg

(5.0 * ureg.millivolt).to(ureg.volt).magnitude   # 0.005
```

`ureg` is **private to TimeNet**. Importing `timenet` does not call `pint.set_application_registry`, so
your own registry is left alone. Everything TimeNet persists or pickles stores units by *name* and
rebuilds them against `ureg`, so nothing in the format depends on process-global pint state: the
manifest codec writes `str(unit)`, and `TimeSeriesSpec` converts every `pint.Unit` attribute (including
ones a subclass adds) in `__getstate__`.

That covers TimeNet's own types. It cannot cover a bare `pint.Unit` or `pint.Quantity` you pickle
yourself, because those store only the unit name and resolve it against pint's *application* registry,
which doesn't know `beat` or `bpm`:

```python
pickle.loads(pickle.dumps(ureg.bpm))   # UndefinedUnitError: 'bpm' is not defined
```

pint's application registry is the only hook for that, so it's opt-in rather than something a library
should do to you on import:

```python
from timenet.types import use_as_application_registry

use_as_application_registry()   # once, at application start
```

It is a global assignment, not a merge: the last call wins, and custom units from a previously
installed registry stop resolving. Prefer it only when you genuinely pickle bare units or quantities.

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

import pint

@dataclass(frozen=True)
class ECGLead(TimeSeriesSpec):
    spec_type: str = "ecg_lead"
    name: str = "ECG Lead"
    unit_sampling_rate: pint.Unit = ureg.hertz
    unit_timestamp: pint.Unit = ureg.second
    unit_value: pint.Unit = ureg.millivolt
```

---

## Annotations

An annotation is extra context attached to a [`Sample`](timef-dataset.md): side information a task
can read as input, or that can itself become a task's question or answer. It is scoped at one of three
levels, and the scopes combine:

- **sample** — the whole sample (a static fact, or a trial-level temporal marker),
- **time range** — a span in the recording timeline (`start_time_s` … `end_time_s`),
- **signal** — one or more specific channels (`time_series_ids`).

Three shapes, one flat frozen dataclass each. `key` / `value` / `unit` / `description` / `id` are
**instance fields**, so connectors author annotations directly (or subclass with field defaults for
reuse) and they round-trip without runtime class synthesis.

| Class | Extra fields | Scope |
| --- | --- | --- |
| `StaticAnnotation` | `value` (required) | Whole sample, time-independent (age, sex, device, ticker). |
| `PointAnnotation` | `start_time_s`, `time_series_ids` | One instant, on specific signals or the whole sample. |
| `IntervalAnnotation` | `start_time_s`, `end_time_s`, `time_series_ids` | A bounded span (`end > start`), on specific signals or the whole sample. |

Shared fields: `key: str`, `value: Any = None`, `unit: str | None = None`,
`description: str | None = None`, `id: str` (auto uuid7). On the temporal shapes,
`time_series_ids=None` means **trial-level** (the whole sample); a non-empty tuple restricts the
annotation to those channels (each id must match a `TimeSeries.time_series_id` on the sample).

```python
from timenet.types import StaticAnnotation, PointAnnotation, IntervalAnnotation

# sample scope
StaticAnnotation(key="age", value=64, unit="years")

# time range on the whole sample (trial-level)
IntervalAnnotation(key="artifact", start_time_s=10.0, end_time_s=12.0)

# signal + time range: leads V1 and V2, seconds 5–6
IntervalAnnotation(
    key="st_elevation",
    value="ST elevation",
    start_time_s=5.0,
    end_time_s=6.0,
    time_series_ids=("lead_v1", "lead_v2"),
)

# one instant on a single channel
PointAnnotation(key="r_peak", start_time_s=4.2, time_series_ids=("lead_v1",))
```

A connector that emits the same key repeatedly can subclass with field defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class Age(StaticAnnotation):
    key: str = "age"
    unit: str | None = "years"
```

`annotation_type_of(ann)` returns the `AnnotationType` (`STATIC` / `POINT` / `INTERVAL`);
`ANNOTATION_BASES` maps each back to its class. `AnnotationDescriptor` is the type-level projection
(`key`, `annotation_type`, `value_type`, `unit`, `description`) hoisted into the schema and manifest at
write time.

---

## Tasks

A task is one labeled training target referencing one or more samples. The class is the type tag
(usable as a search filter, e.g. `search(task=ReasoningTask)`); the instance carries the payload. Tasks
are mutable so [`add_task`](timef-dataset.md) can populate `sample_ids` after construction.

| Class | `task_type` | Payload |
| --- | --- | --- |
| `ClassificationTask` | `classification` | `label`, `label_schema` |
| `LabelingTask` | `labeling` | `label`, `label_schema`, `time_series_ids`, `windows_s` |
| `CaptioningTask` | `captioning` | `answer` |
| `QATask` | `question_and_answer` | `question`, `answer` |
| `ForecastingTask` | `forecasting` | `context_sample_ids`, `target_sample_id` |
| `ReasoningTask` | `reasoning` | `question`, `rationale`, `answer` |

Every task also carries `id` (auto uuid7), `sample_ids`, `from_tasks`, and a `from_task_ids` property.
`TaskType` is the enum of type tags; `TASKS` is **derived** at import by walking the `Task` subclass
tree, so every concrete task in the module is registered by its `task_type` and two classes claiming the
same tag are rejected rather than silently collapsed. Intermediate bases that group tasks without
claiming a `task_type` are skipped. Unlike specs and annotations, task payloads are fixed in code and
resolved on read against `TASKS`, not reconstructed from the manifest.

### Per-type payloads

- **`ClassificationTask`** — one discrete label for the whole sample; `label_schema` names the
  vocabulary the label is drawn from (`None` for free-form).
  ```python
  dataset.add_task(sample, ClassificationTask(label="afib", label_schema="AAMI"))
  ```
- **`LabelingTask`** — a label localized to specific signals and/or time windows: `time_series_ids`
  picks the channels (`None` = all), `windows_s` the spans (`None` = full duration).
  ```python
  dataset.add_task(sample, LabelingTask(
      label="walking",
      time_series_ids=(accel_x.time_series_id, accel_y.time_series_id),
      windows_s=((120.0, 480.0),),
  ))
  ```
- **`CaptioningTask`** — free-form text describing the sample (no question).
  ```python
  dataset.add_task(sample, CaptioningTask(answer="A 10-second sinus rhythm with one PVC."))
  ```
- **`QATask`** — a question and its single-label answer.
  ```python
  dataset.add_task(sample, QATask(question="What happens between 12s and 18s?", answer="ST elevation in V2."))
  ```
- **`ForecastingTask`** — predict a target sample from context samples.
  ```python
  dataset.add_task(target, ForecastingTask(context_sample_ids=("rec_001::history",), target_sample_id="rec_001::future"))
  ```
- **`ReasoningTask`** — a question, the reasoning trace, then the answer. The `answer` is the
  evaluation target; the `rationale` (chain of thought) is the training signal and is optional.
  ```python
  dataset.add_task(sample, ReasoningTask(
      question="Does this ECG show atrial fibrillation?",
      rationale="R-R intervals are irregularly irregular and no P waves precede the QRS complexes.",
      answer="Yes.",
  ))
  ```

### Composition (`from_tasks`)

A task can derive from earlier tasks (or from the annotations that motivated them) via `from_tasks`.
The derived task records the chain it was built from, which is how a handful of base labels multiply
into many higher-level training samples:

```python
base = dataset.add_task(sample, ClassificationTask(label="afib"))
dataset.add_task(sample, ReasoningTask(
    question="Is this recording normal?",
    rationale="The rhythm is classified atrial fibrillation, which is abnormal.",
    answer="No.",
    from_tasks=(base,),
))
```

### Annotations vs tasks

An annotation is context; a task is a learning target. The same annotation can play either role:

- As task input, the annotation is fed to the model as grounding. An `IntervalAnnotation` marking ST
  elevation in lead V1 over seconds 5 to 6 supplies the detail a `QATask` or `ReasoningTask` question
  builds on.
- As the task itself, the annotation's content becomes what the model must produce: a question about
  what happens in lead V1 over that window, answered from the same `st_elevation` annotation.

Because annotations carry signal and time-range scope, one recording yields many targets: a
whole-sample classification, per-lead labelings, windowed QA, and reasoning that composes them via
`from_tasks`.

---

## DatasetMetadata

A dataset's descriptive identity (authored in the card).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | `str` | yes | `org/name` pair (one slash); matches the card / connector module path. |
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

The value types above raise `TimeFValidationError` when a value that would be *stored in a dataset*
violates an invariant: a negative `Version` component, a `unit_value` that isn't a frequency, an
`IntervalAnnotation` that ends before it starts, a `DatasetSchema` whose specs and data sources
disagree. Because it subclasses `ValueError`, `except ValueError` keeps catching all of it.

Plain `ValueError` is reserved for mistakes in *calling code* rather than in dataset content:
`annotation_type_of()` handed something that isn't one of the three shapes, or two `Task` classes
declaring the same `task_type` (a definition bug, raised at import). Those are never data problems, so
tagging them as TimeF validation failures would make the distinction useless.

---

See the [API reference for `timenet.types`](api/types.md) for the full symbol listing.
