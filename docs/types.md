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

DataSource(data_source_type="vib_sensor", name="Vibration Sensor", provider="Acme")
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `data_source_type` | `str` | yes | Dataset-unique type tag, referenced by `TimeSeriesSpec.data_source`. |
| `name` | `str` | yes | Human-readable name. |
| `provider` | `str \| None` | no | Vendor / originator. |

---

## TimeSeriesSpec

The contract for a measurement **modality**: its type tag, display name, axis units, scalar dtype, and
per-timestep shape. One spec is shared across every logical stream of a modality; the stream identifier
lives on [`TimeSeries.channel`](timef-dataset.md), not here.

```python
from timenet.types import TimeSeriesSpec, ureg

vibration = TimeSeriesSpec(
    spec_type="vibration",
    name="Vibration",
    unit_sampling_rate=ureg.hertz,
    unit_timestamp=ureg.second,
    unit_value=ureg.standard_gravity,
)
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `spec_type` | `str` | yes | Dataset-unique modality tag (e.g. `"vibration"`). |
| `name` | `str` | yes | Human-readable modality label. |
| `unit_sampling_rate` | `pint.Unit` | yes | Must be a frequency, else `ValueError`. |
| `unit_timestamp` | `pint.Unit` | yes | Must be a time, else `ValueError`. |
| `unit_value` | `pint.Unit` | yes | Any unit (g, °C, mV, dimensionless, ...). |
| `data_source` | `DataSource \| None` | no | The source that produced this modality. |
| `dtype` | `str` | no | Canonical NumPy scalar dtype; defaults to `"float32"`. |
| `value_shape` | `tuple[int, ...]` | no | Shape of one timestep, excluding time; `()` means scalar. |
| `dimension_names` | `tuple[str, ...]` | no | Optional names matching every dimension in `value_shape`. |

The full logical array shape is `(n_steps, *value_shape)`. For example, an RGB frame stream can use
`dtype="uint8"`, `value_shape=(height, width, 3)`, and
`dimension_names=("height", "width", "color")`. Parquet values currently require the scalar
`float32` defaults; select the Zarr values backend for other dtypes or multidimensional values.

Connectors that reuse a modality can subclass with field defaults:

```python
from dataclasses import dataclass

import pint

@dataclass(frozen=True)
class Vibration(TimeSeriesSpec):
    spec_type: str = "vibration"
    name: str = "Vibration"
    unit_sampling_rate: pint.Unit = ureg.hertz
    unit_timestamp: pint.Unit = ureg.second
    unit_value: pint.Unit = ureg.standard_gravity
```

---

## Annotations

An annotation is extra context attached to a [`Sample`](timef-dataset.md): side information a task
can read as input, or that can itself become a task's question or answer. It is scoped at one of three
levels, and the scopes combine:

- sample: the whole sample (a static fact, or a trial-level temporal marker),
- time range: a span in the recording timeline (`start_time_s` … `end_time_s`),
- signal: one or more specific channels (`time_series_ids`).

Three shapes, one flat frozen dataclass each. `key` / `value` / `unit` / `description` / `id` are
**instance fields**, so connectors author annotations directly (or subclass with field defaults for
reuse) and they round-trip without runtime class synthesis.

| Class | Extra fields | Scope |
| --- | --- | --- |
| `StaticAnnotation` | `value` (required) | Whole sample, time-independent (condition, firmware, device, ticker). |
| `PointAnnotation` | `start_time_s`, `time_series_ids` | One instant, on specific signals or the whole sample. |
| `IntervalAnnotation` | `start_time_s`, `end_time_s`, `time_series_ids` | A bounded span (`end > start`), on specific signals or the whole sample. |

Shared fields: `key: str`, `value: Any = None`, `unit: str | None = None`,
`description: str | None = None`, `id: str` (auto uuid7). On the temporal shapes,
`time_series_ids=None` means **trial-level** (the whole sample); a non-empty tuple restricts the
annotation to those channels (each id must match a `TimeSeries.time_series_id` on the sample).

```python
from timenet.types import StaticAnnotation, PointAnnotation, IntervalAnnotation

# sample scope
StaticAnnotation(key="operating_hours", value=1200, unit="hours")

# time range on the whole sample (trial-level)
IntervalAnnotation(key="artifact", start_time_s=10.0, end_time_s=12.0)

# signal + time range: the vibration and current channels, seconds 5 to 6
IntervalAnnotation(
    key="fault",
    value="bearing fault",
    start_time_s=5.0,
    end_time_s=6.0,
    time_series_ids=("vibration", "current"),
)

# one instant on a single channel
PointAnnotation(key="impact", start_time_s=4.2, time_series_ids=("vibration",))
```

A connector that emits the same key repeatedly can subclass with field defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class OperatingHours(StaticAnnotation):
    key: str = "operating_hours"
    unit: str | None = "hours"
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

- `ClassificationTask`: one discrete label for the whole sample; `label_schema` names the
  vocabulary the label is drawn from (`None` for free-form).
  ```python
  dataset.add_task(sample, ClassificationTask(label="faulty", label_schema="condition"))
  ```
- `LabelingTask`: predict a label for supplied signals and/or time windows: `time_series_ids` picks
  the streams (`None` = all), `windows_s` the spans (`None` = full duration). The region is part of
  the task input contract; finding an unknown region is not represented by this task type.
  ```python
  dataset.add_task(sample, LabelingTask(
      label="fault_episode",
      time_series_ids=(vibration.time_series_id, current.time_series_id),
      windows_s=((120.0, 480.0),),
  ))
  ```
- `CaptioningTask`: free-form text describing the sample (no question).
  ```python
  dataset.add_task(sample, CaptioningTask(answer="A 10-second vibration trace with a bearing-fault signature after 5 s."))
  ```
- `QATask`: a question and its text answer; the answer need not be a categorical label.
  ```python
  dataset.add_task(sample, QATask(question="What happens between 12s and 18s?", answer="A bearing fault on the vibration channel."))
  ```
- `ForecastingTask`: predict a target sample from context samples.
  ```python
  dataset.add_task(target, ForecastingTask(context_sample_ids=("rec_001::history",), target_sample_id="rec_001::future"))
  ```
- `ReasoningTask`: a question, the reasoning trace, then the answer. The `answer` is the
  evaluation target; the `rationale` (chain of thought) is the training signal and is optional.
  ```python
  dataset.add_task(sample, ReasoningTask(
      question="Does this trace show a bearing fault?",
      rationale="The vibration amplitude rises after 5 s with a periodic impact once per shaft revolution.",
      answer="Yes.",
  ))
  ```

### Composition (`from_tasks`)

A task can derive from earlier tasks (or from the annotations that motivated them) via `from_tasks`.
The derived task records the chain it was built from, which is how a handful of base labels multiply
into many higher-level training samples:

```python
base = dataset.add_task(sample, ClassificationTask(label="faulty"))
dataset.add_task(sample, ReasoningTask(
    question="Is this machine healthy?",
    rationale="The trace is classified faulty: a bearing fault is present.",
    answer="No.",
    from_tasks=(base,),
))
```

### Annotations vs tasks

An annotation is sample-level information; a task is a learning target. A connector can use the same
source annotation in either role:

- As task input, the annotation is fed to the model as grounding. An `IntervalAnnotation` marking a
  bearing fault on the vibration channel over seconds 5 to 6 supplies the detail a `QATask` or
  `ReasoningTask` question builds on.
- To derive a task target, the connector copies the relevant information into the task payload: for
  example, a question about what happens on the vibration channel over that window can be answered
  from the source `fault` annotation.

TimeF does not currently mark each attached annotation as input or target for an individual task.
Training adapters must therefore avoid passing a target-derived annotation back to the model as
context, which would leak the answer.

Because annotations carry signal and time-range scope, one recording yields many targets: a
whole-sample classification, per-channel labelings, windowed QA, and reasoning that composes them via
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

- `View`: which slice of a source a sample is: `FULL`, `SINGLE_CHANNEL`, `SUBSET`, `WINDOW`.
- `Domain`: `HEALTH`, `CARDIOLOGY`, `SLEEP`, `ACTIVITY`, `ECONOMICS`, `FINANCE`, `GENERAL`.
- `License`: SPDX-style identifiers (`MIT`, `Apache-2.0`, `CC-BY-4.0`, `CC0-1.0`, ...).

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

`TimeFValidationError` covers both a value that would be *stored in a dataset* violating an invariant (a
negative `Version` component, a `unit_value` that isn't a frequency, an `IntervalAnnotation` that ends
before it starts, a `DatasetSchema` whose specs and data sources disagree) and an *invalid input to the
API* (a malformed dataset ref, a `dataset_id` that isn't an `org/name` pair, a version supplied twice).
Because it subclasses `ValueError`, `except ValueError` keeps catching all of it.

Plain `ValueError` is reserved for genuine programming bugs rather than bad data or input:
`annotation_type_of()` handed something that isn't one of the three shapes, or two `Task` classes
declaring the same `task_type` (a definition bug, raised at import). Those are never data or input
problems, so tagging them as TimeF validation failures would make the distinction useless.

---

See the [API reference for `timenet.types`](api/types.md) for the full symbol listing.
