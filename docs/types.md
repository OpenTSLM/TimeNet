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
# UndefinedUnitError: 'bpm' is not defined
pickle.loads(pickle.dumps(ureg.bpm))
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
| `data_source_type` | `str` | yes | Type tag identifying the kind of source. |
| `name` | `str` | yes | Human-readable name. |
| `provider` | `str \| None` | no | Vendor / originator. |

---

## TimeSeriesSpec

The contract for a measurement **modality**: its type tag, display name, value unit, scalar dtype, and
per-timestep shape. One spec is shared across every logical stream of a modality; the stream identifier
lives on [`TimeSeries.channel`](timef-dataset.md), not here.

```python
from timenet.types import TimeSeriesSpec, ureg

vibration = TimeSeriesSpec(
    spec_type="vibration",
    name="Vibration",
    unit_value=ureg.standard_gravity,
)
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `spec_type` | `str` | yes | Dataset-unique modality tag (e.g. `"vibration"`). |
| `name` | `str` | yes | Human-readable modality label. |
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
    unit_value: pint.Unit = ureg.standard_gravity
```

---

## Annotations

An annotation is extra context attached to a [`Sample`](timef-dataset.md): side information a task
can read as input, or that can itself become a task's question or answer. It is scoped at one of three
levels, and the scopes combine:

- sample: the whole sample (a static fact, or a trial-level temporal marker),
- time range: a `Span` in the recording timeline,
- signal: one or more specific channels (`time_series_ids`).

One flat frozen dataclass, and the optional `span` is what gives it a shape. `key` / `value` / `unit` / `description` / `id` are
**instance fields**, so connectors author annotations directly (or subclass with field defaults for
reuse) and they round-trip without runtime class synthesis.

| `span` | Extra fields | Scope |
| --- | --- | --- |
| absent | `value` (required) | Whole sample, time-independent (condition, firmware, device, ticker). |
| `PointSpan` | — | One time offset, on specific signals or the whole sample. |
| `IntervalSpan` | — | A bounded region, on specific signals or the whole sample. |

Shared fields: `key: str`, `value: Any = None`, `unit: str | pint.Unit | None = None`,
`description: str | None = None`, `id: str` (auto uuid7). `unit` takes either a unit string
(`"years"`) or a `pint.Unit` (`ureg.millivolt`, stored as its canonical name); both are validated
against the shared registry on construction, and an unrecognized unit string raises `ValueError`. On the temporal shapes, `time_series_ids=None` means **trial-level** (the whole
sample); a non-empty tuple restricts the annotation to those channels (each id must match a
`TimeSeries.time_series_id` on the sample).

```python
from timenet.types import Annotation, IntervalSpan, PointSpan

# sample scope
Annotation(key="operating_hours", value=1200, unit="hours")

# time range on the whole sample (trial-level)
Annotation(key="artifact", span=IntervalSpan.seconds(10.0, 12.0))

# signal + time range: the vibration and current channels, seconds 5 to 6
Annotation(
    key="fault",
    value="bearing fault",
    span=IntervalSpan.seconds(5.0, 6.0, time_series_ids=("vibration", "current")),
)

# one time offset on a single channel
Annotation(key="impact", span=PointSpan.seconds(4.2, time_series_ids=("vibration",)))
```

A connector that emits the same key repeatedly can subclass with field defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class OperatingHours(Annotation):
    key: str = "operating_hours"
    unit: str | None = "hours"
```

`annotation_type_of(ann)` returns the `AnnotationType` (`STATIC` / `POINT` / `INTERVAL`);
There is no reverse mapping: one class covers every shape. `AnnotationDescriptor` is the type-level projection
(`key`, `annotation_type`, `value_type`, `unit`, `description`) hoisted into the schema and manifest at
write time.

---

## Tasks

A task is one labeled training target referencing one or more samples. The class is the type tag
(usable as a search filter, e.g. `search(task=AnswerTask)`); the instance carries the payload. Tasks
are mutable so [`add_task`](timef-dataset.md) can populate `sample_ids` after construction.

Every task is `inputs -> one typed answer`, and the shared frame lives on the `Task` base:

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Auto uuid7. |
| `sample_ids` | `tuple[str, ...]` | The samples the task is about; populated by `add_task`. |
| `prompt` | `str \| None` | What the model is asked; `None` for an unprompted task. |
| `scope` | `Span \| None` | The input region; `None` means the whole sample. |
| `input_annotation_ids` | `tuple[str, ...]` | Annotations given to the model as context. |
| `target` | typed per subclass | The answer, inline. |
| `target_annotation_ids` | `tuple[str, ...]` | The answer by reference to stored annotations. |
| `rationale` | `str \| None` | Chain of thought to train on. Any task may carry one. |
| `from_tasks` | `tuple[Task, ...]` | Source tasks this one derives from (plus a `from_task_ids` property). |

A subclass therefore adds only what makes its answer a different *kind* of thing:

| Class | `task_type` | Answer | Extra payload |
| --- | --- | --- | --- |
| `ClassificationTask` | `classification` | `target: str` (a label) | `target_schema` |
| `AnswerTask` | `answer` | `target: str` (free text) | — |
| `ScalarPredictionTask` | `scalar_prediction` | `target: float` | `unit`, `target_name` |
| `TemporalLocalizationTask` | `temporal_localization` | `target: tuple[Span, ...]` | `mode` |
| `ForecastingTask` | `forecasting` | a produced series | `context_sample_ids`, `target_sample_id`, `target_span` |
| `TSEditingTask` | `ts_editing` | a produced series | `source_sample_id`, `target_sample_id` |
| `TSGenerationTask` | `ts_generation` | a produced series | `target_sample_id` |
| `TSCorrespondenceTask` | `ts_correspondence` | `target: tuple[str, ...]` (sample ids) | `candidate_sample_ids` |

`TaskType` is the enum of type tags; `TASKS` is **derived** at import by walking the `Task` subclass
tree, so every concrete task in the module is registered by its `task_type` and two classes claiming the
same tag are rejected rather than silently collapsed. Unlike specs and annotations, task payloads are
fixed in code and resolved on read against `TASKS`, not reconstructed from the manifest.

The first four types carry a scalar-ish `target`, so generic training code reads `task.target` regardless
of type. The three series-output types are the exception: their answer is a *series*, so they set
`answer_is_sample` and point at the sample holding it instead of filling `target`.

### Span

`Span` is the geometry primitive shared by a task's `scope` and a localization target: a point
(`end=None`) or a half-open interval `[start, end)`, optionally scoped to `time_series_ids`
(`None` = every series). A span carries a `frame`. In the default `SpanFrame.SECONDS` its bounds are
whole microseconds on the **source recording timeline**, the same frame as a series' `time_axis`, so
two equal regions compare equal. In `SpanFrame.STEPS` they are step ordinals on the named series, the
only way to name a region of an ordinal series that has no timeline; `time_series_ids` is required
there, since step 5 is a different region on every series.

Build one on the shape you mean: `IntervalSpan` or `PointSpan`, each with `.seconds()` for the seconds
a recording documents itself in, `.micros()` when the source already has integers, `.from_datetime()`
for wall-clock moments, and `.steps()` for a series that counts in steps rather than time. The shape
is named at the call site rather than inferred from how many bounds you passed, so
`IntervalSpan.seconds(5.0)` is an error instead of a point that quietly claims to be an interval.

```python
# an interval on one series; start is stored as 5_000_000
IntervalSpan.seconds(5.0, 8.0, time_series_ids=("vibration",))

# a point, on every series in the sample
PointSpan.seconds(1.2)

# the same interval, written directly in microseconds
IntervalSpan.micros(5_000_000, 8_000_000)

# steps 0..12 on one ordinal series, which has no seconds to name
IntervalSpan.steps(0, 12, time_series_ids=("passengers",))
```

### Per-type payloads

- `ClassificationTask`: one categorical label, for the whole sample or for `scope`; `target_schema` names
  the vocabulary the target is drawn from (`None` for free-form).
  ```python
  dataset.add_task(
      sample, ClassificationTask(target="faulty", target_schema="condition")
  )
  dataset.add_task(
      sample,
      ClassificationTask(target="fault_episode", target_schema="condition"),
      scope=IntervalSpan.seconds(
          120.0, 480.0, time_series_ids=(vibration.time_series_id,)
      ),
  )
  ```
- `AnswerTask`: free text. Without a `prompt` it is a caption; with one it is a question answered, and a
  `rationale` adds the reasoning trace to supervise.
  ```python
  dataset.add_task(sample, AnswerTask(
      target="A 10-second vibration trace with a bearing-fault signature "
      "after 5 s.",
  ))
  dataset.add_task(sample, AnswerTask(
      prompt="What happens between 12s and 18s?",
      target="A bearing fault on the vibration channel.",
  ))
  ```
- `ScalarPredictionTask`: a numeric target that keeps its type. `unit` is validated against the shared
  pint registry (a `pint.Unit` is stored as its name); `target_name` names the quantity.
  ```python
  dataset.add_task(sample, ScalarPredictionTask(
      target=62.0, unit="bpm", target_name="mean_heart_rate"
  ))
  ```
- `TemporalLocalizationTask`: find the regions matching the prompt. `mode` is `SPARSE` (unmarked time is
  unlabeled) or `EXHAUSTIVE` (the spans must tile the region of interest; a gap is an error).
  ```python
  dataset.add_task(sample, TemporalLocalizationTask(
      prompt="Locate all R-peaks in lead II.",
      target=(
          PointSpan.seconds(1.20, time_series_ids=("II",)),
          PointSpan.seconds(2.05, time_series_ids=("II",)),
      ),
  ))
  ```
- `ForecastingTask`: predict a series' future values. The future is a whole separate sample
  (`target_sample_id`) or a region of the attached sample (`target_span`, an interval with an explicit
  `scope` for the context) — exactly one.
  ```python
  dataset.add_task(future, ForecastingTask(
      context_sample_ids=("rec_001::history",),
      target_sample_id="rec_001::future",
  ))
  ```
- `TSEditingTask` / `TSGenerationTask`: produce a series, from a source sample plus an instruction, or
  from the specification alone.
  ```python
  dataset.add_task(source, TSEditingTask(
      prompt="Remove the baseline wander.",
      source_sample_id="ecg-raw",
      target_sample_id="ecg-clean",
  ))
  dataset.add_task(spec_sample, TSGenerationTask(
      prompt="10 s of 150 bpm sinus tachycardia at 500 Hz.",
      target_sample_id="ecg-synth-0001",
  ))
  ```
- `TSCorrespondenceTask`: which candidate sample corresponds to the query. The answer must come from
  `candidate_sample_ids` when that pool is set.
  ```python
  dataset.add_task(query, TSCorrespondenceTask(
      prompt="Which recording is most similar to this one?",
      candidate_sample_ids=("rec-a", "rec-b"),
      target=("rec-b",),
  ))
  ```

### Composition (`from_tasks`)

A task can derive from earlier tasks (or from the annotations that motivated them) via `from_tasks`.
The derived task records the chain it was built from, which is how a handful of base labels multiply
into many higher-level training samples:

```python
base = dataset.add_task(sample, ClassificationTask(target="faulty"))
dataset.add_task(sample, AnswerTask(
    prompt="Is this machine healthy?",
    rationale="The trace is classified faulty: a bearing fault is present.",
    target="No.",
    from_tasks=(base,),
))
```

### Annotations vs tasks

An annotation is sample-level information; a task is a learning target. A connector can use the same
source annotation in either role:

- As task **input**, the annotation is fed to the model as grounding: list it in `input_annotation_ids`.
  An `Annotation` marking a bearing fault on the vibration channel over seconds 5 to 6 supplies
  the detail an `AnswerTask` prompt builds on.
- As the task **target**, either copy the information into the task payload, or point at the stored
  annotations with `target_annotation_ids` and leave `target` unset. The by-reference form avoids
  duplicating, say, a night of sleep-stage intervals into a task row.

A task gives its answer inline **or** by reference, never both; `add_task` rejects a task that sets both,
and one that sets neither unless its answer is a produced series. Because the two roles are separate
fields, a training adapter can tell context from answer instead of guessing, and will not leak a
target-derived annotation back to the model.

Because annotations carry signal and time-range scope, one recording yields many targets: a
whole-sample classification, scoped labels per channel, windowed questions, and follow-up tasks that
compose them via `from_tasks`.

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
manifest. Holds flat descriptors for specs / annotations and the real built-in `Task`
subclasses.

```python
DatasetSchema(
    time_series_specs: tuple[TimeSeriesSpec, ...] = (),
    annotations:       tuple[AnnotationDescriptor, ...] = (),
    tasks:             tuple[type[Task], ...] = (),
)
```

---

## Enums

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
negative `Version` component, a `unit_value` that isn't a frequency, an `Annotation` that ends
before it starts) and an *invalid input to the API* (a malformed dataset ref, a `dataset_id` that
isn't an `org/name` pair, a version supplied twice).
Because it subclasses `ValueError`, `except ValueError` keeps catching all of it.

Plain `ValueError` is reserved for genuine programming bugs rather than bad data or input:
two `Task` classes declaring the same `task_type` (a definition bug, raised at import). That is never
a data or input problem, so tagging it as a TimeF validation failure would make the distinction
useless.

---

See the [API reference for `timenet.types`](api/types.md) for the full symbol listing.
