---
icon: lucide/shapes
description: "TimeF value types: versions, units, specs, tasks, annotations, and metadata."
tags:
  - reference
  - types
---

# Types

The TimeF value types live in `timenet.types`. Each concept has one module, re-exported from the
package. Every schema-carrying type is a **plain frozen dataclass**. Therefore it pickles and
round-trips through [`TimeFReader`](timef-reader.md) without runtime class synthesis. Errors live in
`timenet.errors`.

---

## Version

A semantic `major.minor.patch` version. It is frozen and ordered. Versions compare with the usual
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
every definition and conversion. It also adds two custom units (`beat`, `bpm`) that pint does not
ship. Reference units through `ureg` (`ureg.hertz`, `ureg.millivolt`, `ureg.standard_gravity`,
`ureg.dimensionless`). If you use a second registry, comparisons and conversions fail.

```python
from timenet.types import ureg

(5.0 * ureg.millivolt).to(ureg.volt).magnitude   # 0.005
```

`ureg` is **private to TimeNet**. An import of `timenet` does not call
`pint.set_application_registry`. Your own registry stays untouched. Everything TimeNet persists or
pickles stores units by *name* and rebuilds them against `ureg`. Therefore nothing in the format
depends on process-global pint state. The manifest codec writes `str(unit)`. `TimeSeriesSpec`
converts every `pint.Unit` attribute in `__getstate__`, even the ones a subclass adds.

That covers TimeNet's own types. It cannot cover a bare `pint.Unit` or `pint.Quantity` that you
pickle yourself. Those store only the unit name. They resolve it against pint's *application*
registry, which does not know `beat` or `bpm`:

```python
# UndefinedUnitError: 'bpm' is not defined
pickle.loads(pickle.dumps(ureg.bpm))
```

pint's application registry is the only hook for that. It is opt-in. A library must not apply it for
you on an import:

```python
from timenet.types import use_as_application_registry

use_as_application_registry()   # once, at application start
```

It is a global assignment, not a merge. The last call wins. Custom units from a registry you
installed before stop resolving. It fits one case only: you pickle bare units or quantities
yourself.

---

## DataSource

The origin that produced a modality: a device, an API feed, a model, or an institution. It is a flat
frozen dataclass that you build directly.

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

The contract for a measurement **modality**: its type tag, display name, value unit, scalar dtype,
and per-timestep shape. One spec is shared across every logical stream of a modality. The stream
identifier lives on [`TimeSeries.signal`](timef-dataset.md), not here.

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
| `spec_type` | `str` | yes | Dataset-unique modality tag (for example `"vibration"`). |
| `name` | `str` | yes | Human-readable modality label. |
| `unit_value` | `pint.Unit` | yes | Any unit (g, °C, mV, dimensionless, ...). |
| `data_source` | `DataSource \| None` | no | The source that produced this modality. |
| `dtype` | `str` | no | Canonical NumPy scalar dtype, `"str"` for text, or `"enum"` for a categorical value. Defaults to `"float32"`. |
| `value_shape` | `tuple[int, ...]` | no | Shape of one timestep, excluding time. `()` means scalar. |
| `dimension_names` | `tuple[str, ...]` | no | Optional names matching every dimension in `value_shape`. |
| `nullable` | `bool` | no | Whether a timestep can be missing. The default `False` rejects all nulls. |

The full logical array shape is `(n_steps, *value_shape)`. For example, an RGB frame stream can use
`dtype="uint8"`, `value_shape=(height, width, 3)`, and
`dimension_names=("height", "width", "color")`. Parquet stores scalar values of any `dtype`; scalar
values of a non-`"str"`/`"enum"` dtype preserve their NumPy type on disk, and `"str"`/`"enum"` values
read back as text. An `"enum"` dtype stores values as a PyArrow
dictionary array; the torch bridge maps them to integer codes.
Multidimensional values require the Zarr
values backend.

### Missing values

Python callers use `None` to supply a missing timestep when `nullable=True`.
Constructors reject `None` when `nullable=False`.
Arrow stores missingness in a validity bitmap, a separate bit for each timestep.
The numeric array does not contain Python objects.

For example, `[1.0, None, NaN]` has validity `[True, False, True]`.
The first position contains the measurement `1.0`. The second measurement is absent.
The third contains a floating-point value that represents an undefined numerical result.
TimeF preserves special floating-point values such as NaN and positive or negative infinity.
If a source uses NaN or another marker for missing measurements, its connector must translate those
markers into nulls.

Nullability applies to a whole timestep. A multidimensional value is all present or all missing.
The writer rejects partial nulls. The dtypes `int16`, `bool`, `str`, and `enum` also support nulls.

`TimeSeries.to_arrow()` preserves nulls. `TimeSeries.to_numpy()` raises `TimeFValidationError` when
the loaded array contains nulls. In some cases, the previous conversion lost the distinction between
missing values and NaN.
A nullable spec without actual nulls still supports `to_numpy()`. NaN and infinity remain valid values.

`TimeSeries.to_numpy_and_mask()` returns values and a validity mask, a boolean array that marks
present timesteps. Together, the values and mask preserve the missingness that Arrow stores.
Missing positions hold zero, false, or an empty string. These fill values are not observations.
The PyTorch dataset returns the same pair as `"series"` and `"series_masks"`.
Every series has a boolean mask, including an all-true mask for a non-nullable series.

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

An annotation is extra context on a [`Record`](timef-dataset.md). It is side information. A task can
read it as input, or it can become a task's question or answer. It has one of three scope levels, and
the scopes combine:

- record: the whole record (a static fact, or a trial-level temporal marker),
- time range: a time span (`TimePoint` or `TimeInterval`) in the recording timeline,
- signal: one or more specific signals (`time_series_ids`).

It is one flat frozen dataclass. The optional `span` gives it a shape. `key` / `value` / `unit` /
`description` / `id` are **instance fields**. Therefore connectors author annotations directly, or
subclass with field defaults for reuse. The annotations round-trip without runtime class synthesis.

| `span` | Extra fields | Scope |
| --- | --- | --- |
| absent | `value` (required) | Whole record, time-independent (a subject's age, condition, firmware, device, ticker). |
| `TimePoint` | none | One time offset, on specific signals or the whole record. |
| `TimeInterval` | none | A bounded region, on specific signals or the whole record. |

Shared fields: `key: str`, `value: Any = None`, `unit: str | pint.Unit | None = None`,
`description: str | None = None`, `id: str` (auto uuid7). `unit` takes either a unit string
(`"years"`) or a `pint.Unit` (`ureg.millivolt`, stored as its canonical name). TimeNet validates both
against the shared registry on construction. An unrecognized unit string raises `ValueError`. On the
temporal shapes, `time_series_ids=None` means **trial-level**, that is the whole record. A non-empty
tuple restricts the annotation to those signals. Each id must match a `TimeSeries.time_series_id` on
the record.

```python
from timenet.types import Annotation, TimeInterval, TimePoint

# record scope
Annotation(key="operating_hours", value=1200, unit="hours")

# time range on the whole record (trial-level)
Annotation(key="artifact", span=TimeInterval.seconds(10.0, 12.0))

# signal + time range: the vibration and current signals, seconds 5 to 6
Annotation(
    key="fault",
    value="bearing fault",
    span=TimeInterval.seconds(5.0, 6.0, time_series_ids=("vibration", "current")),
)

# one time offset on a single signal
Annotation(key="impact", span=TimePoint.seconds(4.2, time_series_ids=("vibration",)))
```

A connector that emits the same key repeatedly can subclass with field defaults:

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class OperatingHours(Annotation):
    key: str = "operating_hours"
    unit: str | None = "hours"
```

`annotation_type_of(ann)` returns the `AnnotationType` (`STATIC` / `POINT` / `INTERVAL`). There is no
reverse mapping, because one class covers every shape. `AnnotationDescriptor` is the type-level
projection (`key`, `annotation_type`, `value_type`, `unit`, `description`). TimeNet hoists it into the
schema and manifest at write time.

---

## Tasks

A task is one labeled training target. It references one or more records. The class is the type tag.
You can use it as a search filter, for example `search(task=AnswerTask)`. The instance carries the
payload. Tasks are mutable, so [`add_task`](timef-dataset.md) can populate `record_ids` after
construction.

Every task is `inputs -> one typed answer`, and the shared frame lives on the `Task` base:

| Field | Type | Description |
| --- | --- | --- |
| `id` | `str` | Auto uuid7. |
| `record_ids` | `tuple[str, ...]` | The records the task is about. `add_task` can populate this after construction. |
| `prompt` | `str \| None` | What the model is asked. `None` means an unprompted task. |
| `scope` | `Span \| None` | The input region. `None` means the whole record. |
| `input_annotation_ids` | `tuple[str, ...]` | Annotations given to the model as context. |
| `target` | typed per subclass | The answer, inline. |
| `target_annotation_ids` | `tuple[str, ...]` | The answer by reference to stored annotations. |
| `rationale` | `str \| None` | Chain of thought to train on. Any task can carry one. |
| `from_tasks` | `tuple[Task, ...]` | Source tasks this one derives from (plus a `from_task_ids` property). |

A subclass therefore adds only what makes its answer a different *kind* of thing:

| Class | `task_type` | Answer | Extra payload |
| --- | --- | --- | --- |
| `ClassificationTask` | `classification` | `target: str` (a label) | `target_schema` |
| `AnswerTask` | `answer` | `target: str` (free text) | none |
| `ScalarPredictionTask` | `scalar_prediction` | `target: float` | `unit`, `target_name` |
| `TemporalLocalizationTask` | `temporal_localization` | `target: tuple[TimePoint \| TimeInterval, ...]` | `mode` |
| `ForecastingTask` | `forecasting` | a produced series | `context_record_ids`, `target_record_id`, `target_span` |
| `TSEditingTask` | `ts_editing` | a produced series | `source_record_id`, `target_record_id` |
| `TSGenerationTask` | `ts_generation` | a produced series | `target_record_id` |
| `TSCorrespondenceTask` | `ts_correspondence` | `target: tuple[str, ...]` (record ids) | `candidate_record_ids` |

`TaskType` is the enum of type tags. TimeNet derives `TASKS` at import from a walk of the `Task`
subclass tree. Therefore it registers every concrete task in the module by its `task_type`. If two
classes claim the same tag, TimeNet rejects them instead of a silent collapse. Task payloads are
fixed in code, unlike specs and annotations. TimeNet resolves them on read against `TASKS`, not from
the manifest.

The first four types carry a scalar-ish `target`. Therefore generic training code reads `task.target`
for any of them. The three series-output types are the exception. Their answer is a *series*. They
set `answer_is_record` and point at the record that holds it, instead of a value in `target`.

### Span

A task's `scope` and a localization target are **spans**: one region a task or annotation localizes. A
span has a shape (a point, or a half-open interval) and a **frame**, and the frame is the type. The
frame decides what the bounds mean. A **time** frame reads them as microseconds on the source recording
timeline. A **step** frame reads them as ordinal indices into one series' own array. `Span` is the base
you annotate with for any span. It and the frame bases `TimeSpan` / `StepSpan` are abstract, so you
always build a concrete leaf.

Time spans sit on the recording timeline, the same frame as a series' `time_axis`, so a bound stays
meaningful on a windowed record that starts partway into the recording. `TimePoint(start_us=...)` is one
point. `TimeInterval(start_us=..., end_us=...)` is the half-open range `[start_us, end_us)`. Either
covers the whole record, or a subset of series named by `time_series_ids` (`None` = every series). Build
with `.seconds()` for the seconds a recording documents itself in, or `.micros()` when the source already
has integers. For a wall-clock moment, build it from the record (`record.time_point(at)` /
`record.time_interval(start, end)`), which supplies its own `start_time` as the anchor. Bounds are stored
as whole microseconds, so two equal regions compare equal.

Step spans count a series' own ordinal positions, for a series that has no clock at all. A step index
means nothing without a series to count on, so a step span names exactly one `time_series_id` (a single
str). `StepPoint(time_series_id=..., start=...)` is one step.
`StepInterval(time_series_id=..., start=..., stop=...)` is the half-open range `[start, stop)`. There
are no unit builders. Construct them directly.

Which frame fits is decided by the series' **axis**, not by the caller: a timeline axis (regular or
irregular) takes a time span, an ordinal axis takes a step span. [`add_task`](timef-dataset.md) checks a
span against the axis of every series it names and rejects a mismatch.

```python
# a time interval on one series; start is stored as 5_000_000
TimeInterval.seconds(5.0, 8.0, time_series_ids=("vibration",))

# a time point, on every series in the record
TimePoint.seconds(1.2)

# a region of an ordinal series (order only, no clock, like TSQA):
# the interval covering steps 132 to 143
StepInterval(time_series_id="tsqa", start=132, stop=144)
```

### Per-type payloads

- `ClassificationTask`: one categorical label, for the whole record or for `scope`. `target_schema`
  names the vocabulary of the target (`None` for free-form).
  ```python
  dataset.add_task(
      record, ClassificationTask(target="faulty", target_schema="condition")
  )
  dataset.add_task(
      record,
      ClassificationTask(
          target="fault_episode",
          target_schema="condition",
          scope=TimeInterval.seconds(
              120.0, 480.0, time_series_ids=(vibration.time_series_id,)
          ),
      ),
  )
  ```
- `AnswerTask`: free text. Without a `prompt`, it is a caption. With a `prompt`, it is an answer to a
  question. A `rationale` adds the reasoning trace to supervise.
  ```python
  dataset.add_task(record, AnswerTask(
      target="A 10-second vibration trace with a bearing-fault signature "
      "after 5 s.",
  ))
  dataset.add_task(record, AnswerTask(
      prompt="What happens between 12s and 18s?",
      target="A bearing fault on the vibration signal.",
  ))
  ```
- `ScalarPredictionTask`: a numeric target that keeps its type. TimeNet validates `unit` against the
  shared pint registry, and stores a `pint.Unit` as its name. `target_name` names the quantity.
  ```python
  dataset.add_task(record, ScalarPredictionTask(
      target=62.0, unit="bpm", target_name="mean_heart_rate"
  ))
  ```
- `TemporalLocalizationTask`: find the regions that match the prompt. `mode` is `SPARSE` or
  `EXHAUSTIVE`. `SPARSE` leaves unmarked time unlabeled. `EXHAUSTIVE` needs the spans to tile the
  region of interest, and a gap is an error.
  ```python
  dataset.add_task(record, TemporalLocalizationTask(
      prompt="Locate all R-peaks in lead II.",
      target=(
          TimePoint.seconds(1.20, time_series_ids=("II",)),
          TimePoint.seconds(2.05, time_series_ids=("II",)),
      ),
  ))
  ```
- `ForecastingTask`: predict a series' future values. The future is a whole separate record
  (`target_record_id`) or a region of the attached record (`target_span`, an interval with an explicit
  `scope` for the context). Exactly one of the two is required.
  ```python
  dataset.add_task(future, ForecastingTask(
      context_record_ids=("rec_001::history",),
      target_record_id="rec_001::future",
  ))
  ```
- `TSEditingTask` / `TSGenerationTask`: produce a series. `TSEditingTask` uses a source record plus an
  instruction. `TSGenerationTask` uses the specification alone.
  ```python
  dataset.add_task(source, TSEditingTask(
      prompt="Remove the baseline wander.",
      source_record_id="ecg-raw",
      target_record_id="ecg-clean",
  ))
  dataset.add_task(spec_record, TSGenerationTask(
      prompt="10 s of 150 bpm sinus tachycardia at 500 Hz.",
      target_record_id="ecg-synth-0001",
  ))
  ```
- `TSCorrespondenceTask`: which candidate record corresponds to the query. If that pool is set, the
  answer must come from `candidate_record_ids`.
  ```python
  dataset.add_task(query, TSCorrespondenceTask(
      prompt="Which recording is most similar to this one?",
      candidate_record_ids=("rec-a", "rec-b"),
      target=("rec-b",),
  ))
  ```

### Composition (`from_tasks`)

A task can derive from earlier tasks, or from the annotations that motivated them, via `from_tasks`.
The derived task records its source chain. This is how a handful of base labels multiply into many
higher-level training examples:

```python
base = ClassificationTask(target="faulty")
dataset.add_tasks(record, [
    base,
    AnswerTask(
        prompt="Is this machine healthy?",
        rationale="The trace is classified faulty: a bearing fault is present.",
        target="No.",
        from_tasks=(base,),
    ),
])
```

### Annotations vs tasks

An annotation is record-level information. A task is a learning target. A connector can use the same
source annotation in either role:

- As task **input**, the annotation grounds the model: list it in `input_annotation_ids`. An
  `Annotation` can mark a bearing fault on the vibration signal over seconds 5 to 6. It then supplies
  the detail for an `AnswerTask` prompt.
- As the task **target**, copy the information into the task payload, or point at the stored
  annotations with `target_annotation_ids` and leave `target` unset. The by-reference form avoids a
  copy of, for example, a night of sleep-stage intervals into a task row.

A task gives its answer inline **or** by reference, never both. `add_task` rejects a task that sets
both. It also rejects a task that sets neither, unless its answer is a produced series. The two roles
are separate fields. Therefore a training adapter can tell context from answer, and will not leak a
target-derived annotation back to the model.

Annotations carry signal and time-range scope. Therefore one recording yields many targets: a
whole-record classification, scoped labels per signal, windowed questions, and follow-up tasks that
compose them via `from_tasks`.

---

## DatasetMetadata

A dataset's descriptive identity (authored in the card).

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | `str` | yes | `org/name` pair (one slash) that matches the card / connector module path. |
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

A dataset's type declaration. TimeNet derives it from the data, never by hand, then serializes it
into the manifest. It holds flat descriptors for specs and annotations, and the real built-in `Task`
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

## Errors and warnings

`timenet.errors` defines the exception hierarchy. `TimeNetError` is the base. Validation and manifest
errors also derive from `ValueError`, so existing handlers keep working. It also defines the warning
hierarchy, listed at the end of this section.

| Exception | Base(s) | Raised when |
| --- | --- | --- |
| `TimeNetError` | `Exception` | base for all TimeNet errors |
| `TimeNetRegistryError` | `TimeNetError` | a registry cannot be loaded/reached/served |
| `TimeNetDatasetNotFoundError` | `TimeNetError` | an unknown dataset id/version |
| `TimeFValidationError` | `TimeNetError`, `ValueError` | a dataset/array violates a TimeF invariant |
| `TimeFFormatError` | `TimeNetError` | a corrupt or unsupported on-disk artifact |
| `TimeNetInvalidManifestError` | `TimeFFormatError`, `ValueError` | a malformed `manifest.json` |

`TimeFValidationError` covers two cases. The first is a value for storage in a dataset that violates
an invariant. Examples: a negative `Version` component, a `unit_value` that is not a frequency, or an
`Annotation` that ends before it starts. The second is an invalid input to the API. Examples: a
malformed dataset ref, a `dataset_id` that is not an `org/name` pair, or a version supplied twice. It
subclasses `ValueError`, so `except ValueError` keeps catching all of it.

Plain `ValueError` is for genuine programming bugs, not bad data or input. One example: two `Task`
classes declare the same `task_type`, a definition bug raised at import. That is never a data or input
problem. A tag of TimeF validation failure on it makes the distinction useless.

`timenet.errors` also defines the warnings TimeNet raises. `TimeNetWarning` is the base, and it
derives from `UserWarning`. `SpanOutsideWindowWarning` is raised where a span leaves its window and
is kept rather than refused. Filter it by type to silence a source that states such a region for
every recording.

| Warning | Base(s) | Warned when |
| --- | --- | --- |
| `TimeNetWarning` | `UserWarning` | base for all TimeNet warnings |
| `SpanOutsideWindowWarning` | `TimeNetWarning` | a span leaves its window and is kept |

---

See the [API reference for `timenet.types`](api/types.md) for the full symbol listing.
