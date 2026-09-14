---
icon: lucide/shapes
description: "TimeF value types: versions, units, specs, metadata, enums, and errors."
tags:
  - reference
  - types
---

# Types

The TimeF value types live in `timenet.types`. Each concept has one module, re-exported from the
package. Every type here is a **plain frozen dataclass**, so it pickles and compares by value with
no runtime class synthesis, which is what makes multiprocessing `DataLoader` workers safe. Errors
live in `timenet.errors`.

The types a version actually stores are `TimeSeriesSpec`, `DataSource`, `DatasetMetadata`, and
`Version`. The [data model](data-model/index.md) pages cover the hierarchy itself: records, sources,
signals, annotations, and tasks all live in `timenet.control_plane`.

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
and per-timestep shape. One spec is shared across every signal of a modality. A signal's own name
and id live on the [`Signal`](data-model/signals.md), not here.

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

A signal's values are a 1-D NumPy array, and the dtype it carries is the dtype that comes back out.
The values plane preserves it on disk, and the control plane records it on the spec.

`value_shape` and `dimension_names` describe a per-timestep shape wider than a scalar, and `nullable`
claims that a timestep can be absent. The control plane stores `spec_type`, `name`, `unit`, `dtype`,
and `nullable` per modality; it does not store a shape, and neither values backend writes a validity
mask today. A consumer that wants a per-timestep mask therefore has nothing to build one from, which
is why `timenet.torch` leaves `series_masks` empty rather than asserting an all-present mask that
nothing on disk backs.

A builder that reuses a modality can subclass with field defaults:

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

## Annotations, tasks, and spans

`timenet.types` also carries `Annotation`, the `Task` subclass tree with its `TASKS` registry, and
the `Span` types (`TimePoint`, `TimeInterval`, `StepPoint`, `StepInterval`). These are the
type-level vocabulary a `DatasetSchema` declares, and the manifest codec serializes them.

They are not what a version stores. The control plane has its own
[`Annotation`](data-model/annotations.md), which is a `(name, value, unit)` payload plus a span type
of `static`, `point`, or `interval`, and its own [`Task`](data-model/tasks.md), which is a prompt
with ordered input and target items. Build a dataset with those, from `timenet.control_plane`.

`annotation_type_of(ann)` returns the `AnnotationType` of a `timenet.types.Annotation`
(`STATIC` / `POINT` / `INTERVAL`). `AnnotationDescriptor` is its type-level projection.

---

## DatasetMetadata

A dataset's descriptive identity. It names the version the writer commits, and it is the block a
consumer reads out of the manifest before it downloads anything.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset_id` | `str` | yes | `org/name` pair (one slash). It chooses the output directory. |
| `dataset_version` | `Version` | yes | The upstream source's semantic version. |
| `name` | `str` | yes | Display name. |
| `description` | `str` | yes | One-sentence description. |
| `license` | `License` | yes | SPDX-style license id. |
| `domains` | `tuple[Domain, ...]` | no | Application/clinical domains. |
| `tags` | `tuple[str, ...]` | no | Free-form labels. |
| `source_url` | `str \| None` | no | Canonical source URL. |
| `license_url` | `str \| None` | no | Where the upstream license is stated. |
| `citation` | `str \| None` | no | How to cite the source. |
| `access` | `Access` | no | `OPEN`, `CREDENTIALED`, or `RESTRICTED`. Defaults to `OPEN`. |
| `access_url` | `str \| None` | no | Where to obtain access, for a non-open dataset. |
| `yaml_schema_version` | `int` | no | The card's field-schema version (default `1`). |

A non-`OPEN` dataset cannot be redistributed, so a hosted registry refuses to serve its bytes and
the client raises `TimeNetAccessError` pointing at `access_url`.

---

## DatasetSchema

A dataset's type declaration: flat descriptors for the specs and annotations it uses, plus the
built-in `Task` subclasses it carries. The [manifest](manifest.md) serializes it directly, so there
is no separate set of entry types to keep in sync.

```python
DatasetSchema(
    time_series_specs: tuple[TimeSeriesSpec, ...] = (),
    annotations:       tuple[AnnotationDescriptor, ...] = (),
    tasks:             tuple[type[Task], ...] = (),
)
```

The control plane answers the same questions from the database itself, through the `specs` table and
`reader.counts()`, so `TimeFWriter` leaves the manifest's schema block empty. Registry filters that
read it, `search(task=...)` and `search(time_series_spec=...)`, therefore match nothing on a version
this writer produced.

---

## Enums

- `Domain`: `HEALTH`, `CARDIOLOGY`, `RESPIRATORY`, `SLEEP`, `ACTIVITY`, `MOTION`, `ECONOMICS`,
  `FINANCE`, `ENVIRONMENT`, `ENERGY`, `TRANSPORT`, `OBSERVABILITY`, `AUDIO`, `GENERAL`.
- `License`: SPDX-style identifiers (`MIT`, `Apache-2.0`, `CC-BY-4.0`, `CC0-1.0`, ...).
- `Access`: `OPEN`, `CREDENTIALED`, `RESTRICTED`.
- `AxisType`: `REGULAR`, `IRREGULAR`, `ORDINAL`, in `timenet.dataset`.

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
| `TimeNetAccessError` | `TimeNetError` | a non-open dataset is asked of a registry that does not host its data |
| `TimeNetDownloadError` | `TimeNetError` | a fetch fails or lands bytes that do not match the manifest |
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
