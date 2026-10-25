---
icon: lucide/shapes
description: "Reference for TimeF versions, units, specifications, spans, metadata, and errors."
tags:
  - reference
  - types
---

# Types

TimeF value types live in `timenet.types`. Wire-facing types are Pydantic models; other runtime
objects use typed dataclasses and enums. The full
object model is split across these focused pages:

- [TimeFDataset](timef-dataset.md) covers the complete in-memory hierarchy.
- [Annotations](data-model/annotations.md) covers reusable content and occurrences.
- [Tasks](data-model/tasks.md) covers modelling problems and targets.

This page describes the shared value types used by those objects.

## Version

`Version` is an ordered semantic version with non-negative `major`, `minor`, and `patch` parts.

```python
from timenet.types import Version

version = Version.parse("1.2.3")
assert str(version) == "1.2.3"
assert version > Version(1, 2, 0)
```

## IDs

Sources, axes, annotation contents, and annotation occurrences have no ID in Python. The writer
numbers their rows each time it writes a version. It stores equal annotation contents once, and
equal regular or ordinal axes once. Signals share an irregular axis only if they use the same axis
object.

Two annotation contents are equal if the name, the value and its type, the unit, the description,
and the metadata are equal. `0.0` and `-0.0` are equal. All NaN values are equal. `True`, `1`, and
`1.0` are not equal.

## Units

TimeNet uses one [pint](https://pint.readthedocs.io) registry named `ureg`. Use this registry for
units stored in TimeF objects. A second registry creates units that do not compare reliably with
TimeNet units.

```python
from timenet.types import ureg

millivolts = 5.0 * ureg.millivolt
volts = millivolts.to(ureg.volt)
```

TimeNet serializes unit names and restores them through `ureg`. It does not replace pint's global
application registry during import.

`TimeSeriesSpec.unit_value` is required. Set it to `None` when the source does not establish a
unit. Use `ureg.dimensionless` only when the values have a known dimensionless meaning.

Call `use_as_application_registry()` only when your application pickles bare pint units or
quantities. This call changes process-wide pint state.

```python
from timenet.types import use_as_application_registry

use_as_application_registry()
```

## TimeSeriesSpec

`TimeSeriesSpec` describes one measurement modality. A Signal stores its own identity, display
name, axis, and values. Several Signals can share one specification.

```python
from timenet.types import TimeSeriesSpec, ureg

vibration = TimeSeriesSpec(
    spec_type="vibration",
    name="Vibration",
    unit_value=ureg.standard_gravity,
    dtype="float32",
)
```

| Field | Description |
| --- | --- |
| `spec_type` | Dataset-unique modality tag. |
| `name` | Human-readable modality name. |
| `unit_value` | Physical unit of each value, or `None` when unknown. |
| `dtype` | NumPy scalar dtype, `"str"`, or `"enum"`. |
| `categories` | Allowed labels for an enum specification. |
| `value_shape` | Shape of one timestep, without the time dimension. |
| `dimension_names` | Optional names for every dimension in `value_shape`. |
| `nullable` | Whether a complete timestep can be missing. |
| `modality` | Input kind used for Task inference. Defaults to `TIME_SERIES`; also accepts `IMAGE`, `AUDIO`, or `TEXT`. |

The full logical shape is `(n_steps, *value_shape)`. Parquet stores scalar values. Use the Zarr
backend for multidimensional values.

### Missing values

Pass `None` for a missing timestep when `nullable=True`. Arrow stores missingness in a validity
bitmap. This keeps a missing value distinct from a valid floating-point NaN.

`Signal.to_arrow()` preserves nulls. `Signal.to_numpy()` rejects an array that contains nulls.
The PyTorch adapter also rejects actual nulls, even when `nullable=True`. Consumers must handle
missing data before tensor conversion.

Nullability applies to the complete timestep. TimeF rejects a partially missing multidimensional
value.

## Spans

A span identifies one region for an annotation or task. Its type defines both its shape and its
coordinate frame.

| Type | Meaning |
| --- | --- |
| `TimePoint` | One microsecond position on a recording timeline. |
| `TimeInterval` | One half-open time range, `[start_us, end_us)`. |
| `StepPoint` | One ordinal position in a Signal without a clock. |
| `StepInterval` | One half-open ordinal range, `[start, stop)`. |

Task time spans can select Signals through `time_series`, a tuple of Signal objects. Each task span
must identify exactly one input Record. Annotation spans cannot select Signals. The owner of an
annotation sets its scope.

Step spans always name exactly one Signal through `time_series`. The writer stores the IDs of these
Signals. The reader replaces these IDs with the Signals that it reads back.

```python
from timenet.types import StepInterval, TimeInterval, TimePoint

event = TimePoint.seconds(1.2)
window = TimeInterval.seconds(
    5.0,
    8.0,
    time_series=(vibration,),
)
tokens = StepInterval(time_series=text_series, start=132, stop=144)
```

Use `record.time_point()` or `record.time_interval()` to convert wall-clock values through a
Record's known origin. Relative spans also work when its absolute timestamp is unknown.

## DatasetMetadata

`DatasetMetadata` contains the descriptive identity authored in `dataset.yaml`.

| Field | Required | Description |
| --- | --- | --- |
| `dataset_id` | yes | One `org/name` identifier. |
| `dataset_version` | yes | Semantic version of the source dataset. |
| `name` | yes | Display name. |
| `description` | yes | Short dataset description. |
| `license` | yes | Known SPDX-style license value, `OTHER`, or `NONE`. |
| `domains` | no | Searchable application domains. See [Domains](#domains). |
| `tags` | no | Searchable free-form labels. |
| `source_url` | no | Canonical source location. |
| `license_url` | conditional | License text when `license` is `OTHER`. |
| `citation` | no | Citation requested by the source. |
| `access` | no | `OPEN`, `CREDENTIALED`, or `RESTRICTED`. |
| `access_url` | conditional | Access page for non-open data. |

`DatasetMetadata.from_yaml()` validates a card against the packaged JSON Schema.

## DatasetSchema

`DatasetSchema` is the type summary that TimeNet derives from a populated dataset. Do not author it
by hand. The schema contains:

- one `TimeSeriesSpec` for each measurement modality;
- one `AnnotationDescriptor` for each annotation key and shape;
- the concrete Task classes present in the dataset.

The writer stores this schema in `manifest.json`. The reader uses it to reconstruct the same public
types.

## Enums

The main string enums are:

- `Domain` for application areas such as health, sleep, finance, and general data;
- `License` for supported SPDX-style identifiers;
- `Access` for open, credentialed, and restricted data;
- `TaskType` for the built-in task classes;
- `LocalizationMode` for sparse or exhaustive temporal localization.

They are `StrEnum` values and compare equal to their stored string values.

### Domains

A card declares every domain that a substantial part of its records belongs to.
Finer labels, such as a sensor or a benchmark name, belong in `tags`.

The **Covers** column presents typical examples, and does not claim to be an exhaustive list.

| Value | Covers |
| --- | --- |
| `health` | Human physiology and clinical care. Declare it alongside `cardiology`, `respiratory`, and `sleep`. |
| `cardiology` | The heart: ECG, heart rate, heart sounds. |
| `respiratory` | Breathing and the lungs: respiration, airflow, oxygen saturation, cough. |
| `sleep` | Sleep studies: polysomnography and sleep staging. |
| `activity` | Labeled human activities and daily behavior. Add `motion` when the signals are the movement. |
| `motion` | Movement of a body or object: IMU channels and motion capture. |
| `economics` | Macroeconomic indicators, prices, sales, and demand. |
| `finance` | Asset prices, returns, trading volumes, and exchange rates. |
| `environment` | Weather, climate, air quality, and hydrology. |
| `energy` | Electricity load, generation, power grids, and transformers. |
| `transport` | Transport networks: road traffic, congestion, ridership. Not a vehicle's own machinery. |
| `observability` | Software and IT infrastructure: service metrics and cloud monitoring. |
| `audio` | Recorded sound as content. |
| `industrial` | Machines and industrial processes: condition monitoring, faults, remaining useful life. |
| `general` | Synthetic, demo, or generic data without a stated field. |

## Errors and warnings

TimeNet raises its own exceptions from `timenet.errors`.

| Exception | Meaning |
| --- | --- |
| `TimeNetError` | Base class for TimeNet failures. |
| `TimeNetRegistryError` | A registry cannot complete an operation. |
| `TimeNetDatasetNotFoundError` | A dataset ID or version does not exist. |
| `TimeFValidationError` | Input violates a TimeF invariant. |
| `TimeFFormatError` | A stored TimeF artifact is corrupt or unsupported. |
| `TimeNetInvalidCardError` | `dataset.yaml` cannot be read or is not a mapping. |

Pydantic `ValidationError` reports field and schema errors in a dataset card. The error includes the
native field location and details.

Manifest parsing uses the same error for invalid JSON and invalid fields.

`TimeFValidationError` also subclasses `ValueError`. Existing input-validation handlers can catch it
without losing the TimeNet-specific error type.

`SpanOutsideWindowWarning` reports a span outside a Signal window when the caller has chosen warning
behavior. It derives from `TimeNetWarning` and `UserWarning`.

See the [API reference for `timenet.types`](api/types.md) for all symbols.
