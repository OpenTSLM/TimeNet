---
icon: lucide/shapes
description: "Reference for TimeF versions, units, specifications, spans, metadata, and errors."
tags:
  - reference
  - types
---

# Types

TimeF value types live in `timenet.types`. They are standard typed dataclasses and enums. The full
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
| `unit_value` | Physical unit of each value. |
| `dtype` | NumPy scalar dtype, `"str"`, or `"enum"`. |
| `categories` | Allowed labels for an enum specification. |
| `value_shape` | Shape of one timestep, without the time dimension. |
| `dimension_names` | Optional names for every dimension in `value_shape`. |
| `nullable` | Whether a complete timestep can be missing. |

The full logical shape is `(n_steps, *value_shape)`. Parquet stores scalar values. Use the Zarr
backend for multidimensional values.

### Missing values

Pass `None` for a missing timestep when `nullable=True`. Arrow stores missingness in a validity
bitmap. This keeps a missing value distinct from a valid floating-point NaN.

`Signal.to_arrow()` preserves nulls. `Signal.to_numpy()` rejects an array that contains nulls.
Use `Signal.to_numpy_and_mask()` when NumPy code needs nullable data. It returns filled values and a
boolean validity mask.

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

Time spans can cover a complete Record or selected Signals. Set `time_series_ids` to the selected
Signal IDs. Leave it as `None` to cover the complete owner.

Step spans always name exactly one Signal through `time_series_id`.

```python
from timenet.types import StepInterval, TimeInterval, TimePoint

event = TimePoint.seconds(1.2)
window = TimeInterval.seconds(
    5.0,
    8.0,
    time_series_ids=("vibration",),
)
tokens = StepInterval(time_series_id="text-series", start=132, stop=144)
```

Use `record.time_point()` or `record.time_interval()` to convert wall-clock values through a
Record's `start_time`.

## DatasetMetadata

`DatasetMetadata` contains the descriptive identity authored in `dataset.yaml`.

| Field | Required | Description |
| --- | --- | --- |
| `dataset_id` | yes | One `org/name` identifier. |
| `dataset_version` | yes | Semantic version of the source dataset. |
| `name` | yes | Display name. |
| `description` | yes | Short dataset description. |
| `license` | yes | Known SPDX-style license value. |
| `domains` | no | Searchable application domains. |
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

## Errors and warnings

TimeNet raises its own exceptions from `timenet.errors`.

| Exception | Meaning |
| --- | --- |
| `TimeNetError` | Base class for TimeNet failures. |
| `TimeNetRegistryError` | A registry cannot complete an operation. |
| `TimeNetDatasetNotFoundError` | A dataset ID or version does not exist. |
| `TimeFValidationError` | Input violates a TimeF invariant. |
| `TimeFFormatError` | A stored TimeF artifact is corrupt or unsupported. |
| `TimeNetInvalidManifestError` | `manifest.json` is malformed. |
| `TimeNetInvalidCardError` | `dataset.yaml` is malformed. |

`TimeFValidationError` also subclasses `ValueError`. Existing input-validation handlers can catch it
without losing the TimeNet-specific error type.

`SpanOutsideWindowWarning` reports a span outside a Signal window when the caller has chosen warning
behavior. It derives from `TimeNetWarning` and `UserWarning`.

See the [API reference for `timenet.types`](api/types.md) for all symbols.
