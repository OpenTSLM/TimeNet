# Registry

In-memory dataset catalog. Maps each dataset ID to a `RegistryEntry` — the connector instance plus its optional [precomputed schema](types.md#precomputed-schema). Built once at construction time from one or two YAML files; read-only after that.

---

## At a glance

```python
from dataclasses import dataclass
from pathlib import Path

from timenet.connectors.base import BaseConnector
from timenet.models import QueryCriteria
from timenet.timef.metadata import DatasetMetadata, DatasetSchema


@dataclass(frozen=True)
class RegistryEntry:
    connector: BaseConnector
    precomputed_schema: DatasetSchema | None = None


class DatasetRegistry:
    def __init__(self, entries: dict[str, RegistryEntry]) -> None: ...

    @classmethod
    def from_config(
        cls,
        default_path: Path,
        custom_path: Path | None = None,
    ) -> "DatasetRegistry": ...

    def all(self)    -> list[DatasetMetadata]: ...
    def get(self, dataset_id: str) -> BaseConnector | None: ...
    def ids(self)    -> list[str]: ...
    def filter(self, criteria: QueryCriteria) -> list[DatasetMetadata]: ...
```

---

## Catalog format

```yaml
datasets:
  mit_bih_arrhythmia:
    class: timenet.connectors.MITBIHConnector
    precomputed_schema: timenet/registry/precomputed/mit_bih_arrhythmia.json
  synthetic_ecg:
    class: timenet.connectors.SyntheticECGConnector
```

| Field                     | Type   | Description                                                                                                                                                                                                                                                                        |
| ------------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datasets`                | map    | Top-level key. Each child key is the dataset ID used throughout the system.                                                                                                                                                                                                        |
| `<id>.class`              | string | Fully-qualified Python path to the connector class.                                                                                                                                                                                                                                |
| `<id>.precomputed_schema` | string | **Optional.** Path to a hand-written [precomputed schema](types.md#precomputed-schema) JSON (shape = `manifest.json["schema"]`). Lets the registry answer type-filters for this dataset before any build. Only connectors created by Timenet's mantainers are expected to have one |

The default catalog ships at `timenet/registry/default_datasets.yaml`. A custom catalog is **additive only**: it cannot override or remove entries from the default.

---

## Methods

### `from_config()` { data-toc-label='from_config()' }

```python
@classmethod
def from_config(
    cls,
    default_path: Path,
    custom_path: Path | None = None,
) -> "DatasetRegistry": ...
```

Builds the registry by loading `default_path`, optionally merging `custom_path`, then importing and instantiating each connector class.

**Parameters**

| Name           | Type           | Default  | Description                                                                              |
| -------------- | -------------- | -------- | ---------------------------------------------------------------------------------------- |
| `default_path` | `Path`         | required | Path to the built-in catalog shipped with TimeNet.                                       |
| `custom_path`  | `Path \| None` | `None`   | User-supplied catalog. Entries are merged after the default; duplicate IDs are rejected. |

**Returns:** A fully-loaded `DatasetRegistry`.

**Raises**

| Exception           | Condition                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| `FileNotFoundError` | Either path does not exist.                                                                        |
| `yaml.YAMLError`    | A YAML file is malformed.                                                                          |
| `ImportError`       | The module portion of a `class:` value cannot be imported.                                         |
| `AttributeError`    | The class portion of a `class:` value is not present on the imported module.                       |
| `TypeError`         | The connector class requires constructor arguments.                                                |
| `ValueError`        | `metadata().dataset_id` does not match its YAML key, or a duplicate ID is found across both files. |
| `FileNotFoundError` | A `precomputed_schema` path is set but the file does not exist.                                    |
| `ValueError`        | A `precomputed_schema` file is present but cannot be parsed into a `DatasetSchema`.                |

**Loading sequence**

For each entry, in order (`default_path` first, `custom_path` second):

1. Import the dotted class path.
2. Instantiate with no arguments: `Cls()`.
3. Assert `instance.metadata().dataset_id == yaml_key`, the connector is the source of truth for its own ID.
4. Assert the ID is not already registered.
5. If `precomputed_schema` is set, read and parse the JSON file into a `DatasetSchema` (otherwise `None`).
6. Store a `RegistryEntry(connector, precomputed_schema)` under the dataset ID. `filter()` reaches the schema through `entry.precomputed_schema` for type queries.

**Example**

```python
registry = DatasetRegistry.from_config(
    default_path=Path("timenet/registry/default_datasets.yaml"),
    custom_path=Path("my_datasets.yaml"),  # optional
)
```

---

### `all()` { data-toc-label='all()' }

```python
def all(self) -> list[DatasetMetadata]: ...
```

Returns descriptors for every registered dataset.

**Returns:** Descriptors for every registered dataset, in registration order (default entries first, then custom).

---

### `get()` { data-toc-label='get()' }

```python
def get(self, dataset_id: str) -> BaseConnector | None: ...
```

Looks up a single connector by its dataset ID, unwrapping it from the dataset's `RegistryEntry`.

**Parameters**

| Name         | Type  | Default  | Description                                        |
| ------------ | ----- | -------- | -------------------------------------------------- |
| `dataset_id` | `str` | required | The YAML key / `metadata().dataset_id` to look up. |

**Returns:** The connector instance, or `None` if the ID is not registered.

---

### `ids()` { data-toc-label='ids()' }

```python
def ids(self) -> list[str]: ...
```

Returns the IDs of every registered dataset.

**Returns:** All registered dataset IDs, sorted alphabetically.

---

### `filter()` { data-toc-label='filter()' }

```python
def filter(self, criteria: QueryCriteria) -> list[DatasetMetadata]: ...
```

Returns the descriptors of datasets that match a `QueryCriteria`.

**Parameters**

| Name       | Type            | Default  | Description                                                        |
| ---------- | --------------- | -------- | ------------------------------------------------------------------ |
| `criteria` | `QueryCriteria` | required | Filter specification. Fields are ANDed; `None` fields are ignored. |

**Returns:** Descriptors of datasets that match every non-`None` field in `criteria`.

---

## `QueryCriteria`

Filter specification for `filter()` and the SDK's `query()` / `list()`. Each field is independent: fields set to `None` are ignored, and non-`None` fields are ANDed together to form the final filter.

```python
from dataclasses import dataclass

from timenet.domains import Domain
from timenet.licenses import License
from timenet.tasks import Task


@dataclass(frozen=True)
class QueryCriteria:
    domains: tuple[Domain, ...] | None = None
    tasks: tuple[type[Task], ...] | None = None
    license: License | None = None
    time_series_spec: tuple[str, ...] | None = None
    dataset_ids: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None
```

**Fields**

| Name               | Type                     | Description                                                                                                       |
| ------------------ | ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `domains`          | `tuple[Domain, ...]`     | Match if the dataset's `metadata().domains` shares any value with this tuple.                                     |
| `tasks`            | `tuple[type[Task], ...]` | Match if the dataset's resolved `schema.tasks` includes any of these task classes (e.g. `ClassificationTask`).    |
| `license`          | `License`                | Exact match against `metadata().license`.                                                                         |
| `time_series_spec` | `tuple[str, ...]`        | Match if the dataset's resolved `schema.time_series_specs` declares all of these `TimeSeriesSpec.spec_id` values. |
| `dataset_ids`      | `tuple[str, ...]`        | Match if `metadata().dataset_id` is in this tuple. Use to pin an exact subset by ID.                              |
| `tags`             | `tuple[str, ...]`        | Match if the dataset declares all of these tags in `metadata().tags`.                                             |

#### Schema resolution

`tasks` and `time_series_spec` are **type-filters**: they need the dataset's `DatasetSchema`. The registry resolves it in this precedence order:

1. **Written manifest**: if the dataset has been built, use the `schema` block from its committed `manifest.json`. This is ground truth: it reflects exactly what was written.
2. **Precomputed schema**: else, if the entry's `precomputed_schema` is set, use it.
3. **Unavailable**: else the dataset has no known schema and is **excluded** from any result filtered by `tasks` or `time_series_spec`.

**Example**

```python
from timenet.tasks import ClassificationTask

criteria = QueryCriteria(
    domains=(Domain.CARDIOLOGY,),
    tasks=(ClassificationTask,),
    license=License.CC_BY_4,
    time_series_spec=("ecg_lead",),
)
```
