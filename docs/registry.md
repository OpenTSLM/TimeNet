# Registry

The **registry** is TimeNet's in-memory dataset catalog: a map from dataset id to connector instance. The registry is built once, at client construction time, from one or two YAML files. After that it is read-only.

---

## At a glance

```python
from pathlib import Path

from timenet.connectors.base import DatasetConnector
from timenet.models import DatasetDescriptor, QueryCriteria


class DatasetRegistry:
    def __init__(self, connectors: dict[str, DatasetConnector]) -> None: ...

    @classmethod
    def from_config(
        cls,
        default_path: Path,
        custom_path: Path | None = None,
    ) -> "DatasetRegistry": ...

    def all(self)    -> list[DatasetDescriptor]: ...
    def get(self, dataset_id: str) -> DatasetConnector | None: ...
    def ids(self)    -> list[str]: ...
    def filter(self, criteria: QueryCriteria) -> list[DatasetDescriptor]: ...
```

How the client builds and queries it:

```python

registry = DatasetRegistry.from_config(
    default_path=Path("timenet/registry/default_datasets.yaml"),
    custom_path=Path("datasets.yaml"),     # optional
)

```

The rest of this page explains each method and the loading rules that produce the registry.

---

## Creating the registry

### `DatasetRegistry.from_config(default_path, custom_path=None) -> DatasetRegistry` { data-toc-label='from_config()' }

Builds a registry by reading a default YAML, optionally extending it with a custom YAML, importing every listed connector class dynamically and registering each instance.

- `default_path` : The path to the library's built-in catalog, shipped with TimeNet (`timenet/registry/default_datasets.yaml`). Lists every connector that comes out of the box.

- `custom_path` : Optional. Points at a user-supplied YAML that adds more datasets to the catalog. Same shape as the default.

### YAML shape

```yaml
datasets:
  mit_bih_arrhythmia:
    class: timenet.connectors.MITBIHConnector
  synthetic_ecg:
    class: timenet.connectors.SyntheticECGConnector
```

### Loading sequence

For each entry in the merged `default_path` + `custom_path`:

1. **Import** the dotted class path. Failure → registry load aborts
2. **Instantiate** the class: `instance = Cls()`. A connector that requires constructor args is rejected → load aborts.
3. **Validate identity**: `instance.metadata().dataset_id == yaml_key`. Mismatch → registry load aborts. The connector is the source of truth for its id while the YAML key is the human-readable index
4. **Register** the instance under the key.

### How the YAML becomes a connector instance

Each `class:` value is a dotted Python path (e.g. `timenet.connectors.MITBIHConnector`). The registry resolves it through `importlib`.

```python
import importlib
import yaml


@classmethod
def from_config(
    cls,
    default_path: Path,
    custom_path: Path | None = None,
) -> "DatasetRegistry":
    connectors: dict[str, DatasetConnector] = {}

    for path in (default_path, custom_path):
        if path is None:
            continue
        with open(path) as f:
            config = yaml.safe_load(f)

        for dataset_id, entry in config["datasets"].items():
            # 1. Split dotted path into module + class name.
            module_path, _, class_name = entry["class"].rpartition(".")

            # 2. Dynamically import the module.
            module = importlib.import_module(module_path)

            # 3. Look up the class object.
            connector_cls: type[DatasetConnector] = getattr(module, class_name)

            # 4. Instantiate with no arguments.
            instance: DatasetConnector = connector_cls()

            # 5. Validate that the connector's id matches the YAML key.
            if instance.metadata().dataset_id != dataset_id:
                raise ValueError(
                    f"id mismatch for '{dataset_id}': "
                    f"connector reports '{instance.metadata().dataset_id}'"
                )

            # 6. Reject collisions (default vs custom).
            if dataset_id in connectors:
                raise ValueError(f"duplicate dataset id: {dataset_id}")

            connectors[dataset_id] = instance

    return cls(connectors)
```

---

## Methods

### `all() -> list[DatasetDescriptor]` { data-toc-label='all()' }

Returns every registered dataset's descriptor.

### `get(dataset_id: str) -> DatasetConnector | None` { data-toc-label='get()' }

Returns the connector instance for one dataset id, or `None` if not registered.

### `ids() -> list[str]` { data-toc-label='ids()' }

Just the keys, sorted.

### `filter(criteria: QueryCriteria) -> list[DatasetDescriptor]` { data-toc-label='filter()' }

Returns the descriptors of datasets matching `criteria`.

The filter is implemented by walking `all()` and applying the criteria field-by-field.

---
